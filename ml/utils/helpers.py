"""Shared training utilities: data loading, checkpoints, loaders, history JSON, P/Q fit loop."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from utils.constants import (
    DEVICE, LR_MIN, MAX_GRAD_NORM, MIN_DELTA, NOISE_STD, REL_LOSS_EPS,
    RESTART_LR_DECAY, RESTART_WARMUP_EPOCHS, SEED, SOURCE_AFP, SOURCE_AFP_PROFILE,
    SOURCE_PROFILE, SOURCE_SSRF, SOURCE_UNMANIP, T_0, T_MULT, TEST_FRAC, VAL_FRAC,
    WEIGHT_DECAY,
)

_MANIPULATION_SOURCE = {
    'ssrf': SOURCE_SSRF,
    'afp': SOURCE_AFP,
    'afp_profile': SOURCE_AFP_PROFILE,
    'profile': SOURCE_PROFILE,
    'states_only': SOURCE_UNMANIP,
}


def sanitize_n_steps(n_steps, _source):
    """A missing duration is 0 steps, never NaN. ``int(nan)`` aborts example plots."""
    steps = np.array(n_steps, dtype=np.float64, copy=True).reshape(-1)
    steps[~np.isfinite(steps)] = 0.0
    return steps

def sanitize_applied_power(applied_power, source):
    """AFP / unmanipulated have no continuous RF envelope: power is 0, never NaN/Inf."""
    pwr = np.asarray(applied_power, dtype=np.float64).reshape(-1)
    pwr = np.nan_to_num(pwr, nan=0.0, posinf=0.0, neginf=0.0)
    src = np.asarray(source).reshape(-1)
    pwr[(src == SOURCE_AFP) | (src == SOURCE_UNMANIP)] = 0.0
    return pwr.astype(np.float32, copy=False)

def sanitize_power_profile(power_profile, source):
    """Per-bin RF envelope: AFP / unmanipulated rows are all zeros; NaN/Inf become 0."""
    arr = np.nan_to_num(np.asarray(power_profile, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    src = np.asarray(source).reshape(-1)
    arr[(src == SOURCE_AFP) | (src == SOURCE_UNMANIP)] = 0.0
    return arr.astype(np.float32, copy=False)

def resolve_power_profile(arrays):
    return sanitize_power_profile(arrays['power_profile'], arrays['source'])

def _arrays_from_run5(raw):
    """Map ``run5.npz`` names onto the ``spectra_final_v3`` arrays. Ps only reaches the model."""
    plus = np.asarray(raw['spectrum_plus'], dtype=np.float32)
    minus = np.asarray(raw['spectrum_minus'], dtype=np.float32)
    powers = np.asarray(raw['powers'], dtype=np.float32)
    manip = [str(value) for value in np.asarray(raw['manipulation']).reshape(-1)]
    unknown = sorted({name for name in manip if name not in _MANIPULATION_SOURCE})
    if unknown:
        raise ValueError(f'unknown manipulation values: {unknown}')
    source = np.array([_MANIPULATION_SOURCE[name] for name in manip], dtype=np.int64)
    return {
        'spectra': np.stack((plus, minus), axis=1),
        'power_profile': powers,
        'applied_power': powers.max(axis=1).astype(np.float32, copy=False),
        'n_steps': np.asarray(raw['duration'], dtype=np.float64).reshape(-1),
        'p0': np.asarray(raw['p0']).reshape(-1),
        'P_total': np.asarray(raw['final_P']).reshape(-1),
        'Q_total': np.asarray(raw['final_Q']).reshape(-1),
        'source': source,
    }

def load_npz(path):
    """Load spectra from a ``.npz`` file, fully into RAM.

    ``run5.npz`` uses ``spectrum_plus`` / ``spectrum_minus`` and ``final_P`` /
    ``final_Q`` for the same fields as ``spectra`` and ``P_total`` / ``Q_total``.
    """
    flat = ('spectra', 'power_profile')
    columns = (
        'p0', 'center_bin', 'source', 'burn_steps', 'relax_steps',
        'applied_power', 'n_steps', 'P_total', 'Q_total',
    )
    with np.load(path, allow_pickle=False) as raw:
        if 'spectra' in raw.files:
            out = {k: np.asarray(raw[k]) for k in flat}
            out.update({k: np.asarray(raw[k]).reshape(-1) for k in columns if k in raw.files})
        else:
            out = _arrays_from_run5(raw)
    out['spectra'] = out['spectra'].astype(np.float32, copy=False)
    if 'n_steps' in out:
        out['n_steps'] = sanitize_n_steps(out['n_steps'], out['source'])
    out['applied_power'] = sanitize_applied_power(out['applied_power'], out['source'])
    out['power_profile'] = resolve_power_profile(out)
    return out


def _event_noise(seed, event_idx, n_bins, noise_std):
    """Deterministic per-event noise (same event always gets the same noise)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(event_idx)]))
    return rng.normal(0.0, noise_std, size=int(n_bins)).astype(np.float32)


def _with_noise_copies(clean_rows, base_rows, snr_rows, *, copies, noise_std, seed, stream):
    """Stack ``copies`` noisy spectra per row. Copy 0 is ``base_rows``; later copies are fresh draws."""
    copies = int(copies)
    n_rows, n_bins = base_rows.shape
    if copies == 1:
        return base_rows, np.asarray(snr_rows, dtype=np.float64).reshape(-1)
    aug_rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))
    extra = aug_rng.normal(0.0, float(noise_std), size=(n_rows, copies - 1, n_bins)).astype(np.float32)
    rows = np.empty((n_rows, copies, n_bins), dtype=np.float32)
    rows[:, 0, :] = base_rows
    rows[:, 1:, :] = clean_rows[:, None, :] + extra
    snr_extra = np.stack(
        [spectrum_snr(clean_rows, extra[:, k, :]) for k in range(copies - 1)],
        axis=1,
    )
    snr = np.concatenate([np.asarray(snr_rows, dtype=np.float64).reshape(n_rows, 1), snr_extra], axis=1)
    return rows.reshape(-1, n_bins), snr.reshape(-1)


def prepare_datasets(arrays, *, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED, max_samples=None, noise_std=NOISE_STD, noise_copies=1):
    # ponytail: full-RAM eager load; needs ~20GB for 3M x 500-bin events, chunk again if N grows
    spectra = arrays['spectra']
    n = int(spectra.shape[0])
    n_bins = int(spectra.shape[2])
    power_profile = sanitize_power_profile(arrays['power_profile'], arrays['source'])
    n_steps = np.asarray(arrays['n_steps']).reshape(-1)
    y_p = np.asarray(arrays['P_total'], dtype=np.float32).reshape(-1)
    y_q = np.asarray(arrays['Q_total'], dtype=np.float32).reshape(-1)
    p0 = np.asarray(arrays['p0']).reshape(-1)

    rng = np.random.default_rng(seed)
    if max_samples is not None and max_samples < n:
        keep = np.sort(rng.choice(n, size=int(max_samples), replace=False))
    else:
        keep = np.arange(n, dtype=np.int64)

    n_keep = int(keep.size)
    perm = rng.permutation(n_keep)
    n_test = max(1, round(n_keep * test_frac))
    n_val = max(1, round(n_keep * val_frac))
    local_test = np.sort(perm[:n_test])
    local_val = np.sort(perm[n_test:n_test + n_val])
    local_train = np.sort(perm[n_test + n_val:])
    
    train_global = keep[local_train]

    clean = np.asarray(spectra[keep, 0, :] + spectra[keep, 1, :], dtype=np.float32)
    if noise_std > 0.0:
        noise = np.stack([_event_noise(seed, int(gi), n_bins, noise_std) for gi in keep], axis=0)
        snr_all = spectrum_snr(clean, noise)
        ps = clean + noise
    else:
        snr_all = np.full(keep.size, np.nan, dtype=np.float64)
        ps = clean
    
    pwr = np.asarray(power_profile[keep], dtype=np.float32)
    pwr_mean = float(pwr[local_train].mean())
    pwr_std = float(pwr[local_train].std())
    pwr_std = pwr_std if pwr_std > 1e-08 else 1.0

    n_steps_k = n_steps[keep]
    y_p_k = y_p[keep]
    y_q_k = y_q[keep]
    copies = int(noise_copies)
    if copies < 1:
        raise ValueError('noise_copies must be >= 1')
    if noise_std <= 0.0:
        copies = 1

    def _expand(local_idx, stream):
        ps_rows, snr_rows = _with_noise_copies(
            clean[local_idx], ps[local_idx], snr_all[local_idx],
            copies=copies, noise_std=noise_std, seed=seed, stream=stream,
        )
        return ps_rows, np.repeat(local_idx, copies), snr_rows

    # stream 1 is the historical train draw; 2 and 3 keep val/test from shifting it
    train_ps, train_local, _train_snr = _expand(local_train, 1)
    val_ps, val_local, _val_snr = _expand(local_val, 2)
    test_ps, test_local, snr_test = _expand(local_test, 3)
    val_global = keep[val_local]
    test_global = keep[test_local]

    ps_mean = float(train_ps.mean())
    ps_std = float(train_ps.std())
    ps_std = ps_std if ps_std > 1e-08 else 1.0

    steps_mean = float(n_steps[train_global].mean())
    steps_std = float(n_steps[train_global].std())
    steps_std = steps_std if steps_std > 1e-08 else 1.0
    p_mean = float(y_p[train_global].mean())
    p_std = float(y_p[train_global].std())
    p_std = p_std if p_std > 1e-08 else 1.0
    q_mean = float(y_q[train_global].mean())
    q_std = float(y_q[train_global].std())
    q_std = q_std if q_std > 1e-08 else 1.0

    def _pack(ps_rows, indices):
        t = ps_rows.shape[1]
        ps_n = (ps_rows - ps_mean) / ps_std
        pwr_n = (pwr[indices] - pwr_mean) / pwr_std
        steps_n = (n_steps_k[indices] - steps_mean) / steps_std
        steps_seq = np.broadcast_to(steps_n.reshape(-1, 1), (indices.size, t))
        x = np.stack([ps_n, pwr_n, steps_seq], axis=-1).astype(np.float32)
        yp = ((y_p_k[indices] - p_mean) / p_std).astype(np.float32)
        yq = ((y_q_k[indices] - q_mean) / q_std).astype(np.float32)
        return data.TensorDataset(torch.from_numpy(x), torch.from_numpy(yp), torch.from_numpy(yq))
    train_ds = _pack(train_ps, train_local)
    val_ds = _pack(val_ps, val_local)
    test_ds = _pack(test_ps, test_local)

    p0_test = p0[test_global]
    stats = {
        'ps_mean': ps_mean, 'ps_std': ps_std, 'pwr_mean': pwr_mean, 'pwr_std': pwr_std,
        'steps_mean': steps_mean, 'steps_std': steps_std, 'P_mean': p_mean, 'P_std': p_std,
        'Q_mean': q_mean, 'Q_std': q_std, 'input_size': 3, 'num_bins': n_bins,
        'n_train': int(train_ps.shape[0]), 'n_val': int(val_ps.shape[0]), 'n_test': int(test_ps.shape[0]),
        'train_idx': train_global, 'val_idx': val_global, 'test_idx': test_global,
        'p0_test': p0_test, 'noise_std': noise_std, 'noise_copies': copies, 'snr_test': snr_test,
    }
    return (train_ds, val_ds, test_ds, stats)


def keep_ps_channel(train_ds, val_ds, test_ds, stats):
    """Drop power_profile and n_steps; keep only the Ps channel (index 0)."""
    def _one(ds):
        (x, yp, yq) = ds.tensors
        return data.TensorDataset(x[..., :1].contiguous(), yp, yq)

    return (_one(train_ds), _one(val_ds), _one(test_ds), {**stats, 'input_size': 1})


def keep_ps_and_power_profile(train_ds, val_ds, test_ds, stats):
    """Drop n_steps; keep Ps and power_profile (channels 0 and 1)."""
    def _one(ds):
        (x, yp, yq) = ds.tensors
        return data.TensorDataset(x[..., :2].contiguous(), yp, yq)

    return (_one(train_ds), _one(val_ds), _one(test_ds), {**stats, 'input_size': 2})


def keep_ps_and_n_steps(train_ds, val_ds, test_ds, stats):
    """Drop power_profile; keep Ps and n_steps (channels 0 and 2)."""
    def _one(ds):
        (x, yp, yq) = ds.tensors
        kept = torch.cat([x[..., 0:1], x[..., 2:3]], dim=-1).contiguous()
        return data.TensorDataset(kept, yp, yq)

    return (_one(train_ds), _one(val_ds), _one(test_ds), {**stats, 'input_size': 2})


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}

def spectrum_snr(clean_ps, noise):
    """Per-spectrum SNR: max(clean Ps) / max(injected noise) along frequency."""
    signal_peak = np.max(np.asarray(clean_ps, dtype=np.float64), axis=-1)
    noise_peak = np.max(np.asarray(noise, dtype=np.float64), axis=-1)
    snr = np.full(np.shape(signal_peak), np.nan, dtype=np.float64)
    valid = noise_peak > 0.0
    snr[valid] = signal_peak[valid] / noise_peak[valid]
    return snr

def _stats_for_checkpoint(stats):
    skip = {'train_idx', 'val_idx', 'test_idx', 'p0_test', 'snr_test'}
    return {k: v for (k, v) in stats.items() if k not in skip}

def make_pq_loaders(train_dataset, val_dataset, batch_size):
    kw = {'num_workers': 0, 'pin_memory': DEVICE.type == 'cuda'}
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kw,
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kw,
    )
    return (train_loader, val_loader)


def parse_int_tuple(text):
    values = tuple(int(part.strip()) for part in str(text).split(',') if part.strip())
    if not values:
        raise argparse.ArgumentTypeError('expected a comma-separated list of integers')
    return values


def reload_checkpoint(load_fn, checkpoint_path, device):
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        return None
    model, ckpt = load_fn(checkpoint_path, device=device)
    return (model, ckpt.get('best_val_loss'))

def json_ready(obj):
    """Convert numpy scalars/arrays so json.dump does not raise TypeError."""
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for (k, v) in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj

def save_history_json(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: [x for x in v] for (k, v) in history.items()}
    with path.open('w', encoding='utf-8') as f:
        json.dump(json_ready(payload), f, indent=2)

def load_history_json(path):
    if not path.is_file():
        return None
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or 'train_loss' not in raw:
        return None
    return {key: [x for x in values] for (key, values) in raw.items() if isinstance(values, list)}

def relative_weighted_loss(pred_z, true_z, *, mean, std, eps=REL_LOSS_EPS):
    pred = pred_z * std + mean
    true = true_z * std + mean
    return torch.mean(torch.abs(pred - true) / (torch.abs(true) + eps))

def _step_cosine_restart_scheduler(scheduler, optimizer, *, init_lr, cycle_state, decay=RESTART_LR_DECAY, warmup_epochs=RESTART_WARMUP_EPOCHS, eta_min=LR_MIN):
    """Step CosineAnnealingWarmRestarts with restart LR decay + warmup + Adam reset.

    - Detects a warm restart via ``scheduler.T_cur == 0`` right after ``step()``.
    - Decays the cycle max LR as ``init_lr * decay**cycle`` so late restarts
      (e.g. epoch 310) kick far less than early ones.
    - Clears Adam/AdamW moments at each restart (stale 2nd moments + fresh
      full LR = overshoot into Inf/NaN).
    - Linearly warms LR back up over ``warmup_epochs`` instead of jumping
      straight to max on the first step of a new cycle.
    - Returns ``(lr, restarted)``. ``cycle_state`` is a ``dict(cycle=int,
      base=float)`` mutated in place so train/eval loops stay stateless.
    """
    scheduler.step()
    restarted = bool(scheduler.T_cur == 0)
    if restarted:
        cycle_state['cycle'] += 1
        cycle_state['base'] = float(init_lr * (decay ** cycle_state['cycle']))
        scheduler.base_lrs = [cycle_state['base']] * len(scheduler.base_lrs)
        for pg in optimizer.param_groups:
            pg['lr'] = cycle_state['base']
        optimizer.state.clear()
    if warmup_epochs and warmup_epochs > 0 and cycle_state['cycle'] > 0 and scheduler.T_cur < warmup_epochs:
        warm_lr = float(eta_min + (cycle_state['base'] - eta_min) * (scheduler.T_cur + 1) / warmup_epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = warm_lr
    return optimizer.param_groups[0]['lr'], restarted

def fit_pq_model(model, train_dataset, val_dataset, stats, *, num_epochs, batch_size, learning_rate, on_best=None, load_best=None, device=DEVICE, restart_decay=RESTART_LR_DECAY, warmup_epochs=RESTART_WARMUP_EPOCHS, weight_decay=WEIGHT_DECAY, hold_lr=None):
    """Shared P/Q loop used by the other model scripts (no non-finite guards; see train_model).

    ``hold_lr`` cosines from ``learning_rate`` down to that floor over the first
    cycle and then stays there. The default ``None`` keeps warm restarts.
    """
    (train_loader, val_loader) = make_pq_loaders(train_dataset, val_dataset, batch_size)
    p_mean, p_std = stats['P_mean'], stats['P_std']
    q_mean, q_std = stats['Q_mean'], stats['Q_std']
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if hold_lr is None:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=LR_MIN)
        cycle_state = {'cycle': 0, 'base': float(learning_rate)}
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_0, eta_min=float(hold_lr))
        cycle_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
    best_val = float('inf')
    best_epoch = -1
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for (x_b, y_p, y_q) in train_loader:
            x_b, y_p, y_q = x_b.to(device), y_p.to(device), y_q.to(device)
            (pred_p, pred_q) = model(x_b)
            loss = relative_weighted_loss(pred_p, y_p, mean=p_mean, std=p_std) + relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            train_sum += loss.item()
            train_batches += 1
        avg_train = train_sum / max(train_batches, 1)
        model.eval()
        val_sum = 0.0
        val_batches = 0
        rpe_p_sum = 0.0
        rpe_q_sum = 0.0
        with torch.no_grad():
            for (x_v, y_p, y_q) in val_loader:
                x_v, y_p, y_q = x_v.to(device), y_p.to(device), y_q.to(device)
                (pred_p, pred_q) = model(x_v)
                loss_p = relative_weighted_loss(pred_p, y_p, mean=p_mean, std=p_std)
                loss_q = relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
                val_sum += (loss_p + loss_q).item()
                rpe_p_sum += loss_p.item()
                rpe_q_sum += loss_q.item()
                val_batches += 1
        avg_val = val_sum / max(val_batches, 1)
        rpe_p = rpe_p_sum / max(val_batches, 1) * 100.0
        rpe_q = rpe_q_sum / max(val_batches, 1) * 100.0
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_p_rpe'].append(rpe_p)
        history['val_q_rpe'].append(rpe_q)
        if hold_lr is None:
            lr, restarted = _step_cosine_restart_scheduler(scheduler, optimizer, init_lr=learning_rate, cycle_state=cycle_state, decay=restart_decay, warmup_epochs=warmup_epochs)
        elif epoch < T_0:
            scheduler.step()
            lr, restarted = optimizer.param_groups[0]['lr'], False
        else:
            held = float(hold_lr)
            for group in optimizer.param_groups:
                group['lr'] = held
            lr, restarted = held, False
        restart_note = ' | restart' if restarted else ''
        print(f'epoch {epoch + 1:03d}/{num_epochs} | train {avg_train:.6f} | val {avg_val:.6f} | RPE% P={rpe_p:.3f} Q={rpe_q:.3f} | lr {lr:.2e}{restart_note}', flush=True)
        if avg_val < best_val - MIN_DELTA:
            best_val = avg_val
            best_epoch = epoch + 1
            best_state = clone_state_dict(model)
            if on_best is not None:
                on_best(model, best_val, best_epoch)
    if load_best is not None:
        loaded = load_best()
        if loaded is not None:
            model, loaded_val = loaded
            if loaded_val is not None:
                best_val = loaded_val
        elif best_state is not None:
            model.load_state_dict(best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return (model, best_val, history)
