"""Shared training utilities: data loading, checkpoints, loaders, history JSON."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

from utils.constants import DEVICE, NOISE_STD, SEED, SOURCE_AFP, SOURCE_UNMANIP, TEST_FRAC, VAL_FRAC


def sanitize_applied_power(applied_power, source=None):
    """AFP / unmanipulated have no continuous RF envelope: power is 0, never NaN/Inf."""
    pwr = np.asarray(applied_power, dtype=np.float64).reshape(-1)
    pwr = np.nan_to_num(pwr, nan=0.0, posinf=0.0, neginf=0.0)
    if source is not None:
        src = np.asarray(source).reshape(-1)
        if src.shape[0] == pwr.shape[0]:
            pwr[(src == SOURCE_AFP) | (src == SOURCE_UNMANIP)] = 0.0
    return pwr.astype(np.float32, copy=False)

def sanitize_power_profile(power_profile, source=None):
    """Per-bin RF envelope: AFP / unmanipulated rows are all zeros; NaN/Inf become 0."""
    arr = np.asarray(power_profile, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if arr.size else arr.reshape(0, 0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if source is not None and arr.ndim == 2:
        src = np.asarray(source).reshape(-1)
        if src.shape[0] == arr.shape[0]:
            arr[(src == SOURCE_AFP) | (src == SOURCE_UNMANIP)] = 0.0
    return arr.astype(np.float32, copy=False)

def resolve_power_profile(arrays):
    """Prefer stored (N, bins) envelope; otherwise broadcast scalar applied_power."""
    spectra = np.asarray(arrays['spectra'])
    n = spectra.shape[0]
    n_bins = spectra.shape[2]
    source = arrays.get('source')
    if 'power_profile' in arrays and arrays['power_profile'] is not None:
        prof = np.asarray(arrays['power_profile'])
        if prof.ndim == 1:
            if prof.shape[0] == n:
                prof = np.broadcast_to(prof.reshape(n, 1), (n, n_bins)).copy()
            elif prof.shape[0] == n_bins:
                prof = np.broadcast_to(prof.reshape(1, n_bins), (n, n_bins)).copy()
            else:
                raise ValueError(f'power_profile length {prof.shape[0]} is neither N={n} nor bins={n_bins}')
        if prof.shape != (n, n_bins):
            raise ValueError(f'power_profile shape {prof.shape} != {(n, n_bins)}')
        return sanitize_power_profile(prof, source)
    scalar = sanitize_applied_power(arrays['applied_power'], source)
    return np.broadcast_to(scalar.reshape(n, 1), (n, n_bins)).astype(np.float32, copy=True)

def load_npz(path):
    """Load spectra from a ``.npz`` file, fully into RAM."""
    path = Path(path)
    required = ('spectra', 'applied_power', 'n_steps', 'P_total', 'Q_total')
    with np.load(path, allow_pickle=False) as raw:
        missing = [k for k in required if k not in raw.files]
        if missing:
            raise KeyError(f'{path}: missing fields {missing}; found {raw.files}')
        spectra = np.asarray(raw['spectra'])
        applied_power = np.asarray(raw['applied_power']).reshape(-1)
        n_steps = np.asarray(raw['n_steps']).reshape(-1)
        p_total = np.asarray(raw['P_total']).reshape(-1)
        q_total = np.asarray(raw['Q_total']).reshape(-1)
        optional = {}
        for key in ('p0', 'center_bin', 'source', 'run_id', 'burn_steps', 'relax_steps'):
            if key in raw.files:
                optional[key] = np.asarray(raw[key]).reshape(-1)
        if 'power_profile' in raw.files:
            optional['power_profile'] = np.asarray(raw['power_profile'])
        if 'rf_dose' in raw.files:
            optional['rf_dose'] = np.asarray(raw['rf_dose'])
    applied_power = sanitize_applied_power(applied_power, optional.get('source'))
    if spectra.ndim != 3 or spectra.shape[1] != 2:
        raise ValueError(f'{path}: expected spectra shape (N, 2, num_bins), got {spectra.shape}')
    n = spectra.shape[0]
    num_bins = spectra.shape[2]
    for (name, arr) in (('applied_power', applied_power), ('n_steps', n_steps), ('P_total', p_total), ('Q_total', q_total)):
        if arr.shape[0] != n:
            raise ValueError(f'{path}: {name} length {arr.shape[0]} != N={n}')
    out = {
        'spectra': spectra.astype(np.float32, copy=False),
        'P_total': p_total,
        'Q_total': q_total,
        'applied_power': applied_power,
        'n_steps': n_steps,
        'num_bins': np.asarray(num_bins, dtype=np.int32),
    }
    out.update(optional)
    out['power_profile'] = resolve_power_profile(out)
    if 'rf_dose' not in out or out['rf_dose'] is None:
        out['rf_dose'] = np.zeros((n, num_bins), dtype=np.float32)
    return out


def _event_noise(seed, event_idx, n_bins, noise_std):
    """Deterministic per-event noise (same event always gets the same noise)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(event_idx)]))
    return rng.normal(0.0, noise_std, size=int(n_bins)).astype(np.float32)


def prepare_datasets(arrays, *, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED, max_samples=None, noise_std=NOISE_STD):
    # ponytail: full-RAM eager load; needs ~20GB for 3M x 500-bin events, chunk again if N grows
    spectra = arrays['spectra']
    n = int(spectra.shape[0])
    n_bins = int(spectra.shape[2])
    if 'power_profile' in arrays and arrays['power_profile'] is not None:
        power_profile = sanitize_power_profile(
            np.asarray(arrays['power_profile']), arrays.get('source'),
        )
    else:
        power_profile = resolve_power_profile(arrays)
    
    rf_dose = arrays.get('rf_dose')
    if rf_dose is None:
        rf_dose = np.zeros((n, n_bins), dtype=np.float32)
    else:
        rf_dose = np.asarray(rf_dose, dtype=np.float32)
    
    burn_steps = arrays.get('burn_steps')
    if burn_steps is None:
        burn_steps = np.zeros(n, dtype=np.int32)
    else:
        burn_steps = np.asarray(burn_steps, dtype=np.int32).reshape(-1)
    
    relax_steps = arrays.get('relax_steps')
    if relax_steps is None:
        relax_steps = np.zeros(n, dtype=np.int32)
    else:
        relax_steps = np.asarray(relax_steps, dtype=np.int32).reshape(-1)
    
    applied_power = sanitize_applied_power(arrays['applied_power'], arrays.get('source'))
    
    y_p = np.asarray(arrays['P_total'], dtype=np.float32).reshape(-1)
    y_q = np.asarray(arrays['Q_total'], dtype=np.float32).reshape(-1)
    p0 = arrays.get('p0')
    if p0 is not None:
        p0 = np.asarray(p0).reshape(-1)
    
    run_id = arrays.get('run_id')
    if run_id is None:
        run_id = np.arange(n, dtype=np.int64)
    else:
        run_id = np.asarray(run_id, dtype=np.int64).reshape(-1)
    
    rng = np.random.default_rng(seed)
    if max_samples is not None and max_samples < n:
        keep = np.sort(rng.choice(n, size=int(max_samples), replace=False))
    else:
        keep = np.arange(n, dtype=np.int64)

    unique_runs = np.unique(run_id[keep])
    n_runs = len(unique_runs)
    perm_runs = rng.permutation(n_runs)
    n_test_runs = max(1, round(n_runs * test_frac))
    n_val_runs = max(1, round(n_runs * val_frac))
    
    test_runs = unique_runs[perm_runs[:n_test_runs]]
    val_runs = unique_runs[perm_runs[n_test_runs:n_test_runs + n_val_runs]]
    train_runs = unique_runs[perm_runs[n_test_runs + n_val_runs:]]
    
    train_mask = np.isin(run_id[keep], train_runs)
    val_mask = np.isin(run_id[keep], val_runs)
    test_mask = np.isin(run_id[keep], test_runs)
    
    local_train = np.flatnonzero(train_mask)
    local_val = np.flatnonzero(val_mask)
    local_test = np.flatnonzero(test_mask)
    
    train_global = keep[local_train]
    val_global = keep[local_val]
    test_global = keep[local_test]

    clean = np.asarray(spectra[keep, 0, :] + spectra[keep, 1, :], dtype=np.float32)
    if noise_std > 0.0:
        noise = np.stack([_event_noise(seed, int(gi), n_bins, noise_std) for gi in keep], axis=0)
        snr_all = spectrum_snr(clean, noise)
        ps = clean + noise
    else:
        snr_all = np.full(keep.size, np.nan, dtype=np.float64)
        ps = clean
    
    max_gamma = 10.0
    pwr = np.asarray(power_profile[keep], dtype=np.float32) / max_gamma
    dose = np.asarray(rf_dose[keep], dtype=np.float32)
    burn_k = burn_steps[keep]
    relax_k = relax_steps[keep]
    peak_gamma = np.asarray(applied_power[keep], dtype=np.float32) / max_gamma
    
    y_p_k = y_p[keep]
    y_q_k = y_q[keep]
    ps_mean = float(ps[local_train].mean())
    ps_std = float(ps[local_train].std())
    ps_std = ps_std if ps_std > 1e-08 else 1.0
    
    dose_mean = float(dose[local_train].mean())
    dose_std = float(dose[local_train].std())
    dose_std = dose_std if dose_std > 1e-08 else 1.0
    
    burn_max = float(burn_k[local_train].max()) if burn_k[local_train].size > 0 else 1.0
    burn_max = burn_max if burn_max > 0.0 else 1.0
    relax_max = float(relax_k[local_train].max()) if relax_k[local_train].size > 0 else 1.0
    relax_max = relax_max if relax_max > 0.0 else 1.0
    
    snr_test = snr_all[local_test]

    p_mean = float(y_p[train_global].mean())
    p_std = float(y_p[train_global].std())
    p_std = p_std if p_std > 1e-08 else 1.0
    q_mean = float(y_q[train_global].mean())
    q_std = float(y_q[train_global].std())
    q_std = q_std if q_std > 1e-08 else 1.0

    def _pack(indices):
        t = ps.shape[1]
        ps_n = (ps[indices] - ps_mean) / ps_std
        pwr_n = pwr[indices]
        dose_n = (dose[indices] - dose_mean) / dose_std
        burn_n = burn_k[indices].astype(np.float32) / burn_max
        relax_n = relax_k[indices].astype(np.float32) / relax_max
        peak_g = peak_gamma[indices].astype(np.float32)
        burn_seq = np.broadcast_to(burn_n.reshape(-1, 1), (indices.size, t))
        relax_seq = np.broadcast_to(relax_n.reshape(-1, 1), (indices.size, t))
        peak_seq = np.broadcast_to(peak_g.reshape(-1, 1), (indices.size, t))
        x = np.stack([ps_n, pwr_n, dose_n, burn_seq, relax_seq, peak_seq], axis=-1).astype(np.float32)
        yp = ((y_p_k[indices] - p_mean) / p_std).astype(np.float32)
        yq = ((y_q_k[indices] - q_mean) / q_std).astype(np.float32)
        return data.TensorDataset(torch.from_numpy(x), torch.from_numpy(yp), torch.from_numpy(yq))
    train_ds = _pack(local_train)
    val_ds = _pack(local_val)
    test_ds = _pack(local_test)

    p0_test = (p0[test_global] if p0 is not None else y_p[test_global])
    stats = {
        'ps_mean': ps_mean, 'ps_std': ps_std, 'dose_mean': dose_mean, 'dose_std': dose_std,
        'burn_max': burn_max, 'relax_max': relax_max, 'P_mean': p_mean, 'P_std': p_std,
        'Q_mean': q_mean, 'Q_std': q_std, 'input_size': 6, 'num_bins': n_bins,
        'n_train': int(train_global.size), 'n_val': int(val_global.size), 'n_test': int(test_global.size),
        'train_idx': train_global, 'val_idx': val_global, 'test_idx': test_global,
        'p0_test': p0_test, 'noise_std': noise_std, 'snr_test': snr_test,
    }
    return (train_ds, val_ds, test_ds, stats)


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
