"""
LSTM sequence model: manipulated Ps spectrum → scalar P_total and Q_total.

Input sequence (over frequency bins):
  [Ps, power_profile, n_steps]  (n_steps is broadcast to every bin)

``power_profile`` is the per-bin RF envelope from the NPZ (ssRF Voigt,
AFP zeros, optimal U(R)). Scalar ``applied_power`` is kept for diagnostics
and as a fallback if ``power_profile`` is missing.

Targets:
  P_total, Q_total from ``spectra.npz`` (population n+−n− / n+−2n0+n−)

Usage:
  python ml/lstm.py --spectra ml/data/spectra.npz --max-samples 20000 --epochs 30
  python ml/lstm.py --test-only --checkpoint ml/lstm_results/lstm_best.pth
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SPECTRA_PATH = SCRIPT_DIR / 'data' / 'spectra_v6.npz'
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'lstm_result_v9'
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 15
MIN_DELTA = 1e-06
T_0 = 10
T_MULT = 2
LR_MIN = 1e-07
MAX_GRAD_NORM = 1.0
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
VAL_FRAC = 0.15
TEST_FRAC = 0.15
REL_LOSS_EPS = 0.0001
RPE_ABS_EPS = 1e-10
NOISE_STD = 0.01
N_EXAMPLE_PLOTS = 36
POL_ABS_BANDS = tuple(((lo / 100.0, (lo + 5) / 100.0) for lo in range(5, 95, 5)))
SOURCE_SSRF = 0
SOURCE_AFP = 1
SOURCE_PROFILE = 3
SOURCE_AFP_PROFILE = 4
SOURCE_NAME = {SOURCE_SSRF: 'ssRF', SOURCE_AFP: 'AFP', SOURCE_PROFILE: 'optimal profile', SOURCE_AFP_PROFILE: 'AFP Profile'}
SPECTRUM_R_MIN = -6.0
SPECTRUM_R_MAX = 6.0

class LstmModel(nn.Module):

    def __init__(self, input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        enc_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=enc_dropout, device=DEVICE)
        trunk_dim = 2 * self.hidden_size
        self.head_p = nn.Linear(trunk_dim, 1)
        self.head_q = nn.Linear(trunk_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        (enc_out, _) = self.encoder(x)
        ctx = enc_out.mean(dim=1)
        return (self.head_p(ctx).squeeze(-1), self.head_q(ctx).squeeze(-1))

def resolve_spectra_path(spectra):
    if spectra is not None:
        path = Path(spectra)
        if path.is_file():
            return path
        raise FileNotFoundError(f'Spectra NPZ not found: {path}')
    raise FileNotFoundError(f'Spectra NPZ not found: {path}')

def sanitize_applied_power(applied_power, source=None):
    """AFP has no continuous RF envelope: power is 0, never NaN/Inf."""
    pwr = np.asarray(applied_power, dtype=np.float64).reshape(-1)
    pwr = np.nan_to_num(pwr, nan=0.0, posinf=0.0, neginf=0.0)
    if source is not None:
        src = np.asarray(source).reshape(-1)
        if src.shape[0] == pwr.shape[0]:
            pwr[src == SOURCE_AFP] = 0.0
    return pwr.astype(np.float32, copy=False)

def sanitize_power_profile(power_profile, source=None):
    """Per-bin RF envelope: AFP rows are all zeros; NaN/Inf become 0."""
    arr = np.asarray(power_profile, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1) if arr.size else arr.reshape(0, 0)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if source is not None and arr.ndim == 2:
        src = np.asarray(source).reshape(-1)
        if src.shape[0] == arr.shape[0]:
            arr[src == SOURCE_AFP] = 0.0
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

def load_lstm_npz(path):
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
        for key in ('p0', 'center_bin', 'source'):
            if key in raw.files:
                optional[key] = np.asarray(raw[key]).reshape(-1)
        if 'power_profile' in raw.files:
            optional['power_profile'] = np.asarray(raw['power_profile'])
    applied_power = sanitize_applied_power(applied_power, optional.get('source'))
    if spectra.ndim != 3 or spectra.shape[1] != 2:
        raise ValueError(f'{path}: expected spectra shape (N, 2, num_bins), got {spectra.shape}')
    n = spectra.shape[0]
    num_bins = spectra.shape[2]
    for (name, arr) in (('applied_power', applied_power), ('n_steps', n_steps), ('P_total', p_total), ('Q_total', q_total)):
        if arr.shape[0] != n:
            raise ValueError(f'{path}: {name} length {arr.shape[0]} != N={n}')
    out = {'spectra': spectra.astype(np.float32, copy=False), 'P_total': p_total, 'Q_total': q_total, 'applied_power': applied_power, 'n_steps': n_steps, 'num_bins': np.asarray(num_bins, dtype=np.int32)}
    out.update(optional)
    out['power_profile'] = resolve_power_profile(out)
    return out

def prepare_datasets(arrays, *, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED, max_samples=None, noise_std=NOISE_STD):
    spectra = np.asarray(arrays['spectra'])
    ps = np.asarray(spectra[:, 0, :] + spectra[:, 1, :], dtype=np.float32, copy=True)
    applied_power = sanitize_applied_power(arrays['applied_power'], arrays.get('source'))
    power_profile = resolve_power_profile(arrays)
    n_steps = np.asarray(arrays['n_steps']).reshape(-1)
    y_p = np.asarray(arrays['P_total']).reshape(-1)
    y_q = np.asarray(arrays['Q_total']).reshape(-1)
    p0 = arrays.get('p0')
    n = ps.shape[0]
    rng = np.random.default_rng(seed)
    orig_idx = np.arange(n, dtype=np.int64)
    if max_samples is not None and max_samples < n:
        keep = rng.choice(n, size=max_samples, replace=False)
        ps = ps[keep]
        applied_power = applied_power[keep]
        power_profile = power_profile[keep]
        n_steps = n_steps[keep]
        y_p = y_p[keep]
        y_q = y_q[keep]
        if p0 is not None:
            p0 = np.asarray(p0).reshape(-1)[keep]
        orig_idx = orig_idx[keep]
        n = ps.shape[0]
    if noise_std > 0.0:
        noise = rng.normal(0.0, noise_std, size=ps.shape).astype(np.float32)
        snr_all = spectrum_snr(ps, noise)
        ps = ps + noise
    else:
        snr_all = np.full(n, np.nan, dtype=np.float64)
    perm = rng.permutation(n)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    local_test = perm[:n_test]
    local_val = perm[n_test:n_test + n_val]
    local_train = perm[n_test + n_val:]
    test_idx = orig_idx[local_test]
    val_idx = orig_idx[local_val]
    train_idx = orig_idx[local_train]
    ps_mean = ps[local_train].mean()
    ps_std = ps[local_train].std()
    ps_std = ps_std if ps_std > 1e-08 else 1.0
    pwr_mean = power_profile[local_train].mean()
    pwr_std = power_profile[local_train].std()
    pwr_std = pwr_std if pwr_std > 1e-08 else 1.0
    steps_mean = n_steps[local_train].mean()
    steps_std = n_steps[local_train].std()
    steps_std = steps_std if steps_std > 1e-08 else 1.0
    p_mean = y_p[local_train].mean()
    p_std = y_p[local_train].std()
    p_std = p_std if p_std > 1e-08 else 1.0
    q_mean = y_q[local_train].mean()
    q_std = y_q[local_train].std()
    q_std = q_std if q_std > 1e-08 else 1.0

    def _pack(indices):
        t = ps.shape[1]
        ps_n = (ps[indices] - ps_mean) / ps_std
        pwr_n = (power_profile[indices] - pwr_mean) / pwr_std
        steps_n = (n_steps[indices] - steps_mean) / steps_std
        steps_seq = np.broadcast_to(steps_n.reshape(-1, 1), (indices.size, t))
        x = np.stack([ps_n, pwr_n, steps_seq], axis=-1).astype(np.float32)
        yp = ((y_p[indices] - p_mean) / p_std).astype(np.float32)
        yq = ((y_q[indices] - q_mean) / q_std).astype(np.float32)
        return data.TensorDataset(torch.from_numpy(x), torch.from_numpy(yp), torch.from_numpy(yq))
    stats = {'ps_mean': ps_mean, 'ps_std': ps_std, 'pwr_mean': pwr_mean, 'pwr_std': pwr_std, 'steps_mean': steps_mean, 'steps_std': steps_std, 'P_mean': p_mean, 'P_std': p_std, 'Q_mean': q_mean, 'Q_std': q_std, 'input_size': 3, 'num_bins': ps.shape[1], 'n_train': local_train.size, 'n_val': local_val.size, 'n_test': local_test.size, 'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx, 'p0_test': (p0 if p0 is not None else y_p)[local_test], 'noise_std': noise_std, 'snr_test': snr_all[local_test]}
    return (_pack(local_train), _pack(local_val), _pack(local_test), stats)

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

def relative_weighted_loss(pred_z, true_z, *, mean, std, eps=REL_LOSS_EPS):
    pred = pred_z * std + mean
    true = true_z * std + mean
    return torch.mean(torch.abs(pred - true) / (torch.abs(true) + eps))

def compute_rpe(pred, true):
    pred_a = np.asarray(pred).reshape(-1)
    true_a = np.asarray(true).reshape(-1)
    rpe = np.full_like(true_a, np.nan)
    valid = np.abs(true_a) > RPE_ABS_EPS
    rpe[valid] = np.abs(pred_a[valid] - true_a[valid]) / np.abs(true_a[valid]) * 100.0
    return rpe

def _summary_stats(values):
    v = np.asarray(values).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {'n': 0.0, 'mean': 'nan', 'median': 'nan', 'std': 'nan', 'min': 'nan', 'max': 'nan', 'p25': 'nan', 'p75': 'nan'}
    return {'n': v.size, 'mean': np.mean(v), 'median': np.median(v), 'std': np.std(v), 'min': np.min(v), 'max': np.max(v), 'p25': np.percentile(v, 25), 'p75': np.percentile(v, 75)}

def polarization_range_stats(pred_p, true_p, pred_q, true_q, *, pol_ref, bands=POL_ABS_BANDS, ref_name='abs_P', snr=None):
    pred_p = np.asarray(pred_p).reshape(-1)
    true_p = np.asarray(true_p).reshape(-1)
    pred_q = np.asarray(pred_q).reshape(-1)
    true_q = np.asarray(true_q).reshape(-1)
    ref = np.abs(np.asarray(pol_ref).reshape(-1))
    rpe_p = compute_rpe(pred_p, true_p)
    rpe_q = compute_rpe(pred_q, true_q)
    res_p = pred_p - true_p
    res_q = pred_q - true_q
    rows = []
    for (lo, hi) in bands:
        mask = (ref >= lo) & (ref < hi) if hi < 0.999 else (ref >= lo) & (ref <= hi)
        label = f'{round(lo * 100)}-{round(hi * 100)}%'
        p_rpe = _summary_stats(rpe_p[mask])
        q_rpe = _summary_stats(rpe_q[mask])
        p_res = _summary_stats(res_p[mask])
        q_res = _summary_stats(res_q[mask])
        row = {'ref': ref_name, 'range': label, 'lo': float(lo), 'hi': float(hi), 'n': int(np.sum(mask)), 'P_rpe_mean': p_rpe['mean'], 'P_rpe_median': p_rpe['median'], 'P_rpe_std': p_rpe['std'], 'P_rpe_p25': p_rpe['p25'], 'P_rpe_p75': p_rpe['p75'], 'P_residual_mean': p_res['mean'], 'P_residual_median': p_res['median'], 'P_residual_std': p_res['std'], 'Q_rpe_mean': q_rpe['mean'], 'Q_rpe_median': q_rpe['median'], 'Q_rpe_std': q_rpe['std'], 'Q_rpe_p25': q_rpe['p25'], 'Q_rpe_p75': q_rpe['p75'], 'Q_residual_mean': q_res['mean'], 'Q_residual_median': q_res['median'], 'Q_residual_std': q_res['std']}
        if snr is not None:
            snr_s = _summary_stats(np.asarray(snr, dtype=np.float64).reshape(-1)[mask])
            row['snr_mean'] = float(snr_s['mean'])
            row['snr_median'] = float(snr_s['median'])
            row['snr_std'] = float(snr_s['std'])
        rows.append(row)
    return rows

def print_snr_stats(metrics):
    """Print test-set SNR: max(clean Ps) / max(injected noise), per spectrum."""
    if 'snr_median' not in metrics:
        return
    median = float(metrics['snr_median'])
    mean = float(metrics['snr_mean'])
    std = float(metrics['snr_std'])
    if not (np.isfinite(median) and np.isfinite(mean)):
        print('SNR  n/a (no injected noise)', flush=True)
        return
    print(f'SNR (max clean Ps / max injected noise)  median={median:.3f}  mean={mean:.3f}  std={std:.3f}', flush=True)

def print_range_stats_table(rows, *, title):
    print(f'\n===== {title} =====', flush=True)
    has_snr = bool(rows) and 'snr_median' in rows[0]
    header = f"{'range':>10} {'n':>5} {'P_RPE_med':>10} {'P_RPE_mean':>10} {'P_RPE_std':>10} {'P_res_mean':>11} {'P_res_std':>10} {'Q_RPE_med':>10} {'Q_RPE_mean':>10} {'Q_RPE_std':>10} {'Q_res_mean':>11} {'Q_res_std':>10}"
    if has_snr:
        header += f" {'SNR_med':>10} {'SNR_mean':>10}"
    print(header, flush=True)
    print('-' * len(header), flush=True)
    for row in rows:
        if row['n'] == 0:
            continue
        line = f"{row['range']:>10} {row['n']:5d} {row['P_rpe_median']:10.3f} {row['P_rpe_mean']:10.3f} {row['P_rpe_std']:10.3f} {row['P_residual_mean']:11.5e} {row['P_residual_std']:10.5e} {row['Q_rpe_median']:10.3f} {row['Q_rpe_mean']:10.3f} {row['Q_rpe_std']:10.3f} {row['Q_residual_mean']:11.5e} {row['Q_residual_std']:10.5e}"
        if has_snr:
            line += f" {row['snr_median']:10.3f} {row['snr_mean']:10.3f}"
        print(line, flush=True)

def save_range_stats_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open('w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(('' if row[k] is None or (isinstance(row[k], float) and (not np.isfinite(row[k]))) else str(row[k]) for k in keys)) + '\n')

def _csv_cell(value):
    if value is None:
        return ''
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ''
        return f'{float(value):.8g}'
    return str(int(value) if isinstance(value, (np.integer, int)) else value)

def save_example_info_csv(rows, path):
    """Write per-example metadata that used to appear under the example plots."""
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open('w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join((_csv_cell(row.get(k)) for k in keys)) + '\n')
    return path

def save_predictions_csv(metrics, path, *, test_idx=None, p0=None, source=None, n_steps=None, applied_power=None):
    """Write per-event true/pred P and Q (plus RPE) for the evaluated test set."""
    true_p = np.asarray(metrics['true_P']).reshape(-1)
    pred_p = np.asarray(metrics['pred_P']).reshape(-1)
    true_q = np.asarray(metrics['true_Q']).reshape(-1)
    pred_q = np.asarray(metrics['pred_Q']).reshape(-1)
    n = true_p.size
    if not (pred_p.size == n and true_q.size == n and pred_q.size == n):
        raise ValueError('metrics true/pred P/Q lengths do not match')
    idx = np.arange(n, dtype=np.int64) if test_idx is None else np.asarray(test_idx, dtype=np.int64).reshape(-1)
    if idx.size != n:
        raise ValueError(f'test_idx length {idx.size} != metrics length {n}')
    rpe_p = compute_rpe(pred_p, true_p)
    rpe_q = compute_rpe(pred_q, true_q)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ['event_idx', 'true_P', 'pred_P', 'residual_P', 'RPE_P_pct', 'true_Q', 'pred_Q', 'residual_Q', 'RPE_Q_pct']
    optional = []
    if p0 is not None:
        optional.append(('p0', np.asarray(p0).reshape(-1)))
    if source is not None:
        optional.append(('source', np.asarray(source).reshape(-1)))
    if n_steps is not None:
        optional.append(('n_steps', np.asarray(n_steps).reshape(-1)))
    if applied_power is not None:
        optional.append(('applied_power', np.asarray(applied_power).reshape(-1)))
    if metrics.get('snr') is not None:
        optional.append(('snr', np.asarray(metrics['snr']).reshape(-1)))
    for (name, arr) in optional:
        if arr.size != n:
            raise ValueError(f'{name} length {arr.size} != metrics length {n}')
        header.append(name)

    with path.open('w', encoding='utf-8') as f:
        f.write(','.join(header) + '\n')
        for i in range(n):
            row = [idx[i], true_p[i], pred_p[i], pred_p[i] - true_p[i], rpe_p[i], true_q[i], pred_q[i], pred_q[i] - true_q[i], rpe_q[i]]
            for (_, arr) in optional:
                row.append(arr[i])
            f.write(','.join((_csv_cell(v) for v in row)) + '\n')
    return path

def save_checkpoint(path, *, model, stats, best_val_loss, best_epoch, hidden_size, num_layers, dropout):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': clone_state_dict(model), 'stats': _stats_for_checkpoint(stats), 'best_val_loss': best_val_loss, 'best_epoch': best_epoch, 'hidden_size': hidden_size, 'num_layers': num_layers, 'dropout': dropout, 'input_size': stats['input_size']}, path)

def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LstmModel(input_size=ckpt.get('input_size', ckpt['stats']['input_size']), hidden_size=ckpt['hidden_size'], num_layers=ckpt['num_layers'], dropout=ckpt.get('dropout', DROPOUT)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)

def train_model(train_dataset, val_dataset, stats, *, hidden_size, num_layers, dropout, num_epochs, batch_size, learning_rate, patience, checkpoint_path=None, device=DEVICE):
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    p_mean = stats['P_mean']
    p_std = stats['P_std']
    q_mean = stats['Q_mean']
    q_std = stats['Q_std']
    model = LstmModel(input_size=stats['input_size'], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=LR_MIN)
    history = {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
    best_val = float('inf')
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(num_epochs):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for (x_b, y_p, y_q) in train_loader:
            x_b = x_b.to(device)
            y_p = y_p.to(device)
            y_q = y_q.to(device)
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
                x_v = x_v.to(device)
                y_p = y_p.to(device)
                y_q = y_q.to(device)
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
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch {epoch + 1:03d}/{num_epochs} | train {avg_train:.6f} | val {avg_val:.6f} | RPE% P={rpe_p:.3f} Q={rpe_q:.3f} | lr {lr:.2e}', flush=True)
        if avg_val < best_val - MIN_DELTA:
            best_val = avg_val
            best_epoch = epoch + 1
            best_state = clone_state_dict(model)
            stale = 0
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model=model, stats=stats, best_val_loss=best_val, best_epoch=best_epoch, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        # else:
        #     stale += 1
        #     if stale >= patience:
        #         print(f'Early stopping at epoch {epoch + 1}', flush=True)
        #         break
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        (model, ckpt) = load_checkpoint(checkpoint_path, device=device)
        best_val = ckpt.get('best_val_loss', best_val)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return (model, best_val, history)

@torch.no_grad()
def predict_denormalized(model, dataset, stats, *, batch_size=BATCH_SIZE, device=DEVICE, label_stats=None):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    p_mean = stats['P_mean']
    p_std = stats['P_std']
    q_mean = stats['Q_mean']
    q_std = stats['Q_std']
    label = label_stats if label_stats is not None else stats
    yp_mean = label['P_mean']
    yp_std = label['P_std']
    yq_mean = label['Q_mean']
    yq_std = label['Q_std']
    pred_p_all = []
    pred_q_all = []
    true_p_all = []
    true_q_all = []
    for (x_b, y_p, y_q) in loader:
        (pred_p, pred_q) = model(x_b.to(device))
        pred_p_all.append(pred_p.cpu().numpy() * p_std + p_mean)
        pred_q_all.append(pred_q.cpu().numpy() * q_std + q_mean)
        true_p_all.append(y_p.numpy() * yp_std + yp_mean)
        true_q_all.append(y_q.numpy() * yq_std + yq_mean)
    return (np.concatenate(pred_p_all), np.concatenate(pred_q_all), np.concatenate(true_p_all), np.concatenate(true_q_all))

@torch.no_grad()
def evaluate_model(model, dataset, stats, *, batch_size=BATCH_SIZE, device=DEVICE, label_stats=None):
    (pred_p_arr, pred_q_arr, true_p_arr, true_q_arr) = predict_denormalized(model, dataset, stats, batch_size=batch_size, device=device, label_stats=label_stats)

    def _metrics(pred, true):
        err = pred - true
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 'nan'
        rpe_s = _summary_stats(compute_rpe(pred, true))
        res_s = _summary_stats(err)
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'rpe_mean': rpe_s['mean'], 'rpe_median': rpe_s['median'], 'rpe_std': rpe_s['std'], 'residual_mean': res_s['mean'], 'residual_median': res_s['median'], 'residual_std': res_s['std']}
    p_m = _metrics(pred_p_arr, true_p_arr)
    q_m = _metrics(pred_q_arr, true_q_arr)
    snr = stats.get('snr_test')
    if snr is not None and np.asarray(snr).reshape(-1).size == true_p_arr.size:
        snr = np.asarray(snr, dtype=np.float64).reshape(-1)
    else:
        snr = None
    snr_s = _summary_stats(snr) if snr is not None else None
    range_by_p = polarization_range_stats(pred_p_arr, true_p_arr, pred_q_arr, true_q_arr, pol_ref=true_p_arr, ref_name='abs_P_total', snr=snr)
    p0_test = stats.get('p0_test')
    range_by_p0 = []
    if p0_test is not None and np.asarray(p0_test).size == true_p_arr.size:
        range_by_p0 = polarization_range_stats(pred_p_arr, true_p_arr, pred_q_arr, true_q_arr, pol_ref=np.asarray(p0_test), ref_name='abs_p0', snr=snr)
    metrics = {'P_mae': p_m['mae'], 'P_rmse': p_m['rmse'], 'P_r2': p_m['r2'], 'P_rpe_mean': p_m['rpe_mean'], 'P_rpe_median': p_m['rpe_median'], 'P_rpe_std': p_m['rpe_std'], 'P_residual_mean': p_m['residual_mean'], 'P_residual_median': p_m['residual_median'], 'P_residual_std': p_m['residual_std'], 'Q_mae': q_m['mae'], 'Q_rmse': q_m['rmse'], 'Q_r2': q_m['r2'], 'Q_rpe_mean': q_m['rpe_mean'], 'Q_rpe_median': q_m['rpe_median'], 'Q_rpe_std': q_m['rpe_std'], 'Q_residual_mean': q_m['residual_mean'], 'Q_residual_median': q_m['residual_median'], 'Q_residual_std': q_m['residual_std'], 'pred_P': pred_p_arr, 'pred_Q': pred_q_arr, 'true_P': true_p_arr, 'true_Q': true_q_arr, 'range_stats_by_P': range_by_p, 'range_stats_by_p0': range_by_p0}
    if snr_s is not None:
        metrics['snr'] = snr
        metrics['snr_mean'] = float(snr_s['mean'])
        metrics['snr_median'] = float(snr_s['median'])
        metrics['snr_std'] = float(snr_s['std'])
        metrics['snr_min'] = float(snr_s['min'])
        metrics['snr_max'] = float(snr_s['max'])
    return metrics

def _configure_plot_style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'], 'mathtext.fontset': 'cm', 'axes.unicode_minus': False, 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9, 'axes.linewidth': 1.15})

def _apply_axes_style(ax):
    ax.set_facecolor('#f7f8fa')
    ax.grid(True, which='major', color='white', linewidth=1.2, alpha=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#1f2933')
        spine.set_linewidth(1.15)
    ax.tick_params(colors='#4a5560', labelsize=14)

def _fmt_stat(value, *, percent=False):
    if not np.isfinite(value):
        return 'n/a'
    if percent:
        return f'${value:.3f}\\%$'
    av = abs(value)
    if av == 0.0:
        return '$0$'
    if av >= 0.01:
        return f'${value:.4g}$'
    return f'${value:.3e}$'

def _annotate_stats_box(ax, lines, *, loc='upper right'):
    text = '\n'.join(lines)
    anchors = {'upper right': (0.98, 0.97, 'right', 'top'), 'upper left': (0.02, 0.97, 'left', 'top'), 'lower right': (0.98, 0.03, 'right', 'bottom'), 'lower left': (0.02, 0.03, 'left', 'bottom')}
    (x, y, ha, va) = anchors[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5, color='#24303a', bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#d0d7de', 'alpha': 0.92})

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

def _plot_loss_curves(history, plots_dir, *, best_val_loss=None):
    plots_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()
    color_train = '#2f6fed'
    color_val = '#c45c26'
    color_best = '#0f766e'
    train = np.asarray(history['train_loss'])
    val = np.asarray(history.get('val_loss', []))
    n = train.size
    epochs = np.arange(1, n + 1)
    val = val[:n] if val.size else np.full(n, np.nan)
    has_val = np.isfinite(val).any()
    best_i = np.nanargmin(val) if has_val else -1
    best_ep = epochs[best_i] if best_i >= 0 else -1
    best_v = val[best_i] if best_i >= 0 else 'nan'
    (fig, ax) = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(epochs, train, color=color_train, lw=2.2, label='Train')
    if has_val:
        ax.plot(epochs, val, color=color_val, lw=2.0, label='Val')
        ax.axvline(best_ep, color=color_best, lw=1.1, ls='--', alpha=0.85)
        ax.scatter([best_ep], [best_v], s=54, color=color_best, zorder=5, edgecolors='white', linewidths=0.8, label=f'Best val @ ${best_ep}$')
    positive_parts = [train[train > 0]]
    if has_val:
        positive_parts.append(val[np.isfinite(val) & (val > 0)])
    positive = np.concatenate(positive_parts) if any((p.size for p in positive_parts)) else np.array([])
    if positive.size:
        ax.set_yscale('log')
        ax.set_ylim(np.min(positive) * 0.7, np.max(positive) * 1.25)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative loss ($P + Q$)')
    ax.set_title('Training and validation loss')
    x_pad = max(2.0, 0.04 * max(n, 1))
    ax.set_xlim(1.0 - 0.25 * x_pad, max(n, 1) + x_pad)
    _apply_axes_style(ax)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    final_train = train[-1] if train.size else 'nan'
    final_val = val[-1] if has_val else 'nan'
    shown_best = best_val_loss if best_val_loss is not None and np.isfinite(best_val_loss) else best_v
    _annotate_stats_box(ax, [f'epochs ${n}$', f'best val {_fmt_stat(shown_best)}', f'final train {_fmt_stat(final_train)}', f'final val {_fmt_stat(final_val)}'], loc='lower left')
    fig.tight_layout()
    path = plots_dir / 'loss_curves.png'
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)
    return path

def save_plots(history, metrics, plots_dir, *, best_val_loss=None):
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    _configure_plot_style()
    color_p = '#2f6fed'
    color_q = '#c45c26'
    color_zero = '#1f2933'
    color_mean = '#0f766e'
    color_median = '#b45309'
    color_sigma = '#64748b'
    label_mean_pm = 'mean $\\pm 1\\sigma$'
    if history is not None and history.get('train_loss'):
        path = _plot_loss_curves(history, plots_dir, best_val_loss=best_val_loss)
        saved.append(path)
    (fig, axes) = plt.subplots(1, 2, figsize=(11.2, 5.0))
    for (ax, label, true_key, pred_key, mae_key, r2_key, color) in ((axes[0], '$P_{\\mathrm{total}}$', 'true_P', 'pred_P', 'P_mae', 'P_r2', color_p), (axes[1], '$Q_{\\mathrm{total}}$', 'true_Q', 'pred_Q', 'Q_mae', 'Q_r2', color_q)):
        true = np.asarray(metrics[true_key])
        pred = np.asarray(metrics[pred_key])
        ax.scatter(true, pred, s=14, alpha=0.35, c=color, edgecolors='none', rasterized=True)
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        pad = 0.05 * (hi - lo + 1e-08)
        lims = [lo - pad, hi + pad]
        ax.plot(lims, lims, color=color_zero, ls='--', lw=1.2, label='ideal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f'True {label}')
        ax.set_ylabel(f'Predicted {label}')
        ax.set_title(f'{label}: predicted vs true')
        ax.set_aspect('equal', adjustable='box')
        _apply_axes_style(ax)
        _annotate_stats_box(ax, [f'MAE {_fmt_stat(metrics[mae_key])}', f'$R^2$ {_fmt_stat(metrics[r2_key])}'], loc='lower right')
        ax.legend(loc='upper left', frameon=False, fontsize=8)
    fig.tight_layout()
    path = plots_dir / 'pred_vs_true.png'
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)
    saved.append(path)
    true_p = np.asarray(metrics['true_P'])
    pred_p = np.asarray(metrics['pred_P'])
    true_q = np.asarray(metrics['true_Q'])
    pred_q = np.asarray(metrics['pred_Q'])
    res_p = pred_p - true_p
    res_q = pred_q - true_q
    rpe_p = compute_rpe(pred_p, true_p)
    rpe_q = compute_rpe(pred_q, true_q)
    (fig, axes) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for (ax, res, label, color, mean_key, med_key, std_key) in ((axes[0], res_p, '$P_{\\mathrm{total}}$', color_p, 'P_residual_mean', 'P_residual_median', 'P_residual_std'), (axes[1], res_q, '$Q_{\\mathrm{total}}$', color_q, 'Q_residual_mean', 'Q_residual_median', 'Q_residual_std')):
        finite = res[np.isfinite(res)]
        mean_v = metrics[mean_key]
        med_v = metrics[med_key]
        std_v = metrics[std_key]
        if finite.size:
            ax.hist(finite, bins=55, color=color, edgecolor='white', linewidth=0.4, alpha=0.88)
            ax.axvline(0.0, color=color_zero, lw=1.1, ls='--', label='zero')
            ax.axvline(mean_v, color=color_mean, lw=1.4, label='mean')
            ax.axvline(med_v, color=color_median, lw=1.4, ls='-.', label='median')
            if np.isfinite(std_v) and std_v > 0:
                ax.axvspan(mean_v - std_v, mean_v + std_v, color=color_sigma, alpha=0.18, label=label_mean_pm, zorder=0)
                ax.axvline(mean_v - std_v, color=color_sigma, lw=1.0, ls=':')
                ax.axvline(mean_v + std_v, color=color_sigma, lw=1.0, ls=':')
        ax.set_xlabel(f'Residual (pred $-$ true) {label}')
        ax.set_ylabel('Count')
        ax.set_title(f'{label} residual distribution')
        _apply_axes_style(ax)
        _annotate_stats_box(ax, [f'mean {_fmt_stat(mean_v)}', f'median {_fmt_stat(med_v)}', f'std {_fmt_stat(std_v)}'], loc='upper right')
        ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    path = plots_dir / 'residual_histograms.png'
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)
    saved.append(path)
    (fig, axes) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for (ax, rpe, label, color, mean_key, med_key, std_key) in ((axes[0], rpe_p, '$P_{\\mathrm{total}}$', color_p, 'P_rpe_mean', 'P_rpe_median', 'P_rpe_std'), (axes[1], rpe_q, '$Q_{\\mathrm{total}}$', color_q, 'Q_rpe_mean', 'Q_rpe_median', 'Q_rpe_std')):
        finite = rpe[np.isfinite(rpe)]
        mean_v = metrics[mean_key]
        med_v = metrics[med_key]
        std_v = metrics[std_key]
        if finite.size:
            ax.hist(finite, bins=55, color=color, edgecolor='white', linewidth=0.4, alpha=0.88)
            ax.axvline(mean_v, color=color_mean, lw=1.4, label='mean')
            ax.axvline(med_v, color=color_median, lw=1.4, ls='-.', label='median')
            if np.isfinite(std_v) and std_v > 0:
                ax.axvspan(max(0.0, mean_v - std_v), mean_v + std_v, color=color_sigma, alpha=0.18, label=label_mean_pm, zorder=0)
                ax.axvline(max(0.0, mean_v - std_v), color=color_sigma, lw=1.0, ls=':')
                ax.axvline(mean_v + std_v, color=color_sigma, lw=1.0, ls=':')
            p99 = np.percentile(finite, 99)
            x_hi = max(p99 * 1.05, mean_v + 1.2 * std_v if np.isfinite(std_v) else p99)
            if np.isfinite(x_hi) and x_hi > 0:
                ax.set_xlim(0.0, x_hi)
        ax.set_xlabel(f'RPE (\\%)  {label}')
        ax.set_ylabel('Count')
        ax.set_title(f'{label} relative percent error')
        _apply_axes_style(ax)
        _annotate_stats_box(ax, [f'mean {_fmt_stat(mean_v, percent=True)}', f'median {_fmt_stat(med_v, percent=True)}', f'std {_fmt_stat(std_v, percent=True)}'], loc='upper right')
        ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.tight_layout()
    path = plots_dir / 'rpe_histograms.png'
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)
    saved.append(path)
    (fig, axes) = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for (ax, true, res, label, color, mean_key, std_key) in ((axes[0], true_p, res_p, '$P_{\\mathrm{total}}$', color_p, 'P_residual_mean', 'P_residual_std'), (axes[1], true_q, res_q, '$Q_{\\mathrm{total}}$', color_q, 'Q_residual_mean', 'Q_residual_std')):
        mean_v = metrics[mean_key]
        std_v = metrics[std_key]
        ax.scatter(true, res, s=12, alpha=0.28, c=color, edgecolors='none', rasterized=True)
        ax.axhline(0.0, color=color_zero, lw=1.1, ls='--', label='zero')
        ax.axhline(mean_v, color=color_mean, lw=1.3, label='mean')
        if np.isfinite(std_v) and std_v > 0:
            ax.axhspan(mean_v - std_v, mean_v + std_v, color=color_sigma, alpha=0.16, label=label_mean_pm, zorder=0)
            ax.axhline(mean_v - std_v, color=color_sigma, lw=1.0, ls=':')
            ax.axhline(mean_v + std_v, color=color_sigma, lw=1.0, ls=':')
        ax.set_xlabel(f'True {label}')
        ax.set_ylabel(f'Residual {label}')
        ax.set_title(f'{label}: residual vs true')
        _apply_axes_style(ax)
        _annotate_stats_box(ax, [f'mean {_fmt_stat(mean_v)}', f'std {_fmt_stat(std_v)}'], loc='upper right')
        ax.legend(frameon=False, fontsize=8, loc='lower left')
    fig.tight_layout()
    path = plots_dir / 'residuals_vs_true.png'
    fig.savefig(path, dpi=160, facecolor='white')
    plt.close(fig)
    saved.append(path)
    range_rows = list(metrics.get('range_stats_by_P') or [])
    nonempty = [r for r in range_rows if r['n'] > 0]
    if nonempty:
        labels = [r['range'] for r in nonempty]
        x = np.arange(len(labels))
        n_bands = sum((r['n'] for r in nonempty))
        (fig, axes) = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
        for (ax, prefix, ylabel, title, color) in ((axes[0], 'P', '$P$ RPE (\\%)', '$P$ RPE by $|P_{\\mathrm{total}}|$ band', color_p), (axes[1], 'Q', '$Q$ RPE (\\%)', '$Q$ RPE by $|P_{\\mathrm{total}}|$ band', color_q)):
            med = np.asarray([r[f'{prefix}_rpe_median'] for r in nonempty])
            mean = np.asarray([r[f'{prefix}_rpe_mean'] for r in nonempty])
            std = np.asarray([r[f'{prefix}_rpe_std'] for r in nonempty])
            ax.bar(x - 0.18, med, width=0.32, color='#94a3b8', edgecolor='white', linewidth=0.6, label='median')
            ax.bar(x + 0.18, mean, width=0.32, color=color, edgecolor='white', linewidth=0.6, label='mean', yerr=std, error_kw={'ecolor': color_zero, 'elinewidth': 1.1, 'capsize': 3.0, 'capthick': 1.0, 'alpha': 0.85})
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            _apply_axes_style(ax)
            ax.legend(frameon=False, fontsize=8, loc='upper right')
            _annotate_stats_box(ax, ['error bars: $\\pm 1\\sigma$ of RPE', f'bands $n={n_bands}$'], loc='upper left')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=40, ha='right')
        axes[1].set_xlabel('$|P_{\\mathrm{total}}|$ band')
        fig.tight_layout()
        path = plots_dir / 'rpe_by_polarization_range.png'
        fig.savefig(path, dpi=160, facecolor='white')
        plt.close(fig)
        saved.append(path)
        (fig, axes) = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
        for (ax, prefix, ylabel, title, color) in ((axes[0], 'P', 'Mean residual $P$', '$P$ residuals by $|P_{\\mathrm{total}}|$ band', color_p), (axes[1], 'Q', 'Mean residual $Q$', '$Q$ residuals by $|P_{\\mathrm{total}}|$ band', color_q)):
            mean = np.asarray([r[f'{prefix}_residual_mean'] for r in nonempty])
            std = np.asarray([r[f'{prefix}_residual_std'] for r in nonempty])
            ax.bar(x, mean, width=0.62, color=color, edgecolor='white', linewidth=0.6, yerr=std, error_kw={'ecolor': color_zero, 'elinewidth': 1.1, 'capsize': 3.0, 'capthick': 1.0, 'alpha': 0.85})
            ax.axhline(0.0, color=color_zero, lw=1.0, ls='--')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            _apply_axes_style(ax)
            _annotate_stats_box(ax, ['error bars: $\\pm 1\\sigma$ of residual', f'bands $n={n_bands}$'], loc='upper left')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=40, ha='right')
        axes[1].set_xlabel('$|P_{\\mathrm{total}}|$ band')
        fig.tight_layout()
        path = plots_dir / 'residuals_by_polarization_range.png'
        fig.savefig(path, dpi=160, facecolor='white')
        plt.close(fig)
        saved.append(path)
    return saved

def _select_example_indices(n_test, n_examples, *, true_p, pred_p, true_q, pred_q, seed=SEED, candidate_idx=None):
    if candidate_idx is None:
        pool = np.arange(n_test, dtype=int)
    else:
        pool = np.asarray(candidate_idx, dtype=int).reshape(-1)
        pool = pool[(pool >= 0) & (pool < n_test)]
        if pool.size == 0:
            return np.array([], dtype=int)
    n = pool.size
    k = min(max(1, n_examples), n)
    if k >= n:
        return np.sort(pool)
    true_p = np.asarray(true_p).reshape(-1)[pool]
    pred_p = np.asarray(pred_p).reshape(-1)[pool]
    true_q = np.asarray(true_q).reshape(-1)[pool]
    pred_q = np.asarray(pred_q).reshape(-1)[pool]
    err = np.abs(pred_p - true_p) + np.abs(pred_q - true_q)
    order_err = np.argsort(err)
    err_picks = order_err[np.round(np.linspace(0, n - 1, max(1, k // 2))).astype(int)]
    p_order = np.argsort(np.abs(true_p))
    p_picks = p_order[np.round(np.linspace(0, n - 1, max(1, k - err_picks.size))).astype(int)]
    local_picks = np.unique(np.concatenate([err_picks, p_picks]))
    if local_picks.size < k:
        rng = np.random.default_rng(seed)
        extra = rng.choice(np.setdiff1d(np.arange(n), local_picks, assume_unique=False), size=k - local_picks.size, replace=False)
        local_picks = np.concatenate([local_picks, extra])
    return np.sort(pool[local_picks[:k].astype(int)])

def test_input_ps(test_dataset, stats):
    """Denormalize the Ps channel the model actually evaluated on (includes noise)."""
    x = test_dataset.tensors[0].detach().cpu().numpy()
    return (x[:, :, 0] * float(stats['ps_std']) + float(stats['ps_mean'])).astype(np.float32, copy=False)

def _equilibrium_q0_by_p0(arrays):
    """Map rounded p0 → equilibrium tensor polarization from n_steps==0 rows."""
    if 'p0' not in arrays or 'Q_total' not in arrays or 'n_steps' not in arrays:
        return {}
    p0 = np.asarray(arrays['p0'], dtype=np.float64).reshape(-1)
    q = np.asarray(arrays['Q_total'], dtype=np.float64).reshape(-1)
    n_steps = np.asarray(arrays['n_steps']).reshape(-1)
    mask = n_steps == 0
    out = {}
    for (pv, qv) in zip(p0[mask], q[mask]):
        out[round(float(pv), 6)] = float(qv)
    return out

def _lookup_q0(q0_by_p0, p0_v):
    if p0_v is None or (isinstance(p0_v, float) and not np.isfinite(p0_v)):
        return float('nan')
    return float(q0_by_p0.get(round(float(p0_v), 6), float('nan')))

def _plot_example_lineshape(ax, freq, ps, *, legend=True):
    ax.plot(freq, ps, color='#1f2933', label=r'$P_s$ (input)', lw=1.9)
    ax.set_xlabel('R', fontsize=16)
    ax.set_ylabel('Amplitude', fontsize=16)
    # ax.tick_params(axis='both', which='major', labelsize=18)
    _apply_axes_style(ax)
    # if legend:
        # (handles, labels) = ax.get_legend_handles_labels()
        # if handles:
        #     ax.legend(handles, labels, loc='upper right', fontsize=8, frameon=False)
    ax.set_xlim(float(freq[0]), float(freq[-1]))
    ax.grid(True)

def save_example_signal_plots(arrays, stats, metrics, plots_dir, *, test_dataset=None, input_stats=None, n_examples=N_EXAMPLE_PLOTS, seed=SEED, source_filter=None, examples_subdir='examples', summary_name='examples_summary.png'):
    """Save lineshape example plots and a CSV of the former on-figure metadata.

    When ``source_filter`` is set (e.g. ``SOURCE_PROFILE``), only those test
    events are considered and written under ``examples_subdir``.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = plots_dir / examples_subdir
    examples_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()
    test_idx = np.asarray(stats['test_idx'], dtype=int)
    spectra = np.asarray(arrays['spectra'])
    num_bins = spectra.shape[2]
    freq = np.linspace(SPECTRUM_R_MIN, SPECTRUM_R_MAX, num_bins)
    pred_p = np.asarray(metrics['pred_P']).reshape(-1)
    true_p = np.asarray(metrics['true_P']).reshape(-1)
    pred_q = np.asarray(metrics['pred_Q']).reshape(-1)
    true_q = np.asarray(metrics['true_Q']).reshape(-1)

    denorm_stats = input_stats if input_stats is not None else stats
    noisy_ps = test_input_ps(test_dataset, denorm_stats)
    noise_std = float(denorm_stats.get('noise_std', stats.get('noise_std', 0.0)))
    snr_test = np.asarray(stats['snr_test'], dtype=np.float64).reshape(-1) if 'snr_test' in stats else None
    source = np.asarray(arrays['source'], dtype=np.int64).reshape(-1) if 'source' in arrays else None
    candidate_idx = None
    if source_filter is not None:
        if source is None:
            print(f'No source labels; skipping examples under {examples_subdir}/', flush=True)
            return []
        candidate_idx = np.flatnonzero(source[test_idx] == int(source_filter))
        if candidate_idx.size == 0:
            print(f'No test events with source={source_filter}; skipping {examples_subdir}/', flush=True)
            return []
    pick = _select_example_indices(test_idx.size, n_examples, true_p=true_p, pred_p=pred_p, true_q=true_q, pred_q=pred_q, seed=seed, candidate_idx=candidate_idx)
    if pick.size == 0:
        return []
    applied = np.asarray(arrays['applied_power']).reshape(-1)
    n_steps_arr = np.asarray(arrays['n_steps']).reshape(-1)
    p0_arr = np.asarray(arrays['p0']).reshape(-1) if 'p0' in arrays else None
    q0_by_p0 = _equilibrium_q0_by_p0(arrays)
    saved = []
    info_rows = []
    n_sum = min(4, pick.size)
    sum_ncols = 2 if n_sum > 1 else 1
    sum_nrows = int(np.ceil(n_sum / sum_ncols)) if n_sum else 0
    if n_sum > 0:
        (fig, axes) = plt.subplots(sum_nrows, sum_ncols, figsize=(5.8 * sum_ncols, 3.6 * sum_nrows), squeeze=False, constrained_layout=True)
    else:
        fig = None
        axes = None
    for (panel_i, local_i) in enumerate(pick):
        gi = int(test_idx[local_i])
        ps = np.asarray(noisy_ps[local_i], dtype=np.float64)
        tp = float(true_p[local_i])
        pp = float(pred_p[local_i])
        tq = float(true_q[local_i])
        pq = float(pred_q[local_i])
        rpe_p = float(compute_rpe(np.array([pp]), np.array([tp]))[0])
        rpe_q = float(compute_rpe(np.array([pq]), np.array([tq]))[0])
        p0_v = float(p0_arr[gi]) if p0_arr is not None else float('nan')
        q0_v = _lookup_q0(q0_by_p0, p0_v)
        if int(n_steps_arr[gi]) == 0 and np.isfinite(tq):
            q0_v = tq
        power = float(applied[gi])
        steps = int(n_steps_arr[gi])
        src = SOURCE_NAME.get(int(source[gi]), str(source[gi])) if source is not None else '?'
        (fig_e, ax_s) = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
        _plot_example_lineshape(ax_s, freq, ps, legend=True)
        path_e = examples_dir / f'test_example_{gi:05d}_nsteps{steps:04d}.png'
        fig_e.savefig(path_e, dpi=170, facecolor='white', bbox_inches='tight', pad_inches=0.12)
        plt.close(fig_e)
        saved.append(path_e)
        snr_v = float(snr_test[local_i]) if snr_test is not None else float('nan')
        info_rows.append({'filename': path_e.name, 'event_idx': gi, 'source': src, 'n_steps': steps, 'p0': p0_v, 'q0': q0_v, 'true_P': tp, 'true_Q': tq, 'pred_P': pp, 'pred_Q': pq, 'p0_pct': 100.0 * p0_v, 'q0_pct': 100.0 * q0_v, 'true_P_pct': 100.0 * tp, 'true_Q_pct': 100.0 * tq, 'pred_P_pct': 100.0 * pp, 'pred_Q_pct': 100.0 * pq, 'RPE_P_pct': rpe_p, 'RPE_Q_pct': rpe_q, 'applied_power': power, 'noise_std': noise_std, 'snr': snr_v})
        if axes is not None and panel_i < n_sum:
            ax_sum = axes[panel_i // sum_ncols, panel_i % sum_ncols]
            _plot_example_lineshape(ax_sum, freq, ps, legend=(panel_i == 0))
    if axes is not None:
        for extra_i in range(n_sum, sum_nrows * sum_ncols):
            axes[extra_i // sum_ncols, extra_i % sum_ncols].set_visible(False)
    csv_path = save_example_info_csv(info_rows, examples_dir / 'example_info.csv')
    if csv_path is not None:
        saved.append(csv_path)
        print(f'Saved example metadata -> {csv_path}', flush=True)
    if fig is not None:
        path_sum = plots_dir / summary_name
        fig.savefig(path_sum, dpi=170, facecolor='white', bbox_inches='tight', pad_inches=0.12)
        plt.close(fig)
        saved.append(path_sum)
    print(f'Saved {len(pick)} example plots -> {examples_dir}/', flush=True)
    return saved

def main():
    parser = argparse.ArgumentParser(description='Train/evaluate LSTM seq model: Ps spectrum + per-bin power profile → P_total, Q_total.')
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--hidden-size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--num-layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--max-samples', type=int, default=None, help='Subsample this many events (recommended for first runs)')
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--n-examples', type=int, default=N_EXAMPLE_PLOTS, help='Number of per-sample example plots to save')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint path (default: <out-dir>/lstm_best.pth)')
    parser.add_argument('--test-only', action='store_true', help='Skip training; load --checkpoint and evaluate on a fresh split')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out_dir / 'lstm_best.pth'
    spectra_path = args.spectra or DEFAULT_SPECTRA_PATH
    print(f'Loading spectra from {spectra_path} ...', flush=True)
    arrays = load_lstm_npz(spectra_path)
    print('Preparing datasets...', flush=True)
    (train_ds, val_ds, test_ds, stats) = prepare_datasets(arrays, max_samples=args.max_samples, noise_std=args.noise_std)
    dataset_stats = {k: stats[k] for k in ('ps_mean', 'ps_std', 'pwr_mean', 'pwr_std', 'steps_mean', 'steps_std', 'P_mean', 'P_std', 'Q_mean', 'Q_std', 'noise_std', 'test_idx', 'n_test') if k in stats}
    print(f"Train={stats['n_train']} Val={stats['n_val']} Test={stats['n_test']} bins={stats['num_bins']}", flush=True)
    history_path = args.out_dir / 'history.json'
    if args.test_only:



        print(f'Loading checkpoint {checkpoint_path} ...', flush=True)
        (model, ckpt) = load_checkpoint(checkpoint_path)
        stats = {**stats, **ckpt['stats']}
        # Keep the tensors' own Ps normalization / noise metadata for example plots.
        stats['test_idx'] = dataset_stats['test_idx']
        best_val = ckpt.get('best_val_loss', float('nan'))
        loaded = load_history_json(history_path)
        history = loaded or {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total Params: {total_params:,}")
        print(f"Trainable Params: {trainable_params:,}")
        print(f"Non-Trainable Params: {total_params - trainable_params:,}")
        
        if loaded:
            print(f'Loaded training history from {history_path}', flush=True)
    else:
        print('Training lstm model...', flush=True)
        (model, best_val, history) = train_model(train_ds, val_ds, stats, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout, num_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, patience=args.patience, checkpoint_path=checkpoint_path)
        save_history_json(history, history_path)
        dataset_stats = {k: stats[k] for k in ('ps_mean', 'ps_std', 'pwr_mean', 'pwr_std', 'steps_mean', 'steps_std', 'P_mean', 'P_std', 'Q_mean', 'Q_std', 'noise_std', 'test_idx', 'n_test') if k in stats}
    print('Evaluating on test set...', flush=True)
    metrics = evaluate_model(model, test_ds, stats, batch_size=args.batch_size, label_stats=dataset_stats)
    print(f"Test RPE% median  P={metrics['P_rpe_median']:.3f}  Q={metrics['Q_rpe_median']:.3f}", flush=True)
    print(f"Test MAE  P={metrics['P_mae']:.6f}  Q={metrics['Q_mae']:.6f}  R²  P={metrics['P_r2']:.4f}  Q={metrics['Q_r2']:.4f}", flush=True)
    print_snr_stats(metrics)
    print_range_stats_table(metrics['range_stats_by_P'], title='Performance by |P_total| Range')
    if metrics['range_stats_by_p0']:
        print_range_stats_table(metrics['range_stats_by_p0'], title='Performance by |p0| Range')
    save_range_stats_csv(metrics['range_stats_by_P'], args.out_dir / 'range_stats_by_P.csv')
    if metrics['range_stats_by_p0']:
        save_range_stats_csv(metrics['range_stats_by_p0'], args.out_dir / 'range_stats_by_p0.csv')
    test_idx = np.asarray(dataset_stats.get('test_idx', stats.get('test_idx')), dtype=np.int64)
    pred_csv = args.out_dir / 'test_predictions.csv'
    save_predictions_csv(metrics, pred_csv, test_idx=test_idx, p0=np.asarray(arrays['p0']).reshape(-1)[test_idx] if 'p0' in arrays else None, source=np.asarray(arrays['source']).reshape(-1)[test_idx] if 'source' in arrays else None, n_steps=np.asarray(arrays['n_steps']).reshape(-1)[test_idx], applied_power=np.asarray(arrays['applied_power']).reshape(-1)[test_idx])
    print(f'Saved test predictions -> {pred_csv}', flush=True)
    json_metrics = {k: v for (k, v) in metrics.items() if isinstance(v, (float, int, str, list))}
    json_metrics['best_val_loss'] = best_val
    json_metrics['checkpoint'] = str(checkpoint_path)
    with (args.out_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(json_ready(json_metrics), f, indent=2)
    print('Generating plots...', flush=True)
    save_plots(history, metrics, args.out_dir, best_val_loss=best_val)
    save_example_signal_plots(arrays, stats, metrics, args.out_dir, test_dataset=test_ds, input_stats=dataset_stats, n_examples=args.n_examples)
    save_example_signal_plots(arrays, stats, metrics, args.out_dir, test_dataset=test_ds, input_stats=dataset_stats, n_examples=args.n_examples, source_filter=SOURCE_PROFILE, examples_subdir='examples_optimal', summary_name='examples_optimal_summary.png', seed=SEED + 1)
    print(f'Done. Results in {args.out_dir}', flush=True)
if __name__ == '__main__':
    main()
