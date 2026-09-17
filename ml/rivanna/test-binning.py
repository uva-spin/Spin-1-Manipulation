"""
Evaluate the combined per-bin model on test lineshapes.

Loads Voigt-burn ``spectra.npz`` (N×2×bins I+/I−, plus p0 / P_total / Q_total).

Run:
  python ml/rivanna/test-binning.py
"""
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
RIVANNA_DIR = Path(__file__).resolve().parent
REPO_ROOT = RIVANNA_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RIVANNA_DIR) not in sys.path:
    sys.path.insert(0, str(RIVANNA_DIR))
from single_bin import BinModel as LinearBinModel, build_event_feature_matrix, event_manipulation_features, load_bin_model_state_dict
_DEGENERATE_X_STD = 1e-06
SOURCE_SSRF = 0
SOURCE_AFP = 1
MODEL_PATH = 'models/combined_bin_model.pth'
TEST_FILE = 'data/spectra.npz'
OUTPUT_DIR = 'results/test_binning'
SCALING_FILE = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_BINS = 500
FEATURE_CLIP_Z = 0.0
EXAMPLES = 12
EXAMPLE_SELECTION = 'stratified'
MAX_HEATMAP_SAMPLES = 200
BURN_CONTEXT_FEATURES = ('ps_at_burn_bin', 'P', 'burn_step_norm', 'ps_ratio', 'burn_progress')

@dataclass
class LineshapeEvent:
    burn_bin_idx = None
    gamma_rf = 0.0
    n_steps = 0.0
    burn_step_norm = 0.0
    ps_ratio = 1.0
    burn_progress = 0.0

    def __post_init__(self):
        self.frequency = np.asarray(self.frequency)
        self.ps = np.asarray(self.ps)
        self.iplus = np.asarray(self.iplus)
        self.iminus = np.asarray(self.iminus)
        if self.burn_bin_idx is not None:
            self.burn_bin_idx = self.burn_bin_idx

    @property
    def num_bins(self):
        return self.ps.shape[0]

    def feature_matrix(self, feature_names=None):
        """Per-bin features; ``gamma_rf`` / ``n_steps`` are global event parameters."""
        names = feature_names or list(BURN_CONTEXT_FEATURES)
        if set(names).issubset({'gamma_rf', 'n_steps', 'ps'}):
            return build_event_feature_matrix(self.ps, names, gamma_rf=self.gamma_rf, n_steps=self.n_steps)
        n = self.num_bins
        burn_step_norm = np.zeros(n)
        ps_ratio = np.ones(n)
        burn_progress = np.zeros(n)
        if self.burn_bin_idx is not None and 0 <= self.burn_bin_idx < n:
            b = self.burn_bin_idx
            burn_step_norm[b] = np.float32(self.burn_step_norm)
            ps_ratio[b] = np.float32(self.ps_ratio)
            burn_progress[b] = np.float32(self.burn_progress)
        columns = {'gamma_rf': np.full(n, np.float32(self.gamma_rf)), 'n_steps': np.full(n, np.float32(self.n_steps)), 'ps': self.ps, 'ps_at_burn_bin': self.ps, 'p0': np.full(n, self.polarization), 'P': np.full(n, self.polarization), 'amp': np.abs(self.ps), 'burn_step_norm': burn_step_norm, 'ps_ratio': ps_ratio, 'burn_progress': burn_progress}
        missing = [name for name in names if name not in columns]
        if missing:
            raise KeyError(f'Unknown feature names {missing}; supported: {sorted(columns)}')
        return np.column_stack([columns[name] for name in names]).astype(np.float32)

def event_from_row(row, n_bins):
    if 'P_initial' in row.index:
        p = row['P_initial']
    elif 'P' in row.index:
        p = row['P']
    else:
        raise KeyError('Row missing P_initial or P.')
    burn_bin_idx = row.get('burn_bin_idx')
    if burn_bin_idx is not None and pd.isna(burn_bin_idx):
        burn_bin_idx = None
    elif burn_bin_idx is not None:
        burn_bin_idx = burn_bin_idx
    freq = row['frequency'] if 'frequency' in row.index else np.arange(n_bins)
    return LineshapeEvent(polarization=p, frequency=np.asarray(freq), ps=np.asarray(row['Ps']), iplus=np.asarray(row['Iplus']), iminus=np.asarray(row['Iminus']), burn_bin_idx=burn_bin_idx, gamma_rf=row.get('gamma_rf', row.get('applied_power', 0.0)), n_steps=row.get('n_steps', row.get('burn_step_norm', 0.0) * 100.0), burn_step_norm=row.get('burn_step_norm', 0.0), ps_ratio=row.get('ps_ratio', 1.0), burn_progress=row.get('burn_progress', 0.0))

def _npz_1d(data, key, n, default=0.0):
    if key not in data.files:
        return np.full(n, default)
    return np.asarray(data[key]).reshape(-1)

def load_test_npz(path, *, ssrf_only=True):
    """Load Voigt-burn ``spectra.npz`` (N×2×bins I+/I−) into the row table used below.

    By default keeps only ssRF rows (``source==0``). AFP flips can drive local Ps
    negative at positive p0 and are out of scope for this I+/I− bin model eval.
    """
    with np.load(path, allow_pickle=False) as raw:
        if 'spectra' not in raw.files:
            raise KeyError(f"{path}: missing 'spectra'; found {raw.files}")
        spectra = np.asarray(raw['spectra'])
        if spectra.ndim != 3 or spectra.shape[1] != 2:
            raise ValueError(f'{path}: expected spectra shape (N, 2, num_bins), got {spectra.shape}')
        iplus = spectra[:, 0, :]
        iminus = spectra[:, 1, :]
        (n, n_bins) = (iplus.shape[0], iplus.shape[1])
        p0 = _npz_1d(raw, 'p0', n)
        applied = _npz_1d(raw, 'applied_power', n)
        n_steps_raw = _npz_1d(raw, 'n_steps', n)
        center = _npz_1d(raw, 'center_bin', n, default=np.nan)
        source = _npz_1d(raw, 'source', n, default=SOURCE_SSRF)
        p_total = _npz_1d(raw, 'P_total', n) if 'P_total' in raw.files else None
        q_total = _npz_1d(raw, 'Q_total', n) if 'Q_total' in raw.files else None
        freq = np.asarray(raw['frequency']) if 'frequency' in raw.files else np.arange(n_bins)
        if freq.ndim == 1:
            freq_rows = np.stack([freq] * n, axis=0)
        else:
            freq_rows = np.asarray(freq)
    if ssrf_only:
        keep = source.astype(np.int32) == SOURCE_SSRF
        if not np.any(keep):
            raise ValueError(f'{path}: no ssRF events (source=={SOURCE_SSRF}) to evaluate')
        n_drop = (~keep).sum()
        if n_drop:
            print(f'Keeping {keep.sum()} ssRF events; dropped {n_drop} non-ssRF (AFP/other) rows', flush=True)
        iplus = iplus[keep]
        iminus = iminus[keep]
        p0 = p0[keep]
        applied = applied[keep]
        n_steps_raw = n_steps_raw[keep]
        center = center[keep]
        source = source[keep]
        freq_rows = freq_rows[keep]
        if p_total is not None:
            p_total = p_total[keep]
        if q_total is not None:
            q_total = q_total[keep]
        n = iplus.shape[0]
    gamma_rf = np.empty(n)
    n_steps = np.empty(n)
    for i in range(n):
        (g, s) = event_manipulation_features(source=source[i], applied_power=applied[i], n_steps=n_steps_raw[i])
        gamma_rf[i] = g
        n_steps[i] = s
    ps = iplus + iminus
    df = pd.DataFrame({'P_initial': p0, 'gamma_rf': gamma_rf, 'applied_power': applied, 'n_steps': n_steps, 'burn_step_norm': n_steps / 100.0, 'burn_bin_idx': center, 'source': source.astype(np.int32), 'Ps': list(ps), 'Iplus': list(iplus), 'Iminus': list(iminus), 'frequency': list(freq_rows)})
    if p_total is not None:
        df['true_P'] = p_total
    if q_total is not None:
        df['true_Q'] = q_total
    return df

def load_test_events(path, *, ssrf_only=True):
    df = load_test_npz(path, ssrf_only=ssrf_only)
    missing = {'Ps', 'Iplus', 'Iminus'} - set(df.columns)
    if missing:
        raise KeyError(f'Test file missing columns: {sorted(missing)}')
    if not (df['source'].to_numpy(dtype=np.int32) == SOURCE_SSRF).all():
        raise RuntimeError('Non-ssRF rows present after filter; aborting.')
    n_bins = np.asarray(df['Ps'].iloc[0]).shape[0]
    return ([event_from_row(row, n_bins) for (_, row) in df.iterrows()], df)

def _resolve_input_stats(stats, num_models, input_dim):
    if 'X_mean' in stats:
        x_mean = np.asarray(stats['X_mean'])
        x_std = np.asarray(stats['X_std'])
    elif 'Ps_mean' in stats:
        ps_mean = np.asarray(stats['Ps_mean'])
        ps_std = np.asarray(stats['Ps_std'])
        if ps_mean.ndim == 0:
            raise ValueError('Ps_mean must be per-bin, not a global scalar.')
        x_mean = ps_mean[:, None] if ps_mean.ndim == 1 else ps_mean
        x_std = ps_std[:, None] if ps_std.ndim == 1 else ps_std
    else:
        raise KeyError('Stats need X_mean/X_std or Ps_mean/Ps_std.')
    if x_mean.shape != (num_models, input_dim):
        raise ValueError(f'Expected X_mean shape ({num_models}, {input_dim}), got {x_mean.shape}.')
    return (x_mean, x_std)

def _resolve_output_stats(stats, num_models, target_mode=None):
    mode_hint = (target_mode or '').strip().lower()
    has_pq = 'P_mean' in stats and 'Q_mean' in stats
    has_i = 'Iplus_mean' in stats and 'Iminus_mean' in stats
    if mode_hint in ('iplus_iminus', 'i+', 'iplus') and has_i:
        keys = ('Iplus_mean', 'Iplus_std', 'Iminus_mean', 'Iminus_std')
        mode = 'iplus_iminus'
    elif mode_hint in ('pq', 'p_q') and has_pq:
        keys = ('P_mean', 'P_std', 'Q_mean', 'Q_std')
        mode = 'pq'
    elif has_i and (not has_pq):
        keys = ('Iplus_mean', 'Iplus_std', 'Iminus_mean', 'Iminus_std')
        mode = 'iplus_iminus'
    elif has_pq:
        keys = ('P_mean', 'P_std', 'Q_mean', 'Q_std')
        mode = 'pq'
    else:
        raise KeyError('Stats need Iplus_mean/Iminus_mean or P_mean/Q_mean for output denormalization.')
    out = {}
    for key in keys:
        arr = np.asarray(stats[key])
        if arr.shape != (num_models,):
            raise ValueError(f'Expected per-bin {key} length {num_models}, got {arr.shape}.')
        out[key] = arr
    if mode == 'pq':
        return (mode, out['P_mean'], out['P_std'], out['Q_mean'], out['Q_std'])
    return (mode, out['Iplus_mean'], out['Iplus_std'], out['Iminus_mean'], out['Iminus_std'])

class Combined500BinModel(nn.Module):
    """One bin model per spectrum index; per-bin normalization only."""

    def __init__(self, bin_models, stats, feature_names, feature_clip_z=FEATURE_CLIP_Z, loaded_bin_indices=None, target_mode=None):
        super().__init__()
        self.num_models = len(bin_models)
        self.bin_models = nn.ModuleList(bin_models)
        self.feature_names = list(feature_names)
        self.input_dim = len(self.feature_names)
        self.feature_clip_z = feature_clip_z
        self.loaded_bin_indices = list(range(self.num_models)) if loaded_bin_indices is None else [i for i in loaded_bin_indices]
        (x_mean, x_std) = _resolve_input_stats(stats, self.num_models, self.input_dim)
        (self.target_mode, out0_m, out0_s, out1_m, out1_s) = _resolve_output_stats(stats, self.num_models, target_mode=target_mode)
        self._X_mean = torch.from_numpy(x_mean).float()
        self._X_std = torch.from_numpy(x_std).float()
        self._X_std_degenerate = self._X_std <= _DEGENERATE_X_STD
        self._Out0_mean = torch.from_numpy(out0_m).float()
        self._Out0_std = torch.from_numpy(out0_s).float()
        self._Out1_mean = torch.from_numpy(out1_m).float()
        self._Out1_std = torch.from_numpy(out1_s).float()

    def _predict_bin(self, model_idx, x):
        """Normalize with checkpoint X_mean/X_std, then denormalize outputs."""
        mean = self._X_mean[model_idx].to(x.device)
        std = self._X_std[model_idx].to(x.device)
        degenerate = self._X_std_degenerate[model_idx].to(x.device)
        x_aligned = torch.where(degenerate, mean.expand_as(x), x)
        x_norm = (x_aligned - mean) / (std + 1e-12)
        if self.feature_clip_z > 0:
            x_norm = torch.clamp(x_norm, -self.feature_clip_z, self.feature_clip_z)
        (out0_n, out1_n) = self.bin_models[model_idx](x_norm)
        out0 = out0_n * self._Out0_std[model_idx].to(x.device) + self._Out0_mean[model_idx].to(x.device)
        out1 = out1_n * self._Out1_std[model_idx].to(x.device) + self._Out1_mean[model_idx].to(x.device)
        return (out0, out1)

    def forward(self, features, spectrum_bins=None):
        (batch, n_feat, dim) = features.shape
        if dim != self.input_dim:
            raise ValueError(f'Expected input_dim={self.input_dim}, got {dim}.')
        n_out = n_feat if spectrum_bins is None else spectrum_bins
        pred_out0 = torch.full((batch, n_out), 'nan', device=features.device)
        pred_out1 = torch.full((batch, n_out), 'nan', device=features.device)
        for (model_idx, bin_idx) in enumerate(self.loaded_bin_indices):
            if bin_idx >= n_feat or bin_idx >= n_out:
                continue
            (out0, out1) = self._predict_bin(model_idx, features[:, bin_idx, :])
            pred_out0[:, bin_idx] = out0
            pred_out1[:, bin_idx] = out1
        return (pred_out0, pred_out1)

    def predict_events(self, events, spectrum_bins=NUM_BINS):
        feats = torch.from_numpy(np.stack([e.feature_matrix(self.feature_names) for e in events], axis=0)).float().to(next(self.parameters()).device)
        with torch.no_grad():
            (out0, out1) = self(feats, spectrum_bins=spectrum_bins)
        return (out0.cpu().numpy(), out1.cpu().numpy())

    def predict_iplus_iminus_events(self, events, spectrum_bins=NUM_BINS):
        (out0, out1) = self.predict_events(events, spectrum_bins=spectrum_bins)
        if self.target_mode == 'pq':
            return (0.5 * (out0 + out1), 0.5 * (out0 - out1))
        return (out0, out1)

def load_combined_model(model_path, device, scaling_path=None):
    payload = torch.load(model_path, map_location=device, weights_only=False)
    num_bins = payload['num_bins']
    feature_names = list(payload.get('feature_names', ['ps_at_burn_bin']))
    input_dim = payload.get('input_dim', len(feature_names))
    hidden_dim = payload.get('hidden_dim', 256)
    stats = payload.get('stats')
    if stats is None:
        path = scaling_path or os.path.join(os.path.dirname(model_path), 'scaling_stats.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f'No embedded stats and no file at {path}')
        data = np.load(path)
        stats = {k: np.asarray(data[k]) for k in data.files}
    _resolve_input_stats(stats, num_bins, input_dim)
    target_mode = str(payload.get('target_mode', '')).strip() or None
    _resolve_output_stats(stats, num_bins, target_mode=target_mode)
    models = []
    for i in range(num_bins):
        m = LinearBinModel(input_dim, hidden_dim)
        load_bin_model_state_dict(m, payload['bin_state_dicts'][i])
        m.eval()
        models.append(m)
    combined = Combined500BinModel(models, stats, feature_names, loaded_bin_indices=payload.get('loaded_bin_indices'), target_mode=target_mode)
    meta = {'pq_post_correct': payload.get('pq_post_correct', True), 'targets_precalibrated': payload.get('targets_precalibrated', False), 'target_mode': combined.target_mode, 'feature_names': list(feature_names)}
    return (combined.to(device).eval(), meta)

def integrated_polarization(iplus, iminus):
    return (np.nansum(iplus + iminus, axis=1), np.nansum(iplus - iminus, axis=1))

def _stack_spectrum_column(df, column):
    first = df[column].iloc[0]
    if isinstance(first, (np.ndarray, list, tuple)):
        return np.stack([np.asarray(row[column]) for (_, row) in df.iterrows()], axis=0)
    return np.stack([np.asarray(row) for row in df[column]], axis=0)

def pq_truth_from_dataframe(df):
    """Load pre-calibrated per-bin and integrated P/Q already stored in the test table."""
    if 'P_bins' in df.columns and 'Q_bins' in df.columns:
        p_bins = _stack_spectrum_column(df, 'P_bins')
        q_bins = _stack_spectrum_column(df, 'Q_bins')
    elif 'P' in df.columns and 'Q' in df.columns and isinstance(df['P'].iloc[0], (np.ndarray, list, tuple)):
        p_bins = _stack_spectrum_column(df, 'P')
        q_bins = _stack_spectrum_column(df, 'Q')
    elif 'Iplus' in df.columns and 'Iminus' in df.columns:
        ip = _stack_spectrum_column(df, 'Iplus')
        im = _stack_spectrum_column(df, 'Iminus')
        p_bins = ip + im
        q_bins = ip - im
    else:
        raise KeyError("Test file must include pre-calibrated per-bin truth: 'P_bins'/'Q_bins' or per-row 'P'/'Q' spectrum arrays")
    if 'true_P' in df.columns and 'true_Q' in df.columns:
        p_int = df['true_P'].to_numpy()
        q_int = df['true_Q'].to_numpy()
    elif 'P_total' in df.columns and 'Q_total' in df.columns:
        p_int = df['P_total'].to_numpy()
        q_int = df['Q_total'].to_numpy()
    elif 'P_int' in df.columns and 'Q_int' in df.columns:
        p_int = df['P_int'].to_numpy()
        q_int = df['Q_int'].to_numpy()
    else:
        p_int = np.nanmean(p_bins, axis=1)
        q_int = np.nanmean(q_bins, axis=1)
    return (p_bins, q_bins, p_int, q_int)

def compute_rpe(pred, true, mask):
    rpe = np.full_like(true, np.nan)
    valid = mask & (np.abs(true) > 1e-10)
    rpe[valid] = abs((pred[valid] - true[valid]) / true[valid] * 100.0)
    return rpe

def print_results_table(stats):
    """Print evaluation metrics as aligned tables."""
    sections = [('Per-bin targets', [('L1 P', stats['L1_P'], ''), ('L1 Q', stats['L1_Q'], ''), ('Median RPE P', stats['median_RPE_P'], '%'), ('Median RPE Q', stats['median_RPE_Q'], '%')]), ('Lineshape decomposition', [('L1 I+', stats['L1_Iplus'], ''), ('L1 I-', stats['L1_Iminus'], ''), ('Median RPE I+', stats['median_RPE_Iplus'], '%'), ('Median RPE I-', stats['median_RPE_Iminus'], '%')]), ('Vector polarization P', [('Mean RPE', stats['mean_RPE_P'], '%'), ('Median RPE', stats['median_RPE_P_int'], '%'), ('Std RPE', stats['std_RPE_P'], '%'), ('Mean residual', stats['mean_residual_P'], ''), ('Std residual', stats['std_residual_P'], '')]), ('Tensor polarization Q', [('Mean RPE', stats['mean_RPE_Q'], '%'), ('Median RPE', stats['median_RPE_Q_int'], '%'), ('Std RPE', stats['std_RPE_Q'], '%'), ('Mean residual', stats['mean_residual_Q'], ''), ('Std residual', stats['std_residual_Q'], '')])]
    label_width = max((len(name) for (_, rows) in sections for (name, _, _) in rows))
    print('\n===== Test Results =====')
    print(f"{'Metric':<{label_width}}  {'Value':>12}")
    print('-' * (label_width + 15))
    for (i, (title, rows)) in enumerate(sections):
        if i > 0:
            print()
        print(title)
        for (name, value, unit) in rows:
            print(f'  {name:<{label_width - 2}}  {value:12.4f}{unit}')
    print(f"\nSamples: {stats['n_samples']}  |  Bins: {stats['n_bins']}")

def select_example_indices(n_test, n_examples, residuals, mode):
    n = min(n_examples, n_test)
    if n == 0:
        return np.array([], dtype=int)
    if mode == 'sequential':
        return np.arange(n, dtype=int)
    if mode == 'spread':
        return np.array([n_test // 2], dtype=int) if n == 1 else np.linspace(0, n_test - 1, n, dtype=int)
    err = np.nanmean(np.abs(residuals), axis=1)
    order = np.argsort(err)
    ranks = np.linspace(0, n_test - 1, n)
    return order[np.round(ranks).astype(int)]

def plot_lineshape_examples(out_path, indices, ps, ip_true, im_true, ip_pred, im_pred, title_prefix='Sample'):
    if indices.size == 0:
        return
    x = np.arange(ps.shape[1])
    (fig, axes) = plt.subplots(len(indices), 1, figsize=(12, 3 * len(indices)), squeeze=False)
    for (ax, idx) in zip(axes[:, 0], indices):
        ax.plot(x, ps[idx], 'k-', lw=1.5, label='Ps')
        ax.plot(x, ip_true[idx], color='#d55e00', alpha=0.35, lw=2, label='True I+')
        ax.plot(x, im_true[idx], color='#0072b2', alpha=0.35, lw=2, label='True I-')
        ax.plot(x, ip_pred[idx], color='#d55e00', ls='--', lw=1.3, label='Pred I+')
        ax.plot(x, im_pred[idx], color='#0072b2', ls='--', lw=1.3, label='Pred I-')
        ax.set_title(f'{title_prefix} {idx}')
        ax.set_xlabel('Bin index')
        ax.set_ylabel('Signal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_heatmap(out_path, data, title, label):
    plt.figure(figsize=(12, 6))
    plt.imshow(data, aspect='auto', cmap='coolwarm')
    plt.colorbar(label=label)
    plt.xlabel('Bin index')
    plt.ylabel('Sample index')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    print(f'Device: {device}')
    (model, model_meta) = load_combined_model(MODEL_PATH, device, SCALING_FILE)
    (events, df) = load_test_events(TEST_FILE)
    print(f'Features={model.feature_names}  target_mode={model.target_mode}  (using checkpoint X_mean/X_std)', flush=True)
    if model_meta.get('targets_precalibrated', True):
        print('Evaluating against pre-calibrated NPZ/P/Q test targets', flush=True)
    ps = np.stack([e.ps for e in events])
    ip_true = np.stack([e.iplus for e in events])
    im_true = np.stack([e.iminus for e in events])
    n_bins = ps.shape[1]
    (ip_pred, im_pred) = model.predict_iplus_iminus_events(events, spectrum_bins=n_bins)
    (p_pred, q_pred) = model.predict_events(events, spectrum_bins=n_bins)
    pred_mask = np.isfinite(p_pred) & np.isfinite(q_pred)
    if not pred_mask.any():
        raise ValueError('No finite predictions produced.')
    print(f'Loaded {model.num_models} bin models ({model.loaded_bin_indices[0]}..{model.loaded_bin_indices[-1]}), per-bin normalization.')
    if model.target_mode == 'pq':
        (p_true_bins, q_true_bins, p_true, q_true) = pq_truth_from_dataframe(df)
        (p_pred_bins, q_pred_bins) = (p_pred, q_pred)
    else:
        p_pred_bins = ip_pred + im_pred
        q_pred_bins = ip_pred - im_pred
        p_true_bins = ps
        q_true_bins = ip_true - im_true
        p_true = np.nanmean(p_true_bins, axis=1)
        q_true = np.nanmean(q_true_bins, axis=1)
    p_pred_int = np.nanmean(p_pred_bins, axis=1)
    q_pred_int = np.nanmean(q_pred_bins, axis=1)
    p_bin_mask = pred_mask & (np.abs(p_true_bins) > 1e-10)
    q_bin_mask = pred_mask & (np.abs(q_true_bins) > 1e-10)
    p_bin_rpe = compute_rpe(p_pred_bins, p_true_bins, p_bin_mask)
    q_bin_rpe = compute_rpe(q_pred_bins, q_true_bins, q_bin_mask)
    res_p_bins = np.where(pred_mask, p_pred_bins - p_true_bins, np.nan)
    res_q_bins = np.where(pred_mask, q_pred_bins - q_true_bins, np.nan)
    ip_rpe = compute_rpe(ip_pred, ip_true, pred_mask)
    im_rpe = compute_rpe(im_pred, im_true, pred_mask)
    ip_mask = pred_mask & (np.abs(ip_true) > 1e-10)
    im_mask = pred_mask & (np.abs(im_true) > 1e-10)
    res_ip = np.where(pred_mask, ip_pred - ip_true, np.nan)
    res_im = np.where(pred_mask, im_pred - im_true, np.nan)
    p_mask = np.abs(p_true) > 1e-10
    q_mask = np.abs(q_true) > 1e-10
    p_rpe = compute_rpe(p_pred_int, p_true, p_mask)
    q_rpe = compute_rpe(q_pred_int, q_true, q_mask)
    res_p = p_pred_int - p_true
    res_q = q_pred_int - q_true
    med_p_bin = np.nanmedian(p_bin_rpe, axis=0)
    med_q_bin = np.nanmedian(q_bin_rpe, axis=0)
    med_ip_bin = np.nanmedian(ip_rpe, axis=0)
    med_im_bin = np.nanmedian(im_rpe, axis=0)
    stats = {'n_samples': ps.shape[0], 'n_bins': n_bins, 'L1_P': np.mean(np.abs(p_pred_bins[pred_mask] - p_true_bins[pred_mask])), 'L1_Q': np.mean(np.abs(q_pred_bins[pred_mask] - q_true_bins[pred_mask])), 'median_RPE_P': np.nanmedian(p_bin_rpe[p_bin_mask]), 'median_RPE_Q': np.nanmedian(q_bin_rpe[q_bin_mask]), 'L1_Iplus': np.mean(np.abs(ip_pred[pred_mask] - ip_true[pred_mask])), 'L1_Iminus': np.mean(np.abs(im_pred[pred_mask] - im_true[pred_mask])), 'median_RPE_Iplus': np.nanmedian(ip_rpe[ip_mask]), 'median_RPE_Iminus': np.nanmedian(im_rpe[im_mask]), 'mean_RPE_P': np.nanmean(p_rpe), 'median_RPE_P_int': np.nanmedian(p_rpe), 'std_RPE_P': np.nanstd(p_rpe), 'mean_RPE_Q': np.nanmean(q_rpe), 'median_RPE_Q_int': np.nanmedian(q_rpe), 'std_RPE_Q': np.nanstd(q_rpe), 'mean_residual_P': np.mean(res_p), 'std_residual_P': np.std(res_p), 'mean_residual_Q': np.mean(res_q), 'std_residual_Q': np.std(res_q)}
    with open(os.path.join(OUTPUT_DIR, 'test_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    pd.DataFrame({'bin_idx': np.arange(n_bins), 'median_rpe_p': med_p_bin, 'median_rpe_q': med_q_bin, 'median_rpe_iplus': med_ip_bin, 'median_rpe_iminus': med_im_bin}).to_csv(os.path.join(OUTPUT_DIR, 'median_rpe_per_bin.csv'), index=False)
    print_results_table(stats)
    n_show = min(MAX_HEATMAP_SAMPLES, ps.shape[0])
    plot_heatmap(os.path.join(OUTPUT_DIR, 'residuals_heatmap_iplus.png'), res_ip[:n_show], 'I+ residuals', 'Residual')
    plot_heatmap(os.path.join(OUTPUT_DIR, 'residuals_heatmap_iminus.png'), res_im[:n_show], 'I- residuals', 'Residual')
    example_idx = select_example_indices(ps.shape[0], EXAMPLES, res_ip + res_im, EXAMPLE_SELECTION)
    plot_lineshape_examples(os.path.join(OUTPUT_DIR, 'lineshape_examples.png'), example_idx, ps, ip_true, im_true, ip_pred, im_pred, title_prefix=f'Sample ({EXAMPLE_SELECTION})')
    print(f'\nOutputs written to {OUTPUT_DIR}/')
if __name__ == '__main__':
    main()
