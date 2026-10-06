"""P/Q evaluation and reporting: metrics, plots, CSVs, example figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from utils.constants import (
    BATCH_SIZE, DEVICE, N_EXAMPLE_PLOTS, POL_ABS_BANDS, RPE_ABS_EPS, SEED,
    SOURCE_NAME, SOURCE_PROFILE, SPECTRUM_R_MAX, SPECTRUM_R_MIN,
)
from utils.helpers import json_ready


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

@torch.no_grad()
def predict_denormalized(model, dataset, stats, *, batch_size=BATCH_SIZE, device=DEVICE, label_stats=None):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=DEVICE.type == 'cuda')
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

def _plot_example_lineshape(ax, freq, ps):
    ax.plot(freq, ps, color='#1f2933', lw=1.9)
    ax.set_xlabel('R', fontsize=16)
    ax.set_ylabel('Amplitude', fontsize=16)
    _apply_axes_style(ax)
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
    spectra = arrays['spectra']
    num_bins = int(spectra.shape[2])
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
        _plot_example_lineshape(ax_s, freq, ps)
        path_e = examples_dir / f'test_example_{gi:05d}_nsteps{steps:04d}.png'
        fig_e.savefig(path_e, dpi=170, facecolor='white', bbox_inches='tight', pad_inches=0.12)
        plt.close(fig_e)
        saved.append(path_e)
        snr_v = float(snr_test[local_i]) if snr_test is not None else float('nan')
        info_rows.append({'filename': path_e.name, 'event_idx': gi, 'source': src, 'n_steps': steps, 'p0': p0_v, 'q0': q0_v, 'true_P': tp, 'true_Q': tq, 'pred_P': pp, 'pred_Q': pq, 'p0_pct': 100.0 * p0_v, 'q0_pct': 100.0 * q0_v, 'true_P_pct': 100.0 * tp, 'true_Q_pct': 100.0 * tq, 'pred_P_pct': 100.0 * pp, 'pred_Q_pct': 100.0 * pq, 'RPE_P_pct': rpe_p, 'RPE_Q_pct': rpe_q, 'applied_power': power, 'noise_std': noise_std, 'snr': snr_v})
        if axes is not None and panel_i < n_sum:
            ax_sum = axes[panel_i // sum_ncols, panel_i % sum_ncols]
            _plot_example_lineshape(ax_sum, freq, ps)
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

def write_pq_report(model, test_ds, stats, dataset_stats, arrays, history, best_val, out_dir, checkpoint_path, *, batch_size=BATCH_SIZE, n_examples=N_EXAMPLE_PLOTS, seed=SEED):
    print('Evaluating on test set...', flush=True)
    metrics = evaluate_model(model, test_ds, stats, batch_size=batch_size, label_stats=dataset_stats)
    print(f"Test RPE% median  P={metrics['P_rpe_median']:.3f}  Q={metrics['Q_rpe_median']:.3f}", flush=True)
    print(f"Test MAE  P={metrics['P_mae']:.6f}  Q={metrics['Q_mae']:.6f}  R²  P={metrics['P_r2']:.4f}  Q={metrics['Q_r2']:.4f}", flush=True)
    print_snr_stats(metrics)
    print_range_stats_table(metrics['range_stats_by_P'], title='Performance by |P_total| Range')
    if metrics['range_stats_by_p0']:
        print_range_stats_table(metrics['range_stats_by_p0'], title='Performance by |p0| Range')
    save_range_stats_csv(metrics['range_stats_by_P'], out_dir / 'range_stats_by_P.csv')
    if metrics['range_stats_by_p0']:
        save_range_stats_csv(metrics['range_stats_by_p0'], out_dir / 'range_stats_by_p0.csv')
    test_idx = np.asarray(dataset_stats.get('test_idx', stats.get('test_idx')), dtype=np.int64)
    pred_csv = out_dir / 'test_predictions.csv'
    save_predictions_csv(
        metrics, pred_csv, test_idx=test_idx,
        p0=np.asarray(arrays['p0']).reshape(-1)[test_idx] if 'p0' in arrays else None,
        source=np.asarray(arrays['source']).reshape(-1)[test_idx] if 'source' in arrays else None,
        n_steps=np.asarray(arrays['n_steps']).reshape(-1)[test_idx],
        applied_power=np.asarray(arrays['applied_power']).reshape(-1)[test_idx],
    )
    print(f'Saved test predictions -> {pred_csv}', flush=True)
    json_metrics = {k: v for (k, v) in metrics.items() if isinstance(v, (float, int, str, list))}
    json_metrics['best_val_loss'] = best_val
    json_metrics['checkpoint'] = str(checkpoint_path)
    with (out_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(json_ready(json_metrics), f, indent=2)
    print('Generating plots...', flush=True)
    save_plots(history, metrics, out_dir, best_val_loss=best_val)
    save_example_signal_plots(arrays, stats, metrics, out_dir, test_dataset=test_ds, input_stats=dataset_stats, n_examples=n_examples)
    save_example_signal_plots(
        arrays, stats, metrics, out_dir, test_dataset=test_ds, input_stats=dataset_stats,
        n_examples=n_examples, source_filter=SOURCE_PROFILE, examples_subdir='examples_optimal',
        summary_name='examples_optimal_summary.png', seed=seed + 1,
    )
    print(f'Done. Results in {out_dir}', flush=True)
