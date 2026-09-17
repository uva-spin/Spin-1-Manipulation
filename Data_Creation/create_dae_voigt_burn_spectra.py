"""
Generate full-spectrum manipulated lineshapes for DAE training.

For each polarization on a grid:
  - find burn-window bins where equilibrium Q < 0 (noiseless initial lineshape)
  - ssRF: burn only at those Q < 0 centers; for each center, extract spectra at
    burn lengths 0 .. max_burn_steps from one trajectory (noise added after)
  - AFP: flip at each Q < 0 bin with an AFP window and save the immediately
    post-flip spectrum (n_steps = 0). With ``--afp-relax``, then relax for
    0 .. max_relax_steps, emitting one full-spectrum event per selected
    relax frame (n_steps = relax steps completed after the flip)

ssRF and AFP are never combined in the same event. Toggle modes with
``--ssrf`` / ``--no-ssrf`` and ``--afp`` / ``--no-afp``; when both are on,
events are emitted separately. ``--afp-relax`` implies AFP.

Saved NPZ fields:
  spectra       (N, 2, num_bins)  channel0=I+, channel1=I-
                                  one row per ssRF burn step / AFP post-flip
                                  or AFP relax step
  p0            (N,)              initial vector polarization
  P_total       (N,)              integrated lineshape P after this step
                                  (CC_total * sum(I++I-) with p0 post-correction)
  Q_total       (N,)              integrated lineshape Q after this step
                                  (CC_total * sum(I+-I-) with p0 post-correction)
  applied_power (N,)              gamma_rf for ssRF; 0 for AFP
  n_steps       (N,)              ssRF: burn macro-steps so far;
                                  AFP: relax steps completed after flip
                                  (0 = immediately post-AFP)
  center_bin    (N,)              RF / AFP center bin
  source        (N,)              0=ssRF, 1=AFP

Examples (from repo root):
  python Data_Creation/create_dae_voigt_burn_spectra.py --quick
  python Data_Creation/create_dae_voigt_burn_spectra.py --ssrf --afp
  python Data_Creation/create_dae_voigt_burn_spectra.py --ssrf --no-afp
  python Data_Creation/create_dae_voigt_burn_spectra.py --afp
  python Data_Creation/create_dae_voigt_burn_spectra.py --afp-relax --max-relax-steps 100
  python Data_Creation/create_dae_voigt_burn_spectra.py --ssrf --max-burn-steps 150
"""
import argparse
import sys
from pathlib import Path
import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
DULYA = SCRIPT_DIR / 'rivanna'
if str(DULYA) not in sys.path:
    sys.path.insert(0, str(DULYA))
from afp_bin_traj import run_one_polarization as run_afp_one
from bin_setup import get_shape_params, polarization_grid
from burn_selection import equilibrium_q_profile
from common import BURN_BIN_CHOICES, BURN_R_MAX, BURN_R_MIN, FREQUENCY, NUM_BINS, RF_GAUSSIAN_FWHM_R, RF_LORENTZIAN_FWHM_R, RF_MODE_PHYSICAL_VOIGT, SOURCE_AFP, SOURCE_SSRF
from pq_calibration import integrated_pq_from_bins, load_pq_calibration, post_correct_ratio
from ssrf_bin_traj import run_one_polarization as run_ssrf_one
DEFAULT_OUTPUT = SCRIPT_DIR / 'dae_voigt_burn_spectra' / 'spectra.npz'
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'dae_voigt_burn_spectra' / 'plots'
P_MIN = 0.2
P_MAX = 0.6
P_STEP = 0.05
GAMMA_RF = 10.0
MIN_BURN_STEPS = 0
MAX_BURN_STEPS = 100
BURN_STEPS_STEP = 1
DT = 0.0055
GAUSSIAN_FWHM_R = 3 * RF_GAUSSIAN_FWHM_R
LORENTZIAN_FWHM_R = 3 * RF_LORENTZIAN_FWHM_R
AFP_WINDOW = 8
MIN_RELAX_STEPS = 0
MAX_RELAX_STEPS = 2000
RELAX_STEPS_STEP = 1
STORE_DTYPE = np.float32
DEFAULT_SEED = 42
NOISE_LEVEL = 0.0001

def _burn_window_bins():
    return np.asarray(BURN_BIN_CHOICES, dtype=np.int32)

def q_negative_bins_for_p0(p0, burn_window, *, shape_params):
    """Burn-window bin indices where equilibrium Q = I+ - I- is negative.

    Uses the noiseless Dulya equilibrium lineshape so center selection is
    independent of observation noise added later to saved spectra.
    """
    q = equilibrium_q_profile(p0, shape_params=shape_params)
    mask = q[burn_window] < 0.0
    return burn_window[mask]

def burn_steps_values(min_steps, max_steps, step):
    """Inclusive integer step grid from ``min_steps`` to ``max_steps``."""
    return np.arange(min_steps, max_steps + 1, step, dtype=np.int32)

def _spectrum_totals(iplus, iminus, p0, *, calibration):
    """Total integrated lineshape P and Q after a manipulation step.

    Uses the full-spectrum convention:
      P = CC_total * sum_b (I+_b + I-_b)
      Q = CC_total * sum_b (I+_b - I-_b)
    then applies the equilibrium p0 post-correction ratio so the scale matches
    vector / tensor polarization (same convention as mean of CC_bin-calibrated
    per-bin spectra).
    """
    ip = np.asarray(iplus).reshape(-1)
    im = np.asarray(iminus).reshape(-1)
    ps = ip + im
    q = ip - im
    (p_int, q_int) = integrated_pq_from_bins(ps, q, calibration=calibration)
    ratio = post_correct_ratio(np.asarray([p0]), calibration)[0]
    return (p_int * ratio, q_int * ratio)

def _append_event(rows, *, iplus, iminus, p0, p_total, q_total, applied_power, n_steps, center_bin, source):
    rows['spectra'].append(np.stack([np.asarray(iplus, dtype=STORE_DTYPE).reshape(-1), np.asarray(iminus, dtype=STORE_DTYPE).reshape(-1)], axis=0))
    rows['p0'].append(p0)
    rows['P_total'].append(p_total)
    rows['Q_total'].append(q_total)
    rows['applied_power'].append(applied_power)
    rows['n_steps'].append(n_steps)
    rows['center_bin'].append(center_bin)
    rows['source'].append(source)

def _empty_rows():
    return {'spectra': [], 'p0': [], 'P_total': [], 'Q_total': [], 'applied_power': [], 'n_steps': [], 'center_bin': [], 'source': []}

def generate_ssrf_events(p_values, burn_window, steps_grid, *, shape_params, calibration, gamma_rf, max_centers_per_p=None):
    """ssRF Voigt burns; save the full I+/I- spectrum at every requested burn step.

    Centers are burn-window bins with equilibrium (pre-noise) Q < 0. Runs one
    trajectory of length ``max(steps_grid)`` per (P, center) with
    ``capture_spectrum=True``, then writes a separate event for each macro-step
    index in ``steps_grid`` (index 0 = pre-burn equilibrium). Observation noise
    is added only when saving spectra, after P/Q totals are computed.
    """
    rows = _empty_rows()
    if steps_grid.size == 0:
        return rows
    max_burn = np.max(steps_grid)
    step_set = {s for s in steps_grid}
    for (ip, p0) in enumerate(np.asarray(p_values)):
        centers = q_negative_bins_for_p0(p0, burn_window, shape_params=shape_params)
        if max_centers_per_p is not None and centers.size > max_centers_per_p:
            centers = centers[:max_centers_per_p]
        print(f'  ssRF P={p0:.3f} ({ip + 1}/{len(p_values)}): {centers.size} Q<0 centers × {steps_grid.size} step spectra', flush=True)
        for bin_idx in centers:
            traj = run_ssrf_one(bin_idx, p0, dt=DT, gamma_rf=gamma_rf, n_steps=max_burn, rf_mode=RF_MODE_PHYSICAL_VOIGT, gaussian_fwhm_R=GAUSSIAN_FWHM_R, lorentzian_fwhm_R=LORENTZIAN_FWHM_R, shape_params=shape_params, capture_spectrum=True)
            if traj.get('skipped', False):
                continue
            iplus_full = np.asarray(traj['iplus_full'])
            iminus_full = np.asarray(traj['iminus_full'])
            for burn_steps in sorted(step_set):
                k = burn_steps
                ip_spec = np.asarray(iplus_full[k]).copy()
                im_spec = np.asarray(iminus_full[k]).copy()
                (p_total, q_total) = _spectrum_totals(ip_spec, im_spec, p0, calibration=calibration)
                ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
                im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
                _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=p_total, q_total=q_total, applied_power=gamma_rf, n_steps=burn_steps, center_bin=bin_idx, source=SOURCE_SSRF)
    return rows

def generate_afp_events(p_values, burn_window, relax_steps_grid, *, shape_params, calibration, afp_window, max_centers_per_p=None):
    """AFP flip, then save the full I+/I- spectrum at each requested relax step.

    Runs one trajectory of length ``max(relax_steps_grid)`` per (P, center) with
    ``capture_spectrum=True``. Frame 0 is immediately post-flip; later frames are
    after each relax macro-step. A grid of only ``[0]`` is the post-flip snapshot
    with no relaxation. Emits a separate event for each index in
    ``relax_steps_grid`` with ``n_steps`` equal to that index.
    """
    rows = _empty_rows()
    if relax_steps_grid.size == 0:
        return rows
    max_relax = np.max(relax_steps_grid)
    step_set = {s for s in relax_steps_grid}
    for (ip, p0) in enumerate(np.asarray(p_values)):
        centers = q_negative_bins_for_p0(p0, burn_window, shape_params=shape_params)
        if max_centers_per_p is not None and centers.size > max_centers_per_p:
            centers = centers[:max_centers_per_p]
        print(f"  AFP P={p0:.3f} ({ip + 1}/{len(p_values)}): {centers.size} Q<0 centers × {relax_steps_grid.size} {('relax' if max_relax > 0 else 'post-flip')} spectra (window={afp_window}, n_relax={max_relax})", flush=True)
        for bin_idx in centers:
            traj = run_afp_one(bin_idx, p0, dt=DT, n_relax=max_relax, afp_window=afp_window, shape_params=shape_params, capture_spectrum=True)
            if traj.get('skipped', False):
                continue
            iplus_full = np.asarray(traj['iplus_full'])
            iminus_full = np.asarray(traj['iminus_full'])
            for relax_steps in sorted(step_set):
                k = relax_steps
                ip_spec = np.asarray(iplus_full[k]).copy()
                im_spec = np.asarray(iminus_full[k]).copy()
                (p_total, q_total) = _spectrum_totals(ip_spec, im_spec, p0, calibration=calibration)
                ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
                im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
                _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=p_total, q_total=q_total, applied_power=0.0, n_steps=relax_steps, center_bin=bin_idx, source=SOURCE_AFP)
    return rows

def _merge_rows(*row_groups):
    merged = _empty_rows()
    for rows in row_groups:
        for key in merged:
            merged[key].extend(rows[key])
    n = len(merged['spectra'])
    if n == 0:
        return {'spectra': np.empty((0, 2, NUM_BINS), dtype=STORE_DTYPE), 'p0': np.empty(0, dtype=STORE_DTYPE), 'P_total': np.empty(0, dtype=STORE_DTYPE), 'Q_total': np.empty(0, dtype=STORE_DTYPE), 'applied_power': np.empty(0, dtype=STORE_DTYPE), 'n_steps': np.empty(0, dtype=np.int32), 'center_bin': np.empty(0, dtype=np.int32), 'source': np.empty(0, dtype=np.uint8)}
    return {'spectra': np.stack(merged['spectra'], axis=0).astype(STORE_DTYPE, copy=False), 'p0': np.asarray(merged['p0'], dtype=STORE_DTYPE), 'P_total': np.asarray(merged['P_total'], dtype=STORE_DTYPE), 'Q_total': np.asarray(merged['Q_total'], dtype=STORE_DTYPE), 'applied_power': np.asarray(merged['applied_power'], dtype=STORE_DTYPE), 'n_steps': np.asarray(merged['n_steps'], dtype=np.int32), 'center_bin': np.asarray(merged['center_bin'], dtype=np.int32), 'source': np.asarray(merged['source'], dtype=np.uint8)}

def _event_mask(data, *, source=None, p0=None, center_bin=None, n_steps=None):
    n = data['spectra'].shape[0]
    mask = np.ones(n, dtype=bool)
    if source is not None:
        mask &= data['source'] == source
    if p0 is not None:
        mask &= np.isclose(data['p0'], p0, atol=1e-05, rtol=0.0)
    if center_bin is not None:
        mask &= data['center_bin'] == center_bin
    if n_steps is not None:
        mask &= data['n_steps'] == n_steps
    return mask

def _pick_ssrf_example_keys(data, *, max_examples=2):
    """Select a few (p0, center_bin) pairs with the most ssRF step coverage."""
    mask = data['source'] == SOURCE_SSRF
    if not np.any(mask):
        return []
    p0s = data['p0'][mask]
    bins = data['center_bin'][mask]
    steps = data['n_steps'][mask]
    pairs = {}
    for (p0, b, s) in zip(p0s.tolist(), bins.tolist(), steps.tolist()):
        key = (round(p0, 5), b)
        pairs.setdefault(key, set()).add(s)
    ranked = sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))
    return [k for (k, _) in ranked[:max(0, max_examples)]]

def _pick_afp_example_keys(data, *, max_examples=2):
    """Select a few (p0, center_bin) pairs with the most AFP relax-step coverage."""
    mask = data['source'] == SOURCE_AFP
    if not np.any(mask):
        return []
    p0s = data['p0'][mask]
    bins = data['center_bin'][mask]
    steps = data['n_steps'][mask]
    pairs = {}
    for (p0, b, s) in zip(p0s.tolist(), bins.tolist(), steps.tolist()):
        key = (round(p0, 5), b)
        pairs.setdefault(key, set()).add(s)
    ranked = sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))
    return [k for (k, _) in ranked[:max(0, max_examples)]]

def save_example_plots(data, plot_dir, *, frequency=None, max_ssrf_examples=2, max_afp_examples=2):
    """Write a few diagnostic PNGs for ssRF burn and AFP relax evolution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    f = np.asarray(FREQUENCY if frequency is None else frequency).reshape(-1)
    saved = []
    for (p0, center) in _pick_ssrf_example_keys(data, max_examples=max_ssrf_examples):
        mask = _event_mask(data, source=SOURCE_SSRF, p0=p0, center_bin=center)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        order = np.argsort(data['n_steps'][idx])
        idx = idx[order]
        steps = data['n_steps'][idx]
        spectra = data['spectra'][idx]
        p_tot = data['P_total'][idx]
        q_tot = data['Q_total'][idx]
        power = data['applied_power'][idx[0]]
        (fig, axes) = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
        (ax_ps, ax_q) = axes
        n_show = min(6, idx.size)
        show_i = np.unique(np.round(np.linspace(0, idx.size - 1, n_show)).astype(int))
        cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(show_i)))
        for (color, j) in zip(cmap, show_i):
            ip = spectra[j, 0]
            im = spectra[j, 1]
            ps = ip + im
            q = ip - im
            step = steps[j]
            ax_ps.plot(f, ps, color=color, lw=1.4, label=f'n={step}')
            ax_q.plot(f, q, color=color, lw=1.2, label=f'n={step}')
        for ax in axes:
            ax.axvline(f[center], color='0.35', ls=':', lw=1.0, label='center')
            ax.grid(True, alpha=0.3)
        ax_ps.set_ylabel('$P_s = I_+ + I_-$')
        ax_ps.set_title(f'ssRF Voigt burn  P0={p0:.3f}  center={center}  γ_rf={power:.1f}  P_tot∈[{p_tot.min():.3f},{p_tot.max():.3f}]')
        ax_ps.legend(fontsize=8, ncols=3, loc='upper right')
        ax_q.set_xlabel('R')
        ax_q.set_ylabel('$Q = I_+ - I_-$')
        ax_q.legend(fontsize=8, ncols=3, loc='upper right')
        fig.tight_layout()
        path = plot_dir / f'ssrf_ps_q_steps_P{p0:.2f}_bin{center:04d}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)
        (fig, ax) = plt.subplots(figsize=(10.5, 4.8))
        (j0, j1) = (0, idx.size - 1)
        ax.plot(f, spectra[j0, 0], color='tab:red', ls='--', alpha=0.55, label=f'$I_+$ n={steps[j0]}')
        ax.plot(f, spectra[j0, 1], color='tab:blue', ls='--', alpha=0.55, label=f'$I_-$ n={steps[j0]}')
        ax.plot(f, spectra[j0, 0] + spectra[j0, 1], color='tab:green', ls='--', alpha=0.55, label=f'$P_s$ n={steps[j0]}')
        ax.plot(f, spectra[j1, 0], color='tab:red', lw=1.5, label=f'$I_+$ n={steps[j1]}')
        ax.plot(f, spectra[j1, 1], color='tab:blue', lw=1.5, label=f'$I_-$ n={steps[j1]}')
        ax.plot(f, spectra[j1, 0] + spectra[j1, 1], color='tab:green', lw=1.5, label=f'$P_s$ n={steps[j1]}')
        ax.axvline(f[center], color='0.35', ls=':', lw=1.0)
        ax.set_xlabel('R')
        ax.set_ylabel('intensity (fit scale)')
        ax.set_title(f'ssRF I+/I−  P0={p0:.3f}  bin={center}  P_tot={p_tot[j1]:.4f}  Q_tot={q_tot[j1]:.4f}')
        ax.legend(fontsize=8, ncols=2, loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = plot_dir / f'ssrf_ip_im_P{p0:.2f}_bin{center:04d}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)
    for (p0, center) in _pick_afp_example_keys(data, max_examples=max_afp_examples):
        mask = _event_mask(data, source=SOURCE_AFP, p0=p0, center_bin=center)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        order = np.argsort(data['n_steps'][idx])
        idx = idx[order]
        steps = data['n_steps'][idx]
        spectra = data['spectra'][idx]
        p_tot = data['P_total'][idx]
        q_tot = data['Q_total'][idx]
        eq_mask = _event_mask(data, source=SOURCE_SSRF, p0=p0, center_bin=center, n_steps=0)
        if not np.any(eq_mask):
            eq_mask = _event_mask(data, source=SOURCE_SSRF, p0=p0, n_steps=0)
        eq_idx = np.flatnonzero(eq_mask)
        if idx.size > 1:
            (fig, axes) = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
            (ax_ps, ax_q) = axes
            n_show = min(6, idx.size)
            show_i = np.unique(np.round(np.linspace(0, idx.size - 1, n_show)).astype(int))
            cmap = plt.cm.plasma(np.linspace(0.15, 0.9, len(show_i)))
            for (color, j) in zip(cmap, show_i):
                ip = spectra[j, 0]
                im = spectra[j, 1]
                step = steps[j]
                ax_ps.plot(f, ip + im, color=color, lw=1.4, label=f'n={step}')
                ax_q.plot(f, ip - im, color=color, lw=1.2, label=f'n={step}')
            if eq_idx.size:
                e = eq_idx[0]
                ip0 = data['spectra'][e, 0]
                im0 = data['spectra'][e, 1]
                ax_ps.plot(f, ip0 + im0, color='0.45', ls='--', lw=1.2, label='eq $P_s$')
                ax_q.plot(f, ip0 - im0, color='0.45', ls='--', lw=1.0, label='eq $Q$')
            for ax in axes:
                ax.axvline(f[center], color='green', ls=':', lw=1.1, label='center')
                ax.grid(True, alpha=0.3)
            ax_ps.set_ylabel('$P_s = I_+ + I_-$')
            ax_ps.set_title(f'AFP + relax  P0={p0:.3f}  center={center}  window={AFP_WINDOW}  P_tot∈[{p_tot.min():.3f},{p_tot.max():.3f}]')
            ax_ps.legend(fontsize=8, ncols=3, loc='upper right')
            ax_q.set_xlabel('R')
            ax_q.set_ylabel('$Q = I_+ - I_-$')
            ax_q.legend(fontsize=8, ncols=3, loc='upper right')
            fig.tight_layout()
            path = plot_dir / f'afp_ps_q_relax_P{p0:.2f}_bin{center:04d}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path)
            (fig, ax) = plt.subplots(figsize=(10.5, 4.8))
            (j0, j1) = (0, idx.size - 1)
            ax.plot(f, spectra[j0, 0], color='tab:red', ls='--', alpha=0.55, label=f'$I_+$ n={steps[j0]}')
            ax.plot(f, spectra[j0, 1], color='tab:blue', ls='--', alpha=0.55, label=f'$I_-$ n={steps[j0]}')
            ax.plot(f, spectra[j1, 0], color='tab:red', lw=1.5, label=f'$I_+$ n={steps[j1]}')
            ax.plot(f, spectra[j1, 1], color='tab:blue', lw=1.5, label=f'$I_-$ n={steps[j1]}')
            ax.axvline(f[center], color='green', ls=':', lw=1.1)
            ax.set_xlabel('R')
            ax.set_ylabel('intensity (fit scale)')
            ax.set_title(f'AFP I+/I−  P0={p0:.3f}  bin={center}  P_tot={p_tot[j1]:.4f}  Q_tot={q_tot[j1]:.4f}')
            ax.legend(fontsize=8, ncols=2, loc='upper right')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = plot_dir / f'afp_ip_im_relax_P{p0:.2f}_bin{center:04d}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path)
        else:
            j = idx[0]
            ip = spectra[0, 0]
            im = spectra[0, 1]
            ps = ip + im
            q = ip - im
            (fig, axes) = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
            (ax_ps, ax_ip) = axes
            if eq_idx.size:
                e = eq_idx[0]
                ip0 = data['spectra'][e, 0]
                im0 = data['spectra'][e, 1]
                ax_ps.plot(f, ip0 + im0, color='0.45', ls='--', lw=1.2, label='eq $P_s$')
                ax_ip.plot(f, ip0, color='tab:red', ls='--', alpha=0.5, label='eq $I_+$')
                ax_ip.plot(f, im0, color='tab:blue', ls='--', alpha=0.5, label='eq $I_-$')
            ax_ps.plot(f, ps, color='black', lw=1.5, label='AFP $P_s$')
            ax_ps.plot(f, q, color='tab:orange', lw=1.2, label='AFP $Q$')
            ax_ip.plot(f, ip, color='tab:red', lw=1.5, label='AFP $I_+$')
            ax_ip.plot(f, im, color='tab:blue', lw=1.5, label='AFP $I_-$')
            for ax in axes:
                ax.axvline(f[center], color='green', ls=':', lw=1.1, label='center')
                ax.grid(True, alpha=0.3)
            ax_ps.set_ylabel('$P_s$, $Q$')
            ax_ps.set_title(f'AFP post-flip (n_relax=0)  P0={p0:.3f}  center={center}  window={AFP_WINDOW}  P_tot={p_tot[0]:.4f}  Q_tot={q_tot[0]:.4f}')
            ax_ps.legend(fontsize=8, ncols=3, loc='upper right')
            ax_ip.set_xlabel('R')
            ax_ip.set_ylabel('intensity (fit scale)')
            ax_ip.legend(fontsize=8, ncols=2, loc='upper right')
            fig.tight_layout()
            path = plot_dir / f'afp_apply_P{p0:.2f}_bin{center:04d}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path)
    return saved

def generate_spectra(*, do_ssrf, do_afp, do_afp_relax=False, p_min=P_MIN, p_max=P_MAX, p_step=P_STEP, min_burn_steps=MIN_BURN_STEPS, max_burn_steps=MAX_BURN_STEPS, burn_steps_step=BURN_STEPS_STEP, min_relax_steps=MIN_RELAX_STEPS, max_relax_steps=MAX_RELAX_STEPS, relax_steps_step=RELAX_STEPS_STEP, gamma_rf=GAMMA_RF, afp_window=AFP_WINDOW, max_centers_per_p=None, p_values=None):
    """Build full-spectrum manipulated events for the requested modes."""
    burn_window = _burn_window_bins()
    shape_params = get_shape_params()
    calibration = load_pq_calibration(num_bins=NUM_BINS)
    if p_values is None:
        p_values = polarization_grid(p_min, p_max, p_step)
        p_values = p_values[p_values > 0.0]
    else:
        p_values = np.asarray(p_values).reshape(-1)
    groups = []
    if do_ssrf:
        steps_grid = burn_steps_values(min_burn_steps, max_burn_steps, burn_steps_step)
        groups.append(generate_ssrf_events(p_values, burn_window, steps_grid, shape_params=shape_params, calibration=calibration, gamma_rf=gamma_rf, max_centers_per_p=max_centers_per_p))
    if do_afp:
        relax_grid = _afp_relax_grid(do_afp_relax, min_relax_steps=min_relax_steps, max_relax_steps=max_relax_steps, relax_steps_step=relax_steps_step)
        groups.append(generate_afp_events(p_values, burn_window, relax_grid, shape_params=shape_params, calibration=calibration, afp_window=afp_window, max_centers_per_p=max_centers_per_p))
    return _merge_rows(*groups)

def _afp_relax_grid(do_afp_relax, *, min_relax_steps, max_relax_steps, relax_steps_step):
    """Relax-step indices to save. Without relaxation this is only post-flip (0)."""
    if not do_afp_relax:
        return np.asarray([0], dtype=np.int32)
    return burn_steps_values(min_relax_steps, max_relax_steps, relax_steps_step)

def _resolve_modes(*, ssrf, afp, afp_relax, quick):
    """Resolve ``--ssrf/--no-ssrf``, ``--afp/--no-afp``, and ``--afp-relax``.

    Defaults when no mode flag is given (``--afp-relax`` counts as a mode flag):
      - ``--quick`` → ssRF and AFP on, relaxation off (post-flip only)
      - otherwise → ssRF on, AFP off
    When any mode flag is given, unspecified modes default to off.
    ``--afp-relax`` implies AFP. ``--no-afp --afp-relax`` is an error.
    """
    if afp is False and afp_relax:
        raise ValueError('--afp-relax requires AFP; cannot combine with --no-afp')
    ssrf_set = ssrf is not None
    afp_set = afp is not None
    relax_set = afp_relax
    if not ssrf_set and (not afp_set) and (not relax_set):
        if quick:
            return (True, True, False)
        return (True, False, False)
    do_ssrf = ssrf if ssrf_set else False
    do_afp = afp if afp_set else False
    do_afp_relax = afp_relax
    if do_afp_relax:
        do_afp = True
    if not do_afp:
        do_afp_relax = False
    return (do_ssrf, do_afp, do_afp_relax)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate full-spectrum ssRF and/or AFP manipulated lineshapes centered only at burn-window bins where initial (equilibrium) Q < 0. AFP saves the immediately post-flip spectrum unless --afp-relax is set, in which case one event is emitted per selected relax step. ssRF and AFP are never applied in the same event.')
    parser.add_argument('--ssrf', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable ssRF Voigt-burn events (--ssrf / --no-ssrf; default: on if no mode flags, else off)')
    parser.add_argument('--afp', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable AFP events (--afp / --no-afp; default: off unless --quick with no mode flags). Without --afp-relax, saves only the immediately post-flip spectrum (n_steps = 0)')
    parser.add_argument('--afp-relax', action=argparse.BooleanOptionalAction, default=False, help='After each AFP flip, emit spectra along a relaxation trajectory (--afp-relax / --no-afp-relax; default: off). Implies --afp. Uses --min-relax-steps / --max-relax-steps / --relax-steps-step; n_steps = relax steps after flip')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help=f'Output .npz path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--plot-dir', type=Path, default=None, help=f'Directory for example PNGs (default: <output-parent>/plots, e.g. {DEFAULT_PLOT_DIR})')
    parser.add_argument('--no-plots', action='store_true', help='Skip writing example diagnostic plots')
    parser.add_argument('--p-min', type=float, default=P_MIN, help=f'Minimum polarization (default: {P_MIN})')
    parser.add_argument('--p-max', type=float, default=P_MAX, help=f'Maximum polarization (default: {P_MAX})')
    parser.add_argument('--p-step', type=float, default=P_STEP, help=f'Polarization grid step (default: {P_STEP})')
    parser.add_argument('--min-burn-steps', type=int, default=MIN_BURN_STEPS, help=f'Minimum ssRF burn length (default: {MIN_BURN_STEPS})')
    parser.add_argument('--max-burn-steps', type=int, default=MAX_BURN_STEPS, help=f'Maximum ssRF burn length (default: {MAX_BURN_STEPS})')
    parser.add_argument('--burn-steps-step', type=int, default=BURN_STEPS_STEP, help=f'ssRF burn-length stride (default: {BURN_STEPS_STEP})')
    parser.add_argument('--min-relax-steps', type=int, default=MIN_RELAX_STEPS, help=f'Minimum AFP relax steps after flip when --afp-relax is on (default: {MIN_RELAX_STEPS})')
    parser.add_argument('--max-relax-steps', type=int, default=MAX_RELAX_STEPS, help=f'Maximum AFP relax steps after flip when --afp-relax is on (default: {MAX_RELAX_STEPS})')
    parser.add_argument('--relax-steps-step', type=int, default=RELAX_STEPS_STEP, help=f'AFP relax-step stride when --afp-relax is on (default: {RELAX_STEPS_STEP})')
    parser.add_argument('--gamma-rf', type=float, default=GAMMA_RF, help=f'ssRF applied power gamma_rf (default: {GAMMA_RF})')
    parser.add_argument('--afp-window', type=int, default=AFP_WINDOW, help=f'AFP subset window width in bins (default: {AFP_WINDOW})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'RNG seed reserved for future stochastic options (default: {DEFAULT_SEED})')
    parser.add_argument('--quick', action='store_true', help='Smoke run: 2 polarizations, few centers, short ssRF burn grid, both modes; AFP is post-flip only unless --afp-relax')
    return parser.parse_args()

def main():
    args = parse_args()
    try:
        (do_ssrf, do_afp, do_afp_relax) = _resolve_modes(ssrf=args.ssrf, afp=args.afp, afp_relax=args.afp_relax, quick=args.quick)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.quick:
        p_values = np.asarray([0.3, 0.5])
        min_burn_steps = 0
        max_burn_steps = 10
        burn_steps_step = 5
        min_relax_steps = 0
        max_relax_steps = 10
        relax_steps_step = 5
        max_centers_per_p = 3
        afp_quick = 'AFP relax {0,5,10}' if do_afp_relax else 'AFP post-flip only'
        print(f'Quick smoke: P={{0.30, 0.50}}, 3 centers/P, ssRF steps {{0,5,10}}, {afp_quick}', flush=True)
    else:
        p_values = None
        min_burn_steps = args.min_burn_steps
        max_burn_steps = args.max_burn_steps
        burn_steps_step = args.burn_steps_step
        min_relax_steps = args.min_relax_steps
        max_relax_steps = args.max_relax_steps
        relax_steps_step = args.relax_steps_step
        max_centers_per_p = None
    modes = []
    if do_ssrf:
        modes.append('ssRF')
    if do_afp:
        modes.append('AFP+relax' if do_afp_relax else 'AFP')
    if do_afp_relax:
        afp_relax_desc = f'relax [{min_relax_steps}, {max_relax_steps}] stride {relax_steps_step}'
    elif do_afp:
        afp_relax_desc = 'post-flip only (n_steps=0)'
    else:
        afp_relax_desc = 'off'
    print(f"Generating {'+'.join(modes)} full spectra | P in [{args.p_min}, {args.p_max}] step {args.p_step} | ssRF gamma_rf={args.gamma_rf} steps [{min_burn_steps}, {max_burn_steps}] stride {burn_steps_step} | ssRF centers: equilibrium Q<0 only | AFP window={args.afp_window} {afp_relax_desc} | dt={DT} | burn R in ({BURN_R_MIN}, {BURN_R_MAX}) | Voigt FWHM G={GAUSSIAN_FWHM_R:.3f} L={LORENTZIAN_FWHM_R:.3f}", flush=True)
    data = generate_spectra(do_ssrf=do_ssrf, do_afp=do_afp, do_afp_relax=do_afp_relax, p_min=args.p_min, p_max=args.p_max, p_step=args.p_step, min_burn_steps=min_burn_steps, max_burn_steps=max_burn_steps, burn_steps_step=burn_steps_step, min_relax_steps=min_relax_steps, max_relax_steps=max_relax_steps, relax_steps_step=relax_steps_step, gamma_rf=args.gamma_rf, afp_window=args.afp_window, max_centers_per_p=max_centers_per_p, p_values=p_values)
    np.savez_compressed(output, **data)
    n = data['spectra'].shape[0]
    n_ssrf = np.sum(data['source'] == SOURCE_SSRF)
    n_afp = np.sum(data['source'] == SOURCE_AFP)
    print(f"Saved N={n} events (ssRF={n_ssrf}, AFP={n_afp}) spectra={data['spectra'].shape} dtype={data['spectra'].dtype} -> {output}", flush=True)
    if not args.no_plots and n > 0:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else output.parent / 'plots'
        paths = save_example_plots(data, plot_dir)
        if paths:
            print(f'Wrote {len(paths)} example plots -> {plot_dir}', flush=True)
            for p in paths:
                print(f'  {p.name}', flush=True)
        else:
            print(f'No example plots produced (empty selection) -> {plot_dir}', flush=True)
if __name__ == '__main__':
    main()
