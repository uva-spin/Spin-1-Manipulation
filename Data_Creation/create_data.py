"""
Generate full-spectrum manipulated lineshapes for DAE training.

For each polarization on a grid:
  - find burn-window bins where equilibrium Q < 0 (noiseless initial lineshape)
  - ssRF: burn only at those Q < 0 centers; for each center, extract spectra at
    burn lengths 0 .. max_burn_steps from one trajectory (noise added after)
  - AFP: flip at each Q < 0 bin with an AFP window and save the immediately
    post-flip spectrum (n_steps = 0). With ``--afp-relax``, then relax for
    0 .. max_relax_steps, emitting one full-spectrum event per selected
    relax frame (n_steps = relax steps completed after the flip).
    Relaxation returns the lineshape to the pre-AFP packet, so both vector
    polarization and tensor polarization go back to their initial values.
  - AFP Profile (``--afp-profile``): one sweep over every burn-window bin
    with initial Q < 0, then the same return to the pre-AFP state. Saved as its own
    source (4). center_bin = -1 and power_profile is 1 on the swept bins.
  - optimal profile: apply the ssRF-beta synchronized RF power profile as its
    own event stream; save the spectrum at each Euler step of that program
    (independent of per-bin Voigt burns)
  - unmanipulated (``--unmanipulated``): one equilibrium full-spectrum event per
    polarization with no RF (source=2). n_steps=0, applied_power=0,
    zero power_profile, center_bin=-1.

Separate-mode streams never mix ssRF/AFP/profile in one row unless
``--ssrf-afp-combined`` is set (source=5; see ``combo_scenario`` /
``combo_layout``). Toggle with ``--ssrf`` / ``--no-ssrf``, ``--afp`` /
``--no-afp``, ``--profile`` / ``--no-profile``, ``--unmanipulated`` /
``--no-unmanipulated``, and ``--ssrf-afp-combined``. ``--afp-relax`` implies AFP.
``--afp-profile`` is the AFP Profile source: one Q<0 sweep plus relaxation.

Saved NPZ fields:
  spectra       (N, 2, num_bins)  channel0=I+, channel1=I-
                                  one row per ssRF burn step / AFP post-flip
                                  or AFP relax step / profile Euler step /
                                  unmanipulated equilibrium
  p0            (N,)              initial vector polarization
  P_total       (N,)              population P = n+ − n− after this step
  Q_total       (N,)              population Q = n+ − 2 n0 + n− after this step
  applied_power (N,)              gamma_rf for ssRF; 0 for AFP / unmanipulated;
                                  peak U for profile
  power_profile (N, num_bins)     per-bin RF envelope: Voigt*gamma_rf (ssRF),
                                  zeros (per-bin AFP / unmanipulated), 1 on
                                  swept bins (Q<0 AFP profile), U(R) (optimal profile)
  n_steps       (N,)              ssRF: burn macro-steps so far;
                                  AFP: relax steps (0 = immediately post-AFP);
                                  combined: 0 on the AFP snapshot, then ssRF
                                  burn steps, then relax steps;
                                  unmanipulated: always 0;
                                  profile: Euler steps of the PulseProgram
  center_bin    (N,)              RF / AFP center bin; -1 for whole-line
                                  (profile / AFP Profile / unmanipulated)
  source        (N,)              0=ssRF, 1=AFP, 2=unmanipulated,
                                  3=optimal profile, 4=AFP Profile, 5=ssRF+AFP combined
  combo_scenario (N,) uint8       0 unless source=5 (region-order scenario)
  combo_layout   (N,) uint8       0 unless source=5 (profile vs selective mix)
  meta_json     JSON              source_codes for all enabled sources

Examples (from repo root):
  python Data_Creation/create_data.py --quick
  python Data_Creation/create_data.py --ssrf --afp
  python Data_Creation/create_data.py --ssrf --no-afp
  python Data_Creation/create_data.py --afp
  python Data_Creation/create_data.py --profile
  python Data_Creation/create_data.py --ssrf --profile
  python Data_Creation/create_data.py --unmanipulated
  python Data_Creation/create_data.py --ssrf --unmanipulated
  python Data_Creation/create_data.py --unmanipulated --unmanip-p-step 0.001
  python Data_Creation/create_data.py --afp-relax --max-relax-steps 100
  python Data_Creation/create_data.py --afp-profile --max-relax-steps 100
  python Data_Creation/create_data.py --ssrf --max-burn-steps 150
  python Data_Creation/create_data.py --ssrf-afp-combined --no-ssrf --no-afp --max-burn-steps 50 --max-relax-steps 50

Combined (``--ssrf-afp-combined``) defaults to at most
``COMBO_DEFAULT_MAX_CENTERS`` selective centers per region (zipped, not a
cartesian product) and caps post-AFP relax at ``min(max-relax, max-burn)`` so
it does not inherit AFP Profile's long relax grid. Override with
``--combined-max-centers`` / ``--combined-max-relax-steps``.

The saved spectrum is 500 bins on R in [-6, 6] (main-branch Dulya grid).
Burns are restricted to the Q < 0 subset of the inner window R in (-3, 3).
Manipulated modes default to polarization step 0.025; unmanipulated defaults to
a finer step (0.0005) via ``--unmanip-p-step``.
"""
import argparse
import faulthandler
import json
import sys
import traceback
from pathlib import Path
import numpy as np
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DULYA = SCRIPT_DIR / 'rivanna'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DULYA) not in sys.path:
    sys.path.insert(0, str(DULYA))
from afp_bin_traj import run_one_polarization as run_afp_one
from bin_setup import polarization_grid
from burn_selection import equilibrium_q_profile
from common import BURN_R_MAX, BURN_R_MIN, COMBO_LAYOUT_PROFILE_AFP_SELECTIVE, COMBO_LAYOUT_PROFILE_PROFILE, COMBO_LAYOUT_SELECTIVE_SELECTIVE, COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE, COMBO_SCENARIO_AFP_FIRST_REGION, COMBO_SCENARIO_BOTH_SELECTIVE, COMBO_SCENARIO_SSRF_FIRST_REGION, EXCLUDED_MANIPULATION_BURN_BINS, F_MAX, F_MIN, NUM_BINS, RF_GAUSSIAN_FWHM_R, RF_LORENTZIAN_FWHM_R, RF_MODE_PHYSICAL_VOIGT, SOURCE_AFP, SOURCE_AFP_PROFILE, SOURCE_PROFILE, SOURCE_SSRF, SOURCE_SSRF_AFP, SOURCE_UNMANIP
from ssrf_afp_combined_traj import combined_row_clock, run_combined_polarization, selective_manipulation_overlap, split_q_negative_regions
from model_bridge import build_equilibrium_spin1_model, full_spectrum_intensities, level_pq
from physics.rf import bin_averaged_voigt
from physics.rf.optimal_profile import DATA_GEN_SETTINGS, QUICK_SETTINGS, run_optimal_profile_polarization
from ssrf_bin_traj import run_one_polarization as run_ssrf_one
DEFAULT_OUTPUT = SCRIPT_DIR / 'spectra_data' / 'spectra.npz'
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'spectra_data' / 'plots'
P_MIN = 0.2
P_MAX = 0.6
P_STEP = 0.025
UNMANIP_P_STEP = 0.00005
GAMMA_RF = 10.0
MIN_BURN_STEPS = 0
MAX_BURN_STEPS = 400
BURN_STEPS_STEP = 1
FREQUENCY = np.linspace(F_MIN, F_MAX, NUM_BINS)
DT = 0.0015
GAUSSIAN_FWHM_R = RF_GAUSSIAN_FWHM_R
LORENTZIAN_FWHM_R = RF_LORENTZIAN_FWHM_R
AFP_WINDOW = 8
MIN_RELAX_STEPS = 0
MAX_RELAX_STEPS = 8000
RELAX_STEPS_STEP = 1
# Combined ssRF+AFP must not inherit AFP Profile's long relax / all-center cartesian
# product — that OOMs silently under production grids (see data_creation.log).
COMBO_DEFAULT_MAX_CENTERS = 5
STORE_DTYPE = np.float32
DEFAULT_SEED = 42
NOISE_LEVEL = 0.0
PROFILE_CENTER_BIN = -1
AFP_QNEG_CENTER = -1
UNMANIP_CENTER_BIN = -1

def _burn_window_bins(*, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX):
    """Inner |R| < 3 window, minus main-branch excluded manipulation bins."""
    frequency = np.linspace(r_min, r_max, num_bins)
    bins = np.flatnonzero((frequency > BURN_R_MIN) & (frequency < BURN_R_MAX)).astype(np.int32)
    if EXCLUDED_MANIPULATION_BURN_BINS:
        excluded = np.fromiter(EXCLUDED_MANIPULATION_BURN_BINS, dtype=np.int32)
        bins = bins[~np.isin(bins, excluded)]
    return bins

def _traj_grid_kwargs(*, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT):
    return {'num_bins': int(num_bins), 'r_min': float(r_min), 'r_max': float(r_max), 'dt': float(dt)}

def q_negative_bins_for_p0(p0, burn_window, *, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX):
    """Burn-window bin indices where equilibrium Q = I+ - I- is negative.

    Uses the noiseless ssRF-beta Pake equilibrium so center selection is
    independent of observation noise added later to saved spectra.
    """
    q = equilibrium_q_profile(p0, num_bins=num_bins, r_min=r_min, r_max=r_max)
    mask = q[burn_window] < 0.0
    return burn_window[mask]

def _selective_centers_in_region(region_bins, q_profile, *, max_centers=None):
    """Q<0 bin centers in one contiguous region (deepest Q first)."""
    region = np.asarray(region_bins, dtype=np.int32).reshape(-1)
    if region.size == 0:
        return np.empty(0, dtype=np.int32)
    order = np.argsort(q_profile[region])
    region = region[order]
    if max_centers is not None:
        region = region[: int(max_centers)]
    return region.astype(np.int32, copy=False)

def burn_steps_values(min_steps, max_steps, step):
    """Inclusive integer step grid from ``min_steps`` to ``max_steps``."""
    return np.arange(min_steps, max_steps + 1, step, dtype=np.int32)

def zero_power_profile(num_bins):
    return np.zeros(int(num_bins), dtype=STORE_DTYPE)

def _coerce_power_profile(power_profile, *, source, num_bins=NUM_BINS):
    """Finite per-bin envelope; unmanipulated (and NaN) rows are exactly zero."""
    n = int(num_bins)
    prof = np.asarray(power_profile, dtype=np.float64).reshape(-1)
    if prof.size != n:
        prof = zero_power_profile(n).astype(np.float64, copy=False)
    prof = np.nan_to_num(prof, nan=0.0, posinf=0.0, neginf=0.0)
    if int(source) == int(SOURCE_UNMANIP):
        prof = np.zeros(n, dtype=np.float64)
    return prof.astype(STORE_DTYPE, copy=False)

def _sanitize_saved_arrays(data, *, num_bins=NUM_BINS):
    """Ensure NPZ arrays are finite; unmanipulated power_profile is all zeros."""
    if 'power_profile' not in data or data['power_profile'] is None:
        n = int(data['spectra'].shape[0]) if data['spectra'].size else 0
        data['power_profile'] = np.zeros((n, int(num_bins)), dtype=STORE_DTYPE)
        return data
    pp = np.asarray(data['power_profile'], dtype=np.float64)
    if pp.ndim == 1 and data['spectra'].shape[0] == 1:
        pp = pp.reshape(1, -1)
    pp = np.nan_to_num(pp, nan=0.0, posinf=0.0, neginf=0.0)
    if 'source' in data and pp.ndim == 2:
        src = np.asarray(data['source']).reshape(-1)
        if src.shape[0] == pp.shape[0]:
            unmanip = src == SOURCE_UNMANIP
            if np.any(unmanip):
                pp[unmanip] = 0.0
    data['power_profile'] = pp.astype(STORE_DTYPE, copy=False)
    ap = np.asarray(data['applied_power'], dtype=np.float64).reshape(-1)
    ap = np.nan_to_num(ap, nan=0.0, posinf=0.0, neginf=0.0)
    if 'source' in data and ap.shape[0] == data['source'].shape[0]:
        src = np.asarray(data['source']).reshape(-1)
        ap[(src == SOURCE_UNMANIP) | (src == SOURCE_AFP)] = 0.0
    data['applied_power'] = ap.astype(STORE_DTYPE, copy=False)
    return data

def ssrf_voigt_power_profile(center_bin, gamma_rf, *, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, gaussian_fwhm_R=GAUSSIAN_FWHM_R, lorentzian_fwhm_R=LORENTZIAN_FWHM_R):
    """Physical-R Voigt envelope scaled by gamma_rf (peak ~ gamma_rf at the burn bin)."""
    freq = np.linspace(r_min, r_max, int(num_bins))
    dR = float(np.abs(freq[1] - freq[0])) if freq.size > 1 else 1.0
    center = int(center_bin)
    center_R = float(freq[center])
    shape = bin_averaged_voigt(freq, center_R=center_R, bin_width_R=dR, gaussian_fwhm_R=gaussian_fwhm_R, lorentzian_fwhm_R=lorentzian_fwhm_R, normalization='center_bin')
    return (float(gamma_rf) * np.asarray(shape, dtype=np.float64)).astype(STORE_DTYPE, copy=False)

def _align_power_profile(values, *, num_bins, r_min, r_max, frequency=None):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(num_bins)
    if arr.size == 0:
        return zero_power_profile(n)
    if arr.size == n:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(STORE_DTYPE, copy=False)
    src_f = np.asarray(frequency, dtype=np.float64).reshape(-1) if frequency is not None else np.linspace(r_min, r_max, arr.size)
    if src_f.size != arr.size:
        src_f = np.linspace(r_min, r_max, arr.size)
    dst_f = np.linspace(r_min, r_max, n)
    aligned = np.interp(dst_f, src_f, arr)
    return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0).astype(STORE_DTYPE, copy=False)

def _event_pq(traj, k):
    """Population P/Q at traj frame ``k`` (ssRF-beta n+ − n− / n+ − 2n0 + n−)."""
    p_full = traj.get('p_full')
    q_full = traj.get('q_full')
    if p_full is None or q_full is None:
        raise KeyError('trajectory is missing population p_full/q_full')
    return (float(np.asarray(p_full)[k]), float(np.asarray(q_full)[k]))

def _append_event(rows, *, iplus, iminus, p0, p_total, q_total, applied_power, n_steps, center_bin, source, power_profile, combo_scenario=0, combo_layout=0):
    rows['spectra'].append(np.stack([np.asarray(iplus, dtype=STORE_DTYPE).reshape(-1), np.asarray(iminus, dtype=STORE_DTYPE).reshape(-1)], axis=0))
    rows['p0'].append(p0)
    rows['P_total'].append(p_total)
    rows['Q_total'].append(q_total)
    rows['applied_power'].append(applied_power)
    rows['power_profile'].append(_coerce_power_profile(power_profile, source=source))
    rows['n_steps'].append(n_steps)
    rows['center_bin'].append(center_bin)
    rows['source'].append(source)
    rows['combo_scenario'].append(int(combo_scenario))
    rows['combo_layout'].append(int(combo_layout))

def _empty_rows():
    return {'spectra': [], 'p0': [], 'P_total': [], 'Q_total': [], 'applied_power': [], 'power_profile': [], 'n_steps': [], 'center_bin': [], 'source': [], 'combo_scenario': [], 'combo_layout': []}

def generate_ssrf_events(p_values, burn_window, steps_grid, *, gamma_rf, max_centers_per_p=None, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT):
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
    grid_kwargs = _traj_grid_kwargs(num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt)
    for (ip, p0) in enumerate(np.asarray(p_values)):
        centers = q_negative_bins_for_p0(p0, burn_window, num_bins=num_bins, r_min=r_min, r_max=r_max)
        if max_centers_per_p is not None and centers.size > max_centers_per_p:
            centers = centers[:max_centers_per_p]
        print(f'  ssRF P={p0:.3f} ({ip + 1}/{len(p_values)}): {centers.size} Q<0 centers × {steps_grid.size} step spectra', flush=True)
        for bin_idx in centers:
            traj = run_ssrf_one(bin_idx, p0, gamma_rf=gamma_rf, n_steps=max_burn, rf_mode=RF_MODE_PHYSICAL_VOIGT, gaussian_fwhm_R=GAUSSIAN_FWHM_R, lorentzian_fwhm_R=LORENTZIAN_FWHM_R, capture_spectrum=True, legacy_spectral_recovery=False, **grid_kwargs)
            if traj.get('skipped', False):
                continue
            iplus_full = np.asarray(traj['iplus_full'])
            iminus_full = np.asarray(traj['iminus_full'])
            profile = ssrf_voigt_power_profile(bin_idx, gamma_rf, num_bins=num_bins, r_min=r_min, r_max=r_max)
            for burn_steps in sorted(step_set):
                k = burn_steps
                ip_spec = np.asarray(iplus_full[k]).copy()
                im_spec = np.asarray(iminus_full[k]).copy()
                (p_total, q_total) = _event_pq(traj, k)
                ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
                im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
                _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=p_total, q_total=q_total, applied_power=gamma_rf, n_steps=burn_steps, center_bin=bin_idx, source=SOURCE_SSRF, power_profile=profile)
    return rows

def generate_afp_events(p_values, burn_window, relax_steps_grid, *, afp_window, max_centers_per_p=None, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT):
    """AFP flip, then save the full I+/I- spectrum at each requested relax step.

    Runs one trajectory of length ``max(relax_steps_grid)`` per (P, center) with
    ``capture_spectrum=True``. Frame 0 is immediately post-flip; later frames are
    after each relax macro-step. A grid of only ``[0]`` is the post-flip snapshot
    with no relaxation. Emits a separate event for each index in
    ``relax_steps_grid`` with ``n_steps`` equal to that index.

    Relaxation, when requested, holds the post-flip tensor polarization and
    moves vector polarization toward the Boltzmann state for that Q.
    """
    rows = _empty_rows()
    if relax_steps_grid.size == 0:
        return rows
    max_relax = np.max(relax_steps_grid)
    step_set = {s for s in relax_steps_grid}
    grid_kwargs = _traj_grid_kwargs(num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt)
    afp_profile = zero_power_profile(num_bins)
    for (ip, p0) in enumerate(np.asarray(p_values)):
        centers = q_negative_bins_for_p0(p0, burn_window, num_bins=num_bins, r_min=r_min, r_max=r_max)
        if max_centers_per_p is not None and centers.size > max_centers_per_p:
            centers = centers[:max_centers_per_p]
        print(f"  AFP P={p0:.3f} ({ip + 1}/{len(p_values)}): {centers.size} Q<0 centers × {relax_steps_grid.size} {('relax' if max_relax > 0 else 'post-flip')} spectra (window={afp_window}, n_relax={max_relax})", flush=True)
        for bin_idx in centers:
            traj = run_afp_one(bin_idx, p0, n_relax=max_relax, afp_window=afp_window, capture_spectrum=True, legacy_spectral_recovery=False, **grid_kwargs)
            if traj.get('skipped', False):
                continue
            iplus_full = np.asarray(traj['iplus_full'])
            iminus_full = np.asarray(traj['iminus_full'])
            for relax_steps in sorted(step_set):
                k = relax_steps
                ip_spec = np.asarray(iplus_full[k]).copy()
                im_spec = np.asarray(iminus_full[k]).copy()
                (p_total, q_total) = _event_pq(traj, k)
                ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
                im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
                _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=p_total, q_total=q_total, applied_power=0.0, n_steps=relax_steps, center_bin=bin_idx, source=SOURCE_AFP, power_profile=afp_profile)
    return rows

def generate_afp_qneg_events(p_values, burn_window, relax_steps_grid, *, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT):
    """One AFP sweep over all initial Q < 0 bins, then relaxation to the pre-AFP state.

    A single trajectory per polarization covers the whole Q < 0 profile.
    ``power_profile`` is 1 on swept bins and 0 elsewhere. ``center_bin`` is
    ``AFP_QNEG_CENTER`` (-1). Frame 0 is post-flip; later frames follow the
    relaxation that returns both vector P and tensor Q to their pre-AFP values.
    """
    rows = _empty_rows()
    if relax_steps_grid.size == 0:
        return rows
    max_relax = int(np.max(relax_steps_grid))
    step_set = {int(s) for s in relax_steps_grid}
    grid_kwargs = _traj_grid_kwargs(num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt)
    for (ip, p0) in enumerate(np.asarray(p_values)):
        print(f'  AFP Profile P={p0:.3f} ({ip + 1}/{len(p_values)}): sweep + {relax_steps_grid.size} relax spectra (n_relax={max_relax})', flush=True)
        traj = run_afp_one(0, p0, n_relax=max_relax, qneg_bins=burn_window, capture_spectrum=True, legacy_spectral_recovery=False, **grid_kwargs)
        if traj.get('skipped', False):
            print(f'    skipped (no Q<0 bins)', flush=True)
            continue
        subset = np.asarray(traj['afp_subset'], dtype=int)
        profile = zero_power_profile(num_bins)
        if subset.size:
            profile[subset] = np.float32(1.0)
        iplus_full = np.asarray(traj['iplus_full'])
        iminus_full = np.asarray(traj['iminus_full'])
        print(f'    swept {subset.size} bins, P {float(traj["p_full"][0]):.4f} -> {float(traj["p_full"][-1]):.4f} (initial {float(traj["p_initial"]):.4f}), Q {float(traj["q_full"][0]):.4f} -> {float(traj["q_full"][-1]):.4f} (initial {float(traj["q_initial"]):.4f})', flush=True)
        for relax_steps in sorted(step_set):
            k = int(relax_steps)
            if k < 0 or k >= iplus_full.shape[0]:
                continue
            ip_spec = np.asarray(iplus_full[k]).copy()
            im_spec = np.asarray(iminus_full[k]).copy()
            (p_total, q_total) = _event_pq(traj, k)
            ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
            im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
            _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=p_total, q_total=q_total, applied_power=0.0, n_steps=k, center_bin=AFP_QNEG_CENTER, source=SOURCE_AFP_PROFILE, power_profile=profile)
    return rows

def generate_unmanipulated_events(p_values, *, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT):
    """Equilibrium full-spectrum events with no RF (source=SOURCE_UNMANIP).

    One event per polarization: ``n_steps=0``, ``applied_power=0``, zero
    ``power_profile``, ``center_bin=-1``. Uses the same ssRF-beta Boltzmann
    equilibrium as the manipulated trajectories (no burn / flip).
    """
    rows = _empty_rows()
    profile = zero_power_profile(num_bins)
    for (ip, p0) in enumerate(np.asarray(p_values)):
        print(f'  unmanipulated P={p0:.3f} ({ip + 1}/{len(p_values)})', flush=True)
        model = build_equilibrium_spin1_model(polarization=float(p0), num_bins=num_bins, dt=dt, rf_enabled=False, relax_enabled=True, r_min=r_min, r_max=r_max, legacy_spectral_recovery=False)
        (ip_spec, im_spec, _) = full_spectrum_intensities(model)
        (p_total, q_total) = level_pq(model)
        ip_spec = np.asarray(ip_spec, dtype=np.float64).copy()
        im_spec = np.asarray(im_spec, dtype=np.float64).copy()
        ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
        im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
        _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=float(p_total), q_total=float(q_total), applied_power=0.0, n_steps=0, center_bin=UNMANIP_CENTER_BIN, source=SOURCE_UNMANIP, power_profile=profile)
    return rows

def generate_profile_events(p_values, steps_grid, *, settings=DATA_GEN_SETTINGS, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT):
    """ssRF-beta optimal RF profile; save I+/I- at each requested Euler step.

    Independent of per-bin Voigt burns: one PulseProgram per polarization,
    all candidate bins commanded together on [0, T). Frame 0 is equilibrium;
    later frames follow IdealBinModel.step. The program endpoint is always
    stored even when it lies past ``max(steps_grid)``.

    Returns ``(rows, profile_rates)`` where ``profile_rates`` maps rounded ``p0``
    to the designed applied-power envelope ``U(R)`` for diagnostic plots.
    """
    rows = _empty_rows()
    profile_rates = {}
    if steps_grid.size == 0:
        return (rows, profile_rates)
    max_save = int(np.max(steps_grid))
    for (ip, p0) in enumerate(np.asarray(p_values)):
        print(f'  profile P={p0:.3f} ({ip + 1}/{len(p_values)}): designing synchronized RF program', flush=True)
        traj = run_optimal_profile_polarization(p0, n_steps=max_save, n_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt, settings=settings)
        iplus_full = np.asarray(traj['iplus_full'])
        iminus_full = np.asarray(traj['iminus_full'])
        n_frames = iplus_full.shape[0]
        end_step = int(traj.get('program_end_step', n_frames - 1))
        wanted = {int(s) for s in steps_grid if 0 <= int(s) < n_frames}
        wanted.add(0)
        if 0 <= end_step < n_frames:
            wanted.add(end_step)
        peak_u = float(traj.get('gamma_rf', 0.0))
        rates = np.asarray(traj.get('rates', []), dtype=float).reshape(-1)
        freq = np.asarray(traj.get('frequency'), dtype=float).reshape(-1) if traj.get('frequency') is not None else np.linspace(r_min, r_max, num_bins)
        profile = _align_power_profile(rates, num_bins=num_bins, r_min=r_min, r_max=r_max, frequency=freq)
        profile_rates[round(float(p0), 5)] = {'rates': rates, 'frequency': freq, 'duration': float(traj.get('duration', 0.0)), 'status': str(traj.get('status', '')), 'peak_u': peak_u}
        for burn_steps in sorted(wanted):
            k = burn_steps
            ip_spec = np.asarray(iplus_full[k]).copy()
            im_spec = np.asarray(iminus_full[k]).copy()
            (p_total, q_total) = _event_pq(traj, k)
            ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
            im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
            _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=p_total, q_total=q_total, applied_power=peak_u, n_steps=burn_steps, center_bin=PROFILE_CENTER_BIN, source=SOURCE_PROFILE, power_profile=profile)
    return (rows, profile_rates)

_COMBO_LAYOUTS_REGION_SPLIT = (
    COMBO_LAYOUT_PROFILE_PROFILE,
    COMBO_LAYOUT_PROFILE_AFP_SELECTIVE,
    COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE,
    COMBO_LAYOUT_SELECTIVE_SELECTIVE,
)

def _combined_uses_selective_ssrf(layout):
    return layout in (COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE, COMBO_LAYOUT_SELECTIVE_SELECTIVE)

def _combined_uses_selective_afp(layout):
    return layout in (COMBO_LAYOUT_PROFILE_AFP_SELECTIVE, COMBO_LAYOUT_SELECTIVE_SELECTIVE)

def _combo_centers_cap(max_centers_per_p, combined_max_centers=None):
    """Bound selective centers for combined events (standalone ssRF/AFP stay uncapped)."""
    if combined_max_centers is not None:
        return max(1, int(combined_max_centers))
    if max_centers_per_p is not None:
        return max(1, int(max_centers_per_p))
    return int(COMBO_DEFAULT_MAX_CENTERS)

def _combo_max_relax(max_relax_steps, max_burn_steps, combined_max_relax_steps=None):
    """Post-AFP relax length for combined trajectories.

    AFP Profile often uses thousands of relax steps; reusing that for every
    combined center pair OOMs. Default: ``min(max_relax, max_burn)``.
    """
    if combined_max_relax_steps is not None:
        return max(0, int(combined_max_relax_steps))
    return max(0, min(int(max_relax_steps), int(max_burn_steps)))

def _combined_center_pairs(layout, ssrf_region, afp_region, q_profile, *, max_centers, scenario):
    """Build a bounded list of (ssrf_center, afp_center) for one layout.

    Selective×selective uses depth-ranked zip (not a cartesian product).
    Scenario 3 plans centers inside ``run_combined_polarization``.
    """
    if scenario == COMBO_SCENARIO_BOTH_SELECTIVE:
        return [(None, None)]
    use_s = _combined_uses_selective_ssrf(layout)
    use_a = _combined_uses_selective_afp(layout)
    ssrf_centers = [None]
    afp_centers = [None]
    if use_s:
        picked = _selective_centers_in_region(ssrf_region, q_profile, max_centers=max_centers)
        ssrf_centers = [int(b) for b in picked]
    if use_a:
        picked = _selective_centers_in_region(afp_region, q_profile, max_centers=max_centers)
        afp_centers = [int(b) for b in picked]
    if not ssrf_centers or not afp_centers:
        return []
    if use_s and use_a:
        n = min(len(ssrf_centers), len(afp_centers))
        return list(zip(ssrf_centers[:n], afp_centers[:n]))
    if use_s:
        return [(s, None) for s in ssrf_centers]
    if use_a:
        return [(None, a) for a in afp_centers]
    return [(None, None)]

def generate_combined_ssrf_afp_events(
    p_values,
    burn_window,
    burn_steps_grid,
    relax_steps_grid,
    *,
    gamma_rf,
    max_relax,
    afp_window,
    profile_settings,
    max_centers_per_p=None,
    combined_max_centers=None,
    num_bins=NUM_BINS,
    r_min=F_MIN,
    r_max=F_MAX,
    dt=DT,
):
    """AFP sweep, then ssRF, across the two contiguous Q<0 regions.

    The AFP power profile is saved on the sweep snapshot. ssRF is applied to
    that spectrum and its power profile is saved on the burn frames only, so
    the two envelopes are not added together.

    Scenarios (``combo_scenario``):
      1 — ssRF on the first Q<0 region, AFP on the second
      2 — ssRF on the second region, AFP on the first
      3 — selective ssRF and selective AFP on all Q<0 bins (both regions)

    Layouts (``combo_layout``): profile/profile, profile/selective AFP,
    selective ssRF/profile, selective/selective. Saved with ``source=5``.

    Selective centers are capped (default ``COMBO_DEFAULT_MAX_CENTERS``) and
    selective×selective pairs are zipped by Q depth — never a full cartesian
    product over all Q<0 bins.
    """
    rows = _empty_rows()
    if burn_steps_grid.size == 0:
        return rows
    max_burn = int(np.max(burn_steps_grid))
    burn_step_set = {int(s) for s in burn_steps_grid}
    relax_step_set = {int(s) for s in relax_steps_grid}
    centers_cap = _combo_centers_cap(max_centers_per_p, combined_max_centers)
    frequency = np.linspace(r_min, r_max, num_bins)
    profile_cache = {}
    n_p = len(np.asarray(p_values))
    for (ip, p0) in enumerate(np.asarray(p_values)):
        centers = q_negative_bins_for_p0(p0, burn_window, num_bins=num_bins, r_min=r_min, r_max=r_max)
        regions = split_q_negative_regions(centers, frequency)
        if len(regions) < 2:
            print(f'  combined ssRF+AFP P={p0:.3f} ({ip + 1}/{n_p}): skipped (<2 Q<0 regions)', flush=True)
            continue
        region_a = regions[0]
        region_b = regions[1]
        q_profile = equilibrium_q_profile(p0, num_bins=num_bins, r_min=r_min, r_max=r_max)
        all_qneg = np.concatenate([region_a, region_b])
        scenario_jobs = (
            (COMBO_SCENARIO_SSRF_FIRST_REGION, region_a, region_b, _COMBO_LAYOUTS_REGION_SPLIT),
            (COMBO_SCENARIO_AFP_FIRST_REGION, region_b, region_a, _COMBO_LAYOUTS_REGION_SPLIT),
            (COMBO_SCENARIO_BOTH_SELECTIVE, all_qneg, all_qneg, (COMBO_LAYOUT_SELECTIVE_SELECTIVE,)),
        )
        events_before_p = len(rows['spectra'])
        traj_ok = 0
        traj_skip = 0
        print(
            f'  combined ssRF+AFP P={p0:.3f} ({ip + 1}/{n_p}): Q<0 regions '
            f'[{region_a.size}, {region_b.size}] bins; selective cap={centers_cap}; '
            f'max_burn={max_burn}; max_relax={int(max_relax)}',
            flush=True,
        )
        for (scenario, ssrf_region, afp_region, layouts) in scenario_jobs:
            for layout in layouts:
                center_pairs = _combined_center_pairs(
                    layout,
                    ssrf_region,
                    afp_region,
                    q_profile,
                    max_centers=centers_cap,
                    scenario=scenario,
                )
                if not center_pairs:
                    continue
                print(
                    f'    scenario={int(scenario)} layout={int(layout)}: {len(center_pairs)} traj(s)',
                    flush=True,
                )
                for (pair_i, (ssrf_center, afp_center)) in enumerate(center_pairs):
                    if (
                        ssrf_center is not None
                        and afp_center is not None
                        and _combined_uses_selective_ssrf(layout)
                        and _combined_uses_selective_afp(layout)
                        and selective_manipulation_overlap(
                            ssrf_center,
                            afp_center,
                            num_bins=num_bins,
                            r_min=r_min,
                            r_max=r_max,
                            afp_window=afp_window,
                            gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R,
                            lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R,
                        )
                    ):
                        traj_skip += 1
                        continue
                    try:
                        traj = run_combined_polarization(
                            p0,
                            scenario=scenario,
                            layout=layout,
                            ssrf_region=ssrf_region,
                            afp_region=afp_region,
                            gamma_rf=gamma_rf,
                            max_burn=max_burn,
                            max_relax=max_relax,
                            afp_window=afp_window,
                            profile_settings=profile_settings,
                            num_bins=num_bins,
                            r_min=r_min,
                            r_max=r_max,
                            dt=dt,
                            capture_spectrum=True,
                            max_centers_per_region=centers_cap,
                            legacy_spectral_recovery=False,
                            ssrf_center_bin=ssrf_center,
                            afp_center_bin=afp_center,
                            profile_cache=profile_cache,
                        )
                    except MemoryError:
                        print(
                            f'    ERROR: out of memory on combined traj '
                            f'scenario={int(scenario)} layout={int(layout)} '
                            f'pair={pair_i + 1}/{len(center_pairs)} '
                            f'ssrf_center={ssrf_center} afp_center={afp_center}. '
                            f'Reduce --combined-max-centers / --combined-max-relax-steps '
                            f'or --max-burn-steps.',
                            flush=True,
                        )
                        raise
                    except Exception as exc:
                        print(
                            f'    ERROR: combined traj failed scenario={int(scenario)} '
                            f'layout={int(layout)} pair={pair_i + 1}/{len(center_pairs)} '
                            f'ssrf_center={ssrf_center} afp_center={afp_center}: {exc}',
                            flush=True,
                        )
                        raise
                    if traj.get('skipped', False):
                        traj_skip += 1
                        continue
                    traj_ok += 1
                    iplus_full = np.asarray(traj['iplus_full'])
                    iminus_full = np.asarray(traj['iminus_full'])
                    p_full = np.asarray(traj['p_full'])
                    q_full = np.asarray(traj['q_full'])
                    profiles = np.asarray(traj['power_profiles'])
                    afp_frame = int(traj['afp_frame'])
                    ssrf_start = int(traj.get('ssrf_start', afp_frame + 1))
                    n_ssrf = int(traj['n_ssrf_frames'])
                    relax_start = int(traj.get('relax_start', ssrf_start + n_ssrf))
                    center_bin = int(traj.get('ssrf_center', -2))
                    if _combined_uses_selective_ssrf(layout) and ssrf_center is not None:
                        center_bin = int(ssrf_center)
                    applied = float(traj.get('applied_peak', gamma_rf))
                    wanted = {int(afp_frame)}
                    for step in burn_step_set:
                        if int(step) < n_ssrf:
                            wanted.add(ssrf_start + int(step))
                    for relax_step in relax_step_set:
                        frame = relax_start + int(relax_step)
                        if frame < iplus_full.shape[0]:
                            wanted.add(frame)
                    for frame in sorted(wanted):
                        ip_spec = np.asarray(iplus_full[frame]).copy()
                        im_spec = np.asarray(iminus_full[frame]).copy()
                        ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
                        im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
                        n_steps, applied_power = combined_row_clock(frame, ssrf_start, applied, relax_start=relax_start)
                        _append_event(
                            rows,
                            iplus=ip_spec,
                            iminus=im_spec,
                            p0=p0,
                            p_total=float(p_full[frame]),
                            q_total=float(q_full[frame]),
                            applied_power=applied_power,
                            n_steps=n_steps,
                            center_bin=center_bin,
                            source=SOURCE_SSRF_AFP,
                            power_profile=profiles[frame],
                            combo_scenario=int(scenario),
                            combo_layout=int(layout),
                        )
        print(
            f'    P={p0:.3f} done: traj_ok={traj_ok} traj_skip={traj_skip} '
            f'events+={len(rows["spectra"]) - events_before_p} (total {len(rows["spectra"])})',
            flush=True,
        )
    return rows

def _merge_rows(*row_groups):
    merged = _empty_rows()
    for rows in row_groups:
        for key in merged:
            merged[key].extend(rows[key])
    n = len(merged['spectra'])
    if n == 0:
        return {'spectra': np.empty((0, 2, NUM_BINS), dtype=STORE_DTYPE), 'p0': np.empty(0, dtype=STORE_DTYPE), 'P_total': np.empty(0, dtype=STORE_DTYPE), 'Q_total': np.empty(0, dtype=STORE_DTYPE), 'applied_power': np.empty(0, dtype=STORE_DTYPE), 'power_profile': np.empty((0, NUM_BINS), dtype=STORE_DTYPE), 'n_steps': np.empty(0, dtype=np.int32), 'center_bin': np.empty(0, dtype=np.int32), 'source': np.empty(0, dtype=np.uint8), 'combo_scenario': np.empty(0, dtype=np.uint8), 'combo_layout': np.empty(0, dtype=np.uint8)}
    return {'spectra': np.stack(merged['spectra'], axis=0).astype(STORE_DTYPE, copy=False), 'p0': np.asarray(merged['p0'], dtype=STORE_DTYPE), 'P_total': np.asarray(merged['P_total'], dtype=STORE_DTYPE), 'Q_total': np.asarray(merged['Q_total'], dtype=STORE_DTYPE), 'applied_power': np.asarray(merged['applied_power'], dtype=STORE_DTYPE), 'power_profile': np.stack(merged['power_profile'], axis=0).astype(STORE_DTYPE, copy=False), 'n_steps': np.asarray(merged['n_steps'], dtype=np.int32), 'center_bin': np.asarray(merged['center_bin'], dtype=np.int32), 'source': np.asarray(merged['source'], dtype=np.uint8), 'combo_scenario': np.asarray(merged['combo_scenario'], dtype=np.uint8), 'combo_layout': np.asarray(merged['combo_layout'], dtype=np.uint8)}

def _event_mask(data, *, source=None, p0=None, center_bin=None, n_steps=None, combo_scenario=None, combo_layout=None):
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
    if combo_scenario is not None and 'combo_scenario' in data:
        mask &= data['combo_scenario'] == int(combo_scenario)
    if combo_layout is not None and 'combo_layout' in data:
        mask &= data['combo_layout'] == int(combo_layout)
    return mask

def _combo_scenario_slug(scenario):
    return {COMBO_SCENARIO_SSRF_FIRST_REGION: 'sc1_ssrf_r0_afp_r1', COMBO_SCENARIO_AFP_FIRST_REGION: 'sc2_ssrf_r1_afp_r0', COMBO_SCENARIO_BOTH_SELECTIVE: 'sc3_both_selective'}.get(int(scenario), f'sc{int(scenario)}')

def _combo_layout_slug(layout):
    return {COMBO_LAYOUT_PROFILE_PROFILE: 'lay0_profile_profile', COMBO_LAYOUT_PROFILE_AFP_SELECTIVE: 'lay1_profile_afp_sel', COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE: 'lay2_ssrf_sel_profile', COMBO_LAYOUT_SELECTIVE_SELECTIVE: 'lay3_selective_selective'}.get(int(layout), f'lay{int(layout)}')

def _pick_combined_example_groups(data):
    """Best trajectory per (combo_scenario, combo_layout) by step coverage."""
    mask = data['source'] == SOURCE_SSRF_AFP
    if not np.any(mask):
        return []
    groups = {}
    for i in np.flatnonzero(mask):
        key = (int(data['combo_scenario'][i]), int(data['combo_layout'][i]), round(float(data['p0'][i]), 5), int(data['center_bin'][i]))
        groups.setdefault(key, set()).add(int(data['n_steps'][i]))
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1], kv[0][2], kv[0][3]))
    seen_slots = set()
    picks = []
    for (scen, lay, p0, center), _steps in ranked:
        slot = (scen, lay)
        if slot in seen_slots:
            continue
        seen_slots.add(slot)
        picks.append((float(p0), scen, lay, center))
    return picks

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

def _pick_afp_example_keys(data, *, source=SOURCE_AFP, max_examples=2):
    """Select a few (p0, center_bin) pairs with the most coverage for one AFP source."""
    mask = data['source'] == source
    if not np.any(mask):
        return []
    p0s = data['p0'][mask]
    bins = data['center_bin'][mask]
    steps = data['n_steps'][mask]
    pairs = {}
    for (p0, b, s) in zip(p0s.tolist(), bins.tolist(), steps.tolist()):
        key = (round(p0, 5), int(b))
        pairs.setdefault(key, set()).add(s)
    ranked = sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))
    return [k for (k, _) in ranked[:max(0, max_examples)]]

def _pick_profile_example_keys(data, *, max_examples=2):
    """Select polarizations with the most optimal-profile step coverage."""
    mask = data['source'] == SOURCE_PROFILE
    if not np.any(mask):
        return []
    p0s = data['p0'][mask]
    steps = data['n_steps'][mask]
    groups = {}
    for (p0, s) in zip(p0s.tolist(), steps.tolist()):
        key = round(p0, 5)
        groups.setdefault(key, set()).add(s)
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [k for (k, _) in ranked[:max(0, max_examples)]]

def _style_publication_axes(ax, *, label_size=16, tick_size=14, title_size=15):
    """Apply larger fonts/ticks and clean spines for publication figures."""
    ax.tick_params(axis='both', which='major', labelsize=tick_size, width=1.1, length=5)
    ax.tick_params(axis='both', which='minor', width=0.8, length=3)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_size(title_size)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
    ax.grid(True, alpha=0.28, linewidth=0.7)

def save_example_plots(data, plot_dir, *, frequency=None, max_ssrf_examples=2, max_afp_examples=2, max_afp_profile_examples=2, max_unmanip_examples=2, profile_rates=None):
    """Write diagnostic PNGs for ssRF, AFP, AFP Profile, unmanipulated, optimal profile, and combined ssRF+AFP events."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    f = np.asarray(FREQUENCY if frequency is None else frequency).reshape(-1)
    profile_rates = {} if profile_rates is None else profile_rates
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
    afp_plot_jobs = ((SOURCE_AFP, max_afp_examples, 'AFP', 'afp', plt.cm.plasma), (SOURCE_AFP_PROFILE, max_afp_profile_examples, 'AFP Profile', 'afp_profile', plt.cm.cividis))
    for (source, n_examples, label, file_prefix, cmap_fn) in afp_plot_jobs:
        for (p0, center) in _pick_afp_example_keys(data, source=source, max_examples=n_examples):
            mask = _event_mask(data, source=source, p0=p0, center_bin=center)
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
            is_profile = source == SOURCE_AFP_PROFILE
            whole_line = is_profile or int(center) < 0
            where = 'Q<0 sweep' if whole_line else f'center={center}  window={AFP_WINDOW}'
            tag = 'profile' if whole_line else f'bin{center:04d}'
            if idx.size > 1:
                (fig, axes) = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
                (ax_ps, ax_q) = axes
                n_show = min(6, idx.size)
                show_i = np.unique(np.round(np.linspace(0, idx.size - 1, n_show)).astype(int))
                cmap = cmap_fn(np.linspace(0.15, 0.9, len(show_i)))
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
                    if not whole_line:
                        ax.axvline(f[center], color='green', ls=':', lw=1.1, label='center')
                    ax.grid(True, alpha=0.3)
                ax_ps.set_ylabel('$P_s = I_+ + I_-$')
                ax_ps.set_title(f'{label} + relax  P0={p0:.3f}  {where}  P_tot∈[{p_tot.min():.3f},{p_tot.max():.3f}]  Q_tot∈[{q_tot.min():.3f},{q_tot.max():.3f}]')
                ax_ps.legend(fontsize=8, ncols=3, loc='upper right')
                ax_q.set_xlabel('R')
                ax_q.set_ylabel('$Q = I_+ - I_-$')
                ax_q.legend(fontsize=8, ncols=3, loc='upper right')
                fig.tight_layout()
                path = plot_dir / f'{file_prefix}_ps_q_relax_P{p0:.2f}_{tag}.png'
                fig.savefig(path, dpi=140)
                plt.close(fig)
                saved.append(path)
                (fig, ax) = plt.subplots(figsize=(10.5, 4.8))
                (j0, j1) = (0, idx.size - 1)
                ax.plot(f, spectra[j0, 0], color='tab:red', ls='--', alpha=0.55, label=f'$I_+$ n={steps[j0]}')
                ax.plot(f, spectra[j0, 1], color='tab:blue', ls='--', alpha=0.55, label=f'$I_-$ n={steps[j0]}')
                ax.plot(f, spectra[j1, 0], color='tab:red', lw=1.5, label=f'$I_+$ n={steps[j1]}')
                ax.plot(f, spectra[j1, 1], color='tab:blue', lw=1.5, label=f'$I_-$ n={steps[j1]}')
                if not whole_line:
                    ax.axvline(f[center], color='green', ls=':', lw=1.1)
                ax.set_xlabel('R')
                ax.set_ylabel('intensity (fit scale)')
                ax.set_title(f'{label} I+/I−  P0={p0:.3f}  {where}  P_tot={p_tot[j1]:.4f}  Q_tot={q_tot[j1]:.4f}')
                ax.legend(fontsize=8, ncols=2, loc='upper right')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                path = plot_dir / f'{file_prefix}_ip_im_relax_P{p0:.2f}_{tag}.png'
                fig.savefig(path, dpi=140)
                plt.close(fig)
                saved.append(path)
            else:
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
                ax_ps.plot(f, ps, color='black', lw=1.5, label=f'{label} $P_s$')
                ax_ps.plot(f, q, color='tab:orange', lw=1.2, label=f'{label} $Q$')
                ax_ip.plot(f, ip, color='tab:red', lw=1.5, label=f'{label} $I_+$')
                ax_ip.plot(f, im, color='tab:blue', lw=1.5, label=f'{label} $I_-$')
                for ax in axes:
                    if not whole_line:
                        ax.axvline(f[center], color='green', ls=':', lw=1.1, label='center')
                    ax.grid(True, alpha=0.3)
                ax_ps.set_ylabel('$P_s$, $Q$')
                ax_ps.set_title(f'{label} post-flip (n_relax=0)  P0={p0:.3f}  {where}  P_tot={p_tot[0]:.4f}  Q_tot={q_tot[0]:.4f}')
                ax_ps.legend(fontsize=8, ncols=3, loc='upper right')
                ax_ip.set_xlabel('R')
                ax_ip.set_ylabel('intensity (fit scale)')
                ax_ip.legend(fontsize=8, ncols=2, loc='upper right')
                fig.tight_layout()
                path = plot_dir / f'{file_prefix}_apply_P{p0:.2f}_{tag}.png'
                fig.savefig(path, dpi=140)
                plt.close(fig)
                saved.append(path)
            if is_profile and 'power_profile' in data:
                prof = np.asarray(data['power_profile'][idx[0]], dtype=float)
                (fig, ax) = plt.subplots(figsize=(10.5, 4.2))
                ax.fill_between(f, 0.0, prof, color='0.75', alpha=0.55, label='swept bins')
                ax.plot(f, prof, color='black', lw=1.4, label='AFP Profile')
                ax.set_xlabel('R')
                ax.set_ylabel('power profile')
                ax.set_title(f'AFP Profile envelope  P0={p0:.3f}  ({int(np.count_nonzero(prof))} bins)')
                ax.legend(fontsize=9, loc='upper right')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                path = plot_dir / f'afp_profile_envelope_P{p0:.2f}.png'
                fig.savefig(path, dpi=140)
                plt.close(fig)
                saved.append(path)
    unmanip_mask = data['source'] == SOURCE_UNMANIP
    if np.any(unmanip_mask):
        unmanip_p0 = np.asarray(data['p0'][unmanip_mask], dtype=float)
        uniq_p = np.unique(np.round(unmanip_p0, 5))[:max(0, max_unmanip_examples)]
        for p0 in uniq_p:
            mask = _event_mask(data, source=SOURCE_UNMANIP, p0=float(p0))
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            j = int(idx[0])
            spectra = data['spectra'][j]
            p_tot = float(data['P_total'][j])
            q_tot = float(data['Q_total'][j])
            (fig, axes) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
            (ax_ps, ax_ip) = axes
            ip = spectra[0]
            im = spectra[1]
            ax_ps.plot(f, ip + im, color='tab:green', lw=1.6, label='$P_s$')
            ax_ps.plot(f, ip - im, color='tab:purple', lw=1.2, label='$Q$')
            ax_ip.plot(f, ip, color='tab:red', lw=1.5, label='$I_+$')
            ax_ip.plot(f, im, color='tab:blue', lw=1.5, label='$I_-$')
            for ax in axes:
                ax.grid(True, alpha=0.3)
            ax_ps.set_ylabel('$P_s$, $Q$')
            ax_ps.set_title(f'Unmanipulated equilibrium  P0={p0:.3f}  P_tot={p_tot:.4f}  Q_tot={q_tot:.4f}')
            ax_ps.legend(fontsize=8, ncols=2, loc='upper right')
            ax_ip.set_xlabel('R')
            ax_ip.set_ylabel('intensity (fit scale)')
            ax_ip.legend(fontsize=8, ncols=2, loc='upper right')
            fig.tight_layout()
            path = plot_dir / f'unmanipulated_P{p0:.2f}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path)
            if 'power_profile' in data:
                j = int(idx[0])
                prof = np.nan_to_num(np.asarray(data['power_profile'][j], dtype=float), nan=0.0)
                (fig, ax) = plt.subplots(figsize=(10.5, 4.2))
                ax.plot(f, prof, color='0.35', lw=1.4, label='power profile (all zero)')
                ax.axhline(0.0, color='black', lw=0.8, ls='--')
                ax.set_ylim(-0.05, max(0.05, float(np.max(prof)) * 1.2 + 0.05))
                ax.set_xlabel('R')
                ax.set_ylabel('power profile')
                ax.set_title(f'Unmanipulated power profile  P0={p0:.3f}  (no RF; max={prof.max():.3g})')
                ax.legend(fontsize=9, loc='upper right')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                path = plot_dir / f'unmanipulated_power_profile_P{p0:.2f}.png'
                fig.savefig(path, dpi=140)
                plt.close(fig)
                saved.append(path)
    for (p0, scen, lay, center) in _pick_combined_example_groups(data):
        mask = _event_mask(data, source=SOURCE_SSRF_AFP, p0=p0, center_bin=center, combo_scenario=scen, combo_layout=lay)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        order = np.argsort(data['n_steps'][idx])
        idx = idx[order]
        steps = data['n_steps'][idx]
        spectra = data['spectra'][idx]
        p_tot = data['P_total'][idx]
        q_tot = data['Q_total'][idx]
        power = float(data['applied_power'][idx[0]])
        scen_tag = _combo_scenario_slug(scen)
        lay_tag = _combo_layout_slug(lay)
        center_tag = 'whole' if int(center) < 0 else f'bin{int(center):04d}'
        title_extra = f'scenario={scen} layout={lay}  center={center}'
        (fig, axes) = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
        (ax_ps, ax_q) = axes
        n_show = min(6, idx.size)
        show_i = np.unique(np.round(np.linspace(0, idx.size - 1, n_show)).astype(int))
        cmap = plt.cm.magma(np.linspace(0.15, 0.9, len(show_i)))
        for (color, j) in zip(cmap, show_i):
            ip = spectra[j, 0]
            im = spectra[j, 1]
            step = steps[j]
            ax_ps.plot(f, ip + im, color=color, lw=1.4, label=f'n={step}')
            ax_q.plot(f, ip - im, color=color, lw=1.2, label=f'n={step}')
        if int(center) >= 0:
            for ax in axes:
                ax.axvline(f[int(center)], color='0.35', ls=':', lw=1.0)
        ax_ps.set_ylabel('$P_s = I_+ + I_-$')
        ax_ps.set_title(f'Combined ssRF+AFP  P0={p0:.3f}  {title_extra}  γ/U_peak={power:.2f}  P_tot∈[{p_tot.min():.3f},{p_tot.max():.3f}]')
        ax_ps.legend(fontsize=8, ncols=3, loc='upper right')
        ax_q.set_xlabel('R')
        ax_q.set_ylabel('$Q = I_+ - I_-$')
        ax_q.legend(fontsize=8, ncols=3, loc='upper right')
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = plot_dir / f'combined_ps_q_{scen_tag}_{lay_tag}_P{p0:.2f}_{center_tag}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)
        (fig, ax) = plt.subplots(figsize=(10.5, 4.8))
        (j0, j1) = (0, idx.size - 1)
        ax.plot(f, spectra[j0, 0], color='tab:red', ls='--', alpha=0.55, label=f'$I_+$ n={steps[j0]}')
        ax.plot(f, spectra[j0, 1], color='tab:blue', ls='--', alpha=0.55, label=f'$I_-$ n={steps[j0]}')
        ax.plot(f, spectra[j1, 0], color='tab:red', lw=1.5, label=f'$I_+$ n={steps[j1]}')
        ax.plot(f, spectra[j1, 1], color='tab:blue', lw=1.5, label=f'$I_-$ n={steps[j1]}')
        if int(center) >= 0:
            ax.axvline(f[int(center)], color='0.35', ls=':', lw=1.0)
        ax.set_xlabel('R')
        ax.set_ylabel('intensity (fit scale)')
        ax.set_title(f'Combined I+/I−  P0={p0:.3f}  {title_extra}  P_tot={p_tot[j1]:.4f}  Q_tot={q_tot[j1]:.4f}')
        ax.legend(fontsize=8, ncols=2, loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = plot_dir / f'combined_ip_im_{scen_tag}_{lay_tag}_P{p0:.2f}_{center_tag}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)
        if 'power_profile' in data:
            profiles = np.asarray(data['power_profile'][idx], dtype=float)
            powers = np.asarray(data['applied_power'][idx], dtype=float)
            ssrf_rows = np.flatnonzero(powers > 0.0)
            afp_rows = np.flatnonzero((powers <= 0.0) & (np.max(np.abs(profiles), axis=1) > 0.0))
            (fig, ax) = plt.subplots(figsize=(10.5, 4.2))
            if ssrf_rows.size:
                ssrf_part = profiles[int(ssrf_rows[0])]
                ax.fill_between(f, 0.0, ssrf_part, color='tab:orange', alpha=0.35, label='ssRF envelope')
                ax.plot(f, ssrf_part, color='tab:orange', lw=1.2)
            if afp_rows.size:
                afp_part = profiles[int(afp_rows[0])]
                ax.fill_between(f, 0.0, afp_part, color='0.75', alpha=0.55, label='AFP swept/window')
                ax.plot(f, afp_part, color='black', lw=1.2)
            ax.set_xlabel('R')
            ax.set_ylabel('power profile')
            ax.set_title(f'Combined envelope  P0={p0:.3f}  {title_extra}')
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            path = plot_dir / f'combined_envelope_{scen_tag}_{lay_tag}_P{p0:.2f}_{center_tag}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path)
    for p0 in _pick_profile_example_keys(data, max_examples=max_ssrf_examples):
        mask = _event_mask(data, source=SOURCE_PROFILE, p0=p0)
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
        (fig, ax_ps) = plt.subplots(figsize=(8.5, 5.2))
        n_show = min(6, idx.size)
        show_i = np.unique(np.round(np.linspace(0, idx.size - 1, n_show)).astype(int))
        cmap = plt.cm.cividis(np.linspace(0.15, 0.9, len(show_i)))
        for (color, j) in zip(cmap, show_i):
            ip = spectra[j, 0]
            im = spectra[j, 1]
            step = steps[j]
            ax_ps.plot(f, ip + im, color=color, lw=2.0, label=f'$n={step}$')
        ax_ps.set_xlabel('$R$')
        ax_ps.set_xlim(-6, 6)
        ax_ps.set_ylabel('Intensity (arb. units)')
        ax_ps.set_title(f'Optimal RF profile  $P_0={p0:.3f}$  $U_\\mathrm{{max}}={power:.2f}$  $P_\\mathrm{{tot}}\\in[{p_tot.min():.3f},{p_tot.max():.3f}]$  $Q_\\mathrm{{tot}}\\in[{q_tot.min():.3f},{q_tot.max():.3f}]$')
        ax_ps.legend(fontsize=12, ncols=2, loc='upper right', frameon=False, handlelength=1.6)
        _style_publication_axes(ax_ps)
        fig.tight_layout()
        path = plot_dir / f'profile_ps_q_steps_P{p0:.2f}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)
        rates = None
        f_u = f
        duration = None
        status = ''
        peak = float(power)
        rates_info = profile_rates.get(round(float(p0), 5))
        if rates_info is not None:
            rates = np.asarray(rates_info['rates'], dtype=float).reshape(-1)
            f_u = np.asarray(rates_info.get('frequency', f), dtype=float).reshape(-1)
            duration = float(rates_info.get('duration', 0.0))
            status = str(rates_info.get('status', ''))
            peak = float(rates_info.get('peak_u', power))
        elif 'power_profile' in data:
            rates = np.asarray(data['power_profile'][idx[0]], dtype=float).reshape(-1)
            f_u = f
        if rates is not None and rates.size and f_u.size == rates.size:
            (fig, ax) = plt.subplots(figsize=(8.5, 5.2))
            ax.step(f_u, rates, where='mid', color='#1b7f4e', lw=2.0)
            ax.fill_between(f_u, rates, step='mid', color='#1b7f4e', alpha=0.22)
            ax.set_xlabel('$R$')
            ax.set_ylabel('$U(R)$')
            title = f'Applied RF power profile  $P_0={p0:.3f}$  $U_\\mathrm{{max}}={peak:.3f}$'
            if duration is not None:
                title += f'  $T={duration:.4g}$'
            if status:
                title += f'  ({status})'
            ax.set_title(title)
            _style_publication_axes(ax)
            fig.tight_layout()
            path = plot_dir / f'profile_applied_power_P{p0:.2f}.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)
    return saved

def generate_spectra(*, do_ssrf, do_afp, do_afp_relax=False, do_afp_profile=False, do_profile=False, do_unmanipulated=False, do_ssrf_afp_combined=False, p_min=P_MIN, p_max=P_MAX, p_step=P_STEP, unmanip_p_step=UNMANIP_P_STEP, min_burn_steps=MIN_BURN_STEPS, max_burn_steps=MAX_BURN_STEPS, burn_steps_step=BURN_STEPS_STEP, min_relax_steps=MIN_RELAX_STEPS, max_relax_steps=MAX_RELAX_STEPS, relax_steps_step=RELAX_STEPS_STEP, gamma_rf=GAMMA_RF, afp_window=AFP_WINDOW, max_centers_per_p=None, combined_max_centers=None, combined_max_relax_steps=None, p_values=None, unmanip_p_values=None, profile_settings=DATA_GEN_SETTINGS, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT, profile_rates_out=None):
    """Build full-spectrum events for the requested modes.

    Manipulated modes (ssRF / AFP / profile) use ``p_step`` (or ``p_values``).
    Unmanipulated uses ``unmanip_p_step`` by default (finer grid), unless
    ``unmanip_p_values`` is given, or ``p_values`` is given without a separate
    unmanipulated list (e.g. ``--quick`` smoke polarizations).

    If ``profile_rates_out`` is a dict, it is filled with designed ``U(R)``
    envelopes keyed by rounded polarization for diagnostic plotting.
    """
    burn_window = _burn_window_bins(num_bins=num_bins, r_min=r_min, r_max=r_max)
    if p_values is None:
        manip_p = polarization_grid(p_min, p_max, p_step)
        manip_p = manip_p[manip_p > 0.0]
    else:
        manip_p = np.asarray(p_values).reshape(-1)
    if unmanip_p_values is not None:
        unmanip_p = np.asarray(unmanip_p_values).reshape(-1)
    elif p_values is not None:
        unmanip_p = manip_p
    else:
        unmanip_p = polarization_grid(p_min, p_max, unmanip_p_step)
        unmanip_p = unmanip_p[unmanip_p > 0.0]
    groups = []
    profile_rates = {}
    if do_ssrf:
        steps_grid = burn_steps_values(min_burn_steps, max_burn_steps, burn_steps_step)
        groups.append(generate_ssrf_events(manip_p, burn_window, steps_grid, gamma_rf=gamma_rf, max_centers_per_p=max_centers_per_p, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt))
    if do_afp:
        relax_grid = _afp_relax_grid(do_afp_relax, min_relax_steps=min_relax_steps, max_relax_steps=max_relax_steps, relax_steps_step=relax_steps_step)
        groups.append(generate_afp_events(manip_p, burn_window, relax_grid, afp_window=afp_window, max_centers_per_p=max_centers_per_p, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt))
    if do_afp_profile:
        relax_grid = burn_steps_values(min_relax_steps, max_relax_steps, relax_steps_step)
        groups.append(generate_afp_qneg_events(manip_p, burn_window, relax_grid, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt))
    if do_profile:
        steps_grid = burn_steps_values(min_burn_steps, max_burn_steps, burn_steps_step)
        (profile_rows, profile_rates) = generate_profile_events(manip_p, steps_grid, settings=profile_settings, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt)
        groups.append(profile_rows)
    if do_unmanipulated:
        groups.append(generate_unmanipulated_events(unmanip_p, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt))
    if do_ssrf_afp_combined:
        steps_grid = burn_steps_values(min_burn_steps, max_burn_steps, burn_steps_step)
        combo_relax = _combo_max_relax(max_relax_steps, max_burn_steps, combined_max_relax_steps)
        if combo_relax != int(max_relax_steps):
            print(
                f'  combined ssRF+AFP: capping post-AFP relax {int(max_relax_steps)} -> {combo_relax} '
                f'(override with --combined-max-relax-steps; keeps AFP Profile on the full relax grid)',
                flush=True,
            )
        combo_relax_grid = burn_steps_values(min_relax_steps, combo_relax, relax_steps_step)
        groups.append(
            generate_combined_ssrf_afp_events(
                manip_p,
                burn_window,
                steps_grid,
                combo_relax_grid,
                gamma_rf=gamma_rf,
                max_relax=combo_relax,
                afp_window=afp_window,
                profile_settings=profile_settings,
                max_centers_per_p=max_centers_per_p,
                combined_max_centers=combined_max_centers,
                num_bins=num_bins,
                r_min=r_min,
                r_max=r_max,
                dt=dt,
            )
        )
    if profile_rates_out is not None:
        profile_rates_out.clear()
        profile_rates_out.update(profile_rates)
    return _merge_rows(*groups)

def _afp_relax_grid(do_afp_relax, *, min_relax_steps, max_relax_steps, relax_steps_step):
    """Relax-step indices to save. Without relaxation this is only post-flip (0)."""
    if not do_afp_relax:
        return np.asarray([0], dtype=np.int32)
    return burn_steps_values(min_relax_steps, max_relax_steps, relax_steps_step)

def _resolve_modes(*, ssrf, afp, afp_relax, afp_profile, profile, unmanipulated, quick):
    """Resolve mode flags.

    Defaults when no mode flag is given (``--afp-relax``, ``--afp-profile``,
    ``--profile``, and ``--unmanipulated`` count):
      - ``--quick`` → ssRF, per-bin AFP, and optimal profile on; relaxation /
        unmanipulated off
      - otherwise → ssRF on; AFP, profile, unmanipulated off
    When any mode flag is given, unspecified modes default to off.
    ``--afp-relax`` implies per-bin AFP. ``--no-afp --afp-relax`` is an error.
    ``--afp-profile`` is independent: one Q<0 AFP Profile sweep plus relaxation.
    """
    if afp is False and afp_relax:
        raise ValueError('--afp-relax requires AFP; cannot combine with --no-afp')
    ssrf_set = ssrf is not None
    afp_set = afp is not None
    optimal_set = profile is not None
    unmanip_set = unmanipulated is not None
    relax_set = afp_relax
    afp_profile_set = bool(afp_profile)
    if not ssrf_set and (not afp_set) and (not relax_set) and (not optimal_set) and (not afp_profile_set) and (not unmanip_set):
        if quick:
            return (True, True, False, False, True, False)
        return (True, False, False, False, False, False)
    do_ssrf = ssrf if ssrf_set else False
    do_afp = afp if afp_set else False
    do_profile = profile if optimal_set else False
    do_unmanipulated = unmanipulated if unmanip_set else False
    do_afp_relax = afp_relax
    do_afp_profile = afp_profile_set
    if do_afp_relax:
        do_afp = True
    if not do_afp:
        do_afp_relax = False
    return (do_ssrf, do_afp, do_afp_relax, do_afp_profile, do_profile, do_unmanipulated)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate full-spectrum ssRF, AFP, AFP Profile, unmanipulated, and/or ssRF-beta optimal-profile lineshapes. ssRF/AFP centers are burn-window bins where initial (equilibrium) Q < 0. Unmanipulated (source=2), optimal profile (source=3), and AFP Profile (source=4) are independent event streams. Modes are never combined in the same event.')
    parser.add_argument('--ssrf', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable ssRF Voigt-burn events (--ssrf / --no-ssrf; default: on if no mode flags, else off)')
    parser.add_argument('--afp', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable AFP events (--afp / --no-afp; default: off unless --quick with no mode flags). Without --afp-relax, saves only the immediately post-flip spectrum (n_steps = 0)')
    parser.add_argument('--afp-relax', action=argparse.BooleanOptionalAction, default=False, help='After each per-bin AFP flip, emit spectra along a relaxation trajectory (--afp-relax / --no-afp-relax; default: off). Implies --afp. Relaxation returns P and Q to their pre-AFP values. Uses --min-relax-steps / --max-relax-steps / --relax-steps-step')
    parser.add_argument('--afp-profile', action='store_true', help='AFP Profile: one AFP sweep over all burn-window bins with initial Q < 0, then relaxation back to the pre-AFP P and Q. Saved as source=4. Independent of per-bin --afp. center_bin=-1; power_profile marks swept bins. Uses the relax-step grid')
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable ssRF-beta optimal RF profile events as an independent source (--profile / --no-profile; default: off). Saves spectra at the same n_steps grid as ssRF burns, plus the program endpoint')
    parser.add_argument('--unmanipulated', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable unmanipulated equilibrium spectra as an independent source (--unmanipulated / --no-unmanipulated; default: off). One event per polarization: n_steps=0, applied_power=0, zero power_profile, center_bin=-1, source=2. Uses --unmanip-p-step (finer than --p-step by default)')
    parser.add_argument('--ssrf-afp-combined', action='store_true', help='Combined ssRF+AFP events in one trajectory (source=5). AFP sweeps its region first and that power profile is saved alone; ssRF is then applied to the post-AFP spectrum and its power profile is saved on the burn frames only. Scenario 1: ssRF on first Q<0 region, AFP on second. Scenario 2: regions swapped. Scenario 3: selective ssRF and AFP on all Q<0 bins. Selective centers are capped/zipped; post-sequence relax defaults to min(max-relax, max-burn)')
    parser.add_argument('--combined-max-centers', type=int, default=None, help=f'Max selective centers per Q<0 region for --ssrf-afp-combined (default: {COMBO_DEFAULT_MAX_CENTERS}; --quick uses its own center cap)')
    parser.add_argument('--combined-max-relax-steps', type=int, default=500, help='Post-AFP relax length for --ssrf-afp-combined only (default: min(--max-relax-steps, --max-burn-steps)). AFP Profile still uses the full --max-relax-steps grid')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help=f'Output .npz path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--plot-dir', type=Path, default=None, help=f'Directory for example PNGs (default: <output-parent>/plots, e.g. {DEFAULT_PLOT_DIR})')
    parser.add_argument('--no-plots', action='store_true', help='Skip writing example diagnostic plots')
    parser.add_argument('--p-min', type=float, default=P_MIN, help=f'Minimum polarization (default: {P_MIN})')
    parser.add_argument('--p-max', type=float, default=P_MAX, help=f'Maximum polarization (default: {P_MAX})')
    parser.add_argument('--p-step', type=float, default=P_STEP, help=f'Polarization grid step for manipulated modes ssRF/AFP/profile (default: {P_STEP})')
    parser.add_argument('--unmanip-p-step', type=float, default=UNMANIP_P_STEP, help=f'Polarization grid step for --unmanipulated only (default: {UNMANIP_P_STEP}; finer than --p-step)')
    parser.add_argument('--min-burn-steps', type=int, default=MIN_BURN_STEPS, help=f'Minimum ssRF / profile burn length (default: {MIN_BURN_STEPS})')
    parser.add_argument('--max-burn-steps', type=int, default=MAX_BURN_STEPS, help=f'Maximum ssRF / profile burn length (default: {MAX_BURN_STEPS})')
    parser.add_argument('--burn-steps-step', type=int, default=BURN_STEPS_STEP, help=f'ssRF / profile burn-length stride (default: {BURN_STEPS_STEP})')
    parser.add_argument('--min-relax-steps', type=int, default=MIN_RELAX_STEPS, help=f'Minimum AFP relax steps after flip when --afp-relax is on (default: {MIN_RELAX_STEPS})')
    parser.add_argument('--max-relax-steps', type=int, default=MAX_RELAX_STEPS, help=f'Maximum AFP relax steps after flip when --afp-relax is on (default: {MAX_RELAX_STEPS})')
    parser.add_argument('--relax-steps-step', type=int, default=RELAX_STEPS_STEP, help=f'AFP relax-step stride when --afp-relax is on (default: {RELAX_STEPS_STEP})')
    parser.add_argument('--gamma-rf', type=float, default=GAMMA_RF, help=f'ssRF applied power gamma_rf (default: {GAMMA_RF})')
    parser.add_argument('--afp-window', type=int, default=AFP_WINDOW, help=f'AFP subset window width in bins (default: {AFP_WINDOW})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'RNG seed reserved for future stochastic options (default: {DEFAULT_SEED})')
    parser.add_argument('--quick', action='store_true', help='Smoke run: 2 polarizations, few centers, short ssRF burn grid; includes AFP (post-flip only unless --afp-relax) and the optimal applied-power profile. Use --no-profile / --no-afp to drop modes')
    return parser.parse_args()

def main():
    faulthandler.enable()
    args = parse_args()
    try:
        (do_ssrf, do_afp, do_afp_relax, do_afp_profile, do_profile, do_unmanipulated) = _resolve_modes(ssrf=args.ssrf, afp=args.afp, afp_relax=args.afp_relax, afp_profile=args.afp_profile, profile=args.profile, unmanipulated=args.unmanipulated, quick=args.quick)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_settings = QUICK_SETTINGS if args.quick else DATA_GEN_SETTINGS
    combined_max_centers = args.combined_max_centers
    combined_max_relax_steps = args.combined_max_relax_steps
    if args.quick:
        p_values = np.asarray([0.3, 0.5])
        min_burn_steps = 0
        max_burn_steps = 100
        burn_steps_step = 5
        min_relax_steps = 0
        max_relax_steps = 100
        relax_steps_step = 5
        max_centers_per_p = 3
        if combined_max_centers is None:
            combined_max_centers = max_centers_per_p
        afp_quick = 'AFP relax {0,5,10}' if do_afp_relax else ('AFP post-flip only' if do_afp else 'AFP off')
        if do_afp_profile:
            afp_quick += ', AFP Profile+relax'
        profile_quick = ', profile on' if do_profile else ', profile off'
        unmanip_quick = ', unmanipulated on' if do_unmanipulated else ', unmanipulated off'
        print(f'Quick smoke: P={{0.30, 0.50}}, 3 centers/P, ssRF steps {{0,5,10}}, {afp_quick}{profile_quick}{unmanip_quick}', flush=True)
    else:
        p_values = None
        min_burn_steps = args.min_burn_steps
        max_burn_steps = args.max_burn_steps
        burn_steps_step = args.burn_steps_step
        min_relax_steps = args.min_relax_steps
        max_relax_steps = args.max_relax_steps
        relax_steps_step = args.relax_steps_step
        max_centers_per_p = None
    do_ssrf_afp_combined = bool(args.ssrf_afp_combined)
    modes = []
    if do_ssrf:
        modes.append('ssRF')
    if do_afp:
        modes.append('AFP+relax' if do_afp_relax else 'AFP')
    if do_afp_profile:
        modes.append('AFP Profile')
    if do_profile:
        modes.append('profile')
    if do_unmanipulated:
        modes.append('unmanipulated')
    if do_ssrf_afp_combined:
        modes.append('ssRF+AFP combined')
    if do_afp_relax:
        afp_relax_desc = f'relax [{min_relax_steps}, {max_relax_steps}] stride {relax_steps_step}'
    elif do_afp:
        afp_relax_desc = 'post-flip only (n_steps=0)'
    else:
        afp_relax_desc = 'off'
    profile_desc = f'AFP Profile relax [{min_relax_steps}, {max_relax_steps}] stride {relax_steps_step}' if do_afp_profile else 'AFP Profile off'
    unmanip_desc = f'on (p-step={args.unmanip_p_step})' if do_unmanipulated else 'off'
    print(f"Generating {'+'.join(modes)} full spectra | P in [{args.p_min}, {args.p_max}] manip-step={args.p_step} | ssRF gamma_rf={args.gamma_rf} steps [{min_burn_steps}, {max_burn_steps}] stride {burn_steps_step} | ssRF centers: equilibrium Q<0 only | AFP window={args.afp_window} {afp_relax_desc} | {profile_desc} | profile={'on' if do_profile else 'off'} | unmanipulated={unmanip_desc} | dt={DT} | grid n={NUM_BINS} R in [{F_MIN}, {F_MAX}] | burn R in ({BURN_R_MIN}, {BURN_R_MAX}) | Voigt FWHM G={GAUSSIAN_FWHM_R:.3f} L={LORENTZIAN_FWHM_R:.3f}", flush=True)
    if do_ssrf_afp_combined:
        combo_relax = _combo_max_relax(max_relax_steps, max_burn_steps, combined_max_relax_steps)
        combo_centers = _combo_centers_cap(max_centers_per_p, combined_max_centers)
        print(
            f'  combined ssRF+AFP plan: selective centers/region <= {combo_centers} (zipped pairs); '
            f'post-AFP relax <= {combo_relax}',
            flush=True,
        )
    profile_rates = {}
    try:
        data = generate_spectra(do_ssrf=do_ssrf, do_afp=do_afp, do_afp_relax=do_afp_relax, do_afp_profile=do_afp_profile, do_profile=do_profile, do_unmanipulated=do_unmanipulated, do_ssrf_afp_combined=do_ssrf_afp_combined, p_min=args.p_min, p_max=args.p_max, p_step=args.p_step, unmanip_p_step=args.unmanip_p_step, min_burn_steps=min_burn_steps, max_burn_steps=max_burn_steps, burn_steps_step=burn_steps_step, min_relax_steps=min_relax_steps, max_relax_steps=max_relax_steps, relax_steps_step=relax_steps_step, gamma_rf=args.gamma_rf, afp_window=args.afp_window, max_centers_per_p=max_centers_per_p, combined_max_centers=combined_max_centers, combined_max_relax_steps=combined_max_relax_steps, p_values=p_values, profile_settings=profile_settings, profile_rates_out=profile_rates)
    except MemoryError:
        print(
            'ERROR: ran out of memory while generating spectra. '
            'For --ssrf-afp-combined try smaller --combined-max-centers / '
            '--combined-max-relax-steps / --max-burn-steps, or generate modes separately.',
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from None
    except Exception:
        print('ERROR: create_data.py failed with traceback:', file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1) from None
    data = _sanitize_saved_arrays(data)
    source = data['source']
    metadata = {'dataset': 'spectra', 'source_codes': {'ssRF': int(SOURCE_SSRF), 'AFP': int(SOURCE_AFP), 'unmanipulated': int(SOURCE_UNMANIP), 'optimal profile': int(SOURCE_PROFILE), 'AFP Profile': int(SOURCE_AFP_PROFILE), 'ssRF+AFP combined': int(SOURCE_SSRF_AFP)}, 'combo_scenario_codes': {'ssRF_first_Qneg_region': int(COMBO_SCENARIO_SSRF_FIRST_REGION), 'AFP_first_Qneg_region': int(COMBO_SCENARIO_AFP_FIRST_REGION), 'both_selective_all_Qneg': int(COMBO_SCENARIO_BOTH_SELECTIVE)}, 'combo_layout_codes': {'profile_profile': int(COMBO_LAYOUT_PROFILE_PROFILE), 'profile_afp_selective': int(COMBO_LAYOUT_PROFILE_AFP_SELECTIVE), 'ssrf_selective_profile': int(COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE), 'selective_selective': int(COMBO_LAYOUT_SELECTIVE_SELECTIVE)}, 'n_events': int(source.size), 'n_by_source': {'ssRF': int(np.sum(source == SOURCE_SSRF)), 'AFP': int(np.sum(source == SOURCE_AFP)), 'unmanipulated': int(np.sum(source == SOURCE_UNMANIP)), 'optimal profile': int(np.sum(source == SOURCE_PROFILE)), 'AFP Profile': int(np.sum(source == SOURCE_AFP_PROFILE)), 'ssRF+AFP combined': int(np.sum(source == SOURCE_SSRF_AFP))}}
    data['meta_json'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(output, **data)
    n = data['spectra'].shape[0]
    n_ssrf = np.sum(data['source'] == SOURCE_SSRF)
    n_afp = np.sum(data['source'] == SOURCE_AFP)
    n_unmanip = np.sum(data['source'] == SOURCE_UNMANIP)
    n_afp_profile = np.sum(data['source'] == SOURCE_AFP_PROFILE)
    n_profile = np.sum(data['source'] == SOURCE_PROFILE)
    n_combined = np.sum(data['source'] == SOURCE_SSRF_AFP)
    print(f"Saved N={n} events (ssRF={n_ssrf}, AFP={n_afp}, unmanipulated={n_unmanip}, AFP Profile={n_afp_profile}, optimal profile={n_profile}, ssRF+AFP combined={n_combined}) spectra={data['spectra'].shape} dtype={data['spectra'].dtype} -> {output}", flush=True)
    if not args.no_plots and n > 0:
        plot_dir = Path(args.plot_dir) if args.plot_dir is not None else output.parent / 'plots'
        paths = save_example_plots(data, plot_dir, profile_rates=profile_rates)
        if paths:
            print(f'Wrote {len(paths)} example plots -> {plot_dir}', flush=True)
            for p in paths:
                print(f'  {p.name}', flush=True)
        else:
            print(f'No example plots produced (empty selection) -> {plot_dir}', flush=True)
if __name__ == '__main__':
    main()
