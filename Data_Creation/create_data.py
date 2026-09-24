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

ssRF, AFP, and the optimal profile are never combined in the same event.
Toggle modes with ``--ssrf`` / ``--no-ssrf``, ``--afp`` / ``--no-afp``, and
``--profile`` / ``--no-profile``. ``--afp-relax`` implies AFP.
``--afp-profile`` is the AFP Profile source: one Q<0 sweep plus relaxation.

Saved NPZ fields:
  spectra       (N, 2, num_bins)  channel0=I+, channel1=I-
                                  one row per ssRF burn step / AFP post-flip
                                  or AFP relax step / profile Euler step
  p0            (N,)              initial vector polarization
  P_total       (N,)              population P = n+ − n− after this step
  Q_total       (N,)              population Q = n+ − 2 n0 + n− after this step
  applied_power (N,)              gamma_rf for ssRF; 0 for AFP; peak U for profile
  power_profile (N, num_bins)     per-bin RF envelope: Voigt*gamma_rf (ssRF),
                                  zeros (per-bin AFP), 1 on swept bins
                                  (Q<0 AFP profile), U(R) (optimal profile)
  n_steps       (N,)              ssRF: burn macro-steps so far;
                                  AFP: relax steps completed after flip
                                  (0 = immediately post-AFP);
                                  profile: Euler steps of the PulseProgram
  center_bin    (N,)              RF / AFP center bin; -1 for a whole-line profile
  source        (N,)              0=ssRF, 1=AFP, 3=optimal profile, 4=AFP Profile
  meta_json     JSON              source_codes for ssRF, AFP, optimal profile, AFP Profile

Examples (from repo root):
  python Data_Creation/create_dae_voigt_burn_spectra.py --quick
  python Data_Creation/create_dae_voigt_burn_spectra.py --ssrf --afp
  python Data_Creation/create_dae_voigt_burn_spectra.py --ssrf --no-afp
  python Data_Creation/create_dae_voigt_burn_spectra.py --afp
  python Data_Creation/create_dae_voigt_burn_spectra.py --profile
  python Data_Creation/create_dae_voigt_burn_spectra.py --ssrf --profile
  python Data_Creation/create_data.py --afp-relax --max-relax-steps 100
  python Data_Creation/create_data.py --afp-profile --max-relax-steps 100
  python Data_Creation/create_dae_voigt_burn_spectra.py --ssrf --max-burn-steps 150

The saved spectrum is 500 bins on R in [-6, 6] (main-branch Dulya grid).
Burns are restricted to the Q < 0 subset of the inner window R in (-3, 3).
Polarization sampling defaults to step 0.025.
"""
import argparse
import json
import sys
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
from common import BURN_R_MAX, BURN_R_MIN, EXCLUDED_MANIPULATION_BURN_BINS, F_MAX, F_MIN, NUM_BINS, RF_GAUSSIAN_FWHM_R, RF_LORENTZIAN_FWHM_R, RF_MODE_PHYSICAL_VOIGT, SOURCE_AFP, SOURCE_AFP_PROFILE, SOURCE_PROFILE, SOURCE_SSRF
from physics.rf import bin_averaged_voigt
from physics.rf.optimal_profile import DATA_GEN_SETTINGS, QUICK_SETTINGS, run_optimal_profile_polarization
from ssrf_bin_traj import run_one_polarization as run_ssrf_one
DEFAULT_OUTPUT = SCRIPT_DIR / 'spectra_data' / 'spectra.npz'
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'spectra_data' / 'plots'
P_MIN = 0.2
P_MAX = 0.6
P_STEP = 0.2
GAMMA_RF = 10.0
MIN_BURN_STEPS = 0
MAX_BURN_STEPS = 200
BURN_STEPS_STEP = 1
FREQUENCY = np.linspace(F_MIN, F_MAX, NUM_BINS)
DT = 0.0015
GAUSSIAN_FWHM_R = RF_GAUSSIAN_FWHM_R
LORENTZIAN_FWHM_R = RF_LORENTZIAN_FWHM_R
AFP_WINDOW = 8
MIN_RELAX_STEPS = 0
MAX_RELAX_STEPS = 8000
RELAX_STEPS_STEP = 1
STORE_DTYPE = np.float32
DEFAULT_SEED = 42
NOISE_LEVEL = 0.0001
PROFILE_CENTER_BIN = -1
AFP_QNEG_CENTER = -1

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

def burn_steps_values(min_steps, max_steps, step):
    """Inclusive integer step grid from ``min_steps`` to ``max_steps``."""
    return np.arange(min_steps, max_steps + 1, step, dtype=np.int32)

def zero_power_profile(num_bins):
    return np.zeros(int(num_bins), dtype=STORE_DTYPE)

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

def _append_event(rows, *, iplus, iminus, p0, p_total, q_total, applied_power, n_steps, center_bin, source, power_profile):
    rows['spectra'].append(np.stack([np.asarray(iplus, dtype=STORE_DTYPE).reshape(-1), np.asarray(iminus, dtype=STORE_DTYPE).reshape(-1)], axis=0))
    rows['p0'].append(p0)
    rows['P_total'].append(p_total)
    rows['Q_total'].append(q_total)
    rows['applied_power'].append(applied_power)
    rows['power_profile'].append(np.asarray(power_profile, dtype=STORE_DTYPE).reshape(-1).copy())
    rows['n_steps'].append(n_steps)
    rows['center_bin'].append(center_bin)
    rows['source'].append(source)

def _empty_rows():
    return {'spectra': [], 'p0': [], 'P_total': [], 'Q_total': [], 'applied_power': [], 'power_profile': [], 'n_steps': [], 'center_bin': [], 'source': []}

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

def _merge_rows(*row_groups):
    merged = _empty_rows()
    for rows in row_groups:
        for key in merged:
            merged[key].extend(rows[key])
    n = len(merged['spectra'])
    if n == 0:
        return {'spectra': np.empty((0, 2, NUM_BINS), dtype=STORE_DTYPE), 'p0': np.empty(0, dtype=STORE_DTYPE), 'P_total': np.empty(0, dtype=STORE_DTYPE), 'Q_total': np.empty(0, dtype=STORE_DTYPE), 'applied_power': np.empty(0, dtype=STORE_DTYPE), 'power_profile': np.empty((0, NUM_BINS), dtype=STORE_DTYPE), 'n_steps': np.empty(0, dtype=np.int32), 'center_bin': np.empty(0, dtype=np.int32), 'source': np.empty(0, dtype=np.uint8)}
    return {'spectra': np.stack(merged['spectra'], axis=0).astype(STORE_DTYPE, copy=False), 'p0': np.asarray(merged['p0'], dtype=STORE_DTYPE), 'P_total': np.asarray(merged['P_total'], dtype=STORE_DTYPE), 'Q_total': np.asarray(merged['Q_total'], dtype=STORE_DTYPE), 'applied_power': np.asarray(merged['applied_power'], dtype=STORE_DTYPE), 'power_profile': np.stack(merged['power_profile'], axis=0).astype(STORE_DTYPE, copy=False), 'n_steps': np.asarray(merged['n_steps'], dtype=np.int32), 'center_bin': np.asarray(merged['center_bin'], dtype=np.int32), 'source': np.asarray(merged['source'], dtype=np.uint8)}

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

def save_example_plots(data, plot_dir, *, frequency=None, max_ssrf_examples=2, max_afp_examples=2, max_afp_profile_examples=2, profile_rates=None):
    """Write a few diagnostic PNGs for ssRF, AFP, AFP Profile, and optimal-profile events."""
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

def generate_spectra(*, do_ssrf, do_afp, do_afp_relax=False, do_afp_profile=False, do_profile=False, p_min=P_MIN, p_max=P_MAX, p_step=P_STEP, min_burn_steps=MIN_BURN_STEPS, max_burn_steps=MAX_BURN_STEPS, burn_steps_step=BURN_STEPS_STEP, min_relax_steps=MIN_RELAX_STEPS, max_relax_steps=MAX_RELAX_STEPS, relax_steps_step=RELAX_STEPS_STEP, gamma_rf=GAMMA_RF, afp_window=AFP_WINDOW, max_centers_per_p=None, p_values=None, profile_settings=DATA_GEN_SETTINGS, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT, profile_rates_out=None):
    """Build full-spectrum manipulated events for the requested modes.

    If ``profile_rates_out`` is a dict, it is filled with designed ``U(R)``
    envelopes keyed by rounded polarization for diagnostic plotting.
    """
    burn_window = _burn_window_bins(num_bins=num_bins, r_min=r_min, r_max=r_max)
    if p_values is None:
        p_values = polarization_grid(p_min, p_max, p_step)
        p_values = p_values[p_values > 0.0]
    else:
        p_values = np.asarray(p_values).reshape(-1)
    groups = []
    profile_rates = {}
    if do_ssrf:
        steps_grid = burn_steps_values(min_burn_steps, max_burn_steps, burn_steps_step)
        groups.append(generate_ssrf_events(p_values, burn_window, steps_grid, gamma_rf=gamma_rf, max_centers_per_p=max_centers_per_p, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt))
    if do_afp:
        relax_grid = _afp_relax_grid(do_afp_relax, min_relax_steps=min_relax_steps, max_relax_steps=max_relax_steps, relax_steps_step=relax_steps_step)
        groups.append(generate_afp_events(p_values, burn_window, relax_grid, afp_window=afp_window, max_centers_per_p=max_centers_per_p, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt))
    if do_afp_profile:
        relax_grid = burn_steps_values(min_relax_steps, max_relax_steps, relax_steps_step)
        groups.append(generate_afp_qneg_events(p_values, burn_window, relax_grid, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt))
    if do_profile:
        steps_grid = burn_steps_values(min_burn_steps, max_burn_steps, burn_steps_step)
        (profile_rows, profile_rates) = generate_profile_events(p_values, steps_grid, settings=profile_settings, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt)
        groups.append(profile_rows)
    if profile_rates_out is not None:
        profile_rates_out.clear()
        profile_rates_out.update(profile_rates)
    return _merge_rows(*groups)

def _afp_relax_grid(do_afp_relax, *, min_relax_steps, max_relax_steps, relax_steps_step):
    """Relax-step indices to save. Without relaxation this is only post-flip (0)."""
    if not do_afp_relax:
        return np.asarray([0], dtype=np.int32)
    return burn_steps_values(min_relax_steps, max_relax_steps, relax_steps_step)

def _resolve_modes(*, ssrf, afp, afp_relax, afp_profile, profile, quick):
    """Resolve mode flags.

    Defaults when no mode flag is given (``--afp-relax``, ``--afp-profile``, and
    ``--profile`` count):
      - ``--quick`` → ssRF, per-bin AFP, and optimal profile on; relaxation off
      - otherwise → ssRF on, AFP off, profile off
    When any mode flag is given, unspecified modes default to off.
    ``--afp-relax`` implies per-bin AFP. ``--no-afp --afp-relax`` is an error.
    ``--afp-profile`` is independent: one Q<0 AFP Profile sweep plus relaxation.
    """
    if afp is False and afp_relax:
        raise ValueError('--afp-relax requires AFP; cannot combine with --no-afp')
    ssrf_set = ssrf is not None
    afp_set = afp is not None
    optimal_set = profile is not None
    relax_set = afp_relax
    afp_profile_set = bool(afp_profile)
    if not ssrf_set and (not afp_set) and (not relax_set) and (not optimal_set) and (not afp_profile_set):
        if quick:
            return (True, True, False, False, True)
        return (True, False, False, False, False)
    do_ssrf = ssrf if ssrf_set else False
    do_afp = afp if afp_set else False
    do_profile = profile if optimal_set else False
    do_afp_relax = afp_relax
    do_afp_profile = afp_profile_set
    if do_afp_relax:
        do_afp = True
    if not do_afp:
        do_afp_relax = False
    return (do_ssrf, do_afp, do_afp_relax, do_afp_profile, do_profile)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate full-spectrum ssRF, AFP, AFP Profile, and/or ssRF-beta optimal-profile manipulated lineshapes. ssRF/AFP centers are burn-window bins where initial (equilibrium) Q < 0. Optimal profile (source=3) and AFP Profile (source=4) are independent event streams. Modes are never combined in the same event.')
    parser.add_argument('--ssrf', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable ssRF Voigt-burn events (--ssrf / --no-ssrf; default: on if no mode flags, else off)')
    parser.add_argument('--afp', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable AFP events (--afp / --no-afp; default: off unless --quick with no mode flags). Without --afp-relax, saves only the immediately post-flip spectrum (n_steps = 0)')
    parser.add_argument('--afp-relax', action=argparse.BooleanOptionalAction, default=False, help='After each per-bin AFP flip, emit spectra along a relaxation trajectory (--afp-relax / --no-afp-relax; default: off). Implies --afp. Relaxation returns P and Q to their pre-AFP values. Uses --min-relax-steps / --max-relax-steps / --relax-steps-step')
    parser.add_argument('--afp-profile', action='store_true', help='AFP Profile: one AFP sweep over all burn-window bins with initial Q < 0, then relaxation back to the pre-AFP P and Q. Saved as source=4. Independent of per-bin --afp. center_bin=-1; power_profile marks swept bins. Uses the relax-step grid')
    parser.add_argument('--profile', action=argparse.BooleanOptionalAction, default=None, help='Enable/disable ssRF-beta optimal RF profile events as an independent source (--profile / --no-profile; default: off). Saves spectra at the same n_steps grid as ssRF burns, plus the program endpoint')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help=f'Output .npz path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--plot-dir', type=Path, default=None, help=f'Directory for example PNGs (default: <output-parent>/plots, e.g. {DEFAULT_PLOT_DIR})')
    parser.add_argument('--no-plots', action='store_true', help='Skip writing example diagnostic plots')
    parser.add_argument('--p-min', type=float, default=P_MIN, help=f'Minimum polarization (default: {P_MIN})')
    parser.add_argument('--p-max', type=float, default=P_MAX, help=f'Maximum polarization (default: {P_MAX})')
    parser.add_argument('--p-step', type=float, default=P_STEP, help=f'Polarization grid step (default: {P_STEP}; denser than the 0.05 v4 NPZ grid)')
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
    args = parse_args()
    try:
        (do_ssrf, do_afp, do_afp_relax, do_afp_profile, do_profile) = _resolve_modes(ssrf=args.ssrf, afp=args.afp, afp_relax=args.afp_relax, afp_profile=args.afp_profile, profile=args.profile, quick=args.quick)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_settings = QUICK_SETTINGS if args.quick else DATA_GEN_SETTINGS
    if args.quick:
        p_values = np.asarray([0.3, 0.5])
        min_burn_steps = 0
        max_burn_steps = 100
        burn_steps_step = 5
        min_relax_steps = 0
        max_relax_steps = 100
        relax_steps_step = 5
        max_centers_per_p = 3
        afp_quick = 'AFP relax {0,5,10}' if do_afp_relax else ('AFP post-flip only' if do_afp else 'AFP off')
        if do_afp_profile:
            afp_quick += ', AFP Profile+relax'
        profile_quick = ', profile on' if do_profile else ', profile off'
        print(f'Quick smoke: P={{0.30, 0.50}}, 3 centers/P, ssRF steps {{0,5,10}}, {afp_quick}{profile_quick}', flush=True)
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
    if do_afp_profile:
        modes.append('AFP Profile')
    if do_profile:
        modes.append('profile')
    if do_afp_relax:
        afp_relax_desc = f'relax [{min_relax_steps}, {max_relax_steps}] stride {relax_steps_step}'
    elif do_afp:
        afp_relax_desc = 'post-flip only (n_steps=0)'
    else:
        afp_relax_desc = 'off'
    profile_desc = f'AFP Profile relax [{min_relax_steps}, {max_relax_steps}] stride {relax_steps_step}' if do_afp_profile else 'AFP Profile off'
    print(f"Generating {'+'.join(modes)} full spectra | P in [{args.p_min}, {args.p_max}] step {args.p_step} | ssRF gamma_rf={args.gamma_rf} steps [{min_burn_steps}, {max_burn_steps}] stride {burn_steps_step} | ssRF centers: equilibrium Q<0 only | AFP window={args.afp_window} {afp_relax_desc} | {profile_desc} | profile={'on' if do_profile else 'off'} | dt={DT} | grid n={NUM_BINS} R in [{F_MIN}, {F_MAX}] | burn R in ({BURN_R_MIN}, {BURN_R_MAX}) | Voigt FWHM G={GAUSSIAN_FWHM_R:.3f} L={LORENTZIAN_FWHM_R:.3f}", flush=True)
    profile_rates = {}
    data = generate_spectra(do_ssrf=do_ssrf, do_afp=do_afp, do_afp_relax=do_afp_relax, do_afp_profile=do_afp_profile, do_profile=do_profile, p_min=args.p_min, p_max=args.p_max, p_step=args.p_step, min_burn_steps=min_burn_steps, max_burn_steps=max_burn_steps, burn_steps_step=burn_steps_step, min_relax_steps=min_relax_steps, max_relax_steps=max_relax_steps, relax_steps_step=relax_steps_step, gamma_rf=args.gamma_rf, afp_window=args.afp_window, max_centers_per_p=max_centers_per_p, p_values=p_values, profile_settings=profile_settings, profile_rates_out=profile_rates)
    source = data['source']
    metadata = {'dataset': 'spectra', 'source_codes': {'ssRF': int(SOURCE_SSRF), 'AFP': int(SOURCE_AFP), 'optimal profile': int(SOURCE_PROFILE), 'AFP Profile': int(SOURCE_AFP_PROFILE)}, 'n_events': int(source.size), 'n_by_source': {'ssRF': int(np.sum(source == SOURCE_SSRF)), 'AFP': int(np.sum(source == SOURCE_AFP)), 'optimal profile': int(np.sum(source == SOURCE_PROFILE)), 'AFP Profile': int(np.sum(source == SOURCE_AFP_PROFILE))}}
    data['meta_json'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(output, **data)
    n = data['spectra'].shape[0]
    n_ssrf = np.sum(data['source'] == SOURCE_SSRF)
    n_afp = np.sum(data['source'] == SOURCE_AFP)
    n_afp_profile = np.sum(data['source'] == SOURCE_AFP_PROFILE)
    n_profile = np.sum(data['source'] == SOURCE_PROFILE)
    print(f"Saved N={n} events (ssRF={n_ssrf}, AFP={n_afp}, AFP Profile={n_afp_profile}, optimal profile={n_profile}) spectra={data['spectra'].shape} dtype={data['spectra'].dtype} -> {output}", flush=True)
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
