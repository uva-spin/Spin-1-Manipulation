"""Full-spectrum event generation for create_data.py."""
from __future__ import annotations

import numpy as np

from physics.rf import bin_averaged_voigt
from physics.rf.optimal_profile import DATA_GEN_SETTINGS, run_optimal_profile_polarization

from .dulya import apply_active_fit

from .afp_traj import run_one_polarization as run_afp_one
from .bridge import build_equilibrium_spin1_model, full_spectrum_intensities, level_pq
from .combined_traj import (
    combined_row_clock,
    run_combined_polarization,
    selective_manipulation_overlap,
    split_q_negative_regions,
)
from .constants import (
    BURN_R_MAX,
    BURN_R_MIN,
    COMBO_DEFAULT_MAX_CENTERS,
    COMBO_LAYOUT_PROFILE_AFP_SELECTIVE,
    COMBO_LAYOUT_PROFILE_PROFILE,
    COMBO_LAYOUT_SELECTIVE_SELECTIVE,
    COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE,
    COMBO_SCENARIO_AFP_FIRST_REGION,
    COMBO_SCENARIO_BOTH_SELECTIVE,
    COMBO_SCENARIO_SSRF_FIRST_REGION,
    DT,
    EXCLUDED_MANIPULATION_BURN_BINS,
    F_MAX,
    F_MIN,
    NUM_BINS,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE_PHYSICAL_VOIGT,
    SOURCE_AFP,
    SOURCE_AFP_PROFILE,
    SOURCE_PROFILE,
    SOURCE_SSRF,
    SOURCE_SSRF_AFP,
    SOURCE_UNMANIP,
    STORE_DTYPE,
)
from .eq import equilibrium_q_profile, polarization_grid
from .params import describe_param, ensure_rng, expand_param_product, resolve_for_polarization
from .ssrf_traj import run_one_polarization as run_ssrf_one

NOISE_LEVEL = 0.0
PROFILE_CENTER_BIN = -1
AFP_QNEG_CENTER = -1
UNMANIP_CENTER_BIN = -1
P_MIN = 0.2
P_MAX = 0.6
P_STEP = 0.025
UNMANIP_P_STEP = 0.00005
GAMMA_RF = 10.0
MIN_BURN_STEPS = 0
MAX_BURN_STEPS = 400
BURN_STEPS_STEP = 1
AFP_WINDOW = 8
MIN_RELAX_STEPS = 0
MAX_RELAX_STEPS = 8000
RELAX_STEPS_STEP = 1
GAUSSIAN_FWHM_R = RF_GAUSSIAN_FWHM_R
LORENTZIAN_FWHM_R = RF_LORENTZIAN_FWHM_R

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

def _sanitize_saved_arrays(data):
    """Ensure NPZ arrays are finite; unmanipulated power_profile is all zeros."""
    src = np.asarray(data['source']).reshape(-1)
    pp = np.nan_to_num(np.asarray(data['power_profile'], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    pp[src == SOURCE_UNMANIP] = 0.0
    data['power_profile'] = pp.astype(STORE_DTYPE, copy=False)
    ap = np.nan_to_num(np.asarray(data['applied_power'], dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
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

def _append_event(rows, *, iplus, iminus, p0, p_total, q_total, applied_power, n_steps, center_bin, source, power_profile, combo_scenario=0, combo_layout=0, num_bins=NUM_BINS, rf_dose=None, burn_steps=0, relax_steps=0, run_id=0):
    rows['spectra'].append(np.stack([np.asarray(iplus, dtype=STORE_DTYPE).reshape(-1), np.asarray(iminus, dtype=STORE_DTYPE).reshape(-1)], axis=0))
    rows['p0'].append(p0)
    rows['P_total'].append(p_total)
    rows['Q_total'].append(q_total)
    rows['applied_power'].append(applied_power)
    rows['power_profile'].append(_coerce_power_profile(power_profile, source=source, num_bins=num_bins))
    rows['n_steps'].append(n_steps)
    rows['center_bin'].append(center_bin)
    rows['source'].append(source)
    rows['combo_scenario'].append(int(combo_scenario))
    rows['combo_layout'].append(int(combo_layout))
    if rf_dose is None:
        rf_dose = np.zeros(int(num_bins), dtype=STORE_DTYPE)
    rows['rf_dose'].append(np.asarray(rf_dose, dtype=STORE_DTYPE).reshape(-1))
    rows['burn_steps'].append(int(burn_steps))
    rows['relax_steps'].append(int(relax_steps))
    rows['run_id'].append(int(run_id))

def _empty_rows():
    return {'spectra': [], 'p0': [], 'P_total': [], 'Q_total': [], 'applied_power': [], 'power_profile': [], 'n_steps': [], 'center_bin': [], 'source': [], 'combo_scenario': [], 'combo_layout': [], 'rf_dose': [], 'burn_steps': [], 'relax_steps': [], 'run_id': []}

def generate_ssrf_events(
    p_values,
    burn_window,
    steps_grid,
    *,
    gamma_rf,
    gaussian_fwhm_R=GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R=LORENTZIAN_FWHM_R,
    max_centers_per_p=None,
    num_bins=NUM_BINS,
    r_min=F_MIN,
    r_max=F_MAX,
    dt=DT,
    rng=None,
):
    """ssRF Voigt burns; save the full I+/I- spectrum at every requested burn step.

    Centers are burn-window bins with equilibrium (pre-noise) Q < 0. Runs one
    trajectory of length ``max(steps_grid)`` per (P, center) with
    ``capture_spectrum=True``, then writes a separate event for each macro-step
    index in ``steps_grid`` (index 0 = pre-burn equilibrium). Observation noise
    is added only when saving spectra, after P/Q totals are computed.

    ``gamma_rf`` / FWHM / ``dt`` may be scalars or ``(lo, hi)`` ranges (sampled
    once per center bin and held for every step of that burn). Discrete list
    grids are expanded by ``generate_spectra``.
    """
    rows = _empty_rows()
    if steps_grid.size == 0:
        return rows
    rng = ensure_rng(rng)
    max_burn = np.max(steps_grid)
    step_set = {s for s in steps_grid}
    run_counter = 0
    for (ip, p0) in enumerate(np.asarray(p_values)):
        centers = q_negative_bins_for_p0(p0, burn_window, num_bins=num_bins, r_min=r_min, r_max=r_max)
        if max_centers_per_p is not None and centers.size > max_centers_per_p:
            centers = centers[:max_centers_per_p]
        print(
            f'  ssRF P={p0:.3f} ({ip + 1}/{len(p_values)}): {centers.size} Q<0 centers × '
            f'{steps_grid.size} step spectra (γ={describe_param(gamma_rf)}, '
            f'G={describe_param(gaussian_fwhm_R)}, L={describe_param(lorentzian_fwhm_R)}, '
            f'dt={describe_param(dt)})',
            flush=True,
        )
        for bin_idx in centers:
            run_id = run_counter
            run_counter += 1
            gamma = resolve_for_polarization(gamma_rf, rng)
            g_fwhm = resolve_for_polarization(gaussian_fwhm_R, rng)
            l_fwhm = resolve_for_polarization(lorentzian_fwhm_R, rng)
            dt_p = resolve_for_polarization(dt, rng)
            grid_kwargs = _traj_grid_kwargs(num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt_p)
            traj = run_ssrf_one(
                bin_idx,
                p0,
                gamma_rf=gamma,
                n_steps=max_burn,
                rf_mode=RF_MODE_PHYSICAL_VOIGT,
                gaussian_fwhm_R=g_fwhm,
                lorentzian_fwhm_R=l_fwhm,
                capture_spectrum=True,
                legacy_spectral_recovery=False,
                **grid_kwargs,
            )
            if traj.get('skipped', False):
                continue
            iplus_full = np.asarray(traj['iplus_full'])
            iminus_full = np.asarray(traj['iminus_full'])
            profile = ssrf_voigt_power_profile(
                bin_idx,
                gamma,
                num_bins=num_bins,
                r_min=r_min,
                r_max=r_max,
                gaussian_fwhm_R=g_fwhm,
                lorentzian_fwhm_R=l_fwhm,
            )
            for burn_steps in sorted(step_set):
                k = burn_steps
                ip_spec = np.asarray(iplus_full[k]).copy()
                im_spec = np.asarray(iminus_full[k]).copy()
                (p_total, q_total) = _event_pq(traj, k)
                ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
                im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
                dose = profile * burn_steps * dt_p
                _append_event(
                    rows,
                    iplus=ip_spec,
                    iminus=im_spec,
                    p0=p0,
                    p_total=p_total,
                    q_total=q_total,
                    applied_power=gamma,
                    n_steps=burn_steps,
                    center_bin=bin_idx,
                    source=SOURCE_SSRF,
                    power_profile=profile,
                    num_bins=num_bins,
                    rf_dose=dose,
                    burn_steps=burn_steps,
                    relax_steps=0,
                    run_id=run_id,
                )
    return rows

def generate_afp_events(
    p_values,
    burn_window,
    relax_steps_grid,
    *,
    afp_window,
    max_centers_per_p=None,
    bin_stride=0,
    num_bins=NUM_BINS,
    r_min=F_MIN,
    r_max=F_MAX,
    dt=DT,
    rng=None,
):
    """AFP flip, then save the full I+/I- spectrum at each requested relax step.

    Runs one trajectory of length ``max(relax_steps_grid)`` per (P, center) with
    ``capture_spectrum=True``. Frame 0 is immediately post-flip; later frames are
    after each relax macro-step. A grid of only ``[0]`` is the post-flip snapshot
    with no relaxation. Emits a separate event for each index in
    ``relax_steps_grid`` with ``n_steps`` equal to that index.

    Relaxation, when requested, holds the post-flip tensor polarization and
    moves vector polarization toward the Boltzmann state for that Q.

    ``afp_window`` / ``dt`` may be a scalar or ``(lo, hi)`` range (sampled once per center bin and held through relaxation).

    ``bin_stride`` 0 keeps every Q<0 bin. ``n > 0`` keeps every nth affected bin.
    """
    rows = _empty_rows()
    if relax_steps_grid.size == 0:
        return rows
    rng = ensure_rng(rng)
    max_relax = np.max(relax_steps_grid)
    step_set = {s for s in relax_steps_grid}
    afp_profile = zero_power_profile(num_bins)
    bin_stride = int(bin_stride or 0)
    run_counter = 0
    for (ip, p0) in enumerate(np.asarray(p_values)):
        centers = q_negative_bins_for_p0(p0, burn_window, num_bins=num_bins, r_min=r_min, r_max=r_max)
        if bin_stride > 0:
            centers = centers[::bin_stride]
        if max_centers_per_p is not None and centers.size > max_centers_per_p:
            centers = centers[:max_centers_per_p]
        print(
            f"  AFP P={p0:.3f} ({ip + 1}/{len(p_values)}): {centers.size} Q<0 centers × "
            f"{relax_steps_grid.size} {('relax' if max_relax > 0 else 'post-flip')} spectra "
            f"(window={describe_param(afp_window)}, n_relax={max_relax}, dt={describe_param(dt)})",
            flush=True,
        )
        for bin_idx in centers:
            run_id = run_counter
            run_counter += 1
            window = resolve_for_polarization(afp_window, rng, as_int=True)
            dt_p = resolve_for_polarization(dt, rng)
            grid_kwargs = _traj_grid_kwargs(num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt_p)
            traj = run_afp_one(
                bin_idx,
                p0,
                n_relax=max_relax,
                afp_window=window,
                capture_spectrum=True,
                legacy_spectral_recovery=False,
                **grid_kwargs,
            )
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
                _append_event(
                    rows,
                    iplus=ip_spec,
                    iminus=im_spec,
                    p0=p0,
                    p_total=p_total,
                    q_total=q_total,
                    applied_power=0.0,
                    n_steps=relax_steps,
                    center_bin=bin_idx,
                    source=SOURCE_AFP,
                    power_profile=afp_profile,
                    num_bins=num_bins,
                    rf_dose=afp_profile,
                    burn_steps=0,
                    relax_steps=relax_steps,
                    run_id=run_id,
                )
    return rows

def generate_afp_qneg_events(p_values, burn_window, relax_steps_grid, *, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT, rng=None):
    """One AFP sweep over all initial Q < 0 bins, then relaxation to the pre-AFP state.

    A single trajectory per polarization covers the whole Q < 0 profile.
    ``power_profile`` is 1 on swept bins and 0 elsewhere. ``center_bin`` is
    ``AFP_QNEG_CENTER`` (-1). Frame 0 is post-flip; later frames follow the
    relaxation that returns both vector P and tensor Q to their pre-AFP values.
    """
    rows = _empty_rows()
    if relax_steps_grid.size == 0:
        return rows
    rng = ensure_rng(rng)
    max_relax = int(np.max(relax_steps_grid))
    step_set = {int(s) for s in relax_steps_grid}
    run_counter = 0
    for (ip, p0) in enumerate(np.asarray(p_values)):
        run_id = run_counter
        run_counter += 1
        dt_p = resolve_for_polarization(dt, rng)
        grid_kwargs = _traj_grid_kwargs(num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt_p)
        print(f'  AFP Profile P={p0:.3f} ({ip + 1}/{len(p_values)}): sweep + {relax_steps_grid.size} relax spectra (n_relax={max_relax}, dt={dt_p:.4g})', flush=True)
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
            _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=p_total, q_total=q_total, applied_power=0.0, n_steps=k, center_bin=AFP_QNEG_CENTER, source=SOURCE_AFP_PROFILE, power_profile=profile, num_bins=num_bins, rf_dose=profile, burn_steps=0, relax_steps=k, run_id=run_id)
    return rows

def generate_unmanipulated_events(p_values, *, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT, rng=None):
    """Equilibrium full-spectrum events with no RF (source=SOURCE_UNMANIP).

    One event per polarization: ``n_steps=0``, ``applied_power=0``, zero
    ``power_profile``, ``center_bin=-1``. Uses the same ssRF-beta Boltzmann
    equilibrium as the manipulated trajectories (no burn / flip).
    """
    rows = _empty_rows()
    rng = ensure_rng(rng)
    profile = zero_power_profile(num_bins)
    run_counter = 0
    for (ip, p0) in enumerate(np.asarray(p_values)):
        run_id = run_counter
        run_counter += 1
        dt_p = resolve_for_polarization(dt, rng)
        print(f'  unmanipulated P={p0:.3f} ({ip + 1}/{len(p_values)}) dt={dt_p:.4g}', flush=True)
        model = build_equilibrium_spin1_model(polarization=float(p0), num_bins=num_bins, dt=dt_p, rf_enabled=False, relax_enabled=True, r_min=r_min, r_max=r_max, legacy_spectral_recovery=False)
        (ip_spec, im_spec, _) = full_spectrum_intensities(model)
        (p_total, q_total) = level_pq(model)
        ip_spec = np.asarray(ip_spec, dtype=np.float64).copy()
        im_spec = np.asarray(im_spec, dtype=np.float64).copy()
        ip_spec += np.random.normal(0, NOISE_LEVEL, ip_spec.shape)
        im_spec += np.random.normal(0, NOISE_LEVEL, im_spec.shape)
        _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=float(p_total), q_total=float(q_total), applied_power=0.0, n_steps=0, center_bin=UNMANIP_CENTER_BIN, source=SOURCE_UNMANIP, power_profile=profile, num_bins=num_bins, rf_dose=profile, burn_steps=0, relax_steps=0, run_id=run_id)
    return rows

def generate_profile_events(p_values, steps_grid, *, settings=DATA_GEN_SETTINGS, num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX, dt=DT, rng=None):
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
    rng = ensure_rng(rng)
    max_save = int(np.max(steps_grid))
    run_counter = 0
    for (ip, p0) in enumerate(np.asarray(p_values)):
        run_id = run_counter
        run_counter += 1
        dt_p = resolve_for_polarization(dt, rng)
        print(f'  profile P={p0:.3f} ({ip + 1}/{len(p_values)}): designing synchronized RF program (dt={dt_p:.4g})', flush=True)
        traj = run_optimal_profile_polarization(
            p0,
            n_steps=max_save,
            n_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            dt=dt_p,
            settings=settings,
            equilibrium_loader=apply_active_fit,
        )
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
            dose = profile * burn_steps * dt_p
            _append_event(rows, iplus=ip_spec, iminus=im_spec, p0=p0, p_total=p_total, q_total=q_total, applied_power=peak_u, n_steps=burn_steps, center_bin=PROFILE_CENTER_BIN, source=SOURCE_PROFILE, power_profile=profile, num_bins=num_bins, rf_dose=dose, burn_steps=burn_steps, relax_steps=0, run_id=run_id)
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
    gaussian_fwhm_R=GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R=LORENTZIAN_FWHM_R,
    max_centers_per_p=None,
    combined_max_centers=None,
    num_bins=NUM_BINS,
    r_min=F_MIN,
    r_max=F_MAX,
    dt=DT,
    rng=None,
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

    ``gamma_rf`` is sampled once per selective ssRF burn. Profile ssRF layouts
    ignore it and replay the designed optimal per-bin rates. Other RF knobs
    and ``dt`` may be scalars or ``(lo, hi)`` ranges sampled once per trajectory.
    """
    rows = _empty_rows()
    if burn_steps_grid.size == 0:
        return rows
    rng = ensure_rng(rng)
    max_burn = int(np.max(burn_steps_grid))
    burn_step_set = {int(s) for s in burn_steps_grid}
    relax_step_set = {int(s) for s in relax_steps_grid}
    centers_cap = _combo_centers_cap(max_centers_per_p, combined_max_centers)
    frequency = np.linspace(r_min, r_max, num_bins)
    profile_cache = {}
    n_p = len(np.asarray(p_values))
    run_counter = 0
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
            f'max_burn={max_burn}; max_relax={int(max_relax)}; '
            f'γ={describe_param(gamma_rf)} window={describe_param(afp_window)} '
            f'G={describe_param(gaussian_fwhm_R)} L={describe_param(lorentzian_fwhm_R)} '
            f'dt={describe_param(dt)}',
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
                    run_id = run_counter
                    run_counter += 1
                    # Profile ssRF uses design_optimal_profile rates, not this draw.
                    gamma = (
                        resolve_for_polarization(gamma_rf, rng)
                        if _combined_uses_selective_ssrf(layout)
                        else float(GAMMA_RF)
                    )
                    g_fwhm = resolve_for_polarization(gaussian_fwhm_R, rng)
                    l_fwhm = resolve_for_polarization(lorentzian_fwhm_R, rng)
                    window = resolve_for_polarization(afp_window, rng, as_int=True)
                    dt_p = resolve_for_polarization(dt, rng)
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
                            afp_window=window,
                            gaussian_fwhm_R=g_fwhm,
                            lorentzian_fwhm_R=l_fwhm,
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
                            gamma_rf=gamma,
                            max_burn=max_burn,
                            max_relax=max_relax,
                            afp_window=window,
                            gaussian_fwhm_R=g_fwhm,
                            lorentzian_fwhm_R=l_fwhm,
                            profile_settings=profile_settings,
                            num_bins=num_bins,
                            r_min=r_min,
                            r_max=r_max,
                            dt=dt_p,
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
                    applied = float(traj.get('applied_peak', gamma))
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
                        burn_steps_count = max(0, min(frame - ssrf_start, n_ssrf))
                        relax_steps_count = max(0, frame - relax_start) if relax_start is not None and frame >= relax_start else 0
                        dose = profiles[frame] * burn_steps_count * dt_p
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
                            num_bins=num_bins,
                            rf_dose=dose,
                            burn_steps=burn_steps_count,
                            relax_steps=relax_steps_count,
                            run_id=run_id,
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
        return {'spectra': np.empty((0, 2, NUM_BINS), dtype=STORE_DTYPE), 'p0': np.empty(0, dtype=STORE_DTYPE), 'P_total': np.empty(0, dtype=STORE_DTYPE), 'Q_total': np.empty(0, dtype=STORE_DTYPE), 'applied_power': np.empty(0, dtype=STORE_DTYPE), 'power_profile': np.empty((0, NUM_BINS), dtype=STORE_DTYPE), 'n_steps': np.empty(0, dtype=np.int32), 'center_bin': np.empty(0, dtype=np.int32), 'source': np.empty(0, dtype=np.uint8), 'combo_scenario': np.empty(0, dtype=np.uint8), 'combo_layout': np.empty(0, dtype=np.uint8), 'rf_dose': np.empty((0, NUM_BINS), dtype=STORE_DTYPE), 'burn_steps': np.empty(0, dtype=np.int32), 'relax_steps': np.empty(0, dtype=np.int32), 'run_id': np.empty(0, dtype=np.int64)}
    return {'spectra': np.stack(merged['spectra'], axis=0).astype(STORE_DTYPE, copy=False), 'p0': np.asarray(merged['p0'], dtype=STORE_DTYPE), 'P_total': np.asarray(merged['P_total'], dtype=STORE_DTYPE), 'Q_total': np.asarray(merged['Q_total'], dtype=STORE_DTYPE), 'applied_power': np.asarray(merged['applied_power'], dtype=STORE_DTYPE), 'power_profile': np.stack(merged['power_profile'], axis=0).astype(STORE_DTYPE, copy=False), 'n_steps': np.asarray(merged['n_steps'], dtype=np.int32), 'center_bin': np.asarray(merged['center_bin'], dtype=np.int32), 'source': np.asarray(merged['source'], dtype=np.uint8), 'combo_scenario': np.asarray(merged['combo_scenario'], dtype=np.uint8), 'combo_layout': np.asarray(merged['combo_layout'], dtype=np.uint8), 'rf_dose': np.stack(merged['rf_dose'], axis=0).astype(STORE_DTYPE, copy=False), 'burn_steps': np.asarray(merged['burn_steps'], dtype=np.int32), 'relax_steps': np.asarray(merged['relax_steps'], dtype=np.int32), 'run_id': np.asarray(merged['run_id'], dtype=np.int64)}

def generate_spectra(
    *,
    do_ssrf,
    do_afp,
    do_afp_relax=False,
    do_afp_profile=False,
    do_profile=False,
    do_unmanipulated=False,
    do_ssrf_afp_combined=False,
    p_min=P_MIN,
    p_max=P_MAX,
    p_step=P_STEP,
    unmanip_p_step=UNMANIP_P_STEP,
    min_burn_steps=MIN_BURN_STEPS,
    max_burn_steps=MAX_BURN_STEPS,
    burn_steps_step=BURN_STEPS_STEP,
    min_relax_steps=MIN_RELAX_STEPS,
    max_relax_steps=MAX_RELAX_STEPS,
    relax_steps_step=RELAX_STEPS_STEP,
    gamma_rf=GAMMA_RF,
    afp_window=AFP_WINDOW,
    gaussian_fwhm_R=GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R=LORENTZIAN_FWHM_R,
    max_centers_per_p=None,
    combined_max_centers=None,
    combined_max_relax_steps=None,
    afp_bin_stride=0,
    p_values=None,
    unmanip_p_values=None,
    profile_settings=DATA_GEN_SETTINGS,
    num_bins=NUM_BINS,
    r_min=F_MIN,
    r_max=F_MAX,
    dt=DT,
    profile_rates_out=None,
    seed=None,
    rng=None,
):
    """Build full-spectrum events for the requested modes.

    Manipulated modes (ssRF / AFP / profile) use ``p_step`` (or ``p_values``).
    Unmanipulated uses ``unmanip_p_step`` by default (finer grid), unless
    ``unmanip_p_values`` is given, or ``p_values`` is given without a separate
    unmanipulated list (e.g. ``--quick`` smoke polarizations).

    ``gamma_rf`` varies selective ssRF burns only (scalar, ``(lo, hi)`` once
    per burn, or a list). Optimal-profile ssRF keeps its designed per-bin
    rates. ``afp_window``, FWHM, and ``dt`` accept the same forms and are
    drawn once per burn or whole-spectrum trajectory.

    If ``profile_rates_out`` is a dict, it is filled with designed ``U(R)``
    envelopes keyed by rounded polarization for diagnostic plotting.
    """
    rng = ensure_rng(rng if rng is not None else seed)
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
        for combo in expand_param_product(
            {
                'gamma_rf': gamma_rf,
                'gaussian_fwhm_R': gaussian_fwhm_R,
                'lorentzian_fwhm_R': lorentzian_fwhm_R,
                'dt': dt,
            }
        ):
            groups.append(
                generate_ssrf_events(
                    manip_p,
                    burn_window,
                    steps_grid,
                    gamma_rf=combo['gamma_rf'],
                    gaussian_fwhm_R=combo['gaussian_fwhm_R'],
                    lorentzian_fwhm_R=combo['lorentzian_fwhm_R'],
                    dt=combo['dt'],
                    max_centers_per_p=max_centers_per_p,
                    num_bins=num_bins,
                    r_min=r_min,
                    r_max=r_max,
                    rng=rng,
                )
            )
    if do_afp:
        relax_grid = _afp_relax_grid(
            do_afp_relax,
            min_relax_steps=min_relax_steps,
            max_relax_steps=max_relax_steps,
            relax_steps_step=relax_steps_step,
        )
        for combo in expand_param_product({'afp_window': afp_window, 'dt': dt}):
            groups.append(
                generate_afp_events(
                    manip_p,
                    burn_window,
                    relax_grid,
                    afp_window=combo['afp_window'],
                    dt=combo['dt'],
                    max_centers_per_p=max_centers_per_p,
                    bin_stride=afp_bin_stride,
                    num_bins=num_bins,
                    r_min=r_min,
                    r_max=r_max,
                    rng=rng,
                )
            )
    if do_afp_profile:
        relax_grid = burn_steps_values(min_relax_steps, max_relax_steps, relax_steps_step)
        for combo in expand_param_product({'dt': dt}):
            groups.append(
                generate_afp_qneg_events(
                    manip_p, burn_window, relax_grid, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=combo['dt'], rng=rng
                )
            )
    if do_profile:
        steps_grid = burn_steps_values(min_burn_steps, max_burn_steps, burn_steps_step)
        for combo in expand_param_product({'dt': dt}):
            (profile_rows, rates) = generate_profile_events(
                manip_p, steps_grid, settings=profile_settings, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=combo['dt'], rng=rng
            )
            groups.append(profile_rows)
            profile_rates.update(rates)
    if do_unmanipulated:
        for combo in expand_param_product({'dt': dt}):
            groups.append(
                generate_unmanipulated_events(unmanip_p, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=combo['dt'], rng=rng)
            )
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
        for combo in expand_param_product(
            {
                'gamma_rf': gamma_rf,
                'afp_window': afp_window,
                'gaussian_fwhm_R': gaussian_fwhm_R,
                'lorentzian_fwhm_R': lorentzian_fwhm_R,
                'dt': dt,
            }
        ):
            groups.append(
                generate_combined_ssrf_afp_events(
                    manip_p,
                    burn_window,
                    steps_grid,
                    combo_relax_grid,
                    gamma_rf=combo['gamma_rf'],
                    max_relax=combo_relax,
                    afp_window=combo['afp_window'],
                    gaussian_fwhm_R=combo['gaussian_fwhm_R'],
                    lorentzian_fwhm_R=combo['lorentzian_fwhm_R'],
                    dt=combo['dt'],
                    profile_settings=profile_settings,
                    max_centers_per_p=max_centers_per_p,
                    combined_max_centers=combined_max_centers,
                    num_bins=num_bins,
                    r_min=r_min,
                    r_max=r_max,
                    rng=rng,
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
