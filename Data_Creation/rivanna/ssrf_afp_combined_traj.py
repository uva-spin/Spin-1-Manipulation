"""Combined ssRF + AFP full-spectrum trajectories for create_data.py."""
from __future__ import annotations

import numpy as np

from afp_bin_traj import qneg_sweep_indices
from common import (
    AFP_CENTER_MARGIN,
    AFP_EFFICIENCY,
    COMBO_CENTER_BIN,
    COMBO_LAYOUT_PROFILE_AFP_SELECTIVE,
    COMBO_LAYOUT_PROFILE_PROFILE,
    COMBO_LAYOUT_SELECTIVE_SELECTIVE,
    COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE,
    COMBO_SCENARIO_AFP_FIRST_REGION,
    COMBO_SCENARIO_BOTH_SELECTIVE,
    COMBO_SCENARIO_SSRF_FIRST_REGION,
    DIFFUSION_SCALE,
    DT,
    F_MAX,
    F_MIN,
    NUM_BINS,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE_PHYSICAL_VOIGT,
)
from model_bridge import (
    afp_touched_bins,
    afp_window_indices,
    burn_commit_touched_bins,
    build_equilibrium_spin1_model,
    commit_touched_bins_only,
    configure_afp_recovery,
    configure_ssrf_burn,
    euler_n_sub,
    full_spectrum_intensities,
    level_pq,
    mirror_bin_idx,
)
from physics.rf import bin_averaged_voigt
from physics.rf.optimal_profile import replay_program_spectra, run_optimal_profile_polarization, synchronized_program
from physics.rf.profile_control import make_ideal_model


def split_q_negative_regions(centers, frequency):
    """Split sorted Q<0 bin indices into contiguous spans along the grid."""
    idx = np.asarray(centers, dtype=np.int32).reshape(-1)
    if idx.size == 0:
        return []
    idx = np.sort(idx)
    regions = []
    start = prev = int(idx[0])
    for raw in idx[1:]:
        i = int(raw)
        if i == prev + 1:
            prev = i
            continue
        regions.append(np.arange(start, prev + 1, dtype=np.int32))
        start = prev = i
    regions.append(np.arange(start, prev + 1, dtype=np.int32))
    return regions


def _region_allow_mask(num_bins, region_bins):
    mask = np.zeros(int(num_bins), dtype=bool)
    if region_bins is not None and len(region_bins):
        mask[np.asarray(region_bins, dtype=int)] = True
    return mask


def _deepest_q_center(q, region_bins):
    region = np.asarray(region_bins, dtype=int).reshape(-1)
    if region.size == 0:
        return None
    pick = region[int(np.argmin(q[region]))]
    return int(pick)


def ssrf_influence_bins(
    center_bin,
    *,
    num_bins,
    r_min,
    r_max,
    gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R,
):
    """Bins affected by a physical-R Voigt ssRF burn (support + mirror partners)."""
    grid = np.linspace(float(r_min), float(r_max), int(num_bins))
    touched = burn_commit_touched_bins(
        int(num_bins),
        int(center_bin),
        rf_mode=RF_MODE_PHYSICAL_VOIGT,
        R=grid,
        gaussian_fwhm_R=gaussian_fwhm_R,
        lorentzian_fwhm_R=lorentzian_fwhm_R,
        include_diffusion_spillover=False,
    )
    return {int(i) for i in touched}


def afp_influence_bins(center_bin, *, num_bins, afp_window):
    """Bins in an AFP window plus mirror partners (same convention as shard trajectories)."""
    subset = afp_window_indices(int(center_bin), int(num_bins), int(afp_window))
    return {int(i) for i in afp_touched_bins(int(num_bins), subset)}


def selective_manipulation_overlap(
    ssrf_center,
    afp_center,
    *,
    num_bins,
    r_min,
    r_max,
    afp_window,
    gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R,
):
    """True when ssRF Voigt support and an AFP window share any bin."""
    if ssrf_center is None or afp_center is None:
        return False
    ssrf_bins = ssrf_influence_bins(
        ssrf_center,
        num_bins=num_bins,
        r_min=r_min,
        r_max=r_max,
        gaussian_fwhm_R=gaussian_fwhm_R,
        lorentzian_fwhm_R=lorentzian_fwhm_R,
    )
    afp_bins = afp_influence_bins(afp_center, num_bins=num_bins, afp_window=afp_window)
    return bool(ssrf_bins & afp_bins)


def _forbidden_from_profile_rates(rates, *, num_bins, r_min, r_max, rate_floor=1e-8):
    """Bins receiving ssRF under an optimal profile (nonzero U), plus mirrors."""
    arr = np.asarray(rates, dtype=np.float64).reshape(-1)
    if arr.size != int(num_bins):
        grid = np.linspace(r_min, r_max, int(num_bins))
        src = np.linspace(r_min, r_max, arr.size) if arr.size else grid
        arr = np.interp(grid, src, arr) if arr.size else np.zeros(int(num_bins))
    active = np.flatnonzero(arr > float(rate_floor))
    forbidden = set()
    n = int(num_bins)
    for b in active:
        b = int(b)
        forbidden.add(b)
        forbidden.add(mirror_bin_idx(n, b))
    return forbidden


def _filter_afp_subset(subset, forbidden):
    if not forbidden:
        return [int(i) for i in subset]
    blocked = {int(i) for i in forbidden}
    return [int(i) for i in subset if int(i) not in blocked]


def _ordered_region_centers(region_bins, q, *, max_centers=None):
    region = np.asarray(region_bins, dtype=int).reshape(-1)
    if region.size == 0:
        return []
    order = np.argsort(np.asarray(q)[region])
    ordered = [int(region[i]) for i in order]
    if max_centers is not None:
        ordered = ordered[: int(max_centers)]
    return ordered


def plan_disjoint_selective_centers(
    q,
    ssrf_region,
    afp_region,
    *,
    num_bins,
    r_min,
    r_max,
    afp_window,
    gaussian_fwhm_R,
    lorentzian_fwhm_R,
    ssrf_center_bin=None,
    afp_center_bin=None,
    max_centers_per_region=None,
):
    """Pick ssRF / AFP selective centers whose Voigt support and AFP windows do not overlap."""
    ssrf_candidates = _ordered_region_centers(ssrf_region, q, max_centers=None)
    afp_candidates = _ordered_region_centers(afp_region, q, max_centers=None)

    if ssrf_center_bin is not None and afp_center_bin is not None:
        if selective_manipulation_overlap(
            ssrf_center_bin,
            afp_center_bin,
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            afp_window=afp_window,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
        ):
            return (None, [])
        return (int(ssrf_center_bin), [int(afp_center_bin)])

    if ssrf_center_bin is not None:
        ssrf_pick = int(ssrf_center_bin)
        ssrf_block = ssrf_influence_bins(
            ssrf_pick,
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
        )
        afp_picks = []
        for a in afp_candidates:
            if afp_influence_bins(a, num_bins=num_bins, afp_window=afp_window) & ssrf_block:
                continue
            afp_picks.append(a)
            if max_centers_per_region is not None and len(afp_picks) >= int(max_centers_per_region):
                break
        return (ssrf_pick, afp_picks)

    if afp_center_bin is not None:
        afp_pick = int(afp_center_bin)
        afp_block = afp_influence_bins(afp_pick, num_bins=num_bins, afp_window=afp_window)
        for s in ssrf_candidates:
            if ssrf_influence_bins(
                s,
                num_bins=num_bins,
                r_min=r_min,
                r_max=r_max,
                gaussian_fwhm_R=gaussian_fwhm_R,
                lorentzian_fwhm_R=lorentzian_fwhm_R,
            ) & afp_block:
                continue
            return (int(s), [afp_pick])
        return (None, [])

    for s in ssrf_candidates:
        ssrf_block = ssrf_influence_bins(
            s,
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
        )
        occupied = set(ssrf_block)
        afp_picks = []
        for a in afp_candidates:
            a_block = afp_influence_bins(a, num_bins=num_bins, afp_window=afp_window)
            if a_block & occupied:
                continue
            afp_picks.append(a)
            occupied |= a_block
            if max_centers_per_region is not None and len(afp_picks) >= int(max_centers_per_region):
                break
        if afp_picks:
            return (int(s), afp_picks)
    return (None, [])


def _align_rates_to_grid(rates, frequency_src, *, num_bins, r_min, r_max):
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    src_f = np.asarray(frequency_src, dtype=np.float64).reshape(-1)
    dst_f = np.linspace(r_min, r_max, int(num_bins))
    if rates.size == src_f.size and rates.size == num_bins:
        aligned = rates
    else:
        aligned = np.interp(dst_f, src_f, rates) if rates.size else np.zeros(num_bins)
    return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)


def _run_ssrf_profile_phase(
    p0,
    region_bins,
    *,
    max_save_step,
    num_bins,
    r_min,
    r_max,
    dt,
    settings,
):
    allow = _region_allow_mask(num_bins, region_bins)
    traj = run_optimal_profile_polarization(
        p0,
        n_steps=max_save_step,
        n_bins=num_bins,
        r_min=r_min,
        r_max=r_max,
        dt=dt,
        settings=settings,
        bin_allow_mask=allow,
    )
    if traj.get("skipped", False):
        return None
    return traj


def _ssrf_voigt_profile(center_bin, gamma_rf, *, num_bins, r_min, r_max, gaussian_fwhm_R, lorentzian_fwhm_R):
    freq = np.linspace(r_min, r_max, int(num_bins))
    dR = float(np.abs(freq[1] - freq[0])) if freq.size > 1 else 1.0
    shape = bin_averaged_voigt(
        freq,
        center_R=float(freq[int(center_bin)]),
        bin_width_R=dR,
        gaussian_fwhm_R=gaussian_fwhm_R,
        lorentzian_fwhm_R=lorentzian_fwhm_R,
        normalization="center_bin",
    )
    return float(gamma_rf) * np.asarray(shape, dtype=np.float64)


def equilibrium_q_for_p0(p0, *, num_bins, r_min, r_max, dt=DT):
    model = build_equilibrium_spin1_model(
        polarization=float(p0),
        num_bins=num_bins,
        dt=dt,
        rf_enabled=False,
        relax_enabled=True,
        r_min=r_min,
        r_max=r_max,
        legacy_spectral_recovery=False,
    )
    (ip, im, _) = full_spectrum_intensities(model)
    return np.asarray(ip) - np.asarray(im)


def _apply_afp_profile_phase(model, region_bins, *, num_bins, forbidden_bins=None):
    (iplus0, iminus0, _) = full_spectrum_intensities(model)
    q_eq = np.asarray(iplus0) - np.asarray(iminus0)
    subset = qneg_sweep_indices(q_eq, region_bins)
    subset = _filter_afp_subset(subset, forbidden_bins)
    if not subset:
        return None
    model.params.afp_enabled = True
    model.params.afp_efficiency = AFP_EFFICIENCY
    model.params.afp_center_margin = AFP_CENTER_MARGIN
    model.params.afp_preserve_intensity_area = False
    model.params.afp_subset_indices = [int(i) for i in subset]
    model.afp_sweep()
    model.params.afp_enabled = False
    (ip_sim, im_sim, _) = model.physical_intensities()
    touched = afp_touched_bins(num_bins, subset)
    (base_ip, base_im) = commit_touched_bins_only(iplus0, iminus0, ip_sim, im_sim, touched)
    model.load_from_physical_intensities(base_ip, base_im)
    profile = np.zeros(num_bins, dtype=np.float64)
    profile[np.asarray(subset, dtype=int)] = 1.0
    return (profile, subset)


def _apply_afp_selective_phase(
    model,
    region_bins,
    *,
    afp_window,
    num_bins,
    max_centers=None,
    center_bins=None,
    forbidden_bins=None,
):
    (iplus0, iminus0, _) = full_spectrum_intensities(model)
    q_eq = np.asarray(iplus0) - np.asarray(iminus0)
    if center_bins is not None:
        centers = np.asarray(center_bins, dtype=int).reshape(-1)
    else:
        centers = np.asarray(region_bins, dtype=int).reshape(-1)
        if max_centers is not None and centers.size > max_centers:
            order = np.argsort(q_eq[centers])
            centers = centers[order[:max_centers]]
    blocked = set(int(i) for i in (forbidden_bins or ()))
    kept = []
    for bin_idx in centers:
        bin_idx = int(bin_idx)
        if afp_influence_bins(bin_idx, num_bins=num_bins, afp_window=afp_window) & blocked:
            continue
        kept.append(bin_idx)
    centers = np.asarray(kept, dtype=int)
    if centers.size == 0:
        return None
    profile = np.zeros(num_bins, dtype=np.float64)
    applied = []
    for bin_idx in centers:
        subset = afp_window_indices(int(bin_idx), num_bins, afp_window)
        subset = _filter_afp_subset(subset, forbidden_bins)
        if not subset:
            continue
        model.params.afp_enabled = True
        model.params.afp_efficiency = AFP_EFFICIENCY
        model.params.afp_center_margin = AFP_CENTER_MARGIN
        model.params.afp_preserve_intensity_area = False
        model.params.afp_subset_indices = [int(i) for i in subset]
        model.afp_sweep()
        model.params.afp_enabled = False
        (ip_sim, im_sim, _) = model.physical_intensities()
        touched = afp_touched_bins(num_bins, subset)
        (iplus0, iminus0) = commit_touched_bins_only(iplus0, iminus0, ip_sim, im_sim, touched)
        model.load_from_physical_intensities(iplus0, iminus0)
        profile[np.asarray(subset, dtype=int)] = 1.0
        applied.append(int(bin_idx))
    if not applied:
        return None
    return (profile, int(applied[-1]))


def _mask_ssrf_off_afp(profile, afp_profile):
    """Drop ssRF power on bins already marked by the AFP sweep."""
    out = np.asarray(profile, dtype=np.float64).reshape(-1).copy()
    afp = np.asarray(afp_profile, dtype=np.float64).reshape(-1)
    if afp.size == out.size:
        out[afp > 0.0] = 0.0
    return out


def _record_spectrum_frame(model, frame, iplus_full, iminus_full, p_full, q_full):
    (ip_s, im_s, _) = full_spectrum_intensities(model)
    (p_total, q_total) = level_pq(model)
    iplus_full[frame] = ip_s
    iminus_full[frame] = im_s
    p_full[frame] = p_total
    q_full[frame] = q_total


def _burn_ssrf_on_model(
    model,
    center,
    gamma_rf,
    n_burn,
    *,
    num_bins,
    r_min,
    r_max,
    gaussian_fwhm_R,
    lorentzian_fwhm_R,
    legacy_spectral_recovery,
    afp_profile,
):
    """Step ssRF on the current (post-AFP) spectrum. Returns spectra and the ssRF envelope."""
    configure_ssrf_burn(
        model,
        int(center),
        gamma_rf,
        rf_mode=RF_MODE_PHYSICAL_VOIGT,
        gaussian_fwhm_R=gaussian_fwhm_R,
        lorentzian_fwhm_R=lorentzian_fwhm_R,
        legacy_spectral_recovery=legacy_spectral_recovery,
    )
    prof = _mask_ssrf_off_afp(
        _ssrf_voigt_profile(
            center,
            gamma_rf,
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
        ),
        afp_profile,
    )
    n_burn = int(n_burn)
    t_len = n_burn + 1
    n_bins = int(prof.size)
    iplus = np.empty((t_len, n_bins), dtype=np.float64)
    iminus = np.empty((t_len, n_bins), dtype=np.float64)
    p_full = np.empty(t_len, dtype=np.float64)
    q_full = np.empty(t_len, dtype=np.float64)
    _record_spectrum_frame(model, 0, iplus, iminus, p_full, q_full)
    (n_sub, dt_sub) = euler_n_sub(gamma_rf, model.params.dt)
    for k in range(1, t_len):
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
        _record_spectrum_frame(model, k, iplus, iminus, p_full, q_full)
    return iplus, iminus, p_full, q_full, prof


def _replay_profile_on_state(ip, im, rates, duration, *, polarization, num_bins, r_min, r_max, dt, n_record):
    """Play a designed ssRF envelope starting from post-AFP intensities."""
    n_record = int(n_record)
    play = make_ideal_model(polarization, n_bins=int(num_bins), r_min=float(r_min), r_max=float(r_max))
    play.params.dt = float(dt)
    play.load_from_physical_intensities(np.asarray(ip), np.asarray(im))
    (ip0, im0, _) = full_spectrum_intensities(play)
    (p0, q0) = level_pq(play)
    if n_record <= 0 or float(duration) <= 0.0 or not np.any(np.asarray(rates) > 0.0):
        iplus = np.repeat(np.asarray(ip0, dtype=np.float64).reshape(1, -1), n_record + 1, axis=0)
        iminus = np.repeat(np.asarray(im0, dtype=np.float64).reshape(1, -1), n_record + 1, axis=0)
        return iplus, iminus, np.full(n_record + 1, p0), np.full(n_record + 1, q0)
    program = synchronized_program(np.asarray(play.Rplus), np.asarray(rates, dtype=np.float64), float(duration))
    return replay_program_spectra(play, program, n_record=n_record)


def _write_ssrf_frames(iplus_full, iminus_full, p_full, q_full, power_profiles, ssrf_start, n_ssrf, ip_s, im_s, p_s, q_s, ssrf_prof):
    n_copy = min(int(n_ssrf), int(np.asarray(ip_s).shape[0]))
    if n_copy <= 0:
        return
    stop = int(ssrf_start) + n_copy
    iplus_full[ssrf_start:stop] = np.asarray(ip_s[:n_copy], dtype=np.float64)
    iminus_full[ssrf_start:stop] = np.asarray(im_s[:n_copy], dtype=np.float64)
    p_full[ssrf_start:stop] = np.asarray(p_s[:n_copy], dtype=np.float64)
    q_full[ssrf_start:stop] = np.asarray(q_s[:n_copy], dtype=np.float64)
    if n_copy < int(n_ssrf):
        rest = slice(stop, int(ssrf_start) + int(n_ssrf))
        iplus_full[rest] = iplus_full[stop - 1]
        iminus_full[rest] = iminus_full[stop - 1]
        p_full[rest] = p_full[stop - 1]
        q_full[rest] = q_full[stop - 1]
    power_profiles[int(ssrf_start):int(ssrf_start) + int(n_ssrf)] = ssrf_prof


def _relax_and_capture(
    model,
    *,
    n_relax,
    capture_spectrum,
    relax_start,
    iplus_full,
    iminus_full,
    p_full,
    q_full,
):
    if n_relax <= 0:
        return
    configure_afp_recovery(model)
    model.params.rf_enabled = False
    for k in range(int(n_relax)):
        model.step_once(dt=model.params.dt, rf_on=False, dnp_on=False, copy=False)
        if not capture_spectrum:
            continue
        _record_spectrum_frame(model, int(relax_start) + k, iplus_full, iminus_full, p_full, q_full)


def combined_row_clock(frame, ssrf_start, applied_peak, relax_start=None):
    """AFP snapshot is step 0 with power 0. ssRF rows count burn steps. Later rows count relax steps."""
    frame = int(frame)
    ssrf_start = int(ssrf_start)
    if frame < ssrf_start:
        return 0, 0.0
    if relax_start is not None and frame >= int(relax_start):
        return frame - int(relax_start), 0.0
    return frame - ssrf_start, float(applied_peak)


def run_combined_polarization(
    polarization,
    *,
    scenario,
    layout,
    ssrf_region,
    afp_region,
    gamma_rf,
    max_burn,
    max_relax,
    afp_window,
    profile_settings,
    num_bins=NUM_BINS,
    r_min=F_MIN,
    r_max=F_MAX,
    dt=DT,
    gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R,
    capture_spectrum=True,
    max_centers_per_region=None,
    legacy_spectral_recovery=False,
    ssrf_center_bin=None,
    afp_center_bin=None,
    profile_cache=None,
):
    """One equilibrium trajectory: AFP on ``afp_region``, then ssRF on ``ssrf_region``.

    The AFP envelope is stored on the first frame only. ssRF is applied to that
    post-sweep spectrum and its envelope is stored on the later burn frames,
    with AFP bins removed so the two profiles do not share support.
    ``profile_cache`` reuses the designed ssRF envelope for the same
    polarization and region (the AFP sweep does not change that design).
    """
    p0 = float(polarization)
    q_once = None

    def equilibrium_q():
        nonlocal q_once
        if q_once is None:
            q_once = equilibrium_q_for_p0(p0, num_bins=num_bins, r_min=r_min, r_max=r_max, dt=dt)
        return q_once
    ssrf_region = np.asarray(ssrf_region, dtype=np.int32).reshape(-1)
    afp_region = np.asarray(afp_region, dtype=np.int32).reshape(-1)
    if ssrf_region.size == 0 or afp_region.size == 0:
        return {"polarization": p0, "skipped": True, "reason": "empty_region"}

    use_profile_ssrf = layout in (COMBO_LAYOUT_PROFILE_PROFILE, COMBO_LAYOUT_PROFILE_AFP_SELECTIVE)
    use_profile_afp = layout in (COMBO_LAYOUT_PROFILE_PROFILE, COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE)
    use_selective_ssrf = not use_profile_ssrf
    use_selective_afp = not use_profile_afp
    planned_afp_centers = None

    if use_selective_ssrf and use_selective_afp:
        q_plan = equilibrium_q()
        (ssrf_pick, planned_afp_centers) = plan_disjoint_selective_centers(
            q_plan,
            ssrf_region,
            afp_region,
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            afp_window=afp_window,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
            ssrf_center_bin=ssrf_center_bin,
            afp_center_bin=afp_center_bin,
            max_centers_per_region=max_centers_per_region,
        )
        if ssrf_pick is None or not planned_afp_centers:
            return {"polarization": p0, "skipped": True, "reason": "no_disjoint_selective_centers"}
        ssrf_center_bin = ssrf_pick
    elif use_selective_ssrf and afp_center_bin is not None:
        q_plan = equilibrium_q()
        if selective_manipulation_overlap(
            ssrf_center_bin if ssrf_center_bin is not None else _deepest_q_center(q_plan, ssrf_region),
            afp_center_bin,
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            afp_window=afp_window,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
        ):
            return {"polarization": p0, "skipped": True, "reason": "selective_ssrf_afp_overlap"}
    elif use_selective_afp and ssrf_center_bin is not None:
        q_plan = equilibrium_q()
        if selective_manipulation_overlap(
            ssrf_center_bin,
            afp_center_bin if afp_center_bin is not None else _deepest_q_center(q_plan, afp_region),
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            afp_window=afp_window,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
        ):
            return {"polarization": p0, "skipped": True, "reason": "selective_ssrf_afp_overlap"}

    ssrf_profile_traj = None
    if use_profile_ssrf:
        cache_key = (
            float(p0), int(max_burn), int(num_bins), float(r_min), float(r_max), float(dt),
            np.ascontiguousarray(ssrf_region).tobytes(),
        )
        if profile_cache is not None and cache_key in profile_cache:
            ssrf_profile_traj = profile_cache[cache_key]
        else:
            ssrf_profile_traj = _run_ssrf_profile_phase(
                p0,
                ssrf_region,
                max_save_step=max(0, int(max_burn)),
                num_bins=num_bins,
                r_min=r_min,
                r_max=r_max,
                dt=dt,
                settings=profile_settings,
            )
            if profile_cache is not None:
                profile_cache[cache_key] = ssrf_profile_traj
        if ssrf_profile_traj is None:
            return {"polarization": p0, "skipped": True, "reason": "ssrf_profile_failed"}

    ssrf_rates = None
    ssrf_freq = None
    ssrf_duration = 0.0
    applied_peak = float(gamma_rf)
    ssrf_center = COMBO_CENTER_BIN
    if use_profile_ssrf and ssrf_profile_traj is not None:
        n_ssrf_frames = int(np.asarray(ssrf_profile_traj["iplus_full"]).shape[0])
        ssrf_rates = np.asarray(ssrf_profile_traj.get("rates", []), dtype=float).reshape(-1)
        ssrf_freq = np.asarray(ssrf_profile_traj.get("frequency"), dtype=float).reshape(-1)
        ssrf_duration = float(ssrf_profile_traj.get("duration", 0.0))
        applied_peak = float(ssrf_profile_traj.get("gamma_rf", 0.0))
    else:
        q_eq = equilibrium_q()
        burn_center = ssrf_center_bin
        if burn_center is None:
            for candidate in _ordered_region_centers(ssrf_region, q_eq):
                blocked = False
                if use_selective_afp and afp_center_bin is not None:
                    blocked = selective_manipulation_overlap(
                        candidate,
                        afp_center_bin,
                        num_bins=num_bins,
                        r_min=r_min,
                        r_max=r_max,
                        afp_window=afp_window,
                        gaussian_fwhm_R=gaussian_fwhm_R,
                        lorentzian_fwhm_R=lorentzian_fwhm_R,
                    )
                if not blocked:
                    burn_center = candidate
                    break
            if burn_center is None:
                burn_center = _deepest_q_center(q_eq, ssrf_region)
        if burn_center is None:
            return {"polarization": p0, "skipped": True, "reason": "ssrf_selective_no_center"}
        ssrf_center = int(burn_center)
        n_ssrf_frames = int(max_burn) + 1

    ssrf_forbidden = set()
    if use_profile_ssrf and ssrf_rates is not None and np.asarray(ssrf_rates).size:
        ssrf_forbidden = _forbidden_from_profile_rates(
            ssrf_rates,
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
        )
    elif ssrf_center is not None and int(ssrf_center) >= 0:
        ssrf_forbidden = ssrf_influence_bins(
            ssrf_center,
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
        )

    model = build_equilibrium_spin1_model(
        polarization=p0,
        num_bins=num_bins,
        dt=dt,
        rf_enabled=False,
        relax_enabled=True,
        diffusion_scale=DIFFUSION_SCALE,
        r_min=r_min,
        r_max=r_max,
        legacy_spectral_recovery=legacy_spectral_recovery,
    )
    (p_initial, q_initial) = level_pq(model)

    if use_profile_afp:
        afp_out = _apply_afp_profile_phase(
            model,
            afp_region,
            num_bins=num_bins,
            forbidden_bins=ssrf_forbidden,
        )
        if afp_out is None:
            return {"polarization": p0, "skipped": True, "reason": "afp_profile_failed"}
        (afp_profile, afp_subset) = afp_out
        afp_center = COMBO_CENTER_BIN
    else:
        afp_centers = planned_afp_centers
        if afp_centers is None:
            if afp_center_bin is not None:
                afp_centers = [int(afp_center_bin)]
            else:
                (ip_tmp, im_tmp, _) = full_spectrum_intensities(model)
                q_post = np.asarray(ip_tmp) - np.asarray(im_tmp)
                afp_centers = []
                occupied = set(ssrf_forbidden)
                for a in _ordered_region_centers(afp_region, q_post):
                    infl = afp_influence_bins(a, num_bins=num_bins, afp_window=afp_window)
                    if infl & occupied:
                        continue
                    afp_centers.append(a)
                    occupied |= infl
                    if max_centers_per_region is not None and len(afp_centers) >= int(max_centers_per_region):
                        break
        afp_out = _apply_afp_selective_phase(
            model,
            afp_region,
            afp_window=afp_window,
            num_bins=num_bins,
            max_centers=max_centers_per_region,
            center_bins=afp_centers,
            forbidden_bins=ssrf_forbidden,
        )
        if afp_out is None:
            return {"polarization": p0, "skipped": True, "reason": "afp_selective_failed"}
        (afp_profile, afp_center) = afp_out
        afp_subset = []

    afp_frame = 0
    ssrf_start = 1
    n_relax = int(max_relax)
    relax_start = ssrf_start + int(n_ssrf_frames)
    total_frames = relax_start + n_relax
    iplus_full = iminus_full = None
    p_full = q_full = None
    power_profiles = None
    if capture_spectrum:
        iplus_full = np.empty((total_frames, num_bins), dtype=np.float64)
        iminus_full = np.empty((total_frames, num_bins), dtype=np.float64)
        p_full = np.empty(total_frames, dtype=np.float64)
        q_full = np.empty(total_frames, dtype=np.float64)
        power_profiles = np.zeros((total_frames, num_bins), dtype=np.float64)
        _record_spectrum_frame(model, afp_frame, iplus_full, iminus_full, p_full, q_full)
        power_profiles[afp_frame] = np.asarray(afp_profile, dtype=np.float64)

    if use_profile_ssrf:
        (ip_now, im_now, _) = full_spectrum_intensities(model)
        ssrf_prof = _align_rates_to_grid(
            ssrf_rates if ssrf_rates is not None else np.zeros(num_bins),
            ssrf_freq if ssrf_freq is not None else np.linspace(r_min, r_max, num_bins),
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
        )
        ssrf_prof = _mask_ssrf_off_afp(ssrf_prof, afp_profile)
        (ip_s, im_s, p_s, q_s) = _replay_profile_on_state(
            ip_now,
            im_now,
            ssrf_prof,
            ssrf_duration,
            polarization=p0,
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            dt=dt,
            n_record=max(0, int(n_ssrf_frames) - 1),
        )
        if np.any(ssrf_prof > 0.0):
            applied_peak = float(np.max(ssrf_prof))
        if capture_spectrum:
            _write_ssrf_frames(
                iplus_full, iminus_full, p_full, q_full, power_profiles,
                ssrf_start, n_ssrf_frames, ip_s, im_s, p_s, q_s, ssrf_prof,
            )
        model.load_from_physical_intensities(np.asarray(ip_s[-1]), np.asarray(im_s[-1]))
    else:
        (ip_s, im_s, p_s, q_s, ssrf_prof) = _burn_ssrf_on_model(
            model,
            ssrf_center,
            gamma_rf,
            int(max_burn),
            num_bins=num_bins,
            r_min=r_min,
            r_max=r_max,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
            legacy_spectral_recovery=legacy_spectral_recovery,
            afp_profile=afp_profile,
        )
        if capture_spectrum:
            _write_ssrf_frames(
                iplus_full, iminus_full, p_full, q_full, power_profiles,
                ssrf_start, n_ssrf_frames, ip_s, im_s, p_s, q_s, ssrf_prof,
            )

    _relax_and_capture(
        model,
        n_relax=n_relax,
        capture_spectrum=capture_spectrum,
        relax_start=relax_start,
        iplus_full=iplus_full,
        iminus_full=iminus_full,
        p_full=p_full,
        q_full=q_full,
    )

    (p_final, q_final) = level_pq(model)
    return {
        "polarization": p0,
        "skipped": False,
        "scenario": int(scenario),
        "layout": int(layout),
        "iplus_full": iplus_full,
        "iminus_full": iminus_full,
        "p_full": p_full,
        "q_full": q_full,
        "power_profiles": power_profiles,
        "n_ssrf_frames": n_ssrf_frames,
        "afp_frame": afp_frame,
        "ssrf_start": ssrf_start,
        "relax_start": relax_start,
        "p_initial": p_initial,
        "q_initial": q_initial,
        "p_final": p_final,
        "q_final": q_final,
        "applied_peak": applied_peak,
        "ssrf_center": ssrf_center,
        "afp_center": afp_center,
        "afp_subset": afp_subset,
    }
