"""Bridge Dulya equilibrium lineshapes to physics.rf."""
import numpy as np
from common import D_SAME_0MINUS, D_SAME_PLUS0, D_SPEC_0MINUS, D_SPEC_PLUS0, DIFFUSION_SCALE, F_MAX, F_MIN, MAX_GDT, MAX_NSUB, MIRROR_AMP_EPS, MIRROR_AMP_RTOL, RF_GAUSSIAN_FWHM_R, RF_LORENTZIAN_FWHM_R, RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN, ZQ_WIDTH_R
from physics.rf import Spin1Model, Spin1Params, bin_averaged_voigt, configure_physical_voigt_ssrf, configure_single_bin_ssrf, configure_voigt_burn_spectral_recovery, mirror_bin_idx, ssrf_touched_bins
afp_touched_bins = ssrf_touched_bins
PROFILE_REL_THRESHOLD = 0.01
KERNEL_CUTOFF_WIDTHS = 3.0
INTENSITY_DELTA_ABS_TOL = 1e-15

def bins_within_R_radius(R, seed_bins, radius_R):
    grid = np.asarray(R)
    radius = radius_R
    out = set()
    for i in seed_bins:
        ri = grid[i]
        for j in range(grid.size):
            if abs(grid[j] - ri) <= radius:
                out.add(j)
    return sorted(out)

def diffusion_spillover_bins(R, seed_bins, *, zq_width_R=ZQ_WIDTH_R, kernel_cutoff_widths=KERNEL_CUTOFF_WIDTHS):
    """Bins within the spin-diffusion kernel reach of ``seed_bins``."""
    return bins_within_R_radius(R, seed_bins, kernel_cutoff_widths * zq_width_R)

def bins_with_intensity_delta(iplus, iminus, iplus_sim, iminus_sim, candidates, *, abs_tol=INTENSITY_DELTA_ABS_TOL):
    changed = []
    for i in candidates:
        idx = i
        if abs(iplus_sim[idx] - iplus[idx]) > abs_tol or abs(iminus_sim[idx] - iminus[idx]) > abs_tol:
            changed.append(idx)
    return changed

def physical_voigt_rf_support_bins(R, burn_idx, *, gaussian_fwhm_R, lorentzian_fwhm_R, rel_threshold=PROFILE_REL_THRESHOLD):
    """Bins with non-negligible physical-R Voigt RF at the burn center."""
    grid = np.asarray(R)
    burn_idx = burn_idx
    dR = grid[1] - grid[0] if grid.size > 1 else 1.0
    profile = bin_averaged_voigt(grid, center_R=grid[burn_idx], bin_width_R=dR, gaussian_fwhm_R=gaussian_fwhm_R, lorentzian_fwhm_R=lorentzian_fwhm_R, normalization='center_bin')
    peak = np.max(profile) if profile.size else 0.0
    if peak <= 0.0:
        return [burn_idx]
    floor = rel_threshold * peak
    support = [i for i in np.flatnonzero(np.asarray(profile) >= floor)]
    return support if support else [burn_idx]

def burn_commit_touched_bins(n_bins, burn_idx, *, rf_mode, R=None, gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R, lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R, iplus=None, iminus=None, iplus_sim=None, iminus_sim=None, include_diffusion_spillover=True, zq_width_R=ZQ_WIDTH_R):
    """Bins whose intensities should be committed after a burn trial."""
    burn_idx = burn_idx
    if rf_mode == RF_MODE_SINGLE_BIN:
        return ssrf_touched_bins(n_bins, [burn_idx])
    support = physical_voigt_rf_support_bins(R, burn_idx, gaussian_fwhm_R=gaussian_fwhm_R, lorentzian_fwhm_R=lorentzian_fwhm_R)
    touched = set(ssrf_touched_bins(n_bins, support))
    if include_diffusion_spillover:
        spill_candidates = diffusion_spillover_bins(R, support, zq_width_R=zq_width_R)
        if iplus is not None and iminus is not None and (iplus_sim is not None) and (iminus_sim is not None):
            spill = bins_with_intensity_delta(iplus, iminus, iplus_sim, iminus_sim, spill_candidates)
        else:
            spill = spill_candidates
        for i in spill:
            touched.add(i)
            touched.add(mirror_bin_idx(n_bins, i))
    return sorted(touched)

def commit_touched_bins_only(iplus, iminus, iplus_sim, iminus_sim, touched):
    """Keep baseline intensities except on touched bins."""
    out_ip = np.asarray(iplus).copy()
    out_im = np.asarray(iminus).copy()
    ip_sim = np.asarray(iplus_sim)
    im_sim = np.asarray(iminus_sim)
    for k in touched:
        out_ip[k] = ip_sim[k]
        out_im[k] = im_sim[k]
    return (out_ip, out_im)

def restore_touched_intensity_area(iplus, iminus, touched, area_target):
    """Restore total area on touched bins via common-mode offset."""
    out_ip = np.asarray(iplus).copy()
    out_im = np.asarray(iminus).copy()
    if not touched:
        return (out_ip, out_im)
    n = len(out_ip)
    touched_set = set((k for k in touched))
    unt_idx = [i for i in range(n) if i not in touched_set]
    area_unt = np.sum(out_ip[unt_idx] + out_im[unt_idx]) if unt_idx else 0.0
    ps_touch = out_ip[list(touched)] + out_im[list(touched)]
    area_touch = np.sum(ps_touch)
    missing = area_target - area_unt - area_touch
    if abs(missing) < 1e-15:
        return (out_ip, out_im)
    weights = np.maximum(ps_touch, 0.0)
    wsum = np.sum(weights)
    if wsum > 1e-30:
        for (j, k) in enumerate(touched):
            add = missing * (weights[j] / wsum)
            out_ip[k] += 0.5 * add
            out_im[k] += 0.5 * add
    else:
        add = missing / len(touched)
        for k in touched:
            out_ip[k] += 0.5 * add
            out_im[k] += 0.5 * add
    return (out_ip, out_im)

def afp_window_indices(bin_idx, n_bins, window):
    w = max(1, window)
    half = w // 2
    c = bin_idx
    n = n_bins
    lo = c - half
    hi = c + half
    if lo < 0:
        hi = min(n - 1, hi - lo)
        lo = 0
    if hi >= n:
        lo = max(0, lo - (hi - (n - 1)))
        hi = n - 1
    return list(range(lo, hi + 1))

def build_spin1_model(iplus, iminus, *, polarization, num_bins, dt, rf_enabled=False, relax_enabled=True, diffusion_scale=DIFFUSION_SCALE, rf_gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R, rf_lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R, r_min=None, r_max=None):
    """Build a capacity-weighted Spin1 model loaded from physical intensities."""
    P = polarization
    near_zero_p = abs(P) < 1e-12
    params = Spin1Params(n_bins=num_bins, r_min=F_MIN if r_min is None else r_min, r_max=F_MAX if r_max is None else r_max, p0=P if not near_zero_p else 0.0, q0=None, p_dnp_sat=P if not near_zero_p else 0.0, dnp_enabled=False, rf_enabled=rf_enabled, relax_enabled=relax_enabled, afp_enabled=False, gamma_rf=0.0, dt=dt, capacity_rate_power=1.0, plot_signal_units=not near_zero_p, display_scale=1.0, use_physical_voigt_rf=False, rf_gaussian_fwhm_R=rf_gaussian_fwhm_R, rf_lorentzian_fwhm_R=rf_lorentzian_fwhm_R, rf_profile_normalization='center_bin', diffusion_scale=diffusion_scale, zq_width_R=ZQ_WIDTH_R, d_same_plus0=D_SAME_PLUS0 if relax_enabled else 0.0, d_same_0minus=D_SAME_0MINUS if relax_enabled else 0.0, d_spec_plus0=D_SPEC_PLUS0 if relax_enabled else 0.0, d_spec_0minus=D_SPEC_0MINUS if relax_enabled else 0.0)
    model = Spin1Model(params)
    model.load_from_physical_intensities(np.asarray(iplus), np.asarray(iminus))
    if relax_enabled:
        apply_shared_spectral_recovery(model)
    return model

def apply_shared_spectral_recovery(model):
    """Enable voigt_burn spectral recovery with shared D_SAME_*/D_SPEC_* rates."""
    configure_voigt_burn_spectral_recovery(model)
    model.params.d_same_plus0 = D_SAME_PLUS0
    model.params.d_same_0minus = D_SAME_0MINUS
    model.params.d_spec_plus0 = D_SPEC_PLUS0
    model.params.d_spec_0minus = D_SPEC_0MINUS

def configure_ssrf_burn(model, burn_idx, gamma_rf, *, rf_mode=RF_MODE_PHYSICAL_VOIGT, gaussian_fwhm_R=None, lorentzian_fwhm_R=None):
    """Install RF for one burn bin with shared spectral recovery."""
    mode = str(rf_mode)
    apply_shared_spectral_recovery(model)
    burn_idx = burn_idx
    gamma = gamma_rf
    if mode == RF_MODE_SINGLE_BIN:
        configure_single_bin_ssrf(model, burn_idx, gamma, apply_demo_recovery=False)
    else:
        configure_physical_voigt_ssrf(model, burn_idx, gamma, gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R if gaussian_fwhm_R is None else gaussian_fwhm_R, lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R if lorentzian_fwhm_R is None else lorentzian_fwhm_R, full_spectrum_recovery=True)
    apply_shared_spectral_recovery(model)
    model._sync_level_populations(capture_initial=False)
    p_init = model.n_plus - model.n_minus
    model.set_recovery_boltzmann_P(p_init)
    return mode

def configure_afp_recovery(model):
    """Post-AFP relaxation: Boltzmann at the manipulated (post-AFP) vector P.

    Recovery drives Q → Q_boltz(P_AFP). Uses the same ``D_SAME_*`` / ``D_SPEC_*``
    rates as ssRF, no spin diffusion, and uniform capacity weighting.
    """
    model.params.relax_enabled = True
    model.params.d_same_plus0 = D_SAME_PLUS0
    model.params.d_same_0minus = D_SAME_0MINUS
    model.params.d_spec_plus0 = D_SPEC_PLUS0
    model.params.d_spec_0minus = D_SPEC_0MINUS
    model.params.diffusion_scale = 0.0
    model.params.capacity_rate_power = 0.0
    model._active_idx = None
    model.install_boltzmann_recovery_at_current_P()

def level_pq(model):
    """Return current vector P and tensor Q from stored level populations."""
    lp = model.level_populations()
    return (lp['P'], lp['Q'])

def intensities_at_bins(model, bin_idx, mirror_idx):
    (ip, im, _) = model.physical_intensities()
    iplus = ip[bin_idx]
    iminus = im[bin_idx]
    iplus_m = ip[mirror_idx]
    iminus_m = im[mirror_idx]
    return (iplus, iminus, iplus + iminus, iplus_m, iminus_m, iplus_m + iminus_m)

def intensity_at_bin(model, bin_idx):
    """Return (I+, I-, P) at a single spectral bin."""
    (ip, im, _) = model.physical_intensities()
    i = bin_idx
    iplus = ip[i]
    iminus = im[i]
    return (iplus, iminus, iplus + iminus)

def full_spectrum_intensities(model):
    (ip, im, total) = model.physical_intensities()
    return (np.asarray(ip), np.asarray(im), np.asarray(total))

def mirror_amplitude(ps_m):
    return abs(ps_m)

def mirror_amplitude_decreased(ps_m, ps_m_prev, *, atol=MIRROR_AMP_EPS, rtol=MIRROR_AMP_RTOL):
    cur = mirror_amplitude(ps_m)
    prev = mirror_amplitude(ps_m_prev)
    return cur < prev - max(atol, rtol * prev)

def euler_n_sub(gamma_rf, dt):
    g = abs(gamma_rf)
    dt_f = dt
    if g <= 0.0 or dt_f <= 0.0:
        return (1, dt_f)
    n_sub = min(max(1, int(np.ceil(g * dt_f / MAX_GDT))), int(MAX_NSUB))
    return (n_sub, dt_f / n_sub)

def traj_to_fit_scale(traj, from_spin1):
    scale = from_spin1
    for key in ('ps', 'iplus', 'iminus', 'ps_m', 'iplus_m', 'iminus_m', 'ps_lo', 'iplus_lo', 'iminus_lo', 'ps_hi', 'iplus_hi', 'iminus_hi'):
        if key in traj and traj[key] is not None and np.asarray(traj[key]).size:
            traj[key] = np.asarray(traj[key]) * scale
    if 'ps0' in traj and traj['ps0'] is not None:
        traj['ps0'] = traj['ps0'] * scale
    for key in ('ip_spectrum0', 'im_spectrum0', 'ip_spectrum', 'im_spectrum'):
        if key in traj and traj[key] is not None:
            traj[key] = np.asarray(traj[key]) * scale
    for key in ('ps_full', 'iplus_full', 'iminus_full'):
        if key in traj and traj[key] is not None:
            arr = np.asarray(traj[key])
            if arr.size:
                traj[key] = arr * scale
    return traj
