"""Full-spectrum ssRF Voigt burn trajectories (no per-bin shard I/O)."""
from __future__ import annotations

import numpy as np

from .bridge import (
    build_equilibrium_spin1_model,
    configure_ssrf_burn,
    euler_n_sub,
    full_spectrum_intensities,
    level_pq,
)
from .constants import (
    DIFFUSION_SCALE,
    DT,
    F_MAX,
    F_MIN,
    NUM_BINS,
    PS_ABS_MIN,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE_PHYSICAL_VOIGT,
)


def run_one_polarization(
    bin_idx,
    polarization,
    *,
    num_bins=NUM_BINS,
    dt=DT,
    gamma_rf,
    n_steps,
    rf_mode=RF_MODE_PHYSICAL_VOIGT,
    gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R,
    diffusion_scale=DIFFUSION_SCALE,
    capture_spectrum=True,
    r_min=None,
    r_max=None,
    legacy_spectral_recovery=False,
):
    """Burn ``n_steps`` macro-steps; return full I+/I- frames when capturing."""
    r_lo = F_MIN if r_min is None else r_min
    r_hi = F_MAX if r_max is None else r_max
    model = build_equilibrium_spin1_model(
        polarization=polarization,
        num_bins=num_bins,
        dt=dt,
        rf_enabled=True,
        relax_enabled=True,
        diffusion_scale=diffusion_scale,
        rf_gaussian_fwhm_R=gaussian_fwhm_R,
        rf_lorentzian_fwhm_R=lorentzian_fwhm_R,
        r_min=r_lo,
        r_max=r_hi,
        legacy_spectral_recovery=legacy_spectral_recovery,
    )
    ip0, im0, _ = full_spectrum_intensities(model)
    if abs(float(ip0[bin_idx] + im0[bin_idx])) < PS_ABS_MIN:
        return {'skipped': True, 'iplus_full': None, 'iminus_full': None, 'p_full': None, 'q_full': None}
    configure_ssrf_burn(
        model,
        bin_idx,
        gamma_rf,
        rf_mode=rf_mode,
        gaussian_fwhm_R=gaussian_fwhm_R,
        lorentzian_fwhm_R=lorentzian_fwhm_R,
        legacy_spectral_recovery=legacy_spectral_recovery,
    )
    t_len = int(n_steps) + 1
    p_full = np.empty(t_len)
    q_full = np.empty(t_len)
    iplus_full = iminus_full = None
    if capture_spectrum:
        iplus_full = np.empty((t_len, num_bins))
        iminus_full = np.empty((t_len, num_bins))

    def _record(k):
        p_full[k], q_full[k] = level_pq(model)
        if capture_spectrum:
            ip, im, _ = full_spectrum_intensities(model)
            iplus_full[k] = ip
            iminus_full[k] = im

    _record(0)
    n_sub, dt_sub = euler_n_sub(gamma_rf, dt)
    for k in range(1, t_len):
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
        _record(k)
    return {
        'skipped': False,
        'iplus_full': iplus_full,
        'iminus_full': iminus_full,
        'p_full': p_full,
        'q_full': q_full,
        'p_initial': float(p_full[0]),
        'q_initial': float(q_full[0]),
        'p_final': float(p_full[-1]),
        'q_final': float(q_full[-1]),
        'center_bin': int(bin_idx),
    }
