"""ssRF-beta kinetic defaults shared by data generation and optimal-profile design.

These match the Boltzmann Pake equilibrium and diffusion couplings used in
``Data_Creation/spectrum_gen/bridge.py``. Profile optimization uses the same
population ODE with spectral recovery disabled (``legacy_spectral_recovery=False``).
"""
from __future__ import annotations

from dataclasses import asdict

from .ideal_model import IdealBinParams
from .model import Spin1Params

# Data-generation / ssRF-beta standard values (see Data_Creation/spectrum_gen/constants.py).
SSRF_BETA_DIFFUSION_SCALE = 5.0
SSRF_BETA_ZQ_WIDTH_R = 0.05
SSRF_BETA_CROSS_BRANCH_RATIO = 1.0
SSRF_BETA_DOUBLE_QUANTUM_RATIO = 0.10
SSRF_BETA_KERNEL_CUTOFF_WIDTHS = 0.0
SSRF_BETA_LINE_GAMMA = 0.05
SSRF_BETA_LINE_ASYM = 0.04
SSRF_BETA_LINE_MODEL = "single_pake"
SSRF_BETA_LINE_CD_SCALE_R = 1.0
SSRF_BETA_LINE_OD_SCALE_RATIO = 1.6
SSRF_BETA_LINE_ETA_CD = 0.03
SSRF_BETA_LINE_ETA_OD = 0.12
SSRF_BETA_LINE_OD_FRACTION = 0.07
SSRF_BETA_LINE_PHI_POINTS = 64
SSRF_BETA_PLOT_DIVISOR = 10.0
SSRF_BETA_CALIBRATION_P = 0.50
SSRF_BETA_D_SAME_PLUS0 = 0.18
SSRF_BETA_D_SAME_0MINUS = 0.10
SSRF_BETA_D_SPEC_PLUS0 = 2.0
SSRF_BETA_D_SPEC_0MINUS = 1.0
SSRF_BETA_DEFAULT_DT = 0.0015
# demo_single_pake / ssRF-beta tensor designer (spectral RF on ideal-bin programs)
SSRF_BETA_RF_RESPONSE_MODE = "spectral"
SSRF_BETA_RF_SPECTRAL_STRENGTH = 0.04
SSRF_BETA_RF_HWHM_R = 0.015
SSRF_BETA_RF_PEAK_SELECTIVITY = 0.7
SSRF_BETA_RF_PEAK_CENTER_R = 0.96
SSRF_BETA_RF_PEAK_HALF_WIDTH_R = 0.16
SSRF_BETA_RF_TRANSITION_WIDTH_R = 0.25
SSRF_BETA_PROFILE_SMOOTHING_FWHM_R = 0.03


def ssrf_beta_spin1_params(
    *,
    polarization: float,
    n_bins: int,
    dt: float = SSRF_BETA_DEFAULT_DT,
    rf_enabled: bool = False,
    relax_enabled: bool = True,
    legacy_spectral_recovery: bool = True,
    diffusion_scale: float = SSRF_BETA_DIFFUSION_SCALE,
    r_min: float | None = None,
    r_max: float | None = None,
    rf_gaussian_fwhm_R: float = 0.03,
    rf_lorentzian_fwhm_R: float = 0.015,
    r_min_full: float = -6.0,
    r_max_full: float = 6.0,
) -> Spin1Params:
    """Spin-1 model parameters aligned with ssRF-beta data generation."""
    p = float(polarization)
    near_zero_p = abs(p) < 1e-12
    use_legacy = bool(legacy_spectral_recovery and relax_enabled)
    return Spin1Params(
        n_bins=int(n_bins),
        r_min=float(r_min if r_min is not None else r_min_full),
        r_max=float(r_max if r_max is not None else r_max_full),
        p0=p if not near_zero_p else 0.0,
        q0=None,
        p_dnp_sat=p if not near_zero_p else 0.0,
        dnp_enabled=False,
        rf_enabled=bool(rf_enabled),
        relax_enabled=use_legacy,
        afp_enabled=False,
        gamma_rf=0.0,
        dt=float(dt),
        capacity_rate_power=1.0,
        plot_signal_units=not near_zero_p,
        display_scale=1.0,
        use_physical_voigt_rf=False,
        rf_gaussian_fwhm_R=float(rf_gaussian_fwhm_R),
        rf_lorentzian_fwhm_R=float(rf_lorentzian_fwhm_R),
        rf_profile_normalization="center_bin",
        diffusion_enabled=True,
        diffusion_scale=float(diffusion_scale),
        diffusion_overlap="lorentzian",
        zq_width_R=SSRF_BETA_ZQ_WIDTH_R,
        cross_branch_ratio=SSRF_BETA_CROSS_BRANCH_RATIO,
        double_quantum_ratio=SSRF_BETA_DOUBLE_QUANTUM_RATIO,
        kernel_cutoff_widths=SSRF_BETA_KERNEL_CUTOFF_WIDTHS,
        line_gamma=SSRF_BETA_LINE_GAMMA,
        line_asym=SSRF_BETA_LINE_ASYM,
        line_model=SSRF_BETA_LINE_MODEL,
        line_cd_scale_R=SSRF_BETA_LINE_CD_SCALE_R,
        line_od_scale_ratio=SSRF_BETA_LINE_OD_SCALE_RATIO,
        line_eta_cd=SSRF_BETA_LINE_ETA_CD,
        line_eta_od=SSRF_BETA_LINE_ETA_OD,
        line_od_fraction=SSRF_BETA_LINE_OD_FRACTION,
        line_phi_points=SSRF_BETA_LINE_PHI_POINTS,
        plot_divisor=SSRF_BETA_PLOT_DIVISOR,
        calibration_p=SSRF_BETA_CALIBRATION_P,
        d_same_plus0=SSRF_BETA_D_SAME_PLUS0 if use_legacy else 0.0,
        d_same_0minus=SSRF_BETA_D_SAME_0MINUS if use_legacy else 0.0,
        d_spec_plus0=SSRF_BETA_D_SPEC_PLUS0 if use_legacy else 0.0,
        d_spec_0minus=SSRF_BETA_D_SPEC_0MINUS if use_legacy else 0.0,
    )


def ssrf_beta_ideal_bin_params(
    *,
    polarization: float,
    n_bins: int,
    r_min: float,
    r_max: float,
    dt: float = SSRF_BETA_DEFAULT_DT,
    legacy_spectral_recovery: bool = False,
    diffusion_scale: float = SSRF_BETA_DIFFUSION_SCALE,
) -> IdealBinParams:
    """Ideal-bin profile parameters (Voigt widths zero; same ODE as ssRF-beta burns)."""
    base = ssrf_beta_spin1_params(
        polarization=polarization,
        n_bins=n_bins,
        dt=dt,
        rf_enabled=False,
        relax_enabled=False,
        legacy_spectral_recovery=legacy_spectral_recovery,
        diffusion_scale=diffusion_scale,
        r_min=r_min,
        r_max=r_max,
    )
    fields = asdict(base)
    fields["rf_gaussian_fwhm_R"] = 0.0
    fields["rf_lorentzian_fwhm_R"] = 0.0
    fields["rf_response_mode"] = SSRF_BETA_RF_RESPONSE_MODE
    fields["rf_spectral_strength"] = SSRF_BETA_RF_SPECTRAL_STRENGTH
    fields["rf_homogeneous_hwhm_R"] = SSRF_BETA_RF_HWHM_R
    fields["rf_peak_selectivity"] = SSRF_BETA_RF_PEAK_SELECTIVITY
    fields["rf_peak_center_R"] = SSRF_BETA_RF_PEAK_CENTER_R
    fields["rf_peak_half_width_R"] = SSRF_BETA_RF_PEAK_HALF_WIDTH_R
    fields["rf_transition_width_R"] = SSRF_BETA_RF_TRANSITION_WIDTH_R
    return IdealBinParams(**fields)