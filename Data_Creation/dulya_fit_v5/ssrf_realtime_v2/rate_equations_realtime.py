"""
Headless spin-1 ss-RF model (v2).

Convenience entry point for the capacity-weighted packet-population model in
``physics/ssrf_realtime_v2/``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np

from . import Spin1Model, Spin1Params
from .rf_profile import (
    SIGMA_BINS,
    VOIGT_GAMMA_BINS,
    freeze_rf_profile,
    make_voigt_rf_profile,
    ssrf_touched_bins,
    unfreeze_rf_profile,
)
from .voigt_physical import discrete_bins_to_physical_fwhm
from .model import PLUS, ZERO, MINUS


def voigt_burn_params(**overrides) -> Spin1Params:
    """Return :class:`Spin1Params` aligned with ``spin1_ssrf_realtime_voigt_burn`` defaults."""
    base = Spin1Params(
        n_bins=701,
        r_min=-3.0,
        r_max=3.0,
        line_gamma=0.05,
        line_asym=0.04,
        plot_signal_units=True,
        plot_divisor=10.0,
        display_scale=1.0,
        calibration_p=0.50,
        p0=0.45,
        q0=None,
        rf_burn_R=-0.94,
        rf_enabled=True,
        gamma_rf=2.0,
        use_physical_voigt_rf=True,
        rf_gaussian_fwhm_R=0.030,
        rf_lorentzian_fwhm_R=0.015,
        rf_profile_normalization="center_bin",
        rf_profile_quadrature_order=0,
        diffusion_scale=5.0,
        zq_width_R=0.05,
        cross_branch_ratio=0.0,
        orientation_corr_fraction=0.0,
        orientation_corr_width_deg=20.0,
        kernel_cutoff_widths=4.0,
        microwave_diffusion_factor=1.0,
        capacity_rate_power=1.0,
        capacity_rate_clip=12.0,
        dnp_enabled=False,
        p_dnp_sat=0.58,
        dnp_rate=0.05,
        t1_rate=0.0,
        t1_p_eq=0.0,
        dt=0.0015,
        noise_sigma=0.0,
        relax_enabled=False,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
    )
    if overrides:
        base = replace(base, **overrides)
    return base


def create_voigt_burn_model(**overrides) -> Spin1Model:
    """Build a model with voigt_burn defaults and analytic Boltzmann initialization."""
    model = Spin1Model(voigt_burn_params(**overrides))
    model._active_idx = None
    model._rf_profile_frozen = False
    model.params.ssrf_subset_indices = None
    model.invalidate_rf_profile()
    return model


def configure_voigt_burn_spectral_recovery(model: Spin1Model) -> None:
    """Enable v15-style spectral-neighbor recovery alongside spin diffusion.

    Used for both physical-Voigt and single-bin demo burns so legacy ``d_spec`` /
    ``d_same`` match; spin diffusion comes from ``voigt_burn_params`` (``diffusion_scale``).
    """
    model.params.relax_enabled = True
    model.params.d_same_plus0 = 0.18
    model.params.d_same_0minus = 0.10
    model.params.d_spec_plus0 = 2.0
    model.params.d_spec_0minus = 1.0
    model._active_idx = None


def voigt_burn_recovery_param_snapshot(model: Spin1Model) -> dict[str, float | bool]:
    """Return relaxation / diffusion knobs for parity checks between burn modes."""
    p = model.params
    return {
        "relax_enabled": bool(p.relax_enabled),
        "d_same_plus0": float(p.d_same_plus0),
        "d_same_0minus": float(p.d_same_0minus),
        "d_spec_plus0": float(p.d_spec_plus0),
        "d_spec_0minus": float(p.d_spec_0minus),
        "diffusion_scale": float(p.diffusion_scale),
        "t2_width_R": float(p.t2_width_R),
        "active_idx_is_full_spectrum": model._active_idx is None,
    }


def configure_single_bin_ssrf(
    model: Spin1Model,
    burn_idx: int,
    gamma_rf: float,
    *,
    apply_demo_recovery: bool = True,
) -> int:
    """Install v15-style single-bin RF at ``burn_idx`` (full-spectrum recovery)."""
    if apply_demo_recovery:
        configure_voigt_burn_spectral_recovery(model)
    burn_idx = int(burn_idx)
    n_bins = len(model.Rplus)
    if burn_idx < 0 or burn_idx >= n_bins:
        raise ValueError(f"burn_idx={burn_idx} out of range for n_bins={n_bins}")

    profile = np.zeros(n_bins, dtype=float)
    profile[burn_idx] = float(gamma_rf)
    freeze_rf_profile(model, profile)
    model.params.gamma_rf = float(gamma_rf)
    model.params.ssrf_subset_indices = None
    model.params.rf_burn_R = float(model.Rplus[burn_idx])
    model.params.rf_enabled = True
    model._active_idx = None
    model._invalidate_branch_cache()
    return burn_idx


def configure_discrete_bins_ssrf(
    model: Spin1Model,
    burn_indices: list[int] | tuple[int, ...],
    gamma_rf: float | list[float] | tuple[float, ...],
    *,
    capacity_weighting: str = "separate_branch",
) -> list[int]:
    """Install flat single-bin-style RF on each index in ``burn_indices``.

    ``gamma_rf`` may be a scalar (same rate on every listed bin) or a sequence
    aligned with ``burn_indices``. Uses ``ssrf_subset_indices`` so every listed
    bin is burned; capacity defaults to ``separate_branch`` so each pair uses
    its own ``w[kp]`` / ``w[km]`` like true single-bin burns.
    """
    n_bins = len(model.Rplus)
    indices = [int(i) for i in burn_indices]
    if not indices:
        raise ValueError("burn_indices must be non-empty")
    if isinstance(gamma_rf, (list, tuple, np.ndarray)):
        rates_in = [float(g) for g in gamma_rf]
        if len(rates_in) != len(indices):
            raise ValueError(
                f"gamma_rf length {len(rates_in)} != burn_indices length {len(indices)}"
            )
        rates = {idx: rate for idx, rate in zip(indices, rates_in)}
    else:
        rates = {idx: float(gamma_rf) for idx in indices}

    support = sorted(rates.keys())
    for burn_idx in support:
        if burn_idx < 0 or burn_idx >= n_bins:
            raise ValueError(f"burn_idx={burn_idx} out of range for n_bins={n_bins}")

    profile = np.zeros(n_bins, dtype=float)
    for burn_idx in support:
        profile[burn_idx] = rates[burn_idx]
    freeze_rf_profile(model, profile)
    model.params.gamma_rf = float(max(abs(rates[i]) for i in support))
    model.params.ssrf_subset_indices = list(support)
    model.params.ssrf_multi_bin_capacity = str(capacity_weighting)
    model.params.rf_burn_R = float(model.Rplus[support[0]])
    model.params.rf_enabled = True
    model._active_idx = None
    model._invalidate_branch_cache()
    return support


def configure_physical_voigt_ssrf(
    model: Spin1Model,
    burn_idx: int,
    gamma_rf: float,
    *,
    gaussian_fwhm_R: float | None = None,
    lorentzian_fwhm_R: float | None = None,
    sigma_bins: float = SIGMA_BINS,
    voigt_gamma_bins: float = VOIGT_GAMMA_BINS,
    normalization: str = "center_bin",
    quadrature_order: int = 0,
    full_spectrum_recovery: bool = False,
) -> None:
    """Install bin-averaged physical-R Voigt RF (voigt_burn physics path)."""
    burn_idx = int(burn_idx)
    n_bins = len(model.Rplus)
    if burn_idx < 0 or burn_idx >= n_bins:
        raise ValueError(f"burn_idx={burn_idx} out of range for n_bins={n_bins}")

    unfreeze_rf_profile(model)
    if gaussian_fwhm_R is None or lorentzian_fwhm_R is None:
        mapped_g, mapped_l = discrete_bins_to_physical_fwhm(
            sigma_bins, voigt_gamma_bins, model.dR
        )
        if gaussian_fwhm_R is None:
            gaussian_fwhm_R = mapped_g
        if lorentzian_fwhm_R is None:
            lorentzian_fwhm_R = mapped_l

    model.params.use_physical_voigt_rf = True
    model.params.relax_enabled = False
    model.params.rf_gaussian_fwhm_R = float(gaussian_fwhm_R)
    model.params.rf_lorentzian_fwhm_R = float(lorentzian_fwhm_R)
    model.params.rf_profile_normalization = str(normalization)
    model.params.rf_profile_quadrature_order = int(quadrature_order)
    model.params.gamma_rf = float(gamma_rf)
    model.params.rf_burn_R = float(model.Rplus[burn_idx])
    model.params.rf_enabled = True
    model.invalidate_rf_profile()
    if full_spectrum_recovery:
        model._active_idx = None
    else:
        kp, km = model.branch_indices(model.params.rf_burn_R)
        touched = sorted({i for i in (kp, km) if i is not None})
        model._active_idx = np.asarray(touched, dtype=int) if touched else None
    model._invalidate_branch_cache()


def configure_voigt_ssrf(
    model: Spin1Model,
    burn_idx: int,
    gamma_rf: float,
    *,
    sigma: float = SIGMA_BINS,
    voigt_gamma: float = VOIGT_GAMMA_BINS,
    half_width: int | None = None,
    full_spectrum_recovery: bool = False,
    capacity_weighting: str = "center_shared",
) -> List[int]:
    """Install a rounded Voigt multi-bin RF profile centered on ``burn_idx``."""
    burn_idx = int(burn_idx)
    profile, support = make_voigt_rf_profile(
        len(model.Rplus),
        burn_idx,
        float(gamma_rf),
        sigma=float(sigma),
        lorentz_gamma=float(voigt_gamma),
        half_width=half_width,
    )
    freeze_rf_profile(model, profile)
    model.params.gamma_rf = float(gamma_rf)
    model.params.ssrf_subset_indices = [int(i) for i in support]
    model.params.ssrf_multi_bin_capacity = str(capacity_weighting)
    model.params.rf_burn_R = float(model.Rplus[burn_idx])
    model.params.rf_enabled = True
    if full_spectrum_recovery:
        model._active_idx = None
    else:
        touched = ssrf_touched_bins(len(model.Rplus), support)
        model._active_idx = np.asarray(touched, dtype=int) if touched else None
    return support


def _value_crosses_zero(before: float, after: float) -> bool:
    """True when a quantity moves to the opposite side of zero (or hits zero from one side)."""
    if before > 0:
        return after <= 0
    if before < 0:
        return after >= 0
    return after != 0.0


def burn_preserves_branch_order(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_new: np.ndarray,
    iminus_new: np.ndarray,
    burn_idx: int,
) -> bool:
    """True when I- stays above I+ (or below) at the burn bin as in the initial state."""
    burn_idx = int(burn_idx)
    ip0 = float(iplus[burn_idx])
    im0 = float(iminus[burn_idx])
    ip1 = float(iplus_new[burn_idx])
    im1 = float(iminus_new[burn_idx])
    if im0 > ip0:
        return im1 >= ip1
    if im0 < ip0:
        return im1 <= ip1
    return True


def burn_preserves_ps_sign(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_new: np.ndarray,
    iminus_new: np.ndarray,
    burn_idx: int,
) -> bool:
    """True when I+, I-, and Ps at burn and mirror bins stay on their original side of zero."""
    n = len(iplus)
    burn_idx = int(burn_idx)
    mirror_idx = n - 1 - burn_idx
    for idx in (burn_idx, mirror_idx):
        for before, after in (
            (float(iplus[idx]), float(iplus_new[idx])),
            (float(iminus[idx]), float(iminus_new[idx])),
            (float(iplus[idx] + iminus[idx]), float(iplus_new[idx] + iminus_new[idx])),
        ):
            if _value_crosses_zero(before, after):
                return False
    return True


def build_model_for_intensities(
    Iplus: np.ndarray,
    Iminus: np.ndarray,
    *,
    params: Optional[Spin1Params] = None,
    rf_burn_R: Optional[float] = None,
    p0: Optional[float] = None,
) -> Spin1Model:
    """
    Build a :class:`Spin1Model` whose grid matches ``Iplus`` / ``Iminus`` length.

    The model state is loaded from the supplied intensities.  Set ``p0`` to the
    vector polarization used to generate the lineshape (e.g. ``GenerateVectorLineshape``);
    it sets ``display_cal`` via :attr:`Spin1Params.p0`.
    """
    Iplus = np.asarray(Iplus, dtype=float)
    Iminus = np.asarray(Iminus, dtype=float)
    n_bins = len(Iplus)
    if len(Iminus) != n_bins:
        raise ValueError("Iplus and Iminus must have the same length")

    base = params or Spin1Params()
    p = replace(base, n_bins=n_bins)
    if rf_burn_R is not None:
        p = replace(p, rf_burn_R=float(rf_burn_R))
    if p0 is not None:
        p = replace(p, p0=float(p0))

    model = Spin1Model(p)
    model.load_from_physical_intensities(Iplus, Iminus)
    return model


def solve_rate_equations(
    Iplus,
    Iminus,
    dt: float,
    gamma_rf: float,
    burn_idx: int,
    *,
    params: Optional[Spin1Params] = None,
    p0: Optional[float] = None,
    rf_only: bool = True,
    full_dynamics: bool = False,
):
    step_params = params or Spin1Params()
    if rf_only:
        step_params = replace(
            step_params,
            d_same_plus0=0.0,
            d_same_0minus=0.0,
            d_spec_plus0=0.0,
            d_spec_0minus=0.0,
        )
    model = build_model_for_intensities(
        Iplus, Iminus, params=step_params, p0=p0
    )
    configure_voigt_ssrf(model, int(burn_idx), float(gamma_rf))

    Iplus_cur = np.asarray(Iplus, dtype=float).copy()
    Iminus_cur = np.asarray(Iminus, dtype=float).copy()
    Iplus_new = Iplus_cur.copy()
    Iminus_new = Iminus_cur.copy()

    for _ in range(model.params.steps):
        state_before = model.n.copy()
        model.step_once(
            dt=dt,
            rf_on=True,
            dnp_on=False if rf_only else model.params.dnp_enabled,
        )
        Iplus_new, Iminus_new, _ = model.physical_intensities()
        if not burn_preserves_ps_sign(Iplus_cur, Iminus_cur, Iplus_new, Iminus_new, burn_idx):
            model.n = state_before
            Iplus_new, Iminus_new = Iplus_cur, Iminus_cur
            break
        if not burn_preserves_branch_order(Iplus, Iminus, Iplus_new, Iminus_new, burn_idx):
            model.n = state_before
            Iplus_new, Iminus_new = Iplus_cur, Iminus_cur
            break
        Iplus_cur = np.asarray(Iplus_new, dtype=float).copy()
        Iminus_cur = np.asarray(Iminus_new, dtype=float).copy()
    rho_plus = model.n[:, PLUS].copy()
    rho_zero = model.n[:, ZERO].copy()
    rho_minus = model.n[:, MINUS].copy()
    return Iplus_new, Iminus_new, rho_plus, rho_zero, rho_minus


def verify_burn_response(
    Iplus,
    Iminus,
    Iplus_new,
    Iminus_new,
    burn_idx: int,
    rtol: float = 1e-6,
):
    """
    Check RF response ratios at the burn and mirror bins for given before/after intensities.

    Expected (magnitudes of changes):
        Amp_burn  = 2 * Amp_mirror
        dIplus_burn  = 2 * dIminus_mirror
        dIminus_burn = 2 * dIplus_mirror
    """
    burn_idx = int(burn_idx)
    mirror_idx = len(Iplus) - 1 - burn_idx

    d_ip_burn = Iplus_new[burn_idx] - Iplus[burn_idx]
    d_im_burn = Iminus_new[burn_idx] - Iminus[burn_idx]
    d_ip_mirror = Iplus_new[mirror_idx] - Iplus[mirror_idx]
    d_im_mirror = Iminus_new[mirror_idx] - Iminus[mirror_idx]

    amp_burn = (Iplus_new[burn_idx] + Iminus_new[burn_idx]) - (Iplus[burn_idx] + Iminus[burn_idx])
    amp_mirror = (Iplus_new[mirror_idx] + Iminus_new[mirror_idx]) - (
        Iplus[mirror_idx] + Iminus[mirror_idx]
    )

    checks = {
        "amp_burn_over_amp_mirror": abs(amp_burn) / abs(amp_mirror),
        "iplus_burn_over_iminus_mirror": abs(d_ip_burn) / abs(d_im_mirror),
        "iminus_burn_over_iplus_mirror": abs(d_im_burn) / abs(d_ip_mirror),
    }

    expected = 2.0
    ps_burn_before = Iplus[burn_idx] + Iminus[burn_idx]
    ps_burn_after = Iplus_new[burn_idx] + Iminus_new[burn_idx]
    magnitude_decreased = abs(ps_burn_after) < abs(ps_burn_before)

    passed = magnitude_decreased and all(
        abs(ratio - expected) / expected < rtol for ratio in checks.values()
    )

    return {
        "passed": passed,
        "burn_idx": burn_idx,
        "mirror_idx": mirror_idx,
        "ps_burn_before": ps_burn_before,
        "ps_burn_after": ps_burn_after,
        "magnitude_decreased": magnitude_decreased,
        "amp_burn": amp_burn,
        "amp_mirror": amp_mirror,
        "d_iplus_burn": d_ip_burn,
        "d_iminus_burn": d_im_burn,
        "d_iplus_mirror": d_ip_mirror,
        "d_iminus_mirror": d_im_mirror,
        "ratios": checks,
    }


def verify_rates_response(
    Iplus,
    Iminus,
    burn_idx: int,
    gamma_rf: float,
    dt: float = 1.0,
    rtol: float = 1e-6,
    *,
    params: Optional[Spin1Params] = None,
    p0: Optional[float] = None,
):
    """
    Check RF response ratios at the burn and mirror bins.

    Uses a small ``dt`` relative to ``gamma_rf`` when possible so the linearized
    2:1 burn/mirror relations hold.
    """
    diagnostic_params = replace(params or Spin1Params(), steps=1)
    Iplus_new, Iminus_new, _, _, _ = solve_rate_equations(
        Iplus,
        Iminus,
        dt,
        gamma_rf,
        burn_idx,
        params=diagnostic_params,
        p0=p0,
        rf_only=True,
    )
    return verify_burn_response(Iplus, Iminus, Iplus_new, Iminus_new, burn_idx, rtol=rtol)
