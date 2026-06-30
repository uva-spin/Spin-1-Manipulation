"""
Headless spin-1 ss-RF model (v11).

This module is a convenience entry point for the full packet-population model
in ``physics/ssrf_realtime/``, matching
``physics/simulations/spin1_ssrf_realtime`` but without any GUI.

State, conversions, RF, spin-diffusion redistribution, DNP, and T1 all follow
the realtime simulator conventions:

"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from . import Spin1Model, Spin1Params
from .model import PLUS, ZERO, MINUS

def _value_crosses_zero(before: float, after: float) -> bool:
    """True when a quantity moves to the opposite side of zero (or hits zero from one side)."""
    if before > 0:
        return after <= 0
    if before < 0:
        return after >= 0
    return after != 0.0


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
) -> Spin1Model:
    """
    Build a :class:`Spin1Model` whose grid matches ``Iplus`` / ``Iminus`` length.

    The model state is loaded from the supplied intensities using the same
    packet conversion as the realtime simulator.
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
    rf_only: bool = True,
    full_dynamics: bool = False,
):
    """
    Advance one integration step and return updated intensities and populations.

    Parameters
    ----------
    Iplus, Iminus
        Physical R-grid intensities (same length as the model grid).
    dt
        Integration timestep.
    gamma_rf
        Common ideal-bin RF equalization rate.
    burn_idx
        Index of physical +R on the symmetric grid.
    params
        Optional base parameters.  ``n_bins`` is set from the input length.
    rf_only
        When True (default), apply only the RF burn term for this step.
    full_dynamics
        When True, run the full derivative (RF + diffusion + DNP + T1).  Overrides
        ``rf_only``.

    Returns
    -------
    Iplus_new, Iminus_new, rho_plus, rho_zero, rho_minus
        Updated intensities and per-packet level populations (the latter are the
        three columns of the packet state ``n``).
    """
    model = build_model_for_intensities(Iplus, Iminus, params=params)
    model.params.gamma_rf = float(gamma_rf)
    model.params.rf_burn_R = float(model.Rplus[int(burn_idx)])

    use_rf_only = not full_dynamics and rf_only
    step_dt = float(dt)
    dt_scale = 1.0
    p = model.params
    saved_rates = None
    if use_rf_only:
        saved_rates = (
            p.d_same_plus0,
            p.d_same_0minus,
            p.d_spec_plus0,
            p.d_spec_0minus,
            p.dnp_rate,
            p.t1_rate,
        )
        p.d_same_plus0 = 0.0
        p.d_same_0minus = 0.0
        p.d_spec_plus0 = 0.0
        p.d_spec_0minus = 0.0
        p.dnp_rate = 0.0
        p.t1_rate = 0.0
    for _ in range(model.params.steps):
        model.load_from_physical_intensities(Iplus, Iminus)
        model.step_once(
            dt=step_dt,
            rf_on=True,
            dnp_on=False if use_rf_only else model.params.dnp_enabled,
        )
        Iplus_new, Iminus_new, _ = model.physical_intensities()
        if burn_preserves_ps_sign(Iplus, Iminus, Iplus_new, Iminus_new, burn_idx):
            break
        dt_scale *= 0.5
    if saved_rates is not None:
        (
            p.d_same_plus0,
            p.d_same_0minus,
            p.d_spec_plus0,
            p.d_spec_0minus,
            p.dnp_rate,
            p.t1_rate,
        ) = saved_rates
    # else:
    #     model.load_from_physical_intensities(Iplus, Iminus)
    #     Iplus_new = np.asarray(Iplus, dtype=float).copy()
    #     Iminus_new = np.asarray(Iminus, dtype=float).copy()
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
):
    """
    Check RF response ratios at the burn and mirror bins.

    Uses a small ``dt`` relative to ``gamma_rf`` when possible so the linearized
    2:1 burn/mirror relations hold.
    """
    Iplus_new, Iminus_new, _, _, _ = solve_rate_equations(
        Iplus, Iminus, dt, gamma_rf, burn_idx, rf_only=True
    )
    return verify_burn_response(Iplus, Iminus, Iplus_new, Iminus_new, burn_idx, rtol=rtol)
