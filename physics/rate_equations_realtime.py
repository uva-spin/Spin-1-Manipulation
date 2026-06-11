"""
Headless spin-1 ss-RF model (v11).

This module is a convenience entry point for the full packet-population model
in ``physics/ssrf_realtime/``, matching
``physics/simulations/spin1_ssrf_realtime`` but without any GUI.

State, conversions, RF, spin-diffusion redistribution, DNP, and T1 all follow
the realtime simulator conventions:

* packet populations ``n[k, level]`` with fixed grid weights ``mu[k]``
* intensities from transition differences and display calibration
* positivity-preserving Euler integration with per-packet renormalization

For multi-step simulations, use :class:`Spin1Model` directly.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from ssrf_realtime import Spin1Model, Spin1Params
from ssrf_realtime.model import PLUS, ZERO, MINUS

__all__ = [
    "Spin1Model",
    "Spin1Params",
    "PLUS",
    "ZERO",
    "MINUS",
    "build_model_for_intensities",
    "solve_rate_equations",
    "verify_rates_response",
    "burn_preserves_ps_sign",
]


def _ps_crosses_zero(ps_before: float, ps_after: float) -> bool:
    if ps_before > 0:
        return ps_after <= 0
    if ps_before < 0:
        return ps_after >= 0
    return ps_after != 0.0


def burn_preserves_ps_sign(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_new: np.ndarray,
    iminus_new: np.ndarray,
    burn_idx: int,
) -> bool:
    """True when Ps = I+ + I- at burn and mirror bins stays on its original side of zero."""
    n = len(iplus)
    burn_idx = int(burn_idx)
    mirror_idx = n - 1 - burn_idx
    for idx in (burn_idx, mirror_idx):
        ps_before = float(iplus[idx] + iminus[idx])
        ps_after = float(iplus_new[idx] + iminus_new[idx])
        if _ps_crosses_zero(ps_before, ps_after):
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
    model.step_once(dt=dt, rf_on=True, dnp_on=model.params.dnp_enabled, rf_only=use_rf_only)

    Iplus_new, Iminus_new, _ = model.physical_intensities()
    rho_plus = model.n[:, PLUS].copy()
    rho_zero = model.n[:, ZERO].copy()
    rho_minus = model.n[:, MINUS].copy()
    return Iplus_new, Iminus_new, rho_plus, rho_zero, rho_minus


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
    burn_idx = int(burn_idx)
    mirror_idx = len(Iplus) - 1 - burn_idx

    Iplus_new, Iminus_new, _, _, _ = solve_rate_equations(
        Iplus, Iminus, dt, gamma_rf, burn_idx, rf_only=True
    )

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from ssrf_realtime.lineshape import plot_signal_reference

    params = Spin1Params(
        n_bins=501,
        p0=0.45,
        rf_burn_R=-0.92,
        gamma_rf=10.0,
        dnp_enabled=False,
        dt=0.0015,
    )
    model = Spin1Model(params)
    R = model.Rplus
    _, Iplus, Iminus, _total = model.reference_spectrum()

    burn_idx = model.burn_index(params.rf_burn_R)

    print("Initial P:", model.polarizations()["P"])
    for _ in range(400):
        model.step(1, rf_on=True, dnp_on=False)
    print("After RF P:", model.polarizations()["P"])

    R2, Ip2, Im2, total2 = model.spectrum()
    rates_check = verify_rates_response(Iplus, Iminus, burn_idx, params.gamma_rf, dt=0.0015)
    print(f"single-step verify passed: {rates_check['passed']}")

    fig, axes = plt.subplots(figsize=(12, 6))
    axes.plot(R, _total, label="initial total")
    axes.plot(R, Ip2, label="I+")
    axes.plot(R, Im2, label="I-")
    axes.plot(R2, total2, label="after RF + diffusion")
    axes.axvline(params.rf_burn_R, linestyle="--", alpha=0.5)
    axes.legend()

    plt.tight_layout()
    plt.show()
    plt.close()
