"""
Analytic broadened spin-1 Pake-doublet absorption lineshape.

This module implements the single-branch analytic function used in the
provided Plot_Signal.py example and in the spin-1 Pake-doublet formalism.
The branch label epsilon selects the two mirror-related absorption lines:

    epsilon = +1: I_plus-like branch with the sharp tip on the +R side
    epsilon = -1: I_minus-like branch, equivalently I(R,-1)=I(-R,+1)

The parameters are named to match the theory:

    gamma: Lorentzian/dipolar broadening parameter Γ
    asym:  eta*cos(2 phi), called ``s`` in the original Plot_Signal.py

The returned function is a density in the dimensionless coordinate R.  It is
not a hole profile.  Hole width / RF leakage should be modeled separately by a
Voigt RF kernel after the ideal-bin dynamics are understood.
"""

from __future__ import annotations

import numpy as np


def trapezoid_integral(y, x) -> float:
    """Compatibility wrapper for NumPy < 2.0 and newer NumPy versions."""
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


def boltzmann_Q(P: float) -> float:
    """Spin-1 Boltzmann relation Q(P) = 2 - sqrt(4 - 3 P^2)."""
    return float(2.0 - np.sqrt(max(0.0, 4.0 - 3.0 * P * P)))


def boltzmann_branch_ratio(P: float) -> float:
    """
    Boltzmann equilibrium area ratio I_plus/I_minus for a spin-1 doublet.

    This is Eq. (15)-style ratio used by the attached Plot_Signal.py:

        r = (sqrt(4 - 3 P^2) + P) / (2 - 2 P).
    """
    denom = 2.0 - 2.0 * P
    if abs(denom) < 1e-15:
        return float("inf")
    return float((np.sqrt(max(0.0, 4.0 - 3.0 * P * P)) + P) / denom)


def level_populations_from_PQ(P: float, Q: float | None = None) -> np.ndarray:
    """Return normalized spin-1 level fractions [p_plus, p_zero, p_minus]."""
    if Q is None:
        Q = boltzmann_Q(P)
    p_plus = 1.0 / 3.0 + 0.5 * P + Q / 6.0
    p_zero = (1.0 - Q) / 3.0
    p_minus = 1.0 / 3.0 - 0.5 * P + Q / 6.0
    p = np.array([p_plus, p_zero, p_minus], dtype=float)
    if np.any(p < -1e-12):
        raise ValueError(
            f"Invalid spin-1 populations from P={P}, Q={Q}: {p}. "
            "Check that P,Q are in the physically allowed triangle."
        )
    p = np.maximum(p, 0.0)
    return p / p.sum()


def pake_component_raw(R, epsilon: int = +1, gamma: float = 0.05, asym: float = 0.04):
    """
    Analytic broadened single-branch Pake component I(R, epsilon).

    This is a vectorized, numerically guarded implementation of the formula in
    the attached Plot_Signal.py.  The variable ``asym`` is the script's ``s``
    and corresponds to eta*cos(2 phi).  The variable ``gamma`` is Γ.
    """
    eps = 1 if epsilon >= 0 else -1
    R = np.asarray(R, dtype=float)
    gamma = float(gamma)
    asym = float(asym)

    # Use the notation from the script: bigy = sqrt(3 - s), and the script's
    # bigxsquare is the quantity X^2 = sqrt(Γ^2 + (... )^2).
    bigy = np.sqrt(max(1e-15, 3.0 - asym))
    X2 = np.sqrt(gamma * gamma + (1.0 - eps * R - asym) ** 2)
    sqrt_X2 = np.sqrt(np.maximum(X2, 1e-300))

    cos_alpha = (1.0 - eps * R - asym) / np.maximum(X2, 1e-300)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    cos_half = np.sqrt(np.maximum(0.0, (1.0 + cos_alpha) / 2.0))
    sin_half = np.sqrt(np.maximum(0.0, (1.0 - cos_alpha) / 2.0))

    denom1 = 2.0 * bigy * sqrt_X2 * sin_half
    numer1 = bigy * bigy - X2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio1 = np.divide(numer1, denom1, out=np.full_like(numer1, np.inf), where=np.abs(denom1) > 1e-300)
    term_one = np.pi / 2.0 + np.arctan(ratio1)

    denom2 = bigy * bigy + X2 - 2.0 * bigy * sqrt_X2 * cos_half
    numer2 = bigy * bigy + X2 + 2.0 * bigy * sqrt_X2 * cos_half
    denom2 = np.maximum(denom2, 1e-300)
    numer2 = np.maximum(numer2, 1e-300)
    term_two = np.log(numer2 / denom2)

    mult = 1.0 / (2.0 * np.pi * np.sqrt(np.maximum(X2, 1e-300)))
    out = mult * (2.0 * cos_half * term_one + sin_half * term_two)
    out = np.asarray(out, dtype=float)
    # Numerical guards only; the analytic expression is nonnegative.
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.maximum(out, 0.0)
    return out


def normalized_component(R, epsilon: int = +1, gamma: float = 0.05, asym: float = 0.04):
    """Return the area-normalized analytic component density on grid R."""
    R = np.asarray(R, dtype=float)
    y = pake_component_raw(R, epsilon=epsilon, gamma=gamma, asym=asym)
    area = trapezoid_integral(y, R) if len(R) > 1 else float(np.sum(y))
    if area <= 0 or not np.isfinite(area):
        raise ValueError("Lineshape area is not positive; check gamma/asym/R grid.")
    return y / area


def plot_signal_reference(R, P: float = 0.50, gamma: float = 0.05, asym: float = 0.04, divisor: float = 10.0):
    """
    Reproduce the attached Plot_Signal.py convention.

    Returns (I_plus, I_minus, total) where

        I_plus  = r * I(R,+1) / divisor
        I_minus =     I(-R,+1) / divisor.

    This is useful for validating the static lineshape visually.  The dynamic
    simulator uses the same analytic branch shape but evolves population
    differences bin by bin rather than refitting a global r at every time.
    """
    R = np.asarray(R, dtype=float)
    r = boltzmann_branch_ratio(P)
    y_plus = pake_component_raw(R, +1, gamma=gamma, asym=asym) / divisor
    y_minus = pake_component_raw(-R, +1, gamma=gamma, asym=asym) / divisor
    return r * y_plus, y_minus, r * y_plus + y_minus
