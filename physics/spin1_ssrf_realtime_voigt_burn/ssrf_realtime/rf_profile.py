"""Voigt RF-rate profiles for the real-time spin-1 ss-RF simulation.

The profile in this module is an applied transition-rate field.  It is never
subtracted from the spectrum.  The population ODEs use it to calculate one RF
rate for every frequency bin, and the direct hole, mirror response,
finite-power saturation, diffusion refill, and DNP competition emerge from the
three-level populations.

Widths are full widths at half maximum (FWHM) in dimensionless physical R.
SciPy's exact Voigt evaluator is used when available; a normalized pseudo-Voigt
fallback keeps the package usable with NumPy alone.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np

try:  # pragma: no cover - availability depends on the user's environment
    from scipy.special import voigt_profile as _scipy_voigt_profile  # type: ignore
except Exception:  # pragma: no cover
    _scipy_voigt_profile = None

NormalizationMode = Literal["center_bin", "continuous_peak"]
_SQRT_2LN2 = float(np.sqrt(2.0 * np.log(2.0)))


def gaussian_sigma_from_fwhm(fwhm: float) -> float:
    return max(0.0, float(fwhm)) / (2.0 * _SQRT_2LN2)


def lorentzian_hwhm_from_fwhm(fwhm: float) -> float:
    return 0.5 * max(0.0, float(fwhm))


def approximate_voigt_fwhm(gaussian_fwhm: float, lorentzian_fwhm: float) -> float:
    """Olivero--Longbothum approximation to the Voigt FWHM."""
    g = max(0.0, float(gaussian_fwhm))
    l = max(0.0, float(lorentzian_fwhm))
    if g == 0.0:
        return l
    if l == 0.0:
        return g
    return float(0.5346 * l + np.sqrt(0.2166 * l * l + g * g))


def _pseudo_voigt_peak_normalized(x: np.ndarray, gaussian_fwhm: float, lorentzian_fwhm: float) -> np.ndarray:
    """Peak-normalized Thompson--Cox--Hastings pseudo-Voigt fallback."""
    x = np.asarray(x, dtype=float)
    g = max(0.0, float(gaussian_fwhm))
    l = max(0.0, float(lorentzian_fwhm))
    tiny = 1e-30
    if g <= tiny and l <= tiny:
        return np.zeros_like(x)
    if l <= tiny:
        return np.exp(-4.0 * np.log(2.0) * (x / max(g, tiny)) ** 2)
    if g <= tiny:
        return 1.0 / (1.0 + 4.0 * (x / max(l, tiny)) ** 2)

    f = (
        g**5
        + 2.69269 * g**4 * l
        + 2.42843 * g**3 * l**2
        + 4.47163 * g**2 * l**3
        + 0.07842 * g * l**4
        + l**5
    ) ** 0.2
    ratio = np.clip(l / max(f, tiny), 0.0, 1.0)
    eta = np.clip(1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3, 0.0, 1.0)
    gaussian = np.exp(-4.0 * np.log(2.0) * (x / max(f, tiny)) ** 2)
    lorentzian = 1.0 / (1.0 + 4.0 * (x / max(f, tiny)) ** 2)
    return eta * lorentzian + (1.0 - eta) * gaussian


def voigt_peak_normalized(x: np.ndarray | float, gaussian_fwhm: float, lorentzian_fwhm: float) -> np.ndarray:
    """Evaluate a continuous Voigt profile normalized to one at zero offset."""
    arr = np.asarray(x, dtype=float)
    g = max(0.0, float(gaussian_fwhm))
    l = max(0.0, float(lorentzian_fwhm))
    tiny = 1e-30
    if g <= tiny and l <= tiny:
        return np.zeros_like(arr)
    if _scipy_voigt_profile is None:
        return _pseudo_voigt_peak_normalized(arr, g, l)

    sigma = gaussian_sigma_from_fwhm(g)
    gamma = lorentzian_hwhm_from_fwhm(l)
    values = _scipy_voigt_profile(arr, sigma, gamma)
    peak = float(_scipy_voigt_profile(0.0, sigma, gamma))
    if not np.isfinite(peak) or peak <= 0.0:
        return _pseudo_voigt_peak_normalized(arr, g, l)
    return np.asarray(values / peak, dtype=float)


@lru_cache(maxsize=32)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(max(2, int(order)))
    return nodes.astype(float), weights.astype(float)


def recommended_quadrature_order(
    bin_width: float,
    gaussian_fwhm: float,
    lorentzian_fwhm: float,
    minimum: int = 16,
    maximum: int = 256,
) -> int:
    """Choose enough Gauss--Legendre points to resolve the profile inside a bin."""
    width = approximate_voigt_fwhm(gaussian_fwhm, lorentzian_fwhm)
    if width <= 0.0:
        return int(minimum)
    raw = int(np.ceil(12.0 * abs(float(bin_width)) / max(width, 1e-15)))
    return int(np.clip(max(minimum, raw), minimum, maximum))


def bin_averaged_voigt(
    bin_centers: np.ndarray,
    center_R: float,
    bin_width_R: float,
    gaussian_fwhm_R: float,
    lorentzian_fwhm_R: float,
    normalization: NormalizationMode = "center_bin",
    quadrature_order: int = 0,
) -> np.ndarray:
    r"""Average the continuous Voigt rate field over every finite R bin.

    For bin j,

    .. math::
       \bar v_j = \frac{1}{\Delta R}
       \int_{R_j-\Delta R/2}^{R_j+\Delta R/2}v(R-R_b)\,dR.

    ``center_bin`` rescales the largest finite-bin average to one, so the GUI's
    ``Gamma_RF`` remains the strongest bin-averaged rate and the narrow-profile
    limit approaches the legacy one-bin model smoothly. ``continuous_peak``
    leaves the underlying continuous Voigt normalized at V(0)=1.
    """
    centers = np.asarray(bin_centers, dtype=float)
    if centers.ndim != 1:
        raise ValueError("bin_centers must be one-dimensional")
    dR = abs(float(bin_width_R))
    if dR <= 0.0:
        raise ValueError("bin_width_R must be positive")
    g = max(0.0, float(gaussian_fwhm_R))
    l = max(0.0, float(lorentzian_fwhm_R))
    if g == 0.0 and l == 0.0:
        out = np.zeros_like(centers)
        if out.size:
            out[int(np.argmin(np.abs(centers - float(center_R))))] = 1.0
        return out

    order = int(quadrature_order)
    if order <= 0:
        order = recommended_quadrature_order(dR, g, l)
    nodes, weights = _legendre_rule(order)
    sample_R = centers[:, None] + 0.5 * dR * nodes[None, :]
    values = voigt_peak_normalized(sample_R - float(center_R), g, l)
    averaged = 0.5 * np.sum(values * weights[None, :], axis=1)
    averaged = np.maximum(np.nan_to_num(averaged, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    mode = str(normalization).lower()
    if mode == "center_bin":
        peak = float(np.max(averaged)) if averaged.size else 0.0
        if peak > 0.0 and np.isfinite(peak):
            averaged = averaged / peak
    elif mode != "continuous_peak":
        raise ValueError("normalization must be 'center_bin' or 'continuous_peak'")
    return averaged


def implementation_name() -> str:
    return "SciPy exact Voigt" if _scipy_voigt_profile is not None else "pseudo-Voigt fallback"
