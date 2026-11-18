"""C-D/O-D powder mixture (ssRF-beta / NMR-AI two-site convention)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def azimuth_quadrature(points):
    if isinstance(points, bool) or not isinstance(points, (int, np.integer)) or not 8 <= points <= 512:
        raise ValueError("Multisite azimuth points must be an integer in [8,512]")
    nodes, weights = np.polynomial.legendre.leggauss(points)
    return (nodes + 1) * np.pi / 4, weights / 2


def powder_branch(x, epsilon=1, eta=0.03, gamma=0.05, points=64):
    """Complex branch with unit integrated absorption on the infinite x axis."""
    if epsilon not in (-1, 1) or not np.isfinite([eta, gamma]).all() or not 0 <= eta <= 1 or gamma <= 0:
        raise ValueError("Require epsilon=+/-1, eta in [0,1], and positive linewidth")
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Frequency must be finite")
    phi, weight = azimuth_quadrature(points)
    y = np.sqrt(3 - eta * np.cos(2 * phi))
    root = np.sqrt(1 - epsilon * x[..., None] - eta * np.cos(2 * phi) + 1j * gamma)
    kernel = np.arctanh(y / root) / root
    if epsilon == -1:
        kernel = -kernel.conjugate()
    return (kernel / (np.pi * y)) @ weight


def site_components(
    R,
    epsilon=1,
    cd_scale=1.0,
    od_ratio=1.6,
    hwhm=0.05,
    eta_cd=0.03,
    eta_od=0.12,
    od_fraction=0.07,
    points=64,
):
    """Return weighted complex C-D/O-D branch densities in inverse R."""
    if not np.isfinite([cd_scale, od_ratio, hwhm, od_fraction]).all():
        raise ValueError("Nonfinite site parameter")
    if cd_scale <= 0 or od_ratio <= 1 or hwhm <= 0 or not 0 <= od_fraction <= 1:
        raise ValueError("Require positive scale/width, O-D/C-D ratio >1, and fraction in [0,1]")
    r = np.asarray(R, dtype=float)
    od_scale = cd_scale * od_ratio
    cd = (1 - od_fraction) / cd_scale * powder_branch(r / cd_scale, epsilon, eta_cd, hwhm / cd_scale, points)
    od = od_fraction / od_scale * powder_branch(r / od_scale, epsilon, eta_od, hwhm / od_scale, points)
    return cd, od


def absorption_density(R, **kwargs):
    cd, od = site_components(R, **kwargs)
    return -(cd + od).imag