"""Dulya/Hamada single-site absorption kernel (self-contained, no external physics deps)."""

import numpy as np


def _as_1d_float(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _clip_small_positive(x, floor=1e-15):
    return np.clip(x, floor, None)


def p_to_r(P):
    """Deuteron vector polarization P → Dulya asymmetry r (weak-quadrupole)."""
    P = np.asarray(P, dtype=float)
    disc = np.clip(4.0 - 3.0 * P**2, 0.0, None)
    return (np.sqrt(disc) + P) / (2.0 * (1.0 - P))


def branch_kernel_fixed_phi(R, A, eta, phi, eps):
    """Dipolar-broadened branch kernel f_eps(R, A, eta, phi) — Dulya Eq. (13)/(14)."""
    R = np.asarray(R, dtype=float)
    A = max(float(A), 1e-15)
    phi = np.asarray(phi, dtype=float)

    c2 = np.cos(2.0 * phi)
    b = 1.0 - eps * R - eta * c2
    y_max = np.sqrt(_clip_small_positive(3.0 - eta * c2))

    z = b + 1j * A
    sqrt_z = np.sqrt(z)
    w = (1.0 / sqrt_z) * np.arctanh(y_max / sqrt_z)
    out = (-2.0 / np.pi) * np.imag(w)
    return np.clip(np.real(out), 0.0, None)


def powder_branch(R, A, eta, eps, nphi=64):
    """Powder-averaged branch F_eps(R, A, eta) — Dulya Eq. (15)."""
    R = _as_1d_float(R)
    A = max(float(A), 1e-15)
    eta = float(eta)

    if abs(eta) < 1e-14:
        return branch_kernel_fixed_phi(R, A, 0.0, 0.0, eps)

    phis = np.linspace(0.0, 0.5 * np.pi, int(nphi) + 1)
    c2 = np.cos(2.0 * phis)
    weight = np.sqrt(3.0 / _clip_small_positive(3.0 - eta * c2))

    rr = R[:, None]
    kernels = branch_kernel_fixed_phi(rr, A, eta, phis[None, :], eps)
    return np.mean(weight[None, :] * kernels, axis=1)


def transition_weights(R, P, split, wd, exact_intensity=False):
    """Multiplicative weights for the plus / minus branches."""
    R = _as_1d_float(R)
    r = float(p_to_r(P))

    if not exact_intensity:
        return r * np.ones_like(R), np.ones_like(R)

    vartheta = abs(float(split)) / (3.0 * float(wd))
    plus = (r**2 - r ** (1.0 - 3.0 * vartheta * R)) / (r ** (1.0 - vartheta * R))
    minus = (r ** (1.0 + 3.0 * vartheta * R) - 1.0) / (r ** (1.0 + vartheta * R))
    return plus, minus


def site_transition_components(
    x_eff,
    P,
    split,
    sigma,
    eta,
    *,
    wd=16.35,
    exact_intensity=False,
    nphi=64,
):
    """Transition-resolved contributions of a single deuteron site."""
    x_eff = _as_1d_float(x_eff)
    split = float(split)
    sigma = float(sigma)

    R = x_eff / split
    A = sigma / abs(split)

    F_plus = powder_branch(R, A, eta, eps=+1, nphi=nphi)
    F_minus = powder_branch(R, A, eta, eps=-1, nphi=nphi)
    w_plus, w_minus = transition_weights(
        R, P, split, wd, exact_intensity=exact_intensity
    )

    plus = w_plus * F_plus / abs(split)
    minus = w_minus * F_minus / abs(split)
    return plus, minus


def polynomial_background(x, b0, b1, b2, b3):
    """Residual cubic background (Dulya Eq. 27/29)."""
    x = _as_1d_float(x)
    return b0 + b1 * x + b2 * x**2 + b3 * x**3


def qmeter_gain(x_eff, split_ref, xi):
    """False-asymmetry factor D = 1 + 0.5 * xi * (1 + R)."""
    x_eff = _as_1d_float(x_eff)
    Rq = x_eff / float(split_ref)
    return 1.0 + 0.5 * float(xi) * (1.0 + Rq)
