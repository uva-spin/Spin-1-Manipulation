"""Dulya/Hamada equilibrium from a frozen single-site fit.

``use_fit_params(None)`` (the default) leaves data generation on the analytic
Pake lineshape. A JSON path installs those fit parameters as the equilibrium
for every spectrum built after the call.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SHAPE_KEYS = (
    'amp',
    'center',
    'cc',
    'split',
    'sigma',
    'eta',
    'xi',
    'b0',
    'b1',
    'b2',
    'b3',
)

_ACTIVE: dict | None = None


def _as_1d_float(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _clip_small_positive(x, floor=1e-15):
    return np.clip(x, floor, None)


def p_to_r(P):
    """Deuteron vector polarization P → Dulya asymmetry r (weak-quadrupole)."""
    P = np.asarray(P, dtype=float)
    disc = np.clip(4.0 - 3.0 * P ** 2, 0.0, None)
    return (np.sqrt(disc) + P) / (2.0 * (1.0 - P))


def branch_kernel_fixed_phi(R, A, eta, phi, eps):
    """Dipolar-broadened branch kernel f_eps(R, A, eta, phi) — Dulya Eq. (13)/(14)."""
    R = np.asarray(R)
    A = max(float(A), 1e-15)
    phi = np.asarray(phi)
    c2 = np.cos(2.0 * phi)
    b = 1.0 - eps * R - eta * c2
    y_max = np.sqrt(_clip_small_positive(3.0 - eta * c2))
    z = b + 1j * A
    sqrt_z = np.sqrt(z)
    w = 1.0 / sqrt_z * np.arctanh(y_max / sqrt_z)
    out = -2.0 / np.pi * np.imag(w)
    return np.clip(np.real(out), 0.0, None)


def powder_branch(R, A, eta, eps, nphi=64):
    """Powder-averaged branch F_eps(R, A, eta) — Dulya Eq. (15)."""
    R = _as_1d_float(R)
    A = max(float(A), 1e-15)
    if abs(eta) < 1e-14:
        return branch_kernel_fixed_phi(R, A, 0.0, 0.0, eps)
    phis = np.linspace(0.0, 0.5 * np.pi, nphi + 1)
    c2 = np.cos(2.0 * phis)
    weight = np.sqrt(3.0 / _clip_small_positive(3.0 - eta * c2))
    kernels = branch_kernel_fixed_phi(R[:, None], A, eta, phis[None, :], eps)
    return np.mean(weight[None, :] * kernels, axis=1)


def transition_weights(R, P, split, wd, exact_intensity=False):
    """Multiplicative weights for the plus / minus branches."""
    R = _as_1d_float(R)
    r = p_to_r(P)
    if not exact_intensity:
        return (r * np.ones_like(R), np.ones_like(R))
    vartheta = abs(split) / (3.0 * wd)
    plus = (r ** 2 - r ** (1.0 - 3.0 * vartheta * R)) / r ** (1.0 - vartheta * R)
    minus = (r ** (1.0 + 3.0 * vartheta * R) - 1.0) / r ** (1.0 + vartheta * R)
    return (plus, minus)


def site_transition_components(x_eff, P, split, sigma, eta, *, wd=16.35, exact_intensity=False, nphi=64):
    """Transition-resolved contributions of a single deuteron site."""
    x_eff = _as_1d_float(x_eff)
    R = x_eff / split
    A = sigma / abs(split)
    F_plus = powder_branch(R, A, eta, eps=+1, nphi=nphi)
    F_minus = powder_branch(R, A, eta, eps=-1, nphi=nphi)
    w_plus, w_minus = transition_weights(R, P, split, wd, exact_intensity=exact_intensity)
    plus = w_plus * F_plus / abs(split)
    minus = w_minus * F_minus / abs(split)
    return (plus, minus)


def polynomial_background(x, b0, b1, b2, b3):
    """Residual cubic background (Dulya Eq. 27/29)."""
    x = _as_1d_float(x)
    return b0 + b1 * x + b2 * x ** 2 + b3 * x ** 3


def qmeter_gain(x_eff, split_ref, xi):
    """False-asymmetry factor D = 1 + 0.5 * xi * (1 + R)."""
    x_eff = _as_1d_float(x_eff)
    Rq = x_eff / split_ref
    return 1.0 + 0.5 * xi * (1.0 + Rq)


def load_fit_params(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def shape_params_from_fit(fit_blob):
    """Frozen fit params used for every polarization."""
    missing = [k for k in SHAPE_KEYS if k not in fit_blob]
    if missing:
        raise KeyError(f'fit_params missing required keys {missing}; got {sorted(fit_blob)}')
    return {k: fit_blob[k] for k in SHAPE_KEYS}


def dulya_lineshape(P, x, shape_params, *, wd=16.35, exact_intensity=True, nphi=64):
    """I+/I- at fit signal scale (amp, Q-meter gain, baseline).

    Returns ``(ps, iplus, iminus)``.
    """
    x = _as_1d_float(x)
    if abs(P) < 1e-15:
        z = np.zeros_like(x)
        return (z, z.copy(), z.copy())
    P_clip = float(np.clip(P, -0.999999, 0.999999))
    x_eff = shape_params['cc'] * (x - shape_params['center'])
    plus, minus = site_transition_components(
        x_eff,
        P_clip,
        shape_params['split'],
        shape_params['sigma'],
        shape_params['eta'],
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )
    gain = qmeter_gain(x_eff, shape_params['split'], shape_params['xi'])
    background = polynomial_background(
        x,
        shape_params['b0'],
        shape_params['b1'],
        shape_params['b2'],
        shape_params['b3'],
    )
    iplus = shape_params['amp'] * np.asarray(plus) * gain + 0.5 * background
    iminus = shape_params['amp'] * np.asarray(minus) * gain + 0.5 * background
    return (iplus + iminus, iplus, iminus)


def use_fit_params(path=None):
    """Install a fitted lineshape, or clear it when ``path`` is None.

    Returns the shape dict, or None when the analytic Pake equilibrium is active.
    """
    global _ACTIVE
    if path is None:
        _ACTIVE = None
        return None
    _ACTIVE = shape_params_from_fit(load_fit_params(path))
    return dict(_ACTIVE)


def active_shape():
    return None if _ACTIVE is None else dict(_ACTIVE)


def _minimum_packet_weight(iplus, iminus):
    """Smallest ``mu`` that keeps every level non-negative at ``display_cal = dR``."""
    iplus = np.asarray(iplus, dtype=float)
    iminus = np.asarray(iminus, dtype=float)
    n = iplus.size
    b = iminus[n - 1 - np.arange(n)]
    return np.maximum.reduce((np.zeros(n), -2.0 * iplus - b, iplus - b, iplus + 2.0 * b))


def _seat_packet_weight(mu, iplus, iminus):
    """Move spare packet weight onto bins the fitted curve would otherwise clip.

    The analytic Pake weight is what ``create_data`` uses for an unfitted
    equilibrium, and that curve never asks a bin for a negative level. The
    Dulya horns and the cubic wing do. Zeroing the negative level cuts a notch
    out of an otherwise smooth lineshape. Spare weight on the other packets
    covers that request; the total weight stays 1.
    """
    mu = np.asarray(mu, dtype=float)
    req = _minimum_packet_weight(iplus, iminus)
    deficit = np.maximum(req - mu, 0.0)
    need = float(deficit.sum())
    if need <= 0.0:
        return mu
    slack = np.maximum(mu - req, 0.0)
    slack_sum = float(slack.sum())
    if need >= slack_sum:
        return mu
    seated = np.maximum(mu, req)
    seated = seated - need * slack / slack_sum
    return seated


def apply_active_fit(model):
    """Load the fitted Dulya lineshape at its own amplitude with full spin physics.

    Intensities are normalized so ``sum(I+ + I-) = p0``. Population differences
    are ``I * dR / display_cal``, so the default calibration would shrink that
    into a nearly unpolarized packet and an ssRF burn could not raise Q.
    Setting ``display_cal = dR`` makes those fitted intensities a real ``p0``
    polarization; the saved lineshape stays on the fitted scale. Packet weight
    is then seated so the smooth fitted curve is stored whole, instead of
    chopping bins whose Pake weight is too small.
    """
    shape = _ACTIVE
    if shape is None:
        return model
    freq = np.asarray(model.Rplus, dtype=float)
    _, iplus, iminus = dulya_lineshape(float(model.params.p0), freq, shape)
    area = float(np.sum(iplus + iminus))
    p0 = float(model.params.p0)
    if abs(area) > 1e-30 and abs(p0) > 1e-15:
        scale = p0 / area
        iplus = iplus * scale
        iminus = iminus * scale
    model.display_cal = float(model.dR)
    model.mu = _seat_packet_weight(model.mu, iplus, iminus)
    model.base_density = model.mu / model.dR
    model._diffusion_kernel_key = None
    model.load_from_physical_intensities(iplus, iminus)
    return model
