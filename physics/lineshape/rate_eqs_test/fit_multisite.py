#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone multi-site deuteron NMR fitting script based on the line-shape theory
used by Hamada et al. (1981) and Dulya et al. (1997).

Why this script exists
----------------------
If the extra shoulder in deuterated butanol is really an O-D / heavy-water-like
contribution, fitting it with a generic Voigt can stabilize the fit but it is
not the most physical model.  A better model is the sum of *two deuteron site
contributions*:

    total absorption = (1 - K) * C-D contribution + K * O-D contribution

with:
  • a common deuteron polarization P (spin-temperature assumption),
  • a common *physical* dipolar width sigma,
  • eta_CD ~ 0 for the C-D bond,
  • eta_OD free (or fixed) for the O-D bond,
  • different quadrupole splittings for the C-D and O-D sites,
  • optional Q-meter false-asymmetry and polynomial residual background.

This follows the structure described in:
  • O. Hamada et al., Nucl. Instrum. Methods 189 (1981) 561-568
  • C. Dulya et al., Nucl. Instrum. Methods A398 (1997) 109-125

What is implemented
-------------------
1. The dipolar-broadened powder-pattern branch function F_±(R, A, eta)
   from the Dulya/Hamada theory.
2. A physical two-site model for deuterated butanol-like spectra:
       (1-K) * [C-D] + K * [O-D]
3. Optional exact Dulya frequency-dependent intensity factors (Eq. 24 in the
   Dulya paper), or the usual weak-quadrupole approximation used for deuterons
   in butanol (Eq. 25), where the branch ratio is approximately r.
4. Optional false asymmetry parameter xi and cubic residual background.
5. Transition-resolved component bookkeeping, so you can integrate the plus and
   minus branches *separately* for tensor work.
6. Robust least-squares fitting with optional fixed parameters.
7. A two-stage workflow:
       stage 1: central C-D-only fit
       stage 2: full C-D + O-D fit

Units
-----
All frequency-like quantities must be in the same units.
For example, if your x-axis is in MHz offset, then use:
    x           in MHz offset
    center      in MHz offset
    split_cd    in MHz
    split_od    in MHz
    sigma       in MHz
    wd          in MHz  (for deuterons at 2.5 T, wd ≈ 16.35 MHz)

Here ``split_cd`` and ``split_od`` mean the frequency scale 3*w_q used to form
R = (omega - omega_d) / (3*w_q).  For eta = 0 the main peaks sit near ±split.

Practical defaults for butanol-like spectra
-------------------------------------------
- eta_cd is kept fixed to 0.0 by default.
- K is the relative O-D-like fraction.  Dulya quotes an expected K ~ 0.136 for
  deuterated butanol with added heavy water, but you can either fix or fit it.
- The exact intensity factors matter only at the ~1% level for deuterons in
  butanol at 2.5 T, so the weak-quadrupole approximation is often adequate.
  Still, the exact option is included here.

How to use with real data
-------------------------
Replace the synthetic-data block at the bottom with your own x, y, and yerr
arrays, or load them from text.  Then edit:
    p0_full, bounds_full, fixed_full, clean_window

This file is self-contained and does not depend on your earlier fit scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

N_BINS = 500
def load_event_csv(path: Path) -> tuple[str, np.ndarray]:
    with path.open(newline="") as f:
        row = next(csv.reader(f))
    if len(row) != N_BINS + 1:
        raise ValueError(
            f"Expected {N_BINS + 1} columns (event_id + {N_BINS} bins), got {len(row)}"
        )
    event_id = row[0].strip()
    y = np.array([float(x) for x in row[1:]], dtype=np.float64)
    y_fit = -y
    return event_id, y_fit


# =============================================================================
# Compatibility / numerical helpers
# =============================================================================


def _trapz(y, x):
    """Use np.trapezoid when available, otherwise fall back to np.trapz."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)



def _as_1d_float(x):
    """Return x as a 1-D float array without copying unless needed."""
    return np.asarray(x, dtype=float).reshape(-1)



def _clip_small_positive(x, floor=1e-15):
    """Clip to a small positive floor to avoid division / sqrt problems."""
    return np.clip(x, floor, None)



def p_to_r(P):
    """
    Convert deuteron vector polarization P to Dulya's asymmetry parameter r.

    Using the weak-quadrupole relation:
        P = (r^2 - 1) / (r^2 + r + 1)

    solved for the positive root r(P).
    """
    P = np.asarray(P, dtype=float)
    disc = np.clip(4.0 - 3.0 * P**2, 0.0, None)
    return (np.sqrt(disc) + P) / (2.0 * (1.0 - P))



def r_to_p(r):
    """Inverse of p_to_r under the same weak-quadrupole approximation."""
    r = np.asarray(r, dtype=float)
    return (r**2 - 1.0) / (r**2 + r + 1.0)


# =============================================================================
# Dulya/Hamada branch kernel
# =============================================================================
# Theory summary:
#   R = (omega - omega_d) / (3*w_q)
#   A = sigma / (3*w_q)
#   eta is the quadrupole asymmetry parameter of that site
#
# For a fixed azimuth angle phi and a chosen branch eps = +/-1, the dipolar-
# broadened branch function is the Lorentzian convolution written in Dulya's
# Eq. (13) / (14).  We evaluate it through an equivalent compact complex form,
# which is algebraically the same integral but easier to implement robustly.
#
# Then the powder average over phi is performed numerically as in Eq. (15).
# =============================================================================


def branch_kernel_fixed_phi(R, A, eta, phi, eps):
    """
    Dipolar-broadened branch kernel f_eps(R, A, eta, phi).

    Parameters
    ----------
    R : array-like
        Dimensionless frequency variable of one site:
            R = (omega - omega_d) / (3*w_q)
    A : float
        Dimensionless Lorentzian width parameter for that site:
            A = sigma / (3*w_q)
    eta : float
        Quadrupole asymmetry parameter of that site.
    phi : float or array-like
        Azimuthal angle in radians.
    eps : {+1, -1}
        Branch index.

    Notes
    -----
    This is the closed-form evaluation of the convolution integral appearing in
    Eq. (13) / (14) of the Dulya paper.  The principal-value / branch-cut issues
    are handled automatically by NumPy's complex arithmetic.
    """
    R = np.asarray(R, dtype=float)
    A = max(float(A), 1e-15)
    phi = np.asarray(phi, dtype=float)

    c2 = np.cos(2.0 * phi)
    b = 1.0 - eps * R - eta * c2
    y_max = np.sqrt(_clip_small_positive(3.0 - eta * c2))

    # Complex parameter z = b + i A.
    z = b + 1j * A
    sqrt_z = np.sqrt(z)

    # Closed form of the integral 2A/pi * ∫ dy / ((y^2 - b)^2 + A^2).
    w = (1.0 / sqrt_z) * np.arctanh(y_max / sqrt_z)
    out = (-2.0 / np.pi) * np.imag(w)

    # Small negative values can arise from floating-point roundoff very far out
    # in the tails; clip them away.
    return np.clip(np.real(out), 0.0, None)



def powder_branch(R, A, eta, eps, nphi=64):
    """
    Powder-averaged branch function F_eps(R, A, eta).

    This performs the phi-average from Dulya Eq. (15).
    For eta = 0 the branch is phi-independent, so we skip the average.
    """
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


# =============================================================================
# Intensity factors
# =============================================================================
# In the butanol case at 2.5 T, Dulya notes that the frequency dependence of the
# intensity factors is small because vartheta = w_q / w_d << 1.  The usual
# approximation is then:
#
#     chi'' ~ r * F_plus + F_minus
#
# which is the origin of the simple "asymmetry parameter" picture.
#
# The exact frequency-dependent factors from Dulya Eq. (21)-(24) are included as
# an option for completeness.
# =============================================================================


def transition_weights(R, P, split, wd, exact_intensity=False):
    """
    Return the multiplicative weights for the plus and minus branches.

    Parameters
    ----------
    R : array-like
        Dimensionless frequency variable of one site.
    P : float
        Deuteron vector polarization.
    split : float
        The site frequency scale 3*w_q in the same units as x.
    wd : float
        Larmor frequency in the same units as x.
    exact_intensity : bool
        If True, use the frequency-dependent Dulya factors (Eq. 24).
        If False, use the butanol weak-quadrupole approximation (Eq. 25).

    Notes
    -----
    Any overall R-independent scale factor is left out on purpose because the
    fit already has a free amplitude parameter.  What matters here is the shape
    asymmetry across the line.
    """
    R = _as_1d_float(R)
    r = float(p_to_r(P))

    if not exact_intensity:
        return r * np.ones_like(R), np.ones_like(R)

    # vartheta = w_q / w_d = split / (3 * w_d)
    vartheta = abs(float(split)) / (3.0 * float(wd))

    # Dulya Eq. (24), ignoring the common 1/w_q prefactor that is absorbed into
    # the overall fit amplitude and the site-mixing normalization.
    plus = (r**2 - r**(1.0 - 3.0 * vartheta * R)) / (r**(1.0 - vartheta * R))
    minus = (r**(1.0 + 3.0 * vartheta * R) - 1.0) / (r**(1.0 + vartheta * R))
    return plus, minus


# =============================================================================
# Site and multi-site models
# =============================================================================


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
    """
    Transition-resolved contributions of a single deuteron site.

    Parameters
    ----------
    x_eff : array-like
        Effective frequency offset after any center/axis calibration correction.
    P : float
        Common deuteron vector polarization.
    split : float
        Site frequency scale 3*w_q (same units as x).
    sigma : float
        Common physical dipolar width (same units as x).
    eta : float
        Site asymmetry parameter.
    wd : float
        Larmor frequency.
    exact_intensity : bool
        Use frequency-dependent intensity factors if True.
    nphi : int
        Number of phi steps for the powder average.

    Returns
    -------
    plus, minus : ndarray, ndarray
        The two transition-family contributions for this site.
    """
    x_eff = _as_1d_float(x_eff)
    split = float(split)
    sigma = float(sigma)

    R = x_eff / split
    A = sigma / abs(split)

    F_plus = powder_branch(R, A, eta, eps=+1, nphi=nphi)
    F_minus = powder_branch(R, A, eta, eps=-1, nphi=nphi)
    w_plus, w_minus = transition_weights(R, P, split, wd, exact_intensity=exact_intensity)

    # The 1/w_q prefactor from Dulya's absorption function translates to 1/split
    # up to a common constant factor of 3.  That constant can be absorbed into
    # the overall amplitude, but the *relative* site scaling with split matters.
    plus = w_plus * F_plus / abs(split)
    minus = w_minus * F_minus / abs(split)
    return plus, minus



def butanol_absorption_components(
    x_eff,
    P,
    split_cd,
    split_od,
    sigma,
    eta_od,
    K,
    *,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
):
    """
    Physical two-site absorption model for deuterated-butanol-like spectra.
    Most of the time you can tell from the lineshape if its a contaminent or artifact,
    But you need to pay attention on which transition each area belongs to.

    Parameters
    ----------
    x_eff : array-like
        Effective frequency axis after center/cc correction.
    P : float
        Common deuteron vector polarization.
    split_cd, split_od : float
        Site frequency scales 3*w_q for the C-D and O-D sites.
    sigma : float
        Common physical dipolar width.
    eta_od : float
        O-D quadrupole asymmetry parameter.
    K : float
        Relative O-D-like contribution.  Total model is:
            (1-K) * C-D + K * O-D
        so K must lie between 0 and 1.
    wd : float
        Larmor frequency.
    eta_cd : float
        C-D asymmetry parameter.  Default is 0.0, which is standard for butanol.
    exact_intensity : bool
        Use Dulya Eq. (24) if True, Eq. (25)-style approximation if False.
    nphi : int
        Number of phi points for the powder average.

    Returns
    -------
    dict
        Transition-resolved and site-resolved physical absorption components.
    """
    x_eff = _as_1d_float(x_eff)
    K = float(K)

    cd_plus, cd_minus = site_transition_components(
        x_eff,
        P,
        split_cd,
        sigma,
        eta_cd,
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )
    od_plus, od_minus = site_transition_components(
        x_eff,
        P,
        split_od,
        sigma,
        eta_od,
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )

    cd_plus *= (1.0 - K)
    cd_minus *= (1.0 - K)
    od_plus *= K
    od_minus *= K

    plus_total = cd_plus + od_plus
    minus_total = cd_minus + od_minus
    absorption = plus_total + minus_total

    return {
        "cd_plus": cd_plus,
        "cd_minus": cd_minus,
        "cd_total": cd_plus + cd_minus,
        "od_plus": od_plus,
        "od_minus": od_minus,
        "od_total": od_plus + od_minus,
        "plus_total": plus_total,
        "minus_total": minus_total,
        "absorption": absorption,
    }


# =============================================================================
# Instrument / baseline model
# =============================================================================


def polynomial_background(x, b0, b1, b2, b3):
    """Residual background polynomial, following Dulya's Eq. (27)/(29)."""
    x = _as_1d_float(x)
    return b0 + b1 * x + b2 * x**2 + b3 * x**3



def qmeter_gain(x_eff, split_ref, xi):
    """
    Simple false-asymmetry correction factor.

    Dulya parameterizes the Q-meter distortion as:
        D(omega) = 1 + 0.5 * xi * (1 + R)

    Here R is taken with respect to the larger of the two site splittings (the
    same practical choice Hamada recommends for normalization when one bond has
    the larger quadrupole coupling).
    """
    x_eff = _as_1d_float(x_eff)
    split_ref = float(split_ref)
    xi = float(xi)
    Rq = x_eff / split_ref
    return 1.0 + 0.5 * xi * (1.0 + Rq)



def signal_model(
    x,
    P,
    amp,
    center,
    cc,
    split_cd,
    split_od,
    sigma,
    eta_od,
    K,
    xi,
    b0,
    b1,
    b2,
    b3,
    *,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
):
    """
    Full measurable signal model.

    Model
    -----
    1. Apply a global center shift and optional x-axis calibration coefficient:
           x_eff = cc * (x - center)

    2. Build the physical absorption function:
           chi''_but = (1-K) * chi''_CD + K * chi''_OD

    3. Optionally apply the false-asymmetry Q-meter gain correction:
           D = 1 + 0.5 * xi * (1 + R)

    4. Add a cubic residual background.

    ``amp`` is allowed to be either positive or negative, so the script can fit
    spectra that have or have not been sign-flipped.
    """
    x = _as_1d_float(x)
    x_eff = float(cc) * (x - float(center))

    comps = butanol_absorption_components(
        x_eff,
        P,
        split_cd,
        split_od,
        sigma,
        eta_od,
        K,
        wd=wd,
        eta_cd=eta_cd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )

    split_ref = split_od if abs(split_od) >= abs(split_cd) else split_cd
    gain = qmeter_gain(x_eff, split_ref, xi)
    background = polynomial_background(x, b0, b1, b2, b3)
    return float(amp) * comps["absorption"] * gain + background



def component_curves(
    params,
    x,
    *,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
):
    """
    Return named physical components of the fitted model.

    This is the main bookkeeping function for plotting and area calculations.
    The returned curves are on the *measured* x-grid and are already multiplied
    by the fitted overall amplitude ``amp``.  The baseline is separate.

    Returned keys
    -------------
    cd_plus, cd_minus, cd_total
    od_plus, od_minus, od_total
    plus_total, minus_total, absorption_physical
    qmeter_gain, absorption_measured, background, total
    x_eff
    """
    x = _as_1d_float(x)
    p = dict(params)
    x_eff = float(p["cc"]) * (x - float(p["center"]))

    comps = butanol_absorption_components(
        x_eff,
        p["P"],
        p["split_cd"],
        p["split_od"],
        p["sigma"],
        p["eta_od"],
        p["K"],
        wd=wd,
        eta_cd=eta_cd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )

    # Multiply the physical components by the fitted amplitude.
    amp = float(p["amp"])
    for key in list(comps.keys()):
        comps[key] = amp * comps[key]

    split_ref = p["split_od"] if abs(p["split_od"]) >= abs(p["split_cd"]) else p["split_cd"]
    gain = qmeter_gain(x_eff, split_ref, p["xi"])
    background = polynomial_background(x, p["b0"], p["b1"], p["b2"], p["b3"])

    absorption_measured = comps["absorption"] * gain
    total = absorption_measured + background

    comps.update(
        {
            "absorption_physical": comps["absorption"],
            "qmeter_gain": gain,
            "absorption_measured": absorption_measured,
            "background": background,
            "total": total,
            "x_eff": x_eff,
        }
    )
    return comps


# =============================================================================
# Fitting helpers
# =============================================================================

PARAM_ORDER = (
    "P",
    "amp",
    "center",
    "cc",
    "split_cd",
    "split_od",
    "sigma",
    "eta_od",
    "K",
    "xi",
    "b0",
    "b1",
    "b2",
    "b3",
)


@dataclass
class FitSummary:
    params: Dict[str, float]
    free_names: Tuple[str, ...]
    pcov: Optional[np.ndarray]
    result: object



def _merge_params(p_free, param_order, p0, fixed):
    params = dict(p0)
    params.update(fixed)
    free_names = [name for name in param_order if name not in fixed]
    for name, value in zip(free_names, p_free):
        params[name] = float(value)
    return params



def _estimate_covariance(result):
    """Linearized covariance estimate from the least-squares Jacobian."""
    jac = getattr(result, "jac", None)
    if jac is None:
        return None

    jac = np.asarray(jac, dtype=float)
    if jac.ndim != 2:
        return None

    n_data, n_par = jac.shape
    if n_data <= n_par:
        return None

    try:
        _, svals, vt = np.linalg.svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * svals[0]
        keep = svals > threshold
        if not np.any(keep):
            return None

        svals = svals[keep]
        vt = vt[keep, :]
        jtj_inv = (vt.T / (svals**2)) @ vt

        rss = np.sum(np.asarray(result.fun, dtype=float) ** 2)
        dof = max(n_data - n_par, 1)
        return jtj_inv * (rss / dof)
    except np.linalg.LinAlgError:
        return None



def fit_signal(
    x,
    y,
    yerr=None,
    p0=None,
    bounds=None,
    fixed=None,
    *,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
    loss="soft_l1",
    f_scale=1.0,
    max_nfev=50000,
):
    """
    Generalized fit of the full multi-site signal model.

    Parameters
    ----------
    x, y, yerr : array-like
        Measured data and optional 1-sigma errors.
    p0 : dict
        Initial guesses for all parameters in PARAM_ORDER.
    bounds : dict
        Bounds for all parameters in PARAM_ORDER.
    fixed : dict or None
        Parameters to hold fixed.
        Example:
            fixed={"cc": 0.25, "eta_od": 0.17, "xi": 0.0}
    exact_intensity : bool
        Use the full Dulya frequency-dependent intensity factors.
    loss, f_scale : str, float
        Robust least-squares settings.
    """
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    sigma = np.ones_like(y) if yerr is None else np.maximum(_as_1d_float(yerr), 1e-12)

    if p0 is None:
        p0 = dict(
            P=0.35,
            amp=1.0,
            center=0.0,
            cc=1.0,
            split_cd=0.90,
            split_od=1.15,
            sigma=0.05,
            eta_od=0.15,
            K=0.12,
            xi=0.0,
            b0=0.0,
            b1=0.0,
            b2=0.0,
            b3=0.0,
        )
    if bounds is None:
        bounds = dict(
            P=(-0.99, 0.99),
            amp=(-np.inf, np.inf),
            center=(-0.5, 0.5),
            cc=(0.5, 1.5),
            split_cd=(1e-4, np.inf),
            split_od=(1e-4, np.inf),
            sigma=(1e-5, np.inf),
            eta_od=(0.0, 0.95),
            K=(0.0, 1.0),
            xi=(-0.5, 0.5),
            b0=(-np.inf, np.inf),
            b1=(-np.inf, np.inf),
            b2=(-np.inf, np.inf),
            b3=(-np.inf, np.inf),
        )
    fixed = {} if fixed is None else dict(fixed)

    free_names = tuple(name for name in PARAM_ORDER if name not in fixed)
    if not free_names:
        raise ValueError("At least one parameter must be free.")

    p0_vec = np.array([p0[name] for name in free_names], dtype=float)
    lb = np.array([bounds[name][0] for name in free_names], dtype=float)
    ub = np.array([bounds[name][1] for name in free_names], dtype=float)

    def residuals(p_free):
        params = _merge_params(p_free, PARAM_ORDER, p0, fixed)
        y_model = signal_model(
            x,
            **params,
            wd=wd,
            eta_cd=eta_cd,
            exact_intensity=exact_intensity,
            nphi=nphi,
        )
        return (y_model - y) / sigma

    result = least_squares(
        residuals,
        p0_vec,
        bounds=(lb, ub),
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )

    best = _merge_params(result.x, PARAM_ORDER, p0, fixed)
    pcov = _estimate_covariance(result)
    return FitSummary(params=best, free_names=free_names, pcov=pcov, result=result)



def fit_clean_then_full(
    x,
    y,
    yerr,
    p0_full,
    bounds_full,
    *,
    clean_window=None,
    fixed_main=None,
    fixed_full=None,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
    loss="soft_l1",
    f_scale=1.0,
    max_nfev=50000,
):
    """
    Two-stage workflow tuned for butanol-like spectra.

    Stage 1
    -------
    Fit the cleaner central region with only the C-D site active by fixing K = 0.

    Stage 2
    -------
    Turn the O-D site back on and fit the full range.

    Why this may help
    --------------
    The main C-D doublet is much stronger, so a central-window C-D-only fit gives
    a stable estimate of P, center, cc, split_cd, sigma, and baseline before the
    outer O-D shoulders are introduced.
    """
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    yerr = None if yerr is None else _as_1d_float(yerr)

    if clean_window is None:
        mask = np.ones_like(x, dtype=bool)
    else:
        lo, hi = clean_window
        mask = (x >= lo) & (x <= hi)

    # Stage 1: C-D only.  K=0 removes the O-D site.  The O-D parameters are then
    # irrelevant, so it is safest to keep them fixed too.
    default_fixed_main = {
        "K": 0.0,
        "eta_od": p0_full["eta_od"],
        "split_od": p0_full["split_od"],
        "xi": 0.0,
    }
    if fixed_main is not None:
        default_fixed_main.update(dict(fixed_main))

    main_fit = fit_signal(
        x[mask],
        y[mask],
        None if yerr is None else yerr[mask],
        p0=p0_full,
        bounds=bounds_full,
        fixed=default_fixed_main,
        wd=wd,
        eta_cd=eta_cd,
        exact_intensity=exact_intensity,
        nphi=nphi,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )

    # Seed the full fit from the stage-1 result for the parameters that overlap.
    seeded = dict(p0_full)
    seeded.update(main_fit.params)

    full_fit = fit_signal(
        x,
        y,
        yerr,
        p0=seeded,
        bounds=bounds_full,
        fixed=fixed_full,
        wd=wd,
        eta_cd=eta_cd,
        exact_intensity=exact_intensity,
        nphi=nphi,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
    )
    return main_fit, full_fit


# =============================================================================
# Area bookkeeping and stuff
# =============================================================================


def compute_transition_areas(
    x,
    fit,
    *,
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
    n_mc=300,
    seed=123,
):
    """
    Just compute transition- and site-resolved areas from a fit.

    Areas are taken from the *underlying physical absorption components* before
    any Q-meter gain correction and before the polynomial background is added.

    Reported keys
    -------------
    cd_plus, cd_minus, od_plus, od_minus
    plus_total, minus_total
    cd_total, od_total, total_physical
    diff_total = plus_total - minus_total

    This is the quantity you generally want for tensor-analysis bookkeeping.
    Keep in mind that this is where monkeys really fly.
    """
    x = _as_1d_float(x)
    curves = component_curves(
        fit.params,
        x,
        wd=wd,
        eta_cd=eta_cd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )

    out = {
        "area_cd_plus": _trapz(curves["cd_plus"], x),
        "area_cd_minus": _trapz(curves["cd_minus"], x),
        "area_od_plus": _trapz(curves["od_plus"], x),
        "area_od_minus": _trapz(curves["od_minus"], x),
    }
    out["area_plus_total"] = out["area_cd_plus"] + out["area_od_plus"]
    out["area_minus_total"] = out["area_cd_minus"] + out["area_od_minus"]
    out["area_cd_total"] = out["area_cd_plus"] + out["area_cd_minus"]
    out["area_od_total"] = out["area_od_plus"] + out["area_od_minus"]
    out["area_total_physical"] = out["area_plus_total"] + out["area_minus_total"]
    out["area_diff_total"] = out["area_plus_total"] - out["area_minus_total"]

    if fit.pcov is None:
        return out

    # Monte-Carlo uncertainty propagation through the approximate covariance.
    rng = np.random.default_rng(seed)
    free_names = list(fit.free_names)
    try:
        if not np.all(np.isfinite(fit.pcov)):
            return out
        if np.linalg.cond(fit.pcov) > 1e12:
            return out
        L = np.linalg.cholesky(fit.pcov)
    except np.linalg.LinAlgError:
        return out

    base = np.array([fit.params[name] for name in free_names], dtype=float)
    draws = base + rng.standard_normal((n_mc, len(free_names))) @ L.T

    plus_vals, minus_vals = [], []
    for draw in draws:
        trial = dict(fit.params)
        for name, value in zip(free_names, draw):
            trial[name] = float(value)

        # Basic clipping into a reasonable physical region.
        trial["P"] = float(np.clip(trial["P"], -0.999, 0.999))
        trial["cc"] = float(np.clip(trial["cc"], 1e-9, np.inf))
        trial["split_cd"] = float(np.clip(trial["split_cd"], 1e-9, np.inf))
        trial["split_od"] = float(np.clip(trial["split_od"], 1e-9, np.inf))
        trial["sigma"] = float(np.clip(trial["sigma"], 1e-9, np.inf))
        trial["eta_od"] = float(np.clip(trial["eta_od"], 0.0, 0.999))
        trial["K"] = float(np.clip(trial["K"], 0.0, 1.0))

        curves_i = component_curves(
            trial,
            x,
            wd=wd,
            eta_cd=eta_cd,
            exact_intensity=exact_intensity,
            nphi=nphi,
        )
        plus_vals.append(_trapz(curves_i["plus_total"], x))
        minus_vals.append(_trapz(curves_i["minus_total"], x))

    plus_vals = np.asarray(plus_vals)
    minus_vals = np.asarray(minus_vals)
    out["area_plus_total_std"] = np.std(plus_vals, ddof=1)
    out["area_minus_total_std"] = np.std(minus_vals, ddof=1)
    out["area_diff_total_std"] = np.std(plus_vals - minus_vals, ddof=1)
    return out


# =============================================================================
# Plotting
# =============================================================================


def plot_fit(
    x,
    y,
    yerr,
    fit,
    *,
    title="Multi-site Dulya/Hamada fit",
    wd=16.35,
    eta_cd=0.0,
    exact_intensity=False,
    nphi=64,
    savepath="multisite_dulya_fit.png",
):
    """
    Plot data, fitted total, site-resolved curves, and residuals.

    Top panel shows:
      - data ± errors
      - total fitted signal (including Q-meter gain + baseline)
      - physical C-D site contribution (baseline excluded)
      - physical O-D site contribution (baseline excluded)
      - total plus-branch and minus-branch physical absorption
      - background

    Bottom panel shows residuals.
    """
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    yerr = None if yerr is None else _as_1d_float(yerr)

    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    yerrs = None if yerr is None else yerr[order]

    dense = np.linspace(xs.min(), xs.max(), max(800, 3 * len(xs)))
    curves = component_curves(
        fit.params,
        dense,
        wd=wd,
        eta_cd=eta_cd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )

    y_model_data = signal_model(
        x,
        **fit.params,
        wd=wd,
        eta_cd=eta_cd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )
    residuals = y - y_model_data

    areas = compute_transition_areas(
        dense,
        fit,
        wd=wd,
        eta_cd=eta_cd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )

    fig = plt.figure(figsize=(9.0, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    if yerrs is not None:
        ax.errorbar(xs, ys, yerr=yerrs, fmt="o", ms=3, alpha=0.85, label="data")
    else:
        ax.plot(xs, ys, "o", ms=3, alpha=0.85, label="data")

    ax.plot(dense, curves["total"], lw=2.2, label="total fit")
    ax.plot(dense, curves["cd_total"], lw=1.5, linestyle="--", label="C-D site")
    ax.plot(dense, curves["od_total"], lw=1.5, linestyle=":", label="O-D site")
    ax.plot(dense, curves["plus_total"], lw=1.2, linestyle="-.", label="plus branch")
    ax.plot(dense, curves["minus_total"], lw=1.2, linestyle=(0, (5, 2, 1, 2)), label="minus branch")
    ax.plot(dense, curves["background"], lw=1.0, alpha=0.8, label="background")

    ax.set_ylabel("Signal")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)

    txt = (
        f"P = {fit.params['P']:.5f}\n"
        f"r = {float(p_to_r(fit.params['P'])):.5f}\n"
        f"split_cd = {fit.params['split_cd']:.5g}\n"
        f"split_od = {fit.params['split_od']:.5g}\n"
        f"sigma = {fit.params['sigma']:.5g}\n"
        f"eta_od = {fit.params['eta_od']:.5g}\n"
        f"K = {fit.params['K']:.5g}\n"
        f"A_plus = {areas['area_plus_total']:.5g}\n"
        f"A_minus = {areas['area_minus_total']:.5g}\n"
        f"A_plus - A_minus = {areas['area_diff_total']:.5g}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.78, lw=0.5),
    )

    axr.axhline(0.0, lw=1.0, alpha=0.8)
    if yerrs is not None:
        axr.errorbar(x, residuals, yerr=yerr, fmt="o", ms=3, alpha=0.85)
    else:
        axr.plot(x, residuals, "o", ms=3, alpha=0.85)
    axr.set_xlabel("Frequency / offset")
    axr.set_ylabel("resid")
    axr.grid(alpha=0.25)

    for label in ax.get_xticklabels():
        label.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=170)
    plt.show()
    return areas


# =============================================================================
# Small demo
# =============================================================================
# This synthetic example is here only to make the file runnable as-is. Replace it
# with your real x/y/yerr arrays in actual analysis.
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Load one event row (500 bins on the usual R = offset grid), then keep
    # only samples with offset in [-1.6, 1.6] MHz (same units as split_*, wd).
    # -------------------------------------------------------------------------
    x_full = np.linspace(-6.0, 6.0, N_BINS)
    # _event_id, y_full = load_event_csv(Path("event_1432659714.csv"))
    _event_id, y_full = load_event_csv(Path("event_1432659714.csv"))
    fit_window = (-6, 6)
    m = (x_full >= fit_window[0]) & (x_full <= fit_window[1])
    x = x_full[m]
    y = y_full[m]
    yerr = np.full_like(x, 1.816364e-05, dtype=np.float64)
    # true = dict(
    #     P=0.44,
    #     amp=1.0,
    #     center=0.02,
    #     cc=1.0,
    #     split_cd=0.92,
    #     split_od=1.17,
    #     sigma=0.06,
    #     eta_od=0.17,
    #     K=0.13,
    #     xi=0.00,
    #     b0=0.0,
    #     b1=0.0,
    #     b2=0.0,
    #     b3=0.0,
    # )
    # y_true = signal_model(x, **true, exact_intensity=True)
    # yerr = np.full_like(x, 0.04)
    # y = y_true + rng.normal(0.0, yerr)

    # -------------------------------------------------------------------------
    # User-editable fit settings
    # -------------------------------------------------------------------------
    p0_full = dict(
        P=0.35,
        amp=0.9,
        center=0.0,
        cc=1.0,
        split_cd=0.88,
        split_od=1.12,
        sigma=0.05,
        eta_od=0.12,
        K=0.10,
        xi=0.0,
        b0=0.0,
        b1=0.0,
        b2=0.0,
        b3=0.0,
    )
    bounds_full = dict(
        P=(0.0, 0.95),
        amp=(-np.inf, np.inf),
        center=(-0.2, 0.2),
        cc=(0.0, 1.2),
        split_cd=(0.0, 1.5),
        split_od=(0.0, 1.8),
        sigma=(0.01, 0.20),
        eta_od=(0.0, 0.50),
        K=(0.0, 0.40),
        xi=(-0.15, 0.15),
        b0=(-1.0, 1.0),
        b1=(-1.0, 1.0),
        b2=(-1.0, 1.0),
        b3=(-1.0, 1.0),
    )

    # Example of how to freeze parameters:
    #   fixed_full = {"cc": 0.25}
    # or maybe after a TE study:
    #   fixed_full = {"xi": 0.048}
    fixed_full = {
        "cc": 1.0,
        "xi": 0.0,
        "b2": 0.0,
        "b3": 0.0,
    }

    # Central window used for the stage-1 C-D-only fit.
    clean_window = (-1.05, 1.05)

    # -------------------------------------------------------------------------
    # Two-stage fit
    # -------------------------------------------------------------------------
    main_fit, full_fit = fit_clean_then_full(
        x,
        y,
        yerr,
        p0_full,
        bounds_full,
        clean_window=clean_window,
        fixed_main={"cc": 1.0, "b2": 0.0, "b3": 0.0},
        fixed_full=fixed_full,
        wd=16.35,
        eta_cd=0.0,
        exact_intensity=True,
        nphi=48,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=300,
    )

    print("Stage-1 C-D-only fit:")
    for name in PARAM_ORDER:
        if name in main_fit.params:
            print(f"  {name:>8s} = {main_fit.params[name]:.8g}")

    print("\nStage-2 full physical C-D + O-D fit:")
    for name in PARAM_ORDER:
        if name in full_fit.params:
            print(f"  {name:>8s} = {full_fit.params[name]:.8g}")

    print("\nOptimizer success:", full_fit.result.success)
    print("Optimizer message:", full_fit.result.message)

    areas = plot_fit(
        x,
        y,
        yerr,
        full_fit,
        title="Physical multi-site fit: C-D + O-D composition",
        wd=16.35,
        eta_cd=0.0,
        exact_intensity=True,
        nphi=48,
        savepath="./multisite_fit_demo.png",
    )

    print("\nTransition-resolved areas (physical absorption, no baseline):")
    for key in sorted(areas):
        print(f"  {key:>22s} : {areas[key]}")
