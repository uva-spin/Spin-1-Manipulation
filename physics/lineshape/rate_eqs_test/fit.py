#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-site deuteron NMR lineshape fit (Dulya/Hamada formulation).

Same physical kernel as ``fit_multisite.py``, but for a clean single deuteron
site only — no C-D / O-D mixing and no impurity components.

Model
-----
1. Axis calibration:
       x_eff = cc * (x - center)
2. Single-site absorption from powder-averaged Dulya branch functions:
       chi'' = w_+ F_+ + w_- F_-
   with R = x_eff / split, A = sigma / |split|.
3. Optional Q-meter false-asymmetry:
       D = 1 + 0.5 * xi * (1 + R)
4. Optional cubic residual background.

Units
-----
Keep frequency-like quantities consistent (e.g. MHz offset):
    x, center, split, sigma, wd
For deuterons at ~2.5 T, wd ≈ 16.35 MHz.  ``split`` is 3*w_q; for eta ≈ 0
the main peaks sit near ±split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]  # physics/lineshape/rate_eqs_test -> repo root
_DULYA_FIT_PARAMS_PATH = _REPO_ROOT / "Data_Creation" / "dulya_fit" / "fit_params.json"

CSV_PATH = _HERE / "data" / "2014-08-03_17h59m47s-PolySignal.csv"
N_BINS = 500
N_COLS = 1 + N_BINS
# EVENT_NUMBER = 1406922665
EVENT_NUMBER = 1406922906
# Fixed TE polarization for this event (as in the previous fit.py / fit-dulya).
P_FIXED = 0.4612505

# Dimensionless R-axis used by the PolySignal CSV (same as prior fit.py).
R_AXIS = np.linspace(-6.0, 6.1, N_BINS)

PARAM_ORDER = (
    "P",
    "amp",
    "center",
    "cc",
    "split",
    "sigma",
    "eta",
    "xi",
    "b0",
    "b1",
    "b2",
    "b3",
)


# =============================================================================
# Data I/O
# =============================================================================


def load_polysignal(
    csv_path: Path = CSV_PATH,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Load PolySignal CSV: col0 = event number, cols1..500 = signal bins."""
    rows = []
    with csv_path.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != N_COLS:
                continue
            rows.append([float(x) for x in parts])
    raw = np.asarray(rows, dtype=np.float64)
    event_numbers = raw[:, 0].astype(np.int64)
    signal = raw[:, 1:]
    signal_by_event = {int(event): signal[i] for i, event in enumerate(event_numbers)}
    return event_numbers, signal, signal_by_event


# =============================================================================
# Numerical helpers
# =============================================================================


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def _as_1d_float(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _clip_small_positive(x, floor=1e-15):
    return np.clip(x, floor, None)


def p_to_r(P):
    """
    Deuteron vector polarization P → Dulya asymmetry r (weak-quadrupole).

        P = (r^2 - 1) / (r^2 + r + 1)
    """
    P = np.asarray(P, dtype=float)
    disc = np.clip(4.0 - 3.0 * P**2, 0.0, None)
    return (np.sqrt(disc) + P) / (2.0 * (1.0 - P))


def r_to_p(r):
    r = np.asarray(r, dtype=float)
    return (r**2 - 1.0) / (r**2 + r + 1.0)


# =============================================================================
# Dulya/Hamada branch kernel
# =============================================================================


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


# =============================================================================
# Intensity factors
# =============================================================================


def transition_weights(R, P, split, wd, exact_intensity=False):
    """
    Multiplicative weights for the plus / minus branches.

    exact_intensity=False → weak-quadrupole approx (constant r, 1).
    exact_intensity=True  → Dulya Eq. (24) frequency-dependent factors.
    """
    R = _as_1d_float(R)
    r = float(p_to_r(P))

    if not exact_intensity:
        return r * np.ones_like(R), np.ones_like(R)

    vartheta = abs(float(split)) / (3.0 * float(wd))
    plus = (r**2 - r ** (1.0 - 3.0 * vartheta * R)) / (r ** (1.0 - vartheta * R))
    minus = (r ** (1.0 + 3.0 * vartheta * R) - 1.0) / (r ** (1.0 + vartheta * R))
    return plus, minus


# =============================================================================
# Single-site absorption
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


def absorption_components(
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
    """Single-site physical absorption (plus / minus / total)."""
    plus, minus = site_transition_components(
        x_eff,
        P,
        split,
        sigma,
        eta,
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )
    return {
        "plus": plus,
        "minus": minus,
        "absorption": plus + minus,
    }


# =============================================================================
# Instrument / baseline
# =============================================================================


def polynomial_background(x, b0, b1, b2, b3):
    """Residual cubic background (Dulya Eq. 27/29)."""
    x = _as_1d_float(x)
    return b0 + b1 * x + b2 * x**2 + b3 * x**3


def qmeter_gain(x_eff, split_ref, xi):
    """False-asymmetry factor D = 1 + 0.5 * xi * (1 + R)."""
    x_eff = _as_1d_float(x_eff)
    Rq = x_eff / float(split_ref)
    return 1.0 + 0.5 * float(xi) * (1.0 + Rq)


def signal_model(
    x,
    P,
    amp,
    center,
    cc,
    split,
    sigma,
    eta,
    xi,
    b0,
    b1,
    b2,
    b3,
    *,
    wd=16.35,
    exact_intensity=False,
    nphi=64,
):
    """
    Full measurable single-site signal:

        amp * chi''(x_eff) * D(xi) + poly(x)
    """
    x = _as_1d_float(x)
    x_eff = float(cc) * (x - float(center))

    comps = absorption_components(
        x_eff,
        P,
        split,
        sigma,
        eta,
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )
    gain = qmeter_gain(x_eff, split, xi)
    background = polynomial_background(x, b0, b1, b2, b3)
    return float(amp) * comps["absorption"] * gain + background


def component_curves(
    params,
    x,
    *,
    wd=16.35,
    exact_intensity=False,
    nphi=64,
):
    """Named physical components on the measured x-grid (amp applied)."""
    x = _as_1d_float(x)
    p = dict(params)
    x_eff = float(p["cc"]) * (x - float(p["center"]))

    comps = absorption_components(
        x_eff,
        p["P"],
        p["split"],
        p["sigma"],
        p["eta"],
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )

    amp = float(p["amp"])
    for key in list(comps.keys()):
        comps[key] = amp * comps[key]

    gain = qmeter_gain(x_eff, p["split"], p["xi"])
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
# Fitting
# =============================================================================


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


def default_p0(*, p_guess: float = P_FIXED) -> dict[str, float]:
    return dict(
        P=float(p_guess),
        amp=1.0,
        center=0.0,
        cc=1.0,
        split=0.99,
        sigma=0.02,
        eta=0.05,
        xi=0.0,
        b0=0.0,
        b1=0.0,
        b2=0.0,
        b3=0.0,
    )


def default_bounds() -> dict[str, tuple[float, float]]:
    return dict(
        P=(0.0, 1.0),
        amp=(-np.inf, np.inf),
        center=(-0.3, 0.3),
        cc=(0.5, 1.5),
        split=(0.2, 2.0),
        sigma=(1e-5, 0.5),
        eta=(0.0, 0.5),
        # Wide enough that false-asymmetry is not forced onto a bound.
        xi=(-2.0, 2.0),
        b0=(-np.inf, np.inf),
        b1=(-np.inf, np.inf),
        b2=(-np.inf, np.inf),
        b3=(-np.inf, np.inf),
    )


def seed_p0_from_data(x, y, *, p_guess: float = P_FIXED) -> dict[str, float]:
    """Build a better starting point from the observed horn spacing / polarity."""
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    p0 = default_p0(p_guess=p_guess)

    # Prefer the two strongest extrema of opposite sign relative to baseline.
    # For the usual negative-going PolySignal doublet, use the deepest bins
    # near the expected horn locations |R| ~ 1.
    near = (np.abs(x) > 0.4) & (np.abs(x) < 1.8)
    if np.any(near):
        left = near & (x < 0.0)
        right = near & (x > 0.0)
        if np.any(left) and np.any(right):
            x_l = float(x[left][np.argmin(y[left])])
            x_r = float(x[right][np.argmin(y[right])])
            split0 = 0.5 * abs(x_r - x_l)
            if 0.3 < split0 < 1.8:
                p0["split"] = split0
            p0["center"] = 0.5 * (x_l + x_r)

    return p0


def residual_metrics(x, y, params, *, wd=16.35, exact_intensity=True, nphi=64):
    """Return rms / max-abs / SSE of model - data."""
    y_model = signal_model(
        x,
        **{k: params[k] for k in PARAM_ORDER},
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )
    resid = _as_1d_float(y_model) - _as_1d_float(y)
    return {
        "rms": float(np.sqrt(np.mean(resid**2))),
        "max_abs": float(np.max(np.abs(resid))),
        "sse": float(np.sum(resid**2)),
        "residuals": resid,
        "model": y_model,
    }


def fit_signal(
    x,
    y,
    yerr=None,
    p0=None,
    bounds=None,
    fixed=None,
    *,
    wd=16.35,
    exact_intensity=True,
    nphi=64,
    loss="linear",
    f_scale=1.0,
    max_nfev=80000,
):
    """
    Least-squares fit of the single-site Dulya/Hamada signal model.

    Defaults target residual minimization: ordinary least squares (``loss='linear'``)
    with Dulya's exact intensity weights.  Pass ``fixed={'P': P_TE}`` to hold
    polarization at a known TE value (usually worsens lineshape residuals).
    """
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    sigma_y = np.ones_like(y) if yerr is None else np.maximum(_as_1d_float(yerr), 1e-12)

    if p0 is None:
        p0 = seed_p0_from_data(x, y)
    else:
        p0 = dict(p0)
    if bounds is None:
        bounds = default_bounds()
    fixed = {} if fixed is None else dict(fixed)

    free_names = tuple(name for name in PARAM_ORDER if name not in fixed)
    if not free_names:
        raise ValueError("At least one parameter must be free.")

    # Always re-seed amplitude from a unit-amp trial when amp is free.
    if "amp" not in fixed:
        trial = dict(p0)
        trial.update(fixed)
        trial["amp"] = 1.0
        seed = signal_model(
            x,
            **{k: trial[k] for k in PARAM_ORDER},
            wd=wd,
            exact_intensity=exact_intensity,
            nphi=nphi,
        )
        a0 = float(np.dot(y, seed) / (np.dot(seed, seed) + 1e-30))
        if abs(a0) < 1e-12:
            a0 = -1.0 if float(np.sum(y)) < 0.0 else 1.0
        p0["amp"] = a0

    p0_vec = np.array([p0[name] for name in free_names], dtype=float)
    lb = np.array([bounds[name][0] for name in free_names], dtype=float)
    ub = np.array([bounds[name][1] for name in free_names], dtype=float)

    # Keep the start strictly inside bounds (least_squares is sensitive to this).
    p0_vec = np.minimum(np.maximum(p0_vec, lb + 1e-12), ub - 1e-12)

    def residuals(p_free):
        params = _merge_params(p_free, PARAM_ORDER, p0, fixed)
        y_model = signal_model(
            x,
            **{k: params[k] for k in PARAM_ORDER},
            wd=wd,
            exact_intensity=exact_intensity,
            nphi=nphi,
        )
        return (y_model - y) / sigma_y

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


def compute_transition_areas(
    x,
    fit,
    *,
    wd=16.35,
    exact_intensity=False,
    nphi=64,
):
    """Areas of physical plus/minus absorption (before Q-meter gain / baseline)."""
    x = _as_1d_float(x)
    curves = component_curves(
        fit.params,
        x,
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )
    area_plus = float(_trapz(curves["plus"], x))
    area_minus = float(_trapz(curves["minus"], x))
    return {
        "area_plus": area_plus,
        "area_minus": area_minus,
        "area_total_physical": area_plus + area_minus,
        "area_diff": area_plus - area_minus,
    }


# =============================================================================
# Plotting
# =============================================================================


def plot_fit(
    x,
    y,
    fit,
    *,
    title="Single-site Dulya/Hamada deuteron fit",
    wd=16.35,
    exact_intensity=True,
    nphi=64,
    savepath=None,
    yerr=None,
):
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    yerr = None if yerr is None else _as_1d_float(yerr)

    curves = component_curves(
        fit.params,
        x,
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )
    metrics = residual_metrics(
        x, y, fit.params, wd=wd, exact_intensity=exact_intensity, nphi=nphi
    )
    residuals = metrics["residuals"]
    areas = compute_transition_areas(
        x, fit, wd=wd, exact_intensity=exact_intensity, nphi=nphi
    )

    fig = plt.figure(figsize=(10.0, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="-", lw=1.0, alpha=0.85, label="data")
    else:
        ax.plot(x, y, lw=1.2, label="data")

    ax.plot(x, curves["total"], lw=2.0, label="total fit")
    ax.plot(x, curves["plus"], "--", lw=1.3, alpha=0.9, label="plus branch")
    ax.plot(x, curves["minus"], "--", lw=1.3, alpha=0.9, label="minus branch")
    ax.plot(x, curves["background"], lw=1.0, alpha=0.75, label="background")

    ax.set_ylabel("signal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", ncols=2)

    txt = (
        f"P = {fit.params['P']:.5g}\n"
        f"r = {float(p_to_r(fit.params['P'])):.5g}\n"
        f"amp = {fit.params['amp']:.5g}\n"
        f"split = {fit.params['split']:.5g}\n"
        f"sigma = {fit.params['sigma']:.5g}\n"
        f"eta = {fit.params['eta']:.5g}\n"
        f"xi = {fit.params['xi']:.5g}\n"
        f"rms = {metrics['rms']:.3e}\n"
        f"A+ = {areas['area_plus']:.5g}\n"
        f"A- = {areas['area_minus']:.5g}"
    )
    ax.text(
        0.98,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    axr.axhline(0.0, lw=1.0, alpha=0.8)
    axr.plot(x, residuals, lw=1.0, alpha=0.85)
    axr.set_xlabel("R / frequency offset")
    axr.set_ylabel("resid")
    axr.set_title(f"residuals  rms={metrics['rms']:.3e}  max={metrics['max_abs']:.3e}", fontsize=10)
    axr.grid(True, alpha=0.3)

    for label in ax.get_xticklabels():
        label.set_visible(False)

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=120)
    return areas, fig, metrics


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    _event_numbers, _signal, signal_by_event = load_polysignal()
    y = signal_by_event[EVENT_NUMBER]
    x = R_AXIS.copy()

    # Residual-minimizing setup:
    #   - free P (fixed TE P forces the wrong branch ratio and large peak residuals)
    #   - exact Dulya intensity weights
    #   - ordinary least squares
    #   - free cubic baseline; keep axis scale cc=1 on the PolySignal R-grid
    p0 = seed_p0_from_data(x, y, p_guess=P_FIXED)
    bounds = default_bounds()
    fixed = {"cc": 1.0}

    fit = fit_signal(
        x,
        y,
        p0=p0,
        bounds=bounds,
        fixed=fixed,
        wd=16.35,
        exact_intensity=True,
        nphi=64,
        loss="linear",
        max_nfev=80000,
    )
    metrics = residual_metrics(
        x, y, fit.params, wd=16.35, exact_intensity=True, nphi=64
    )

    print(f"event {EVENT_NUMBER}")
    print(f"  success = {fit.result.success}")
    print(f"  message = {fit.result.message}")
    print(f"  TE P ref = {P_FIXED:.6g}  (not fixed; lineshape P is free)")
    print(f"  rms      = {metrics['rms']:.6e}")
    print(f"  max|r|   = {metrics['max_abs']:.6e}")
    print(f"  sse      = {metrics['sse']:.6e}")
    for name in PARAM_ORDER:
        tag = " (fixed)" if name in fixed else ""
        print(f"  {name:8s} = {fit.params[name]:.6g}{tag}")

    # Freeze all non-P fit params for Dulya MC data generation (P is varied later).
    shape_export = {
        "amp": float(fit.params["amp"]),
        "center": float(fit.params["center"]),
        "cc": float(fit.params["cc"]),
        "split": float(fit.params["split"]),
        "sigma": float(fit.params["sigma"]),
        "eta": float(fit.params["eta"]),
        "xi": float(fit.params["xi"]),
        "b0": float(fit.params["b0"]),
        "b1": float(fit.params["b1"]),
        "b2": float(fit.params["b2"]),
        "b3": float(fit.params["b3"]),
    }
    _DULYA_FIT_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _DULYA_FIT_PARAMS_PATH.open("w", encoding="utf-8") as f:
        json.dump(shape_export, f, indent=2)
        f.write("\n")
    print(f"  saved fit params = {_DULYA_FIT_PARAMS_PATH}")

    out_path = _HERE / "data" / "fit_diag.png"
    areas, _fig, _metrics = plot_fit(
        x,
        y,
        fit,
        title=(
            f"Single-site Dulya fit — event {EVENT_NUMBER}\n"
        ),
        wd=16.35,
        exact_intensity=True,
        nphi=64,
        savepath=out_path,
    )
    print(f"  saved plot = {out_path}")
    print("  areas:")
    for key, val in areas.items():
        print(f"    {key:22s} = {val:.6g}")
    plt.show()
