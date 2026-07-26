"""Dulya/Hamada lineshape from frozen single-site fit.py parameters."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.rate_eqs_test.fit import (  # noqa: E402
    polynomial_background,
    qmeter_gain,
    site_transition_components,
)

DEFAULT_FIT_PARAMS_PATH = _HERE / "fit_params.json"

# All non-P knobs from fit.py PARAM_ORDER (P is varied by the generators).
SHAPE_KEYS = (
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


def load_fit_params(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_FIT_PARAMS_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shape_params_from_fit(fit_blob: dict[str, Any] | None = None) -> dict[str, float]:
    """Frozen fit.py params used for all polarizations."""
    p = fit_blob if fit_blob is not None else load_fit_params()
    missing = [k for k in SHAPE_KEYS if k not in p]
    if missing:
        raise KeyError(f"fit_params missing required keys {missing}; got {sorted(p)}")
    return {k: float(p[k]) for k in SHAPE_KEYS}


def _build_dulya_intensities(
    P: float,
    x: np.ndarray,
    shape_params: dict[str, float],
    *,
    wd: float = 16.35,
    exact_intensity: bool = True,
    nphi: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit-scale Ps, I+, I- before optional P normalization."""
    P_clip = float(np.clip(P, -0.999999, 0.999999))
    amp = float(shape_params["amp"])
    split = float(shape_params["split"])
    sigma = float(shape_params["sigma"])
    eta = float(shape_params["eta"])
    xi = float(shape_params["xi"])

    x_eff = float(shape_params["cc"]) * (x - float(shape_params["center"]))
    plus, minus = site_transition_components(
        x_eff,
        P_clip,
        split,
        sigma,
        eta,
        wd=float(wd),
        exact_intensity=bool(exact_intensity),
        nphi=int(nphi),
    )
    gain = qmeter_gain(x_eff, split, xi)
    background = polynomial_background(
        x,
        shape_params["b0"],
        shape_params["b1"],
        shape_params["b2"],
        shape_params["b3"],
    )

    iplus = amp * np.asarray(plus, dtype=float) * gain + 0.5 * background
    iminus = amp * np.asarray(minus, dtype=float) * gain + 0.5 * background
    ps = iplus + iminus
    return ps, iplus, iminus


def GenerateDulyaLineshape(
    P: float,
    x: np.ndarray,
    shape_params: dict[str, float] | None = None,
    *,
    wd: float = 16.35,
    exact_intensity: bool = True,
    nphi: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    I+/I- at fit.py signal scale (``signal_model`` / ``component_curves``).

    Polarization ``P`` sets Dulya branch weights only. Intensities use the fitted
    ``amp``, Q-meter gain, and baseline — **no** rescaling of ``sum(I+ + I-)`` to
    ``P``. Compare integrated polarizations to ``P_input`` via ``amp`` (see
    ``polarization.integrated_polarizations``).
    """
    P = float(P)
    x = np.asarray(x, dtype=float).reshape(-1)
    if shape_params is None:
        shape_params = shape_params_from_fit()

    if abs(P) < 1e-15:
        z = np.zeros_like(x, dtype=float)
        return z, z.copy(), z.copy()

    return _build_dulya_intensities(
        P,
        x,
        shape_params,
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )


def GenerateDulyaLineshapeFitScale(
    P: float,
    x: np.ndarray,
    shape_params: dict[str, float] | None = None,
    *,
    wd: float = 16.35,
    exact_intensity: bool = True,
    nphi: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alias for :func:`GenerateDulyaLineshape` (fit-scale, no area rescaling)."""
    return GenerateDulyaLineshape(
        P,
        x,
        shape_params,
        wd=wd,
        exact_intensity=exact_intensity,
        nphi=nphi,
    )
