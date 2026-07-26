"""Shared Dulya-fit lineshape helpers for per-bin traj workers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import F_MAX, F_MIN  # noqa: E402
from lineshape import GenerateDulyaLineshape, shape_params_from_fit  # noqa: E402

_SHAPE_PARAMS: dict[str, float] | None = None

LINESHape_MODEL = "dulya_fit"


def get_shape_params() -> dict[str, float]:
    """Frozen fit params from ``fit_params.json``."""
    global _SHAPE_PARAMS
    if _SHAPE_PARAMS is None:
        _SHAPE_PARAMS = shape_params_from_fit()
    return dict(_SHAPE_PARAMS)


def equilibrium_lineshape(
    P: float,
    f: np.ndarray,
    shape_params: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Equilibrium I± from the fitted Dulya/Hamada formulation (``fit.py`` kernel).

    All bin workers (unmanipulated / ssRF / AFP) must use this for initial spectra.
    """
    return GenerateDulyaLineshape(
        float(P),
        np.asarray(f, dtype=float),
        shape_params if shape_params is not None else get_shape_params(),
    )


def spin1_scale_factors(P: float, iplus: np.ndarray, iminus: np.ndarray) -> tuple[float, float]:
    """Map fit-scale intensities to Spin1 units (sum ~ P) and back."""
    area_fit = float(np.sum(np.asarray(iplus, dtype=float) + np.asarray(iminus, dtype=float)))
    P = float(P)
    if abs(area_fit) > 1e-30 and abs(P) > 1e-15:
        return P / area_fit, area_fit / P
    return 1.0, 1.0


def generate_unmanipulated_cube(
    *,
    num_bins: int,
    p_min: float,
    p_max: float,
    p_step: float,
    shape_params: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """(n_p, n_bins) equilibrium cubes via ``GenerateDulyaLineshape``."""
    shape = shape_params if shape_params is not None else get_shape_params()
    p_values = polarization_grid(p_min, p_max, p_step)
    f = np.linspace(float(F_MIN), float(F_MAX), int(num_bins))
    n_p = int(p_values.size)
    n_bins_i = int(num_bins)

    ps = np.zeros((n_p, n_bins_i), dtype=float)
    iplus = np.zeros((n_p, n_bins_i), dtype=float)
    iminus = np.zeros((n_p, n_bins_i), dtype=float)

    for j, p0 in enumerate(p_values):
        if (j + 1) % 50 == 0 or j == 0 or j == n_p - 1:
            print(
                f"  GenerateDulyaLineshape P={p0:+.3f} ({j + 1}/{n_p})",
                flush=True,
            )
        signal, ip, im = equilibrium_lineshape(float(p0), f, shape)
        ps[j] = np.asarray(signal, dtype=float)
        iplus[j] = np.asarray(ip, dtype=float)
        iminus[j] = np.asarray(im, dtype=float)

    return {
        "p_values": p_values,
        "ps": ps,
        "iplus": iplus,
        "iminus": iminus,
        "amp": np.abs(ps),
        "R": f,
        "shape_params": shape,
    }


def shape_meta(shape_params: dict[str, float] | None = None) -> dict[str, Any]:
    """Provenance blob stored in NPZ ``meta_json`` fields."""
    shape = shape_params if shape_params is not None else get_shape_params()
    return {
        "lineshape_model": LINESHape_MODEL,
        "equilibrium_kernel": "GenerateDulyaLineshape",
        "fit_params_path": str(_HERE / "fit_params.json"),
        "shape_params": {k: float(v) for k, v in shape.items()},
    }


def resolve_bin_idx(cli_bin_idx: int | None) -> int | None:
    if cli_bin_idx is not None:
        return int(cli_bin_idx)
    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_idx is not None and str(env_idx).strip() != "":
        return int(env_idx)
    return None


def polarization_grid(p_min: float, p_max: float, p_step: float) -> np.ndarray:
    g = np.arange(float(p_min), float(p_max) + 1e-12, float(p_step), dtype=float)
    return g[np.abs(g) >= 1e-12]


def print_shape_banner(shape: dict[str, float], *, num_bins: int) -> None:
    print(
        "Dulya fitted lineshape:",
        ", ".join(f"{k}={v:.6g}" for k, v in shape.items()),
        flush=True,
    )
    print(f"R grid: [{F_MIN}, {F_MAX}]  n={num_bins}", flush=True)
