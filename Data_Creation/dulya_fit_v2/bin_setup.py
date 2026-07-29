"""Dulya-fit lineshape helpers for v2 per-bin workers."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

import _bootstrap  # noqa: F401
from common import (
    D_SAME_0MINUS,
    D_SAME_PLUS0,
    D_SPEC_0MINUS,
    D_SPEC_PLUS0,
    DIFFUSION_SCALE,
    F_MAX,
    F_MIN,
    FIT_PARAMS_PATH,
    NUM_BINS,
    PHYSICS_MODEL,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE,
    ZQ_WIDTH_R,
)
from lineshape import GenerateDulyaLineshape, shape_params_from_fit

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
    """Equilibrium I± from the fitted Dulya/Hamada formulation."""
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


def shape_meta(
    shape_params: dict[str, float] | None = None,
    *,
    rf_mode: str | None = None,
    gaussian_fwhm_R: float | None = None,
    lorentzian_fwhm_R: float | None = None,
    diffusion_scale: float | None = None,
) -> dict[str, Any]:
    """Provenance blob stored in NPZ ``meta_json`` fields."""
    shape = shape_params if shape_params is not None else get_shape_params()
    return {
        "lineshape_model": LINESHape_MODEL,
        "equilibrium_kernel": "GenerateDulyaLineshape",
        "fit_params_path": str(FIT_PARAMS_PATH),
        "physics_model": PHYSICS_MODEL,
        "rf_mode": str(RF_MODE if rf_mode is None else rf_mode),
        "rf_gaussian_fwhm_R": float(
            RF_GAUSSIAN_FWHM_R if gaussian_fwhm_R is None else gaussian_fwhm_R
        ),
        "rf_lorentzian_fwhm_R": float(
            RF_LORENTZIAN_FWHM_R if lorentzian_fwhm_R is None else lorentzian_fwhm_R
        ),
        "diffusion_scale": float(
            DIFFUSION_SCALE if diffusion_scale is None else diffusion_scale
        ),
        "zq_width_R": float(ZQ_WIDTH_R),
        "d_same_plus0": float(D_SAME_PLUS0),
        "d_same_0minus": float(D_SAME_0MINUS),
        "d_spec_plus0": float(D_SPEC_PLUS0),
        "d_spec_0minus": float(D_SPEC_0MINUS),
        "shape_params": {k: float(v) for k, v in shape.items()},
    }


def resolve_bin_idx(
    cli_bin_idx: int | None,
    *,
    num_bins: int = NUM_BINS,
) -> int | None:
    """Resolve a zero-indexed spectral bin (0 .. num_bins-1) from CLI or SLURM."""
    if cli_bin_idx is not None:
        bin_idx = int(cli_bin_idx)
    else:
        env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_idx is None or str(env_idx).strip() == "":
            return None
        bin_idx = int(env_idx)

    nb = int(num_bins)
    if bin_idx < 0 or bin_idx >= nb:
        raise ValueError(
            f"bin_idx={bin_idx} out of range for num_bins={nb} "
            f"(zero-indexed valid range 0..{nb - 1})"
        )
    return bin_idx


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
