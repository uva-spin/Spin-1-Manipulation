"""Integrated vector / tensor polarization from I± lineshapes."""

from __future__ import annotations

from typing import Any

import numpy as np

from lineshape import GenerateDulyaLineshape, shape_params_from_fit


def integrated_polarizations_raw(
    iplus: np.ndarray,
    iminus: np.ndarray,
) -> tuple[float, float]:
    """Direct sum of I± (fit signal units)."""
    ip = np.asarray(iplus, dtype=float)
    im = np.asarray(iminus, dtype=float)
    return float(np.sum(ip + im)), float(np.sum(ip - im))


def polarization_from_amp(
    p_sum: float,
    q_sum: float,
    amp: float,
    *,
    n_bins: int | None = None,
) -> tuple[float, float]:
    """
    Naive amp-scaled integration: ``sum(I±) / (amp * n_bins)``.

    At equilibrium this is proportional to ``P`` but nonlinear in |P|; apply
    :func:`apply_amp_post_correction` for the calibrated polarization.
    """
    denom = float(amp)
    if n_bins is not None:
        denom *= float(n_bins)
    if abs(denom) < 1e-30:
        raise ValueError("amp normalization factor is zero")
    return float(p_sum / denom), float(q_sum / denom)


def build_amp_integration_calibration(
    f: np.ndarray,
    shape_params: dict[str, float],
    *,
    p_min: float = -0.9,
    p_max: float = 0.9,
    p_step: float = 0.05,
) -> dict[str, np.ndarray | float | int]:
    """
    Equilibrium calibration for amp integration post-correction.

    Fit-scale Dulya lineshapes give ``p_amp = sum/(amp*n)`` that is monotonic
    in ``P`` but not equal to ``P``. The post-correction factor
    ``P / p_amp(P)`` (or equivalently inversion on ``p_amp``) recovers ``P``.
    """
    amp = float(shape_params["amp"])
    n_bins = int(np.asarray(f, dtype=float).size)
    p_input: list[float] = []
    p_amp_naive: list[float] = []
    correction: list[float] = []

    for p0 in np.arange(float(p_min), float(p_max) + 1e-12, float(p_step)):
        if abs(p0) < 1e-12:
            continue
        _, ip, im = GenerateDulyaLineshape(float(p0), f, shape_params)
        p_sum, q_sum = integrated_polarizations_raw(ip, im)
        p_amp, _ = polarization_from_amp(p_sum, q_sum, amp, n_bins=n_bins)
        p_input.append(float(p0))
        p_amp_naive.append(p_amp)
        correction.append(float(p0) / p_amp if abs(p_amp) > 1e-30 else 1.0)

    return {
        "p_input": np.asarray(p_input, dtype=float),
        "p_amp_naive": np.asarray(p_amp_naive, dtype=float),
        "correction": np.asarray(correction, dtype=float),
        "amp": amp,
        "n_bins": n_bins,
    }


def apply_amp_post_correction(
    p_naive: float,
    q_naive: float,
    calibration: dict[str, np.ndarray | float | int],
) -> tuple[float, float]:
    """
    Post-correct amp integration using the equilibrium calibration curve.

    Inverts ``p_amp = sum/(amp*n)`` via the precomputed monotone map, and scales
    ``Q`` by the same factor as ``P``.
    """
    p_amp_axis = np.asarray(calibration["p_amp_naive"], dtype=float)
    p_axis = np.asarray(calibration["p_input"], dtype=float)
    p_corr = float(np.interp(float(p_naive), p_amp_axis, p_axis))
    ratio = p_corr / float(p_naive) if abs(float(p_naive)) > 1e-30 else 1.0
    return p_corr, float(q_naive * ratio)


def integrated_polarizations(
    iplus: np.ndarray,
    iminus: np.ndarray,
    *,
    amp: float | None = None,
    n_bins: int | None = None,
    post_correct: bool = False,
    calibration: dict[str, np.ndarray | float | int] | None = None,
) -> tuple[float, float]:
    """
    Lineshape-integrated polarizations.

    Without ``amp``: raw sums ``sum(I+ + I-)``, ``sum(I+ - I-)``.
    With ``amp``: divide by ``amp`` (and ``n_bins`` on discrete grids).
    With ``post_correct=True``: apply equilibrium amp post-correction (requires
    ``calibration`` from :func:`build_amp_integration_calibration`).
    """
    p_sum, q_sum = integrated_polarizations_raw(iplus, iminus)
    if amp is None:
        return p_sum, q_sum
    p_naive, q_naive = polarization_from_amp(p_sum, q_sum, amp, n_bins=n_bins)
    if not post_correct:
        return p_naive, q_naive
    if calibration is None:
        raise ValueError("post_correct=True requires calibration")
    return apply_amp_post_correction(p_naive, q_naive, calibration)


def polarization_before_after(
    row: dict[str, Any],
    shape_params: dict[str, float] | None = None,
    *,
    post_correct: bool = True,
) -> dict[str, float]:
    """
    Before/after integrated P and Q for one dulya_fit event row.

    *Before* = equilibrium Dulya lineshape at ``P_initial`` (fit scale).
    *After*  = stored ``Iplus`` / ``Iminus`` (manipulated or same if none).

    ``P_vec`` / ``Q_tensor`` use amp integration with optional post-correction.
    """
    shape_params = shape_params or shape_params_from_fit()
    amp = float(shape_params["amp"])
    f = np.asarray(row["frequency"], dtype=float)
    n_bins = int(f.size)
    p0 = float(row["P_initial"])

    calibration = None
    if post_correct:
        calibration = build_amp_integration_calibration(
            f,
            shape_params,
            p_min=-0.9,
            p_max=0.9,
            p_step=0.01,
        )

    _, ip0, im0 = GenerateDulyaLineshape(p0, f, shape_params)
    p_before_raw, q_before_raw = integrated_polarizations_raw(ip0, im0)
    p_before, q_before = integrated_polarizations(
        ip0,
        im0,
        amp=amp,
        n_bins=n_bins,
        post_correct=post_correct,
        calibration=calibration,
    )

    ip1 = np.asarray(row["Iplus"], dtype=float)
    im1 = np.asarray(row["Iminus"], dtype=float)
    p_after_raw, q_after_raw = integrated_polarizations_raw(ip1, im1)
    p_after, q_after = integrated_polarizations(
        ip1,
        im1,
        amp=amp,
        n_bins=n_bins,
        post_correct=post_correct,
        calibration=calibration,
    )

    p_naive_before, _ = integrated_polarizations(ip0, im0, amp=amp, n_bins=n_bins)
    p_naive_after, _ = integrated_polarizations(ip1, im1, amp=amp, n_bins=n_bins)

    return {
        "P_input": p0,
        "amp": amp,
        "n_bins": n_bins,
        "P_vec_before": p_before,
        "Q_tensor_before": q_before,
        "P_vec_after": p_after,
        "Q_tensor_after": q_after,
        "P_vec_before_raw": p_before_raw,
        "Q_tensor_before_raw": q_before_raw,
        "P_vec_after_raw": p_after_raw,
        "Q_tensor_after_raw": q_after_raw,
        "P_amp_naive_before": p_naive_before,
        "P_amp_naive_after": p_naive_after,
        "dP_vec": p_after - p_before,
        "dQ_tensor": q_after - q_before,
    }
