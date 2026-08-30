import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from common import FIT_PARAMS_PATH, NUM_BINS

PQ_CALIBRATION_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "ml" / "data" / "pq_amp_calibration.npz",
    Path(__file__).resolve().parent / "data" / "pq_amp_calibration.npz",
)

P_NAIVE_EQ_P_STEP = 0.01
P_NAIVE_EQ_P_MIN = -0.9
P_NAIVE_EQ_P_MAX = 0.9

_PQ_CALIB_CACHE: Dict[tuple, Dict[str, Union[float, int, np.ndarray]]] = {}


def _load_shape_amp(fit_params_path: Optional[Path] = None) -> float:
    path = Path(fit_params_path) if fit_params_path is not None else FIT_PARAMS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"fit_params not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        blob = json.load(handle)
    if "amp" not in blob:
        raise KeyError(f"{path}: missing 'amp'")
    return float(blob["amp"])


def _build_p_naive_eq_table(
    *,
    num_bins: int,
    fit_params_path: Optional[Path] = None,
    p_min: float = P_NAIVE_EQ_P_MIN,
    p_max: float = P_NAIVE_EQ_P_MAX,
    p_step: float = P_NAIVE_EQ_P_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    from lineshape import GenerateDulyaLineshape, shape_params_from_fit

    amp = _load_shape_amp(fit_params_path)
    cc_total = 1.0 / (float(amp) * float(num_bins))
    shape = shape_params_from_fit()
    freq_axis = np.linspace(-6.0, 6.0, int(num_bins))
    p_input: list[float] = []
    p_naive_eq: list[float] = []
    for p0 in np.arange(float(p_min), float(p_max) + 1e-12, float(p_step)):
        if abs(float(p0)) < 1e-12:
            continue
        _, ip, im = GenerateDulyaLineshape(float(p0), freq_axis, shape)
        ip_arr = np.asarray(ip, dtype=np.float64)
        im_arr = np.asarray(im, dtype=np.float64)
        p_n = float(cc_total * np.sum(ip_arr + im_arr))
        p_input.append(float(p0))
        p_naive_eq.append(p_n)
    return (
        np.asarray(p_input, dtype=np.float64),
        np.asarray(p_naive_eq, dtype=np.float64),
    )


def load_pq_calibration(
    *,
    num_bins: int = NUM_BINS,
    fit_params_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    build_p_naive_eq: bool = True,
) -> Dict[str, Union[float, int, np.ndarray]]:
    """Discretized CC constants plus equilibrium naive-P table for post-correction."""
    cache_key = (
        int(num_bins),
        str(Path(fit_params_path).resolve()) if fit_params_path is not None else None,
        str(Path(calibration_path).resolve()) if calibration_path is not None else None,
        bool(build_p_naive_eq),
    )
    if cache_key in _PQ_CALIB_CACHE:
        return _PQ_CALIB_CACHE[cache_key]

    amp = _load_shape_amp(fit_params_path)
    cc_total = 1.0 / (float(amp) * float(num_bins))
    cc_bin = cc_total * float(num_bins)
    cal: Dict[str, Union[float, int, np.ndarray]] = {
        "amp": float(amp),
        "num_bins": int(num_bins),
        "cc": float(cc_total),
        "cc_bin": float(cc_bin),
    }

    cal_path: Optional[Path] = None
    if calibration_path is not None:
        cal_path = Path(calibration_path)
    else:
        for candidate in PQ_CALIBRATION_CANDIDATES:
            if candidate.is_file():
                cal_path = candidate
                break

    if cal_path is not None and cal_path.is_file():
        with np.load(cal_path, allow_pickle=False) as data:
            if "p_input" in data.files and "p_naive_eq" in data.files:
                cal["p_input"] = np.asarray(data["p_input"], dtype=np.float64)
                cal["p_naive_eq"] = np.asarray(data["p_naive_eq"], dtype=np.float64)
            loaded_amp = float(np.asarray(data["amp"]).reshape(()))
            if abs(loaded_amp - float(amp)) > 1e-12 * max(abs(float(amp)), 1.0):
                raise ValueError(
                    f"{cal_path}: amp={loaded_amp} != fit_params amp={amp}; regenerate table"
                )

    if build_p_naive_eq and "p_naive_eq" not in cal:
        p_input, p_naive_eq = _build_p_naive_eq_table(
            num_bins=int(num_bins),
            fit_params_path=fit_params_path,
        )
        cal["p_input"] = p_input
        cal["p_naive_eq"] = p_naive_eq

    _PQ_CALIB_CACHE[cache_key] = cal
    return cal


def equilibrium_naive_p_at_p0(
    p0: np.ndarray,
    calibration: Dict[str, Union[float, int, np.ndarray]],
) -> np.ndarray:
    p0_arr = np.asarray(p0, dtype=np.float64)
    if "p_input" not in calibration or "p_naive_eq" not in calibration:
        raise KeyError("calibration missing p_naive_eq table")
    p_axis = np.asarray(calibration["p_input"], dtype=np.float64)
    p_naive_axis = np.asarray(calibration["p_naive_eq"], dtype=np.float64)
    return np.interp(p0_arr, p_axis, p_naive_axis)


def post_correct_ratio(
    p0: np.ndarray,
    calibration: Dict[str, Union[float, int, np.ndarray]],
) -> np.ndarray:
    p0_arr = np.asarray(p0, dtype=np.float64)
    p_naive_eq = equilibrium_naive_p_at_p0(p0_arr, calibration)
    return np.where(np.abs(p_naive_eq) > 1e-30, p0_arr / p_naive_eq, 1.0)


def calibrate_bin_pq_targets(
    ps: np.ndarray,
    q: np.ndarray,
    p0: np.ndarray,
    *,
    calibration: Dict[str, Union[float, int, np.ndarray]],
    post_correct: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin CC_bin scaling with optional p0 post-correction (same ratio on P and Q).

    Each output element is the calibrated polarization at **one spectral bin**:
      P_b = CC_bin * ps_b  (optionally * post_correct_ratio(p0))
    where ``ps_b = I+(b) + I-(b)`` at that bin only.

    Do **not** use ``CC_total`` here; that scale applies only to full-spectrum sums.
    """
    cc_bin = float(calibration["cc_bin"])
    ps_arr = np.asarray(ps, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if ps_arr.ndim != q_arr.ndim:
        raise ValueError("ps and q must have the same shape")
    p_out = cc_bin * ps_arr
    q_out = cc_bin * q_arr
    if post_correct:
        ratio = post_correct_ratio(np.asarray(p0, dtype=np.float64), calibration)
        p_out = p_out * ratio
        q_out = q_out * ratio
    return p_out.astype(np.float64, copy=False), q_out.astype(np.float64, copy=False)


def integrated_pq_from_bins(
    ps: np.ndarray,
    q: np.ndarray,
    *,
    calibration: Dict[str, Union[float, int, np.ndarray]],
) -> tuple[float, float]:
    """Full-spectrum integrated P/Q: CC_total * sum_b(ps_b), sum_b(q_b)."""
    cc_total = float(calibration["cc"])
    ps_arr = np.asarray(ps, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    return (
        float(cc_total * np.sum(ps_arr)),
        float(cc_total * np.sum(q_arr)),
    )


def validate_stored_per_bin_pq(
    ps: np.ndarray,
    q: np.ndarray,
    p0: np.ndarray,
    p_cal: np.ndarray,
    q_cal: np.ndarray,
    *,
    calibration: Dict[str, Union[float, int, np.ndarray]],
    post_correct: bool = True,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> None:
    """Assert stored P/Q match per-bin CC_bin calibration (not integrated lineshape P/Q)."""
    ps_arr = np.asarray(ps, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    if ps_arr.ndim != 1:
        raise ValueError(
            f"validate_stored_per_bin_pq expects 1d per-bin samples, got ps.ndim={ps_arr.ndim}"
        )
    exp_p, exp_q = calibrate_bin_pq_targets(
        ps_arr,
        q_arr,
        np.asarray(p0, dtype=np.float64),
        calibration=calibration,
        post_correct=post_correct,
    )
    np.testing.assert_allclose(p_cal, exp_p, rtol=rtol, atol=atol)
    np.testing.assert_allclose(q_cal, exp_q, rtol=rtol, atol=atol)
    cc_total = float(calibration["cc"])
    wrong_total_scale = cc_total * ps_arr
    if np.any(np.abs(wrong_total_scale) > 1e-12):
        assert not np.allclose(
            p_cal,
            wrong_total_scale,
            rtol=0.05,
            atol=1e-12,
        ), "P looks like CC_total*ps (integrated scale) instead of CC_bin*ps (per-bin)"
    p0_arr = np.asarray(p0, dtype=np.float64)
    if np.any(np.abs(p0_arr) > 1e-6):
        assert not np.allclose(
            p_cal,
            p0_arr,
            rtol=0.01,
            atol=1e-6,
        ), "P must be per-bin calibrated values, not input polarization p0"


def calibrated_pq_fields(
    arrays: dict[str, np.ndarray],
    *,
    num_bins: int = NUM_BINS,
    calibration: Optional[Dict[str, Union[float, int, np.ndarray]]] = None,
    post_correct: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CC-scaled true P and Q at **this bin** (one value per sample row).

    Uses ``CC_bin = 1/amp`` on each row's ``ps`` / ``q`` (local I± sums at this bin).
    Integrated lineshape P/Q would use ``CC_total`` on the sum over all bins instead.
    """
    ps = np.asarray(arrays["ps"], dtype=np.float64)
    if ps.ndim != 1:
        raise ValueError(
            f"calibrated_pq_fields expects scalar per-bin ps (ndim=1), got shape {ps.shape}"
        )
    if "q" in arrays:
        q = np.asarray(arrays["q"], dtype=np.float64)
    elif "iplus" in arrays and "iminus" in arrays:
        q = np.asarray(arrays["iplus"], dtype=np.float64) - np.asarray(
            arrays["iminus"], dtype=np.float64
        )
    else:
        raise KeyError("arrays need 'q' or both 'iplus' and 'iminus'")
    p0 = np.asarray(arrays["p0"], dtype=np.float64)
    cal = calibration or load_pq_calibration(num_bins=int(num_bins))
    return calibrate_bin_pq_targets(
        ps,
        q,
        p0,
        calibration=cal,
        post_correct=post_correct,
    )


def calibrated_pq_spectrum(
    ps: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    p0: float,
    *,
    num_bins: int = NUM_BINS,
    calibration: Optional[Dict[str, Union[float, int, np.ndarray]]] = None,
    post_correct: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """CC-calibrated P and Q at every spectral bin for one full-spectrum event."""
    ps_arr = np.asarray(ps, dtype=np.float64).reshape(-1)
    ip_arr = np.asarray(iplus, dtype=np.float64).reshape(-1)
    im_arr = np.asarray(iminus, dtype=np.float64).reshape(-1)
    if not (ps_arr.shape == ip_arr.shape == im_arr.shape):
        raise ValueError("ps, iplus, iminus must have the same shape")
    q_arr = ip_arr - im_arr
    p0_arr = np.full(ps_arr.shape, float(p0), dtype=np.float64)
    cal = calibration or load_pq_calibration(num_bins=int(num_bins))
    return calibrate_bin_pq_targets(
        ps_arr,
        q_arr,
        p0_arr,
        calibration=cal,
        post_correct=post_correct,
    )
