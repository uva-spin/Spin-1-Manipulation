"""Per-bin P/Q calibration for organized train NPZs (not integrated lineshape P/Q)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from bin_setup import equilibrium_lineshape, get_shape_params  # noqa: E402
from common import F_MAX, F_MIN, NUM_BINS  # noqa: E402
from pq_calibration import (  # noqa: E402
    calibrate_bin_pq_targets,
    calibrated_pq_fields,
    integrated_pq_from_bins,
    load_pq_calibration,
    validate_stored_per_bin_pq,
)


@pytest.fixture(scope="module")
def calibration() -> dict:
    return load_pq_calibration(num_bins=NUM_BINS)


@pytest.fixture(scope="module")
def freq_axis() -> np.ndarray:
    return np.linspace(float(F_MIN), float(F_MAX), int(NUM_BINS))


def test_stored_pq_uses_cc_bin_not_cc_total(calibration: dict, freq_axis: np.ndarray) -> None:
    p0 = 0.48
    shape = get_shape_params()
    _, ip, im = equilibrium_lineshape(float(p0), freq_axis, shape)
    bin_idx = 210
    ps = np.asarray([float(ip[bin_idx] + im[bin_idx])], dtype=float)
    q = np.asarray([float(ip[bin_idx] - im[bin_idx])], dtype=float)
    p0_arr = np.asarray([p0], dtype=float)
    p_cal, q_cal = calibrate_bin_pq_targets(
        ps, q, p0_arr, calibration=calibration, post_correct=True
    )
    validate_stored_per_bin_pq(
        ps, q, p0_arr, p_cal, q_cal, calibration=calibration, post_correct=True
    )
    cc_total = float(calibration["cc"])
    cc_bin = float(calibration["cc_bin"])
    assert cc_bin == pytest.approx(1.0 / float(calibration["amp"]))
    assert cc_total * float(ps[0]) != pytest.approx(float(p_cal[0]), rel=0.05)
    assert float(p0) != pytest.approx(float(p_cal[0]), rel=0.01)
    p_int, _ = integrated_pq_from_bins(ip, im, calibration=calibration)
    assert float(p_cal[0]) != pytest.approx(p_int, rel=0.05)


def test_calibrated_pq_fields_one_row_per_sample(calibration: dict, freq_axis: np.ndarray) -> None:
    p0 = 0.48
    shape = get_shape_params()
    _, ip, im = equilibrium_lineshape(float(p0), freq_axis, shape)
    n_p = 3
    bin_idx = 210
    arrays = {
        "ps": np.full(n_p, float(ip[bin_idx] + im[bin_idx]), dtype=float),
        "q": np.full(n_p, float(ip[bin_idx] - im[bin_idx]), dtype=float),
        "p0": np.full(n_p, p0, dtype=float),
    }
    p_out, q_out = calibrated_pq_fields(arrays, num_bins=NUM_BINS, calibration=calibration)
    assert p_out.shape == (n_p,)
    assert q_out.shape == (n_p,)
    validate_stored_per_bin_pq(
        arrays["ps"],
        arrays["q"],
        arrays["p0"],
        p_out,
        q_out,
        calibration=calibration,
    )
