"""Tests for gamma_rf / n_steps feature plumbing in single_bin."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "ml"
RIVANNA = REPO_ROOT / "Data_Creation" / "rivanna"
for path in (ML_DIR, RIVANNA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from combine_spectrum_train_bins import _slice_rows_at_bin
from pq_calibration import load_pq_calibration
from single_bin import (
    build_event_feature_matrix,
    build_feature_row,
    build_features,
    event_manipulation_features,
    resolve_gamma_rf_from_npz,
    resolve_n_steps_from_npz,
)


def test_event_manipulation_features_ssrf_vs_afp() -> None:
    g, s = event_manipulation_features(source=0, applied_power=10.0, n_steps=500.0)
    assert g == 10.0
    assert s == 500.0
    g2, s2 = event_manipulation_features(source=1, applied_power=10.0, n_steps=1.0)
    assert g2 == 0.0
    assert s2 == 0.0


def test_build_event_feature_matrix_global_params() -> None:
    ps = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    feats = build_event_feature_matrix(
        ps, ["gamma_rf", "n_steps", "ps"], gamma_rf=10.0, n_steps=500.0
    )
    assert feats.shape == (3, 3)
    assert np.allclose(feats[:, 0], 10.0)
    assert np.allclose(feats[:, 1], 500.0)
    assert np.allclose(feats[:, 2], ps)


def test_slice_rows_at_bin_includes_manipulation_fields() -> None:
    n = 4
    nb = 8
    rows = {
        "p0": np.linspace(0.1, 0.4, n, dtype=np.float32),
        "step": np.array([0, 50, 100, 150], dtype=np.int32),
        "center_bin": np.full(n, 200, dtype=np.int32),
        "source": np.zeros(n, dtype=np.uint8),
        "gamma_rf": np.full(n, 10.0, dtype=np.float32),
        "burn_steps": np.full(n, 150, dtype=np.int32),
        "ps": np.random.randn(n, nb).astype(np.float32),
        "iplus": np.random.rand(n, nb).astype(np.float32) * 0.01,
        "iminus": np.random.rand(n, nb).astype(np.float32) * 0.01,
    }
    rows["ps"] = rows["iplus"] + rows["iminus"]
    cal = load_pq_calibration(num_bins=nb)
    sliced = _slice_rows_at_bin(rows, 3, num_bins=nb, pq_calibration=cal)
    assert sliced["gamma_rf"].shape == (n,)
    assert np.allclose(sliced["gamma_rf"], 10.0)
    assert np.allclose(sliced["n_steps"], rows["step"].astype(np.float32))
    arrays = {
        "gamma_rf": sliced["gamma_rf"],
        "n_steps": sliced["n_steps"],
        "ps": sliced["ps"],
    }
    feats, names, ps_col = build_features(arrays)
    assert names == ["gamma_rf", "n_steps", "ps"]
    assert ps_col == 2
    assert feats.shape == (n, 3)


def test_resolve_fields_from_npz_keys(tmp_path: Path) -> None:
    path = tmp_path / "train_bin_0000.npz"
    np.savez(
        path,
        ps=np.array([1.0, 2.0], dtype=np.float32),
        p0=np.array([0.3, 0.5], dtype=np.float32),
        gamma_rf=np.array([10.0, 0.0], dtype=np.float32),
        n_steps=np.array([100.0, 0.0], dtype=np.float32),
        iplus=np.array([0.5, 0.6], dtype=np.float32),
        iminus=np.array([0.5, 0.4], dtype=np.float32),
    )
    with np.load(path) as data:
        assert np.allclose(resolve_gamma_rf_from_npz(data, path), [10.0, 0.0])
        assert np.allclose(resolve_n_steps_from_npz(data, path), [100.0, 0.0])


def test_build_feature_row() -> None:
    row = build_feature_row(
        ["gamma_rf", "n_steps", "ps"],
        gamma_rf=10.0,
        n_steps=500.0,
        ps=0.05,
    )
    assert np.allclose(row, [10.0, 500.0, 0.05])
