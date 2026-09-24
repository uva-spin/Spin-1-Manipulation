"""Synchronized ssRF-beta profile design (no ssRF-beta imports)."""
import ast
from pathlib import Path

import numpy as np
import pytest

from physics.rf.optimal_profile import (
    OptimizerSettings,
    design_optimal_profile,
    run_optimal_profile_polarization,
)
from physics.rf.profile_control import make_ideal_model


def test_optimal_profile_module_does_not_import_ssrf_beta():
    path = Path(__file__).resolve().parents[1] / "optimal_profile.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            assert "ssRF-beta" not in name
            assert "ssrf-beta" not in name.lower().replace("_", "-")
            assert "ssrf_realtime" not in name


def test_design_optimal_profile_returns_simultaneous_program():
    model = make_ideal_model(0.45, n_bins=21)
    q0 = float(model.polarizations()["Q"])
    settings = OptimizerSettings(
        duration_samples=3,
        duration_refinements=0,
        max_iterations=2,
        starts=1,
        max_wall_seconds=20.0,
        search_dt=0.01,
    )
    design = design_optimal_profile(model, settings=settings)
    assert design.rates.shape == (21,)
    assert design.mask.shape == (21,)
    assert design.duration >= 0.0
    assert np.isfinite(design.q_final)
    if design.duration > 0.0:
        pulses = [pulse for profile in design.program.profiles for pulse in profile.pulses if pulse.enabled and pulse.rate > 0.0]
        starts = {pulse.start for pulse in pulses}
        stops = {pulse.stop for pulse in pulses}
        assert starts <= {0.0}
        if stops:
            assert len(stops) == 1
            assert pytest.approx(design.duration) == next(iter(stops))
        assert design.q_final >= q0 - 1e-6


def test_run_optimal_profile_polarization_saves_each_step():
    settings = OptimizerSettings(
        duration_samples=3,
        duration_refinements=0,
        max_iterations=2,
        starts=1,
        max_wall_seconds=20.0,
        search_dt=0.01,
    )
    traj = run_optimal_profile_polarization(
        0.45,
        n_steps=4,
        n_bins=21,
        dt=0.02,
        settings=settings,
    )
    assert traj["skipped"] is False
    assert traj["iplus_full"].shape[1] == 21
    assert traj["iminus_full"].shape == traj["iplus_full"].shape
    assert traj["p_full"].shape[0] == traj["iplus_full"].shape[0]
    assert traj["q_full"].shape == traj["p_full"].shape
    assert traj["center_bin"] == -1
    np.testing.assert_allclose(traj["p_full"][0], 0.45, atol=1e-5)
    assert np.all(np.isfinite(traj["q_full"]))
    assert traj["n_steps"] >= 5
