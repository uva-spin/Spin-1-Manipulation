"""LabVIEW Dulya / ssRF sim node — InitSim / main command API."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from physics.rf.optimal_profile import OptimizerSettings

from ml.labview_sim import labview_sim as lv

FIT = Path(__file__).resolve().parents[3] / 'Data_Creation' / 'fit_params.json'
N_BINS = 81


@pytest.fixture(autouse=True)
def _reset_session():
    lv.model = None
    lv.sim_time = 0.0
    lv._init_kwargs = None
    lv._last_freq = None
    lv._last_intensity = None
    lv._last_p = 0.0
    lv._last_q = 0.0
    lv._optimizer_settings = OptimizerSettings(
        duration_samples=3,
        duration_refinements=0,
        max_iterations=2,
        starts=1,
        max_wall_seconds=20.0,
        search_dt=0.01,
    )
    yield
    lv.model = None
    lv._optimizer_settings = None


def _init(p0=0.45, n_bins=N_BINS, fit_params_path=''):
    """Default: analytic Pake (create_data without --fit-params)."""
    return lv.InitSim(fit_params_path, p0, Q0=-1.0, n_bins=n_bins)


def test_main_without_init_returns_error_status():
    freq, intensity, p, q, t, status = lv.main(0)
    assert status.startswith('error:')
    assert freq.size == 0
    assert intensity.size == 0


def test_init_sim_returns_finite_spectrum():
    freq, intensity, p, q, t, status = _init()
    assert status == 'ok'
    assert freq.shape == intensity.shape
    assert freq.size == N_BINS
    assert np.all(np.isfinite(freq))
    assert np.all(np.isfinite(intensity))
    assert abs(p - 0.45) < 0.08
    assert np.isfinite(q)
    assert lv._init_kwargs['fit_params_path'] is None


def test_init_sim_optional_dulya_fit():
    assert FIT.is_file(), f'missing fit params: {FIT}'
    freq, intensity, p, q, t, status = _init(fit_params_path=str(FIT))
    assert status == 'ok'
    assert freq.size == N_BINS
    assert np.all(np.isfinite(intensity))
    assert abs(p - 0.45) < 0.08
    assert lv._init_kwargs['fit_params_path'] == str(FIT)


def test_ssrf_changes_spectrum():
    _init()
    freq0, i0, *_ = lv.main(0)
    burn_R = float(freq0[len(freq0) // 4])
    freq1, i1, p1, q1, t1, status = lv.main(
        2, burn_R=burn_R, gamma_rf=2.0, n_steps=15, rf_mode=0
    )
    assert status == 'ok'
    assert freq1.shape == i1.shape
    assert not np.allclose(i0, i1)


def test_afp_preserves_vector_p():
    """Match physics/rf AFP invariant: |ΔP| < 0.2 on a Q<0 sweep."""
    _init()
    _, _, p0, _, _, _ = lv.main(0)
    _, _, p1, _, _, status = lv.main(3, afp_mode=0, efficiency=1.0)
    assert status == 'ok'
    assert abs(p1 - p0) < 0.2


def test_step_moves_state_after_burn():
    _init()
    freq0, i0, *_ = lv.main(0)
    burn_R = float(freq0[len(freq0) // 4])
    lv.main(2, burn_R=burn_R, gamma_rf=2.0, n_steps=12)
    _, i_burn, p_burn, q_burn, _, _ = lv.main(0)
    _, i_step, p_step, q_step, _, status = lv.main(1, dt=0.0015, n_steps=25)
    assert status == 'ok'
    moved = (
        not np.allclose(i_burn, i_step)
        or abs(p_step - p_burn) > 1e-8
        or abs(q_step - q_burn) > 1e-8
    )
    assert moved


def test_afp_profile_and_reset():
    _init()
    _, i0, p0, _, _, _ = lv.main(0)
    _, _, _, _, _, status = lv.main(5, efficiency=1.0)
    assert status == 'ok'
    _, i_reset, p_reset, _, _, status = lv.main(6)
    assert status == 'ok'
    assert p_reset == pytest.approx(p0, rel=0.05, abs=0.05)
    assert np.allclose(i_reset, i0, rtol=1e-5, atol=1e-8)


def test_unknown_cmd_status():
    _init()
    _, _, _, _, _, status = lv.main(99)
    assert 'unknown cmd' in status


def test_ssrf_profile_runs():
    _init(n_bins=21)
    _, _, _, _, _, status = lv.main(4, n_steps=2)
    assert status.startswith('ok')
    freq, intensity, p, q, t, status2 = lv.main(0)
    assert status2 == 'ok'
    assert freq.size == 21
    assert np.all(np.isfinite(intensity))
    assert np.isfinite(p) and np.isfinite(q)
