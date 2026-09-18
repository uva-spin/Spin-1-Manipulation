"""Tests for IdealBinModel pulse playback (no ssRF-beta imports)."""
import numpy as np
import pytest
from physics.rf.ideal_model import IdealBinModel, IdealBinParams
from physics.rf.pulse_program import BinPulse, PulseProgram, RFProfile


def _small_params(**overrides):
    base = dict(
        n_bins=21,
        r_min=-3.0,
        r_max=3.0,
        p0=0.45,
        dt=0.01,
        diffusion_enabled=False,
        diffusion_scale=0.0,
        relax_enabled=False,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
        rf_enabled=False,
    )
    base.update(overrides)
    return IdealBinParams(**base)


def test_ideal_params_zero_voigt_widths():
    p = IdealBinParams()
    assert p.rf_gaussian_fwhm_R == 0.0
    assert p.rf_lorentzian_fwhm_R == 0.0


def test_single_bin_pulse_changes_commanded_bin():
    params = _small_params()
    model = IdealBinModel(params)
    k = int(np.argmin(np.abs(model.Rplus - 0.9)))
    n0 = model.n.copy()
    q0 = model.polarizations()["Q"]
    program = PulseProgram(
        n_bins=21,
        r_min=-3.0,
        r_max=3.0,
        profiles=[RFProfile("one", pulses=[BinPulse(k, 4.0, 0.0, 0.2)])],
    )
    t_before = model.t
    n_before = model.n.copy()
    model.set_program(program)
    assert model.t == pytest.approx(t_before)
    assert np.allclose(model.n, n_before)
    model.start_program()
    model.step(25, rf_on=True, dnp_on=False)
    assert not np.allclose(model.n[k], n0[k])
    assert model.polarizations()["Q"] != pytest.approx(q0)


def test_mirror_populations_follow_shared_n():
    params = _small_params()
    model = IdealBinModel(params)
    k = int(np.argmin(np.abs(model.Rplus - 1.2)))
    program = PulseProgram(
        n_bins=21,
        r_min=-3.0,
        r_max=3.0,
        profiles=[RFProfile("one", pulses=[BinPulse(k, 5.0, 0.0, 0.15)])],
    )
    model.set_program(program)
    model.start_program()
    plus, minus = model.rf_profile_arrays()
    u = model.commanded_rf_field()
    assert u[k] > 0.0
    assert np.allclose(plus, u)
    assert np.allclose(minus, u[::-1])


def test_program_start_does_not_reset_n():
    model = IdealBinModel(_small_params())
    model.n *= 0.9
    n_saved = model.n.copy()
    k = 4
    program = PulseProgram(
        n_bins=21,
        r_min=-3.0,
        r_max=3.0,
        profiles=[RFProfile("one", pulses=[BinPulse(k, 2.0, 0.0, 0.1)])],
    )
    model.set_program(program)
    model.start_program()
    assert np.allclose(model.n, n_saved)


def test_finished_program_applies_zero_rf_while_time_advances():
    model = IdealBinModel(_small_params(dt=0.02))
    k = 6
    program = PulseProgram(
        n_bins=21,
        r_min=-3.0,
        r_max=3.0,
        profiles=[RFProfile("one", pulses=[BinPulse(k, 3.0, 0.0, 0.04)])],
    )
    model.set_program(program)
    model.start_program()
    model.step(5)
    assert model.program_state == "finished"
    t_fin = model.t
    field = model.commanded_rf_field()
    assert np.allclose(field, 0.0)
    n_fin = model.n.copy()
    model.step(3, rf_on=False, dnp_on=False)
    assert model.t > t_fin
    assert np.allclose(model.n, n_fin)
