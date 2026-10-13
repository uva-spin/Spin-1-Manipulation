"""Tests for ideal-bin pulse programs (no ssRF-beta imports)."""
import numpy as np
import pytest
from physics.rf.pulse_program import BinPulse, PulseProgram, RFProfile, make_profile


def _program():
    grid = np.linspace(-3.0, 3.0, 21)
    program = PulseProgram(n_bins=21, r_min=-3.0, r_max=3.0, gain=1.0)
    program.profiles = [
        RFProfile(
            "A",
            pulses=[
                BinPulse(3, 2.0, 0.0, 0.5),
                BinPulse(5, 1.0, 0.2, 0.5),
            ],
        )
    ]
    program.validate()
    return program, grid


def test_json_round_trip(tmp_path):
    program, _ = _program()
    path = tmp_path / "prog.json"
    program.save(path)
    loaded = PulseProgram.load(path)
    assert loaded.n_bins == program.n_bins
    assert loaded.to_dict()["schema"] == "ssrf-ideal-bin-program-1"
    assert len(loaded.profiles[0].pulses) == 2


def test_csv_round_trip(tmp_path):
    program, _ = _program()
    path = tmp_path / "prog.csv"
    program.save_csv(path)
    loaded = PulseProgram.load_csv(path)
    assert loaded.n_bins == 21
    assert loaded.profiles[0].pulses[0].bin_index == 3


def test_additive_overlap():
    program, _ = _program()
    compiled = program.compile()
    field = compiled.field_at(0.25)
    assert field[3] == pytest.approx(2.0)
    assert field[5] == pytest.approx(1.0)
    assert compiled.field_at(0.6)[3] == pytest.approx(0.0)


def test_field_zero_after_last_stop():
    program, _ = _program()
    compiled = program.compile()
    assert compiled.end_time == pytest.approx(0.7)
    assert np.allclose(compiled.field_at(compiled.end_time), 0.0)
    assert np.allclose(compiled.field_at(compiled.end_time + 1.0), 0.0)


def test_asymmetric_grid_rejected():
    with pytest.raises(ValueError, match="symmetric"):
        PulseProgram(n_bins=21, r_min=-3.0, r_max=2.5).validate()


def test_make_profile_flat():
    grid = np.linspace(-3.0, 3.0, 21)
    profile = make_profile(grid, "flat", -0.3, 0.3, "flat", rate=4.0, duration=0.2)
    assert profile.pulses
    assert all(p.rate == pytest.approx(4.0) for p in profile.pulses)
