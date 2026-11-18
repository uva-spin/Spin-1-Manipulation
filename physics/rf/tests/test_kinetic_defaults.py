"""Optimal profile uses the same ssRF-beta defaults as data generation."""
import numpy as np
import pytest

from Data_Creation.spectrum_gen.bridge import build_equilibrium_spin1_model
from physics.rf.lineshape import boltzmann_Q
from physics.rf.profile_control import make_ideal_model
from physics.rf.kinetic_defaults import ssrf_beta_spin1_params


def test_make_ideal_model_matches_bridge_spin1_params():
    p0 = 0.45
    n_bins = 101
    r_min, r_max = -3.0, 3.0
    dt = 0.0015
    bridge = build_equilibrium_spin1_model(
        polarization=p0,
        num_bins=n_bins,
        dt=dt,
        r_min=r_min,
        r_max=r_max,
        legacy_spectral_recovery=False,
    )
    ideal = make_ideal_model(p0, n_bins=n_bins, r_min=r_min, r_max=r_max, dt=dt)
    skip = {
        "rf_enabled",
        "use_physical_voigt_rf",
        "rf_gaussian_fwhm_R",
        "rf_lorentzian_fwhm_R",
        "rf_response_mode",
        "rf_spectral_strength",
        "rf_homogeneous_hwhm_R",
        "rf_peak_selectivity",
        "rf_peak_center_R",
        "rf_peak_half_width_R",
        "rf_transition_width_R",
        "rf_density_power",
        "rf_density_clip",
        "rf_pedestal_gain",
    }
    for name in bridge.params.__dataclass_fields__:
        if name in skip:
            continue
        assert getattr(bridge.params, name) == getattr(ideal.params, name), name


def test_ssrf_beta_equilibrium_pq():
    params = ssrf_beta_spin1_params(polarization=0.45, n_bins=101)
    ideal = make_ideal_model(0.45, n_bins=101, r_min=-3.0, r_max=3.0)
    pol = ideal.polarizations()
    assert pol["P"] == pytest.approx(0.45, rel=1e-8)
    assert pol["Q"] == pytest.approx(boltzmann_Q(0.45), rel=1e-8)
    assert params.p0 == pytest.approx(0.45)