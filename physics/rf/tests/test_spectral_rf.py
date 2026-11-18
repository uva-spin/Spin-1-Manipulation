"""Spectral RF operator matches ssRF-beta ideal-bin spreading."""
import numpy as np

from physics.rf.kinetic_defaults import ssrf_beta_ideal_bin_params
from physics.rf.ideal_model import IdealBinModel
from physics.rf.optimal_profile import smooth_bounded_commands


def test_spectral_rf_spreads_point_command():
    params = ssrf_beta_ideal_bin_params(
        polarization=0.45, n_bins=201, r_min=-3.0, r_max=3.0
    )
    model = IdealBinModel(params)
    center = int(np.argmin(np.abs(model.Rplus + 0.9)))
    u = np.zeros(model.Rplus.size)
    u[center] = 10.0
    gp, gm = model.rf_rate_operator().rates(u)
    spread = np.maximum(gp, gm)
    assert np.count_nonzero(spread > 0.1 * spread.max()) > 3
    assert spread[center] > 0.0
    assert spread.max() > 1.0


def test_smooth_bounded_commands_reduces_roughness():
    upper = np.full(50, 20.0)
    powers = np.zeros(50)
    powers[10:15] = 20.0
    smooth = smooth_bounded_commands(powers, upper, sigma_bins=2.0)
    assert np.sum(smooth) == np.sum(powers)
    assert np.sum(np.diff(smooth) ** 2) < np.sum(np.diff(powers) ** 2)