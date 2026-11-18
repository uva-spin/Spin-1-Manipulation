"""GUI intensity units match ssRF-beta Plot_Signal scale."""
import numpy as np
import pytest

from physics.rf.profile_control import make_ideal_model


def test_intensities_plot_units_peak_near_four_percent():
    model = make_ideal_model(0.45)
    _, _, total = model.intensities_plot_units()
    assert float(np.max(total)) == pytest.approx(0.035, rel=0.15)
    assert float(np.max(total)) < 0.06