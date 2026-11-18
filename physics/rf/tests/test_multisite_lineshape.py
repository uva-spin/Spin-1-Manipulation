"""Powder lineshape capacity and display scaling match ssRF-beta."""
import numpy as np
import pytest

from physics.rf.kinetic_defaults import ssrf_beta_spin1_params
from physics.rf.lineshape import plot_signal_reference
from physics.rf.model import Spin1Model
from physics.rf.multisite_lineshape import absorption_density


def test_single_pake_equilibrium_matches_plot_signal_shape():
    p0 = 0.45
    params = ssrf_beta_spin1_params(polarization=p0, n_bins=201, r_min=-3.0, r_max=3.0)
    model = Spin1Model(params)
    ip, im, total = model.physical_intensities()
    ref_ip, ref_im, ref_total = plot_signal_reference(
        model.Rplus,
        P=p0,
        gamma=params.line_gamma,
        asym=params.line_asym,
        divisor=params.plot_divisor,
    )
    scale = float(np.max(total) / max(np.max(ref_total), 1e-30))
    np.testing.assert_allclose(ip, ref_ip * scale, rtol=0.02, atol=1e-6)
    np.testing.assert_allclose(im, ref_im * scale, rtol=0.02, atol=1e-6)


def test_multisite_density_normalizes_to_unity():
    r = np.linspace(-3.0, 3.0, 500)
    density = absorption_density(
        r,
        cd_scale=1.0,
        od_ratio=1.6,
        hwhm=0.05,
        eta_cd=0.03,
        eta_od=0.12,
        od_fraction=0.07,
        points=64,
    )
    d_r = r[1] - r[0]
    assert float(np.sum(density) * d_r) == pytest.approx(1.0, rel=0.02)


def test_multisite_butanol_model_builds():
    params = ssrf_beta_spin1_params(polarization=0.45, n_bins=101).replace(
        line_model="multisite_butanol"
    )
    model = Spin1Model(params)
    assert float(model.mu.sum()) == pytest.approx(1.0, rel=1e-10)
    ip, im, total = model.physical_intensities()
    assert np.all(np.isfinite(total))
    assert np.max(total) > 0.0