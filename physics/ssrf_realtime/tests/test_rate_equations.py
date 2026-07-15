import numpy as np
import pytest

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime import Spin1Params
from physics.ssrf_realtime.rate_equations_realtime import (
    build_model_for_intensities,
    solve_rate_equations,
    verify_burn_response,
    verify_rates_response,
)


def _lineshape(polarization: float = 0.45, n_bins: int = 500):
    f = np.linspace(-3.0, 3.0, n_bins)
    _, Iplus, Iminus = GenerateVectorLineshape(polarization, f)
    return f, Iplus, Iminus


def test_verify_rates_response_with_realistic_lineshape():
    _, Iplus, Iminus = _lineshape()
    burn_idx = 180
    result = verify_rates_response(Iplus, Iminus, burn_idx=burn_idx, gamma_rf=2.0, dt=0.001)
    assert result["passed"] is True
    assert result["ratios"]["amp_burn_over_amp_mirror"] == pytest.approx(2.0, rel=1e-5)


def test_solve_rate_equations_rf_only_preserves_ps_sign():
    _, Iplus, Iminus = _lineshape()
    burn_idx = 250
    Iplus_new, Iminus_new, _, _, _ = solve_rate_equations(
        Iplus,
        Iminus,
        dt=0.001,
        gamma_rf=2.0,
        burn_idx=burn_idx,
        initial_polarization=0.45,
        rf_only=True,
    )
    check = verify_burn_response(Iplus, Iminus, Iplus_new, Iminus_new, burn_idx)
    assert check["magnitude_decreased"]


def test_build_model_for_intensities_matches_grid_length():
    f, Iplus, Iminus = _lineshape(n_bins=200)
    model = build_model_for_intensities(
        Iplus,
        Iminus,
        initial_polarization=0.45,
    )
    assert len(model.Rplus) == 200
    assert model.display_cal == pytest.approx(0.45)
    Iplus_out, Iminus_out, _ = model.physical_intensities()
    assert np.allclose(Iplus, Iplus_out, rtol=1e-10)
    assert np.allclose(Iminus, Iminus_out, rtol=1e-10)


def test_solve_rate_equations_full_dynamics_advances_state():
    _, Iplus, Iminus = _lineshape()
    burn_idx = 180
    params = Spin1Params(
        d_same_plus0=0.25,
        d_same_0minus=0.15,
        d_spec_plus0=1.5,
        d_spec_0minus=0.8,
        t2_width_R=0.05,
        dnp_enabled=False,
        steps=1,
    )
    Iplus_new, Iminus_new, rho_plus, rho_zero, rho_minus = solve_rate_equations(
        Iplus,
        Iminus,
        dt=0.001,
        gamma_rf=2.0,
        burn_idx=burn_idx,
        params=params,
        initial_polarization=0.45,
        full_dynamics=True,
    )
    assert not np.allclose(Iplus, Iplus_new)
    assert rho_plus.shape == (len(Iplus),)
    assert rho_zero.shape == (len(Iplus),)
    assert rho_minus.shape == (len(Iplus),)
