"""Tests for v14 position-dependent recovery and observability APIs in ssrf_realtime."""

from __future__ import annotations

import numpy as np
import pytest

from physics.ssrf_realtime import Spin1Model, Spin1Params


def _recovery_params(**overrides) -> Spin1Params:
    """Spin1Params with non-zero recovery scales (defaults are zero)."""
    base = dict(
        d_same_plus0=0.25,
        d_same_0minus=0.15,
        d_spec_plus0=1.5,
        d_spec_0minus=0.8,
        t2_width_R=0.05,
        dnp_enabled=False,
    )
    base.update(overrides)
    return Spin1Params(**base)


def test_no_dnp_internal_recovery_conserves_p_when_rf_off():
    m = Spin1Model(_recovery_params(rf_burn_R=-0.9, gamma_rf=5.0))
    m.step(400, rf_on=True, dnp_on=False)
    P_after_burn = m.polarizations()["P"]
    m.step(1500, rf_on=False, dnp_on=False)
    P_after_recovery = m.polarizations()["P"]
    assert abs(P_after_recovery - P_after_burn) < 5e-5


def test_rf_depolarizes_positive_p_no_dnp():
    m = Spin1Model(_recovery_params(rf_burn_R=-0.9, gamma_rf=5.0, p0=0.45))
    P0 = m.polarizations()["P"]
    m.step(400, rf_on=True, dnp_on=False)
    assert m.polarizations()["P"] < P0


def test_dnp_rebuilds_toward_saturation_after_burn():
    m = Spin1Model(
        Spin1Params(
            rf_burn_R=-0.9,
            p0=0.1,
            p_dnp_sat=0.58,
            dnp_rate=0.2,
            gamma_rf=3.0,
            dnp_enabled=True,
        )
    )
    m.step(400, rf_on=True, dnp_on=False)
    P_after_burn = m.polarizations()["P"]
    m.step(3000, rf_on=False, dnp_on=True)
    assert m.polarizations()["P"] > P_after_burn
    assert m.polarizations()["P"] < 0.58 + 1e-3


def test_direct_holes_and_mirror_peaks_signs_short_burn():
    m = Spin1Model(
        Spin1Params(
            rf_burn_R=-0.9,
            gamma_rf=2.0,
            dnp_enabled=False,
            d_same_plus0=0.0,
            d_same_0minus=0.0,
            d_spec_plus0=0.0,
            d_spec_0minus=0.0,
        )
    )
    m.step(200, rf_on=True, dnp_on=False)
    vals = m.response_values(-0.9)
    print(vals['dIplus_R'], vals['dIminus_R'], vals['dIplus_minusR'], vals['dIminus_minusR'])
    assert vals["dIplus_R"] < 0.0
    assert vals["dIminus_R"] < 0.0
    assert vals["dIplus_minusR"] > 0.0
    assert vals["dIminus_minusR"] > 0.0


def test_negative_initial_polarization_allowed():
    m = Spin1Model(Spin1Params(p0=-0.45, rf_burn_R=0.4))
    assert m.polarizations()["P"] < 0.0


def test_effective_recovery_rates_depend_on_selected_R():
    m = Spin1Model(_recovery_params(p0=0.45))
    r1 = m.recovery_pathway_rates(-0.90)
    r2 = m.recovery_pathway_rates(0.40)
    diff = abs(r1["same_plus0_eff"] - r2["same_plus0_eff"]) + abs(
        r1["same_0minus_eff"] - r2["same_0minus_eff"]
    )
    assert diff > 1e-6


def test_same_theta_effective_rate_increases_after_rf_created_recoil():
    m = Spin1Model(_recovery_params(rf_burn_R=-0.90, gamma_rf=4.0))
    before = m.recovery_pathway_rates(-0.90)
    m.step(300, rf_on=True, dnp_on=False)
    after = m.recovery_pathway_rates(-0.90)
    assert after["same_plus0_eff"] >= before["same_plus0_eff"]
    assert after["same_0minus_eff"] >= before["same_0minus_eff"]


def test_neighbor_left_and_right_effective_rates_are_reported_for_selected_bin():
    m = Spin1Model(_recovery_params(rf_burn_R=-0.90))
    rates = m.recovery_pathway_rates(-0.90)
    assert rates["left_plus0_eff"] > 0.0
    assert rates["right_plus0_eff"] > 0.0
    assert rates["left_0minus_eff"] > 0.0
    assert rates["right_0minus_eff"] > 0.0


def test_live_local_recovery_rates_change_during_burn_not_just_with_R():
    m = Spin1Model(_recovery_params(rf_burn_R=-0.90, gamma_rf=4.0))
    before = m.local_recovery_rates(-0.90)
    m.step(250, rf_on=True, dnp_on=False)
    after = m.local_recovery_rates(-0.90)
    delta_same = abs(after["Iplus_same_theta"] - before["Iplus_same_theta"]) + abs(
        after["Iminus_same_theta"] - before["Iminus_same_theta"]
    )
    delta_refill = abs(after["Iplus_refill_dt_no_rf"] - before["Iplus_refill_dt_no_rf"]) + abs(
        after["Iminus_refill_dt_no_rf"] - before["Iminus_refill_dt_no_rf"]
    )
    assert delta_same + delta_refill > 1e-8


def test_branch_indices_outside_grid_returns_none():
    m = Spin1Model(Spin1Params(r_min=-3.0, r_max=3.0, n_bins=101))
    kp, km = m.branch_indices(10.0)
    assert kp is None
    assert km is None


def test_branch_indices_inside_grid_returns_valid_indices():
    m = Spin1Model(Spin1Params(rf_burn_R=0.4))
    kp, km = m.branch_indices(0.4)
    assert kp is not None
    assert km is not None
    assert 0 <= kp < m.params.n_bins
    assert 0 <= km < m.params.n_bins


def test_pair_intensities_reports_direct_and_mirror():
    m = Spin1Model(Spin1Params(rf_burn_R=0.4, p0=0.45))
    pair = m.pair_intensities(0.4)
    direct = m.local_intensities(0.4)
    mirror = m.local_intensities(-0.4)
    assert pair["Iplus_R"] == direct["Iplus"]
    assert pair["Iminus_R"] == direct["Iminus"]
    assert pair["Iplus_minusR"] == mirror["Iplus"]
    assert pair["Iminus_minusR"] == mirror["Iminus"]


def test_response_values_zero_at_reference():
    m = Spin1Model(Spin1Params(rf_burn_R=0.4, p0=0.45))
    vals = m.response_values(0.4)
    assert np.isclose(vals["dIplus_R"], 0.0, atol=1e-12)
    assert np.isclose(vals["dIminus_R"], 0.0, atol=1e-12)
    assert np.isclose(vals["dIplus_minusR"], 0.0, atol=1e-12)
    assert np.isclose(vals["dIminus_minusR"], 0.0, atol=1e-12)


def test_spectrum_from_state_total_equals_branch_sum():
    m = Spin1Model(Spin1Params(p0=0.45))
    _, Iplus, Iminus, total = m.spectrum_from_state(m.n)
    assert np.allclose(total, Iplus + Iminus)


def test_reference_spectrum_matches_initial_state_spectrum():
    m = Spin1Model(Spin1Params(p0=0.50, initial_polarization=0.50, rf_enabled=False))
    ref = m.reference_spectrum()
    from_state = m.spectrum_from_state(m.n_ref)
    for a, b in zip(ref, from_state):
        assert np.allclose(a, b)


def test_initial_polarization_sets_display_cal():
    m = Spin1Model(Spin1Params(p0=0.45, initial_polarization=0.60))
    assert m.display_cal == pytest.approx(0.60)


def test_load_from_physical_intensities_roundtrip():
    m = Spin1Model(Spin1Params(p0=0.45, initial_polarization=0.45, n_bins=101))
    Iplus, Iminus, _ = m.physical_intensities()
    m.load_from_physical_intensities(Iplus, Iminus)
    Iplus2, Iminus2, _ = m.physical_intensities()
    assert np.allclose(Iplus, Iplus2, rtol=1e-10)
    assert np.allclose(Iminus, Iminus2, rtol=1e-10)


def test_stored_level_populations_match_vector_p():
    m = Spin1Model(Spin1Params(p0=0.50, initial_polarization=0.50, n_bins=101))
    assert m.n_plus - m.n_minus == pytest.approx(0.50, abs=1e-12)
    assert m.n_plus + m.n_zero + m.n_minus == pytest.approx(1.0, abs=1e-12)
    pops = m.level_populations()
    assert pops["P"] == pytest.approx(0.50, abs=1e-12)
    assert pops["P_initial"] == pytest.approx(0.50, abs=1e-12)


def test_stored_level_populations_from_event_intensities():
    from physics.lineshape.Lineshape import GenerateVectorLineshape

    p0 = 0.50
    f = np.linspace(-3.0, 3.0, 101)
    _, ip, im = GenerateVectorLineshape(p0, f)
    m = Spin1Model(Spin1Params(p0=p0, initial_polarization=p0, n_bins=101, r_min=-3.0, r_max=3.0))
    m.load_from_physical_intensities(ip, im)
    assert m.n_plus - m.n_minus == pytest.approx(p0, abs=1e-9)
    assert m.n_plus_initial - m.n_minus_initial == pytest.approx(p0, abs=1e-9)
    m.step(50, rf_on=True, dnp_on=False)
    # After RF, stored levels stay synced and still sum to 1.
    assert m.n_plus + m.n_zero + m.n_minus == pytest.approx(1.0, abs=1e-9)
    assert m.level_populations()["P"] == pytest.approx(m.n_plus - m.n_minus, abs=1e-12)


def test_same_theta_mirror_gain_increases_recovery_rate():
    low_gain = Spin1Model(_recovery_params(same_theta_mirror_gain=0.0, rf_burn_R=-0.9))
    high_gain = Spin1Model(_recovery_params(same_theta_mirror_gain=3.0, rf_burn_R=-0.9))
    r_low = low_gain.recovery_pathway_rates(-0.9)
    r_high = high_gain.recovery_pathway_rates(-0.9)
    assert r_high["same_plus0_eff"] >= r_low["same_plus0_eff"]
    assert r_high["same_0minus_eff"] >= r_low["same_0minus_eff"]


def test_position_dependent_recovery_uses_local_rates_in_derivative():
    m = Spin1Model(_recovery_params(rf_burn_R=-0.9))
    m.step(200, rf_on=True, dnp_on=False)
    _, parts = m.derivative(rf_on=False, dnp_on=False, breakdown=True)
    same = parts["same_theta_mirror_backpath"]
    spec = parts["spectral_neighbor_diffusion"]
    assert np.isfinite(same["dIplus_R_dt"])
    assert np.isfinite(spec["dIminus_R_dt"])
    assert same["dP_dt"] == pytest.approx(0.0, abs=1e-12)
    assert spec["dP_dt"] == pytest.approx(0.0, abs=1e-12)
