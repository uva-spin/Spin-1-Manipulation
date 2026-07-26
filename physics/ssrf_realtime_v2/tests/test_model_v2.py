"""Tests for capacity-weighted v2 ssrf_realtime model."""

from __future__ import annotations

import numpy as np
import pytest

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime_v2 import Spin1Model, Spin1Params
from physics.ssrf_realtime_v2.rate_equations_realtime import verify_burn_response, verify_rates_response


def _recovery_params(**overrides) -> Spin1Params:
    base = dict(
        d_same_plus0=0.18,
        d_same_0minus=0.10,
        d_spec_plus0=2.0,
        d_spec_0minus=1.0,
        t2_width_R=0.05,
        dnp_enabled=False,
    )
    base.update(overrides)
    return Spin1Params(**base)


def test_capacity_power_zero_returns_uniform_rate_weights():
    m = Spin1Model(Spin1Params(capacity_rate_power=0.0))
    w = m.capacity_rate_weights()
    assert np.allclose(w, 1.0)


def test_capacity_weighting_identifies_horn_branch_at_same_physical_R():
    m = Spin1Model(Spin1Params(capacity_rate_power=1.0))
    right = m.local_capacity_factors(0.90)
    left = m.local_capacity_factors(-0.90)
    assert right["w_Iplus_R"] > 3.0 * right["w_Iminus_R"]
    assert left["w_Iminus_R"] > 3.0 * left["w_Iplus_R"]


def test_capacity_weighted_rf_drains_horn_faster_than_overlap_branch():
    m = Spin1Model(Spin1Params(p0=0.45, gamma_rf=1.0, capacity_rate_power=1.0))
    rates = m.effective_local_rates(0.90)
    assert rates["gamma_rf_Iplus_R"] > 4.0 * rates["gamma_rf_Iminus_R"]


def test_capacity_weighted_dnp_has_larger_local_effective_rate_at_horn():
    m = Spin1Model(Spin1Params(p0=0.10, p_dnp_sat=0.58, dnp_rate=0.2, capacity_rate_power=1.0))
    rates = m.effective_local_rates(0.90)
    assert rates["dnp_Iplus_R"] > 3.0 * rates["dnp_Iminus_R"]


def test_dnp_builds_toward_selected_saturation_without_overshoot():
    p = Spin1Params(
        p0=0.10,
        p_dnp_sat=0.58,
        dnp_enabled=True,
        dnp_rate=1.2,
        dt=5e-4,
        capacity_rate_power=0.0,
    )
    m = Spin1Model(p)
    for _ in range(6000):
        m.step(rf_on=False, dnp_on=True)
    P = m.polarizations()["P"]
    assert 0.55 < P < 0.581


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
    m.params.rf_profile = np.full(m.params.n_bins, 2.0, dtype=float)
    m.step(200, rf_on=True, dnp_on=False)
    vals = m.response_values(-0.9)
    assert vals["dIplus_R"] < 0.0
    assert vals["dIminus_R"] < 0.0
    assert vals["dIplus_minusR"] > 0.0
    assert vals["dIminus_minusR"] > 0.0


AFP_SUBSET = list(range(250, 300))


def test_afp_preserves_vector_polarization():
    f = np.linspace(-3.0, 3.0, 500)
    _, ip, im = GenerateVectorLineshape(0.48, f)
    m = Spin1Model(
        Spin1Params(
            p0=0.48,
            n_bins=500,
            r_min=-3.0,
            r_max=3.0,
            afp_subset_indices=AFP_SUBSET,
        )
    )
    m.load_from_physical_intensities(ip, im)
    P_before = m.level_populations()["P"]
    m.afp_sweep()
    P_after = m.level_populations()["P"]
    # Multi-bin AFP redistributes Q locally; vector P should stay approximately unchanged.
    assert abs(P_after - P_before) < 0.20


def test_afp_one_shot_in_step():
    m = Spin1Model(Spin1Params(p0=0.48, afp_enabled=True, afp_subset_indices=[50]))
    assert m.params.afp_enabled is True
    m.step(1, rf_on=False, dnp_on=False)
    assert m.params.afp_enabled is False


def test_load_from_physical_intensities_roundtrip():
    m = Spin1Model(Spin1Params(p0=0.45, n_bins=101))
    assert m.display_cal == pytest.approx(0.45)
    Iplus, Iminus, _ = m.physical_intensities()
    m.load_from_physical_intensities(Iplus, Iminus)
    Iplus2, Iminus2, _ = m.physical_intensities()
    assert np.allclose(Iplus, Iplus2, rtol=1e-10)
    assert np.allclose(Iminus, Iminus2, rtol=1e-10)


def test_ssrf_mirror_burn_ratios_approximate_half():
    f = np.linspace(-3.0, 3.0, 500)
    _, Iplus, Iminus = GenerateVectorLineshape(0.48, f)
    burn_idx = int(np.argmin(np.abs(f - (-0.92))))
    result = verify_rates_response(
        Iplus,
        Iminus,
        burn_idx=burn_idx,
        gamma_rf=2.0,
        dt=0.001,
        rtol=0.10,
        params=Spin1Params(
            capacity_rate_power=1.0,
            d_same_plus0=0.0,
            d_same_0minus=0.0,
            d_spec_plus0=0.0,
            d_spec_0minus=0.0,
        ),
        p0=0.48,
    )
    assert result["magnitude_decreased"]
    assert abs(result["ratios"]["iplus_burn_over_iminus_mirror"] - 2.0) / 2.0 < 0.10
    assert abs(result["ratios"]["iminus_burn_over_iplus_mirror"] - 2.0) / 2.0 < 0.10


def test_integrated_burn_area_mirror_ratios():
    f = np.linspace(-3.0, 3.0, 500)
    _, Iplus, Iminus = GenerateVectorLineshape(0.48, f)
    burn_idx = int(np.argmin(np.abs(f - (-0.92))))
    mirror_idx = len(Iplus) - 1 - burn_idx

    params = Spin1Params(
        capacity_rate_power=1.0,
        gamma_rf=2.0,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
        steps=80,
        dt=0.0015,
    )
    from physics.ssrf_realtime_v2.rate_equations_realtime import solve_rate_equations

    Iplus_new, Iminus_new, _, _, _ = solve_rate_equations(
        Iplus,
        Iminus,
        dt=0.0015,
        gamma_rf=2.0,
        burn_idx=burn_idx,
        params=params,
        p0=0.48,
        rf_only=True,
    )
    check = verify_burn_response(Iplus, Iminus, Iplus_new, Iminus_new, burn_idx, rtol=0.10)
    d_ip_burn = check["d_iplus_burn"]
    d_im_burn = check["d_iminus_burn"]
    d_ip_mirror = check["d_iplus_mirror"]
    d_im_mirror = check["d_iminus_mirror"]

    ratio_imirror_over_iburn = abs(d_ip_mirror) / max(abs(d_im_burn), 1e-30)
    ratio_mmirror_over_pburn = abs(d_im_mirror) / max(abs(d_ip_burn), 1e-30)
    assert abs(ratio_imirror_over_iburn - 0.5) / 0.5 < 0.12
    assert abs(ratio_mmirror_over_pburn - 0.5) / 0.5 < 0.12


def test_voigt_burn_preserves_iminus_above_iplus_at_burn_bin():
    f = np.linspace(-3.0, 3.0, 500)
    _, Iplus, Iminus = GenerateVectorLineshape(0.48, f)
    burn_idx = int(np.argmin(np.abs(f - (-0.92))))
    assert Iminus[burn_idx] > Iplus[burn_idx]

    params = Spin1Params(
        p0=0.48,
        n_bins=500,
        capacity_rate_power=1.0,
        gamma_rf=1.0,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
        steps=400,
        dt=0.0015,
    )
    from physics.ssrf_realtime_v2.rate_equations_realtime import (
        burn_preserves_branch_order,
        solve_rate_equations,
    )

    Iplus_new, Iminus_new, _, _, _ = solve_rate_equations(
        Iplus,
        Iminus,
        dt=0.0015,
        gamma_rf=1.0,
        burn_idx=burn_idx,
        params=params,
        p0=0.48,
        rf_only=True,
    )
    assert burn_preserves_branch_order(Iplus, Iminus, Iplus_new, Iminus_new, burn_idx)
    assert Iminus_new[burn_idx] >= Iplus_new[burn_idx]


def test_voigt_multi_bin_uses_center_pair_capacity_weights():
    from physics.ssrf_realtime_v2.rate_equations_realtime import configure_voigt_ssrf

    f = np.linspace(-3.0, 3.0, 500)
    _, ip, im = GenerateVectorLineshape(0.48, f)
    burn_idx = int(np.argmin(np.abs(f - (-0.92))))

    params = Spin1Params(
        p0=0.48,
        n_bins=500,
        capacity_rate_power=1.0,
        gamma_rf=1.0,
        dt=0.0015,
        rf_burn_R=-0.92,
    )
    model = Spin1Model(params)
    model.load_from_physical_intensities(ip, im)
    configure_voigt_ssrf(model, burn_idx, 1.0, half_width=3, full_spectrum_recovery=True)

    w = model.capacity_rate_weights()
    ckp, ckm = model.cached_branch_indices(-0.92)
    profile = np.asarray(model.params.rf_profile, dtype=float)

    for _ in range(400):
        model.step_once(dt=0.0015, rf_on=True, dnp_on=False)
    ip2, im2, _ = model.physical_intensities()
    assert im2[burn_idx] >= ip2[burn_idx]
    assert float(w[ckp]) != float(w[ckm])
    assert profile[burn_idx] == pytest.approx(1.0)


def test_voigt_rf_profile_peak_at_center_rounded_falloff():
    from physics.ssrf_realtime_v2.rf_profile import make_voigt_rf_profile

    profile, support = make_voigt_rf_profile(500, 250, 10.0, half_width=5)
    assert 250 in support
    assert profile[250] == pytest.approx(10.0)
    assert profile[249] == pytest.approx(profile[251])
    linear_neighbor = 10.0 * (1.0 - 1.0 / 6.0)
    assert profile[249] > linear_neighbor
    assert profile[249] > profile[253]
    assert np.all(profile[np.asarray(support)] > 0.0)
    assert np.all(profile[np.setdiff1d(np.arange(500), support)] == 0.0)


def test_recovery_rates_use_capacity_weights():
    m = Spin1Model(_recovery_params(rf_burn_R=0.90, capacity_rate_power=1.0))
    rates = m.recovery_pathway_rates(0.90)
    assert rates["same_plus0_eff"] > 2.0 * m.params.d_same_plus0
    assert rates["capacity_weight_plus0"] > 2.0
