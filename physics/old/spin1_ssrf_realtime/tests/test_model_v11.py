import numpy as np

from ssrf_realtime.model import Spin1Model, Spin1Params
from ssrf_realtime.lineshape import plot_signal_reference


def test_plot_signal_reference_is_matched_at_calibration_p():
    p = Spin1Params(p0=0.50, calibration_p=0.50, line_gamma=0.05, line_asym=0.04, rf_enabled=False)
    m = Spin1Model(p)
    R, Ip, Im, total = m.reference_spectrum()
    Ip_ref, Im_ref, total_ref = plot_signal_reference(R, P=0.50, gamma=0.05, asym=0.04, divisor=10.0)
    assert np.max(np.abs(Ip - Ip_ref)) < 3e-3
    assert np.max(np.abs(Im - Im_ref)) < 3e-3
    assert np.max(np.abs(total - total_ref)) < 4e-3


def test_signed_initial_vector_polarization_is_supported():
    for P in [0.10, 0.58, -0.10, -0.58]:
        m = Spin1Model(Spin1Params(p0=P))
        pol = m.polarizations()
        assert np.isclose(pol["P"], P, atol=5e-7)
        areas = m.branch_areas()
        assert np.sign(areas["A_total"]) == np.sign(P)


def test_common_rf_parameter_only():
    p = Spin1Params()
    assert hasattr(p, "gamma_rf")
    assert not hasattr(p, "gamma_plus")
    assert not hasattr(p, "gamma_minus")


def test_pure_rf_direct_holes_and_mirror_peaks_have_correct_signs_for_positive_p():
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=-0.9,
        gamma_rf=5.0,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
        dt=1e-4,
    )
    m = Spin1Model(p)
    for _ in range(50):
        m.step(rf_on=True, dnp_on=False)
    vals = m.response_values(p.rf_burn_R)
    assert vals["dIplus_R"] < 0
    assert vals["dIminus_R"] < 0
    assert vals["dIplus_minusR"] > 0
    assert vals["dIminus_minusR"] > 0


def test_pure_rf_half_gain_rule_in_display_units():
    p = Spin1Params(
        rf_burn_R=0.4,
        gamma_rf=1.7,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
    )
    m = Spin1Model(p)
    _, parts = m.derivative(rf_on=True, dnp_on=False, breakdown=True)
    rf = parts["RF"]
    assert np.isclose(rf["dIminus_minusR_dt"], -0.5 * rf["dIplus_R_dt"], rtol=1e-10, atol=1e-12)
    assert np.isclose(rf["dIplus_minusR_dt"], -0.5 * rf["dIminus_R_dt"], rtol=1e-10, atol=1e-12)


def test_rf_depolarizes_total_area_when_dnp_is_off():
    p = Spin1Params(p0=0.45, rf_burn_R=0.4, gamma_rf=5.0, dnp_enabled=False, dt=5e-4)
    m = Spin1Model(p)
    P0 = m.polarizations()["P"]
    for _ in range(200):
        m.step(rf_on=True, dnp_on=False)
    P1 = m.polarizations()["P"]
    assert P1 < P0

    p_neg = Spin1Params(p0=-0.45, rf_burn_R=0.4, gamma_rf=5.0, dnp_enabled=False, dt=5e-4)
    m_neg = Spin1Model(p_neg)
    P0n = m_neg.polarizations()["P"]
    for _ in range(200):
        m_neg.step(rf_on=True, dnp_on=False)
    P1n = m_neg.polarizations()["P"]
    assert P1n > P0n
    assert abs(P1n) < abs(P0n)


def test_dnp_off_recovery_redistributes_without_restoring_lost_total_p():
    p = Spin1Params(p0=0.45, rf_burn_R=0.4, gamma_rf=6.0, dnp_enabled=False, dt=5e-4)
    m = Spin1Model(p)
    P_initial = m.polarizations()["P"]
    for _ in range(250):
        m.step(rf_on=True, dnp_on=False)
    P_after_burn = m.polarizations()["P"]
    assert P_after_burn < P_initial
    for _ in range(2000):
        m.step(rf_on=False, dnp_on=False)
    P_after_recovery = m.polarizations()["P"]
    # Internal diffusion/recovery should conserve the reduced vector polarization.
    assert np.isclose(P_after_recovery, P_after_burn, atol=2e-4)
    assert P_after_recovery < P_initial - 1e-4


def test_dnp_builds_toward_selected_saturation_without_overshoot():
    p = Spin1Params(p0=0.10, p_dnp_sat=0.58, dnp_enabled=True, dnp_rate=1.2, dt=5e-4)
    m = Spin1Model(p)
    for _ in range(6000):
        m.step(rf_on=False, dnp_on=True)
    P = m.polarizations()["P"]
    assert 0.55 < P < 0.581


def test_negative_dnp_saturation_builds_negative_polarization():
    p = Spin1Params(p0=0.05, p_dnp_sat=-0.40, dnp_enabled=True, dnp_rate=1.0, dt=5e-4)
    m = Spin1Model(p)
    for _ in range(5000):
        m.step(rf_on=False, dnp_on=True)
    P = m.polarizations()["P"]
    assert -0.405 < P < -0.35
