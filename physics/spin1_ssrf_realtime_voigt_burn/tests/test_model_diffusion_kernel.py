from pathlib import Path

import numpy as np

from ssrf_realtime.lineshape import plot_signal_reference
from ssrf_realtime.model import Spin1Model, Spin1Params


DATA = Path(__file__).with_name("data") / "previous_working_dynamics.npz"


def test_plot_signal_reference_is_unchanged():
    p = Spin1Params(p0=0.50, calibration_p=0.50, line_gamma=0.05, line_asym=0.04)
    m = Spin1Model(p)
    R, Ip, Im, total = m.reference_spectrum()
    Ip_ref, Im_ref, total_ref = plot_signal_reference(
        R, P=0.50, gamma=0.05, asym=0.04, divisor=10.0
    )
    assert np.max(np.abs(Ip - Ip_ref)) < 3e-3
    assert np.max(np.abs(Im - Im_ref)) < 3e-3
    assert np.max(np.abs(total - total_ref)) < 4e-3


def test_signed_initial_vector_polarization_is_unchanged():
    for P in (0.10, 0.58, -0.10, -0.58):
        m = Spin1Model(Spin1Params(p0=P))
        assert np.isclose(m.polarizations()["P"], P, atol=5e-7)
        assert np.sign(m.branch_areas()["A_total"]) == np.sign(P)


def test_rf_only_trajectory_matches_previous_working_package_exactly():
    ref = np.load(DATA)
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=0.4,
        gamma_rf=2.0,
        diffusion_scale=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
        dt=0.0015,
        capacity_rate_power=1.0,
        rf_gaussian_fwhm_R=0.0,
        rf_lorentzian_fwhm_R=0.0,
    )
    m = Spin1Model(p)
    assert np.allclose(m.n, ref["rf_initial"], rtol=2e-13, atol=2e-15)
    for _ in range(1000):
        m.step(rf_on=True, dnp_on=False)
    assert np.allclose(m.n, ref["rf_after"], rtol=2e-13, atol=2e-15)
    assert np.allclose(np.array(m.spectrum()[1:]), ref["rf_spec"], rtol=2e-13, atol=2e-14)


def test_dnp_only_trajectory_matches_previous_working_package_exactly():
    ref = np.load(DATA)
    p = Spin1Params(
        p0=0.10,
        p_dnp_sat=0.58,
        dnp_enabled=True,
        dnp_rate=0.2,
        capacity_rate_power=1.0,
        diffusion_scale=0.0,
        t1_rate=0.0,
        dt=0.0015,
    )
    m = Spin1Model(p)
    assert np.allclose(m.n, ref["dnp_initial"], rtol=2e-13, atol=2e-15)
    for _ in range(1000):
        m.step(rf_on=False, dnp_on=True)
    assert np.allclose(m.n, ref["dnp_after"], rtol=2e-13, atol=2e-15)
    assert np.allclose(np.array(m.spectrum()[1:]), ref["dnp_spec"], rtol=2e-13, atol=2e-14)


def test_isolated_ideal_bin_mirror_relation_emerges_from_population_equations():
    p = Spin1Params(
        rf_burn_R=0.4,
        gamma_rf=1.7,
        diffusion_scale=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
        rf_gaussian_fwhm_R=0.0,
        rf_lorentzian_fwhm_R=0.0,
    )
    m = Spin1Model(p)
    _, parts = m.derivative(rf_on=True, dnp_on=False, breakdown=True)
    rf = parts["RF"]
    assert np.isclose(
        rf["dIminus_minusR_dt"], -0.5 * rf["dIplus_R_dt"], rtol=1e-12, atol=1e-14
    )
    assert np.isclose(
        rf["dIplus_minusR_dt"], -0.5 * rf["dIminus_R_dt"], rtol=1e-12, atol=1e-14
    )


def test_diffusion_is_zero_in_a_uniform_boltzmann_packet_state():
    m = Spin1Model(Spin1Params(diffusion_scale=80.0, cross_branch_ratio=1.0))
    terms = m._spin_diffusion_terms(dnp_on=False)
    dn = sum(terms.values())
    assert np.max(np.abs(dn)) < 2e-16


def test_internal_diffusion_conserves_every_packet_mass_and_total_vector_p():
    p = Spin1Params(
        diffusion_scale=80.0,
        cross_branch_ratio=0.4,
        orientation_corr_fraction=0.35,
    )
    m = Spin1Model(p)
    rng = np.random.default_rng(20260724)
    frac = rng.random(m.n.shape)
    frac /= frac.sum(axis=1, keepdims=True)
    m.n = m.mu[:, None] * frac
    dn = sum(m._spin_diffusion_terms(dnp_on=False).values())
    assert np.max(np.abs(np.sum(dn, axis=1))) < 1e-14
    assert abs(float(np.sum(dn[:, 0] - dn[:, 2]))) < 1e-13
    assert abs(float(np.sum(dn))) < 1e-13


def _burned_model(burn_steps: int, R: float = 0.4) -> Spin1Model:
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=R,
        gamma_rf=6.0,
        diffusion_scale=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
        dt=5e-4,
    )
    m = Spin1Model(p)
    for _ in range(burn_steps):
        m.step(rf_on=True, dnp_on=False)
    m.params.diffusion_scale = 80.0
    return m


def test_post_burn_population_currents_fill_both_direct_holes_and_reduce_mirror_peaks():
    m = _burned_model(250, R=0.4)
    _, parts = m.derivative(rf_on=False, dnp_on=False, breakdown=True)
    d = parts["net"]
    assert d["dIplus_R_dt"] > 0.0
    assert d["dIminus_R_dt"] > 0.0
    assert d["dIplus_minusR_dt"] < 0.0
    assert d["dIminus_minusR_dt"] < 0.0
    assert abs(d["dP_dt"]) < 1e-12


def test_deeper_burn_produces_larger_initial_recovery_current():
    shallow = _burned_model(60)
    deep = _burned_model(250)
    _, ps = shallow.derivative(rf_on=False, dnp_on=False, breakdown=True)
    _, pd = deep.derivative(rf_on=False, dnp_on=False, breakdown=True)
    assert pd["net"]["dIplus_R_dt"] > ps["net"]["dIplus_R_dt"]
    assert pd["net"]["dIminus_R_dt"] > ps["net"]["dIminus_R_dt"]


def test_connectivity_changes_with_selected_R_and_identifies_the_horn_branch():
    m = Spin1Model(Spin1Params(diffusion_scale=80.0))
    right = m.local_diffusion_diagnostics(0.90)
    left = m.local_diffusion_diagnostics(-0.90)
    assert right["conn_Iplus"] > 3.0 * right["conn_Iminus"]
    assert left["conn_Iminus"] > 3.0 * left["conn_Iplus"]


def test_no_dnp_recovery_conserves_the_reduced_vector_polarization():
    m = _burned_model(250)
    P_burn = m.polarizations()["P"]
    for _ in range(2000):
        m.step(rf_on=False, dnp_on=False)
    assert np.isclose(m.polarizations()["P"], P_burn, atol=2e-12)


def test_dnp_saturation_remains_bounded_when_diffusion_is_disabled():
    p = Spin1Params(
        p0=0.10,
        p_dnp_sat=0.58,
        dnp_enabled=True,
        dnp_rate=1.2,
        dt=5e-4,
        capacity_rate_power=0.0,
        diffusion_scale=0.0,
    )
    m = Spin1Model(p)
    for _ in range(6000):
        m.step(rf_on=False, dnp_on=True)
    assert 0.55 < m.polarizations()["P"] < 0.581


def test_cross_branch_channel_conserves_P_but_can_change_Q():
    p = Spin1Params(diffusion_scale=80.0, cross_branch_ratio=1.0)
    m = Spin1Model(p)
    # Perturb the tensor distribution without changing packet masses.
    k = m.branch_indices(0.25)[0]
    assert k is not None
    eps = 0.05 * m.mu[k]
    m.n[k, 0] += eps
    m.n[k, 1] -= 2.0 * eps
    m.n[k, 2] += eps
    terms = m._spin_diffusion_terms(dnp_on=False)
    dn = sum(terms.values())
    dP = float(np.sum(dn[:, 0] - dn[:, 2]))
    dQ = float(np.sum(dn[:, 0] - 2.0 * dn[:, 1] + dn[:, 2]))
    assert abs(dP) < 1e-12
    assert abs(dQ) > 1e-10



def test_zero_width_profile_is_exact_legacy_one_bin_selector():
    p = Spin1Params(
        rf_burn_R=0.417,
        rf_gaussian_fwhm_R=0.0,
        rf_lorentzian_fwhm_R=0.0,
    )
    m = Spin1Model(p)
    vp, vm = m.rf_profile_arrays()
    kp, km = m.branch_indices()
    assert kp is not None and km is not None
    assert np.count_nonzero(vp) == 1
    assert np.count_nonzero(vm) == 1
    assert vp[kp] == 1.0
    assert vm[km] == 1.0


def test_voigt_is_one_common_physical_profile_for_both_transitions():
    p = Spin1Params(
        rf_burn_R=0.37,
        rf_gaussian_fwhm_R=0.08,
        rf_lorentzian_fwhm_R=0.04,
        rf_profile_normalization="center_bin",
    )
    m = Spin1Model(p)
    R, physical = m.rf_profile_physical()
    vp, vm = m.rf_profile_arrays()
    assert np.allclose(vp, physical, rtol=0.0, atol=0.0)
    assert np.array_equal(vm, physical[::-1])
    kp, km = m.branch_indices()
    assert kp is not None and km is not None
    assert np.isclose(vp[kp], vm[km], rtol=1e-13, atol=1e-14)
    assert vp[kp] > 0.99


def test_voigt_rf_term_is_full_three_level_rate_equation():
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=0.23,
        gamma_rf=1.9,
        rf_gaussian_fwhm_R=0.35,
        rf_lorentzian_fwhm_R=0.18,
        capacity_rate_power=0.7,
        diffusion_scale=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
    )
    m = Spin1Model(p)
    gp, gm = m.rf_rate_fields()
    dp = m.n[:, 0] - m.n[:, 1]
    dm = m.n[:, 1] - m.n[:, 2]
    jp = gp * dp
    jm = gm * dm
    expected = np.zeros_like(m.n)
    expected[:, 0] = -jp
    expected[:, 1] = jp - jm
    expected[:, 2] = jm
    actual = m._rf_population_term(True)
    assert np.allclose(actual, expected, rtol=2e-14, atol=2e-18)
    assert np.max(np.abs(actual.sum(axis=1))) < 1e-18


def test_finite_voigt_profile_does_not_impose_a_fixed_mirror_ratio():
    p = Spin1Params(
        rf_burn_R=0.20,
        gamma_rf=2.0,
        rf_gaussian_fwhm_R=0.65,
        rf_lorentzian_fwhm_R=0.30,
        capacity_rate_power=0.0,
        diffusion_scale=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
    )
    m = Spin1Model(p)
    _, parts = m.derivative(rf_on=True, dnp_on=False, breakdown=True)
    rf = parts["RF"]
    # Broad wings drive both transitions of the same packet.  The complete
    # three-level ODE therefore determines the mirror slope, not a fixed rule.
    ratio = rf["dIminus_minusR_dt"] / (-rf["dIplus_R_dt"])
    assert np.isfinite(ratio)
    assert abs(ratio - 0.5) > 1e-3


def test_center_of_voigt_burn_depletes_faster_than_wings():
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=0.55,
        gamma_rf=2.0,
        rf_gaussian_fwhm_R=0.12,
        rf_lorentzian_fwhm_R=0.04,
        capacity_rate_power=0.0,
        diffusion_scale=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
    )
    m = Spin1Model(p)
    gp, _ = m.rf_rate_fields()
    k0 = int(np.argmin(np.abs(m.Rplus - p.rf_burn_R)))
    k1 = int(np.argmin(np.abs(m.Rplus - (p.rf_burn_R + 0.08))))
    k2 = int(np.argmin(np.abs(m.Rplus - (p.rf_burn_R + 0.25))))
    assert gp[k0] > gp[k1] > gp[k2]

    initial = m.packet_intensities(density=False)[0].copy()
    for _ in range(200):
        m.step(rf_on=True, dnp_on=False)
    final = m.packet_intensities(density=False)[0]
    frac0 = 1.0 - final[k0] / initial[k0]
    frac1 = 1.0 - final[k1] / initial[k1]
    frac2 = 1.0 - final[k2] / initial[k2]
    assert frac0 > frac1 > frac2 >= 0.0


def test_continuous_peak_profile_bin_integral_converges_with_grid():
    widths = (0.10, 0.05)
    areas = []
    for n_bins in (351, 701, 1401):
        p = Spin1Params(
            n_bins=n_bins,
            rf_burn_R=0.31,
            rf_gaussian_fwhm_R=widths[0],
            rf_lorentzian_fwhm_R=widths[1],
            rf_profile_normalization="continuous_peak",
        )
        m = Spin1Model(p)
        areas.append(m.rf_profile_summary()["equivalent_width_R"])
    areas = np.asarray(areas, dtype=float)
    assert np.max(np.abs(areas - areas[-1])) / areas[-1] < 2e-3
