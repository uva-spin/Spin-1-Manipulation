"""Tests for voigt_burn physics ported into ssrf_realtime_v2."""

from __future__ import annotations

import numpy as np
import pytest

from physics.ssrf_realtime_v2 import Spin1Model, Spin1Params
from physics.ssrf_realtime_v2.rate_equations_realtime import (
    configure_single_bin_ssrf,
    configure_voigt_burn_spectral_recovery,
    create_voigt_burn_model,
    verify_burn_response,
    voigt_burn_recovery_param_snapshot,
)


def _physical_params(**overrides) -> Spin1Params:
    base = dict(use_physical_voigt_rf=True, diffusion_scale=0.0, dnp_enabled=False, t1_rate=0.0)
    base.update(overrides)
    return Spin1Params(**base)


def _configure_single_bin_with_neighbors(model: Spin1Model, *, gamma_rf: float = 2.0) -> int:
    """Single-bin RF with the same demo recovery as physical Voigt."""
    burn_idx = model.burn_index(model.params.rf_burn_R)
    configure_single_bin_ssrf(model, burn_idx, float(gamma_rf))
    return burn_idx


def _setup_voigt_burn_model(*, single_bin: bool, gamma_rf: float = 2.0) -> Spin1Model:
    model = create_voigt_burn_model(gamma_rf=gamma_rf)
    configure_voigt_burn_spectral_recovery(model)
    if single_bin:
        configure_single_bin_ssrf(
            model, model.burn_index(model.params.rf_burn_R), gamma_rf, apply_demo_recovery=False
        )
    return model


def _run_burn(model: Spin1Model, *, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    burn_idx = model.burn_index(model.params.rf_burn_R)
    ip0, im0, _ = model.physical_intensities()
    for _ in range(steps):
        model.step(1, rf_on=True, dnp_on=False)
    ip, im, _ = model.physical_intensities()
    return ip0, im0, ip, im, burn_idx


def _assert_integrated_mirror_half_ratios(
    ip0: np.ndarray,
    im0: np.ndarray,
    ip: np.ndarray,
    im: np.ndarray,
    burn_idx: int,
    *,
    rtol: float = 0.12,
) -> None:
    """Assert burn/mirror integrated-area and branch-delta ratios (~2:1 and ~1:2)."""
    check = verify_burn_response(ip0, im0, ip, im, burn_idx, rtol=rtol)
    assert check["magnitude_decreased"]
    assert check["passed"]

    d_ip_burn = check["d_iplus_burn"]
    d_im_burn = check["d_iminus_burn"]
    d_ip_mirror = check["d_iplus_mirror"]
    d_im_mirror = check["d_iminus_mirror"]

    ratio_ip_mirror_over_im_burn = abs(d_ip_mirror) / max(abs(d_im_burn), 1e-30)
    ratio_im_mirror_over_ip_burn = abs(d_im_mirror) / max(abs(d_ip_burn), 1e-30)
    assert ratio_ip_mirror_over_im_burn == pytest.approx(0.5, rel=rtol)
    assert ratio_im_mirror_over_ip_burn == pytest.approx(0.5, rel=rtol)
    assert check["ratios"]["amp_burn_over_amp_mirror"] == pytest.approx(2.0, rel=rtol)


def test_voigt_burn_integrated_mirror_area_ratios():
    model = create_voigt_burn_model(gamma_rf=2.0)
    configure_voigt_burn_spectral_recovery(model)
    ip0, im0, ip, im, burn_idx = _run_burn(model, steps=80)
    _assert_integrated_mirror_half_ratios(ip0, im0, ip, im, burn_idx)


def test_single_bin_integrated_mirror_area_ratios():
    model = create_voigt_burn_model(gamma_rf=2.0)
    _configure_single_bin_with_neighbors(model, gamma_rf=2.0)
    ip0, im0, ip, im, burn_idx = _run_burn(model, steps=80)
    _assert_integrated_mirror_half_ratios(ip0, im0, ip, im, burn_idx)


def test_voigt_and_single_bin_use_identical_recovery_params():
    voigt = _setup_voigt_burn_model(single_bin=False)
    single = _setup_voigt_burn_model(single_bin=True)
    assert voigt_burn_recovery_param_snapshot(voigt) == voigt_burn_recovery_param_snapshot(
        single
    )


def test_voigt_and_single_bin_share_spectral_recovery_during_burn():
    for single_bin in (False, True):
        model = _setup_voigt_burn_model(single_bin=single_bin)
        for _ in range(30):
            model.step(1, rf_on=True, dnp_on=False)
        _, parts = model.derivative(rf_on=True, dnp_on=False, breakdown=True)
        spec = parts["spectral_neighbors"]
        assert abs(spec["dIminus_R_dt"]) > 0.0 or abs(spec["dIplus_R_dt"]) > 0.0


def test_zero_width_profile_is_exact_legacy_one_bin_selector():
    m = Spin1Model(
        _physical_params(
            rf_burn_R=0.417,
            rf_gaussian_fwhm_R=0.0,
            rf_lorentzian_fwhm_R=0.0,
        )
    )
    vp, vm = m.rf_profile_arrays()
    kp, km = m.branch_indices()
    assert kp is not None and km is not None
    assert np.count_nonzero(vp) == 1
    assert np.count_nonzero(vm) == 1
    assert vp[kp] == 1.0
    assert vm[km] == 1.0


def test_voigt_is_one_common_physical_profile_for_both_transitions():
    m = Spin1Model(
        _physical_params(
            rf_burn_R=0.37,
            rf_gaussian_fwhm_R=0.08,
            rf_lorentzian_fwhm_R=0.04,
            rf_profile_normalization="center_bin",
        )
    )
    R, physical = m.rf_profile_physical()
    vp, vm = m.rf_profile_arrays()
    assert np.allclose(vp, physical, rtol=0.0, atol=0.0)
    assert np.array_equal(vm, physical[::-1])
    kp, km = m.branch_indices()
    assert kp is not None and km is not None
    assert np.isclose(vp[kp], vm[km], rtol=1e-13, atol=1e-14)
    assert vp[kp] > 0.99


def test_voigt_rf_term_is_full_three_level_rate_equation():
    m = Spin1Model(
        _physical_params(
            p0=0.45,
            rf_burn_R=0.23,
            gamma_rf=1.9,
            rf_gaussian_fwhm_R=0.35,
            rf_lorentzian_fwhm_R=0.18,
            capacity_rate_power=0.7,
        )
    )
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


def test_diffusion_is_zero_in_a_uniform_boltzmann_packet_state():
    m = Spin1Model(Spin1Params(diffusion_scale=80.0, cross_branch_ratio=1.0))
    terms = m._spin_diffusion_terms(dnp_on=False)
    dn = sum(terms.values())
    assert np.max(np.abs(dn)) < 2e-16


def test_internal_diffusion_conserves_every_packet_mass_and_total_vector_p():
    p = Spin1Params(
        diffusion_scale=5.0,
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
    p = _physical_params(
        p0=0.45,
        rf_burn_R=R,
        gamma_rf=6.0,
        rf_gaussian_fwhm_R=0.0,
        rf_lorentzian_fwhm_R=0.0,
        dt=5e-4,
    )
    m = Spin1Model(p)
    for _ in range(burn_steps):
        m.step(1, rf_on=True, dnp_on=False)
    m.params.diffusion_scale = 80.0
    m._diffusion_kernel_key = None
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


def test_voigt_burn_package_trajectory_parity():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    pkg = root / "physics" / "spin1_ssrf_realtime_voigt_burn"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from ssrf_realtime.model import Spin1Model as RefModel, Spin1Params as RefParams

    from physics.ssrf_realtime_v2.rate_equations_realtime import create_voigt_burn_model

    ref = RefModel(RefParams(gamma_rf=1.0, diffusion_scale=5.0, dnp_enabled=False, t1_rate=0.0))
    v2 = create_voigt_burn_model(gamma_rf=1.0)
    burn_R = float(v2.params.rf_burn_R)
    for _ in range(100):
        ref.step(1, rf_on=True, dnp_on=False)
        v2.step(1, rf_on=True, dnp_on=False)
    assert np.allclose(ref.n, v2.n, rtol=1e-12, atol=1e-14)
    loc = v2.local_intensities(burn_R)
    assert loc["Iminus"] >= loc["Iplus"]


def test_branch_order_preserved_at_moderate_burn():
    from physics.ssrf_realtime_v2.rate_equations_realtime import create_voigt_burn_model

    m = create_voigt_burn_model(gamma_rf=1.0)
    burn_R = float(m.params.rf_burn_R)
    for _ in range(100):
        m.step(1, rf_on=True, dnp_on=False)
    loc = m.local_intensities(burn_R)
    assert loc["Iminus"] >= loc["Iplus"]


def test_isolated_ideal_bin_mirror_relation_emerges_from_population_equations():
    m = Spin1Model(
        _physical_params(
            rf_burn_R=0.4,
            gamma_rf=1.7,
            rf_gaussian_fwhm_R=0.0,
            rf_lorentzian_fwhm_R=0.0,
        )
    )
    _, parts = m.derivative(rf_on=True, dnp_on=False, breakdown=True)
    rf = parts["RF"]
    assert np.isclose(
        rf["dIminus_minusR_dt"], -0.5 * rf["dIplus_R_dt"], rtol=1e-12, atol=1e-14
    )
    assert np.isclose(
        rf["dIplus_minusR_dt"], -0.5 * rf["dIminus_R_dt"], rtol=1e-12, atol=1e-14
    )
