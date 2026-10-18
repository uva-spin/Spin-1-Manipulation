"""Unit tests for scalar / range / list RF knob resolution."""
from __future__ import annotations

import numpy as np
import pytest

from Data_Creation.spectrum_gen import events as ev
from Data_Creation.spectrum_gen.constants import (
    COMBO_LAYOUT_PROFILE_AFP_SELECTIVE,
    COMBO_LAYOUT_PROFILE_PROFILE,
    COMBO_LAYOUT_SELECTIVE_SELECTIVE,
    COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE,
)
from Data_Creation.spectrum_gen.params import (
    describe_param,
    expand_param_product,
    is_list_spec,
    is_range_spec,
    resolve_for_polarization,
)


def test_classify_scalar_range_list():
    assert not is_range_spec(10.0)
    assert not is_list_spec(10.0)
    assert is_range_spec((1.0, 2.0))
    assert not is_list_spec((1.0, 2.0))
    assert is_list_spec([1.0, 2.0])
    assert not is_range_spec([1.0, 2.0])


def test_expand_product_mixed():
    combos = expand_param_product(
        {
            'gamma_rf': [5.0, 10.0],
            'gaussian_fwhm_R': (0.02, 0.05),
            'lorentzian_fwhm_R': 0.015,
        }
    )
    assert len(combos) == 2
    assert combos[0]['gamma_rf'] == 5.0
    assert combos[1]['gamma_rf'] == 10.0
    assert combos[0]['gaussian_fwhm_R'] == (0.02, 0.05)
    assert combos[0]['lorentzian_fwhm_R'] == 0.015


def test_resolve_scalar_and_range():
    rng = np.random.default_rng(0)
    assert resolve_for_polarization(10.0, rng) == 10.0
    samples = [resolve_for_polarization((2.0, 4.0), rng) for _ in range(20)]
    assert all(2.0 <= s <= 4.0 for s in samples)
    assert len({round(s, 6) for s in samples}) > 1


def test_resolve_afp_window_int_range():
    rng = np.random.default_rng(1)
    samples = [resolve_for_polarization((4, 8), rng, as_int=True) for _ in range(30)]
    assert all(isinstance(s, (int, np.integer)) for s in samples)
    assert all(4 <= int(s) <= 8 for s in samples)


def test_resolve_rejects_unexpanded_list():
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        resolve_for_polarization([1.0, 2.0], rng)


def test_describe_param():
    assert 'fixed' in describe_param(10.0)
    assert 'range' in describe_param((1.0, 2.0))
    assert 'grid' in describe_param([1.0, 2.0])


def test_afp_bin_stride_keeps_every_nth_affected_bin(monkeypatch):
    seen = []

    def centers(*_a, **_k):
        return np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int32)

    def run(bin_idx, _p0, **_k):
        seen.append(int(bin_idx))
        z = np.zeros((1, 8))
        return {'iplus_full': z, 'iminus_full': z, 'p_full': np.zeros(1), 'q_full': np.zeros(1)}

    monkeypatch.setattr(ev, 'q_negative_bins_for_p0', centers)
    monkeypatch.setattr(ev, 'run_afp_one', run)
    kwargs = dict(afp_window=8, num_bins=8, dt=0.001, rng=np.random.default_rng(0))
    grid = np.asarray([0])
    window = np.arange(8)
    ev.generate_afp_events([0.4], window, grid, bin_stride=0, **kwargs)
    assert seen == [10, 11, 12, 13, 14, 15]
    seen.clear()
    ev.generate_afp_events([0.4], window, grid, bin_stride=2, **kwargs)
    assert seen == [10, 12, 14]


def test_ssrf_range_sampled_once_per_bin(monkeypatch):
    """A range tuple is drawn once per center and reused for every step of that burn."""
    gammas = []

    def centers(*_a, **_k):
        return np.asarray([1, 2], dtype=np.int32)

    def run(bin_idx, _p0, *, gamma_rf, n_steps, **_k):
        gammas.append(float(gamma_rf))
        n = int(n_steps) + 1
        z = np.zeros((n, 8))
        return {'iplus_full': z, 'iminus_full': z, 'p_full': np.zeros(n), 'q_full': np.zeros(n)}

    monkeypatch.setattr(ev, 'q_negative_bins_for_p0', centers)
    monkeypatch.setattr(ev, 'run_ssrf_one', run)
    rows = ev.generate_ssrf_events(
        [0.4],
        np.arange(8),
        np.asarray([0, 1, 2]),
        gamma_rf=(1.0, 9.0),
        gaussian_fwhm_R=0.03,
        lorentzian_fwhm_R=0.015,
        dt=0.001,
        num_bins=8,
        rng=np.random.default_rng(0),
    )
    assert len(gammas) == 2
    assert gammas[0] != gammas[1]
    powers = np.asarray(rows['applied_power'], dtype=float)
    bins = np.asarray(rows['center_bin'])
    for gamma, center in zip(gammas, (1, 2)):
        got = powers[bins == center]
        assert got.size == 3
        assert np.all(got == gamma)


def test_combined_gamma_range_only_on_selective_ssrf(monkeypatch):
    """Profile ssRF layouts keep the default gamma; selective burns draw the range."""
    seen = []

    def run(_p0, **kwargs):
        seen.append((int(kwargs['layout']), float(kwargs['gamma_rf'])))
        return {'skipped': True}

    monkeypatch.setattr(ev, 'run_combined_polarization', run)
    ev.generate_combined_ssrf_afp_events(
        [0.4],
        ev._burn_window_bins(),
        np.asarray([0, 1]),
        np.asarray([0]),
        gamma_rf=(1.0, 9.0),
        max_relax=0,
        afp_window=8,
        profile_settings=None,
        combined_max_centers=1,
        rng=np.random.default_rng(0),
    )
    profile_layouts = {COMBO_LAYOUT_PROFILE_PROFILE, COMBO_LAYOUT_PROFILE_AFP_SELECTIVE}
    selective_layouts = {COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE, COMBO_LAYOUT_SELECTIVE_SELECTIVE}
    profile = [g for layout, g in seen if layout in profile_layouts]
    selective = [g for layout, g in seen if layout in selective_layouts]
    assert profile
    assert selective
    assert all(g == ev.GAMMA_RF for g in profile)
    assert all(1.0 <= g <= 9.0 for g in selective)
    assert any(g != ev.GAMMA_RF for g in selective)
