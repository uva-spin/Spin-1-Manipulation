"""Unit tests for scalar / range / list RF knob resolution."""
from __future__ import annotations

import numpy as np
import pytest

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
