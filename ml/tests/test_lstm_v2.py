"""Tests for recommendation A: decouple RF profile shape from strength."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lstm_v2 import DecoupledLstmModel, profile_shape_and_strength


def _zscored_profile(gamma, *, mu=0.4, sigma=2.0):
    """Profile channel as prepare_datasets emits it: (gamma * Voigt - mu) / sigma."""
    bins = torch.arange(50, dtype=torch.float32)
    voigt = torch.exp(-0.5 * ((bins - 25.0) / 2.0) ** 2)
    return (gamma * voigt - mu) / sigma


def test_shape_is_gamma_invariant_but_strength_is_not():
    shape_low, strength_low = profile_shape_and_strength(_zscored_profile(2.0).unsqueeze(0))
    shape_high, strength_high = profile_shape_and_strength(_zscored_profile(20.0).unsqueeze(0))

    assert shape_low.shape == (1, 50)
    torch.testing.assert_close(shape_low, shape_high, rtol=1e-5, atol=1e-6)
    assert shape_low.max() == pytest.approx(1.0)
    # Strength (log peak / log integral) grows with gamma even though shape does not.
    assert strength_high[0, 0] > strength_low[0, 0] > 0.0
    assert strength_high[0, 1] > strength_low[0, 1]


def test_zero_profile_has_zero_shape_and_strength():
    shape, strength = profile_shape_and_strength(torch.full((2, 50), -0.2))

    assert torch.count_nonzero(shape) == 0
    torch.testing.assert_close(strength, torch.zeros(2, 2))


def test_forward_is_finite_for_zero_and_strong_profiles():
    torch.manual_seed(0)
    model = DecoupledLstmModel(hidden_size=16, num_layers=1, proj_dim=8, strength_dim=4)
    model.eval()
    ps = torch.randn(3, 50)
    steps = torch.zeros(3, 50)
    profiles = torch.stack([
        torch.full((50,), -0.2),
        _zscored_profile(2.0),
        _zscored_profile(20.0),
    ])
    x = torch.stack([ps, profiles, steps], dim=-1)

    with torch.no_grad():
        pred_p, pred_q = model(x)

    assert pred_p.shape == pred_q.shape == (3,)
    assert torch.isfinite(pred_p).all() and torch.isfinite(pred_q).all()


def test_forward_rejects_wrong_channel_count():
    model = DecoupledLstmModel(hidden_size=16, num_layers=1)
    with pytest.raises(ValueError, match='input_size=3'):
        model(torch.randn(2, 10, 4))
