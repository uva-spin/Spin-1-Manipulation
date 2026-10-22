"""Smoke tests for Ps-only LSTM v3 (projected) and v4 (plain)."""
import sys
from pathlib import Path

import pytest
import torch
import torch.utils.data as data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lstm_v3 import PsProjLstmModel
from lstm import LstmModel
from utils.helpers import keep_ps_channel


def test_keep_ps_channel_slices_to_one():
    x = torch.randn(4, 10, 3)
    yp = torch.randn(4)
    yq = torch.randn(4)
    ds = data.TensorDataset(x, yp, yq)
    stats = {'input_size': 3, 'n_train': 4}
    (train_ds, val_ds, test_ds, out) = keep_ps_channel(ds, ds, ds, stats)
    assert out['input_size'] == 1
    assert train_ds.tensors[0].shape == (4, 10, 1)
    assert val_ds.tensors[0].shape == test_ds.tensors[0].shape == (4, 10, 1)
    torch.testing.assert_close(train_ds.tensors[0], x[..., :1])


def test_v3_forward_ps_only():
    torch.manual_seed(0)
    model = PsProjLstmModel(hidden_size=16, num_layers=1, proj_dim=8)
    model.eval()
    x = torch.randn(2, 20, 1)
    with torch.no_grad():
        pred_p, pred_q = model(x)
    assert pred_p.shape == pred_q.shape == (2,)
    assert torch.isfinite(pred_p).all() and torch.isfinite(pred_q).all()


def test_v3_rejects_three_channel_input():
    model = PsProjLstmModel(hidden_size=16, num_layers=1, proj_dim=8)
    with pytest.raises(ValueError, match=r'\[B, L, 1\]'):
        model(torch.randn(2, 10, 3))


def test_v4_forward_ps_only():
    torch.manual_seed(0)
    model = LstmModel(input_size=1, hidden_size=16, num_layers=1)
    model.eval()
    x = torch.randn(2, 20, 1)
    with torch.no_grad():
        pred_p, pred_q = model(x)
    assert pred_p.shape == pred_q.shape == (2,)
    assert torch.isfinite(pred_p).all() and torch.isfinite(pred_q).all()
