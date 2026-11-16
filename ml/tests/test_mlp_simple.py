"""Smoke test for the Ps-only residual MLP."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlp_simple import MlpV2Plain, ResidualBlock, attach_clean_targets, load_spectra_npz


def test_blocks_use_layernorm_not_batchnorm():
    model = MlpV2Plain(input_size=1, hidden=8, num_blocks=3, num_bins=20)
    types = {type(module) for module in model.modules()}
    assert nn.LayerNorm in types
    assert nn.BatchNorm1d not in types
    assert len(model.blocks) == 3
    assert all(isinstance(block, ResidualBlock) for block in model.blocks)
    for block in model.blocks:
        assert isinstance(block.ln1, nn.LayerNorm)
        assert isinstance(block.ln2, nn.LayerNorm)


def test_residual_block_starts_as_identity():
    model = MlpV2Plain(input_size=1, hidden=8, num_blocks=3, num_bins=20)
    x = torch.randn(4, 8)
    with torch.no_grad():
        assert torch.allclose(model.blocks[0](x), x)


def test_mlp_simple_splits_heads():
    model = MlpV2Plain(input_size=1, hidden=8, num_blocks=3, num_bins=20).cpu()
    model.eval()
    with torch.no_grad():
        pred_p, pred_q = model(torch.randn(4, 20, 1))
    assert pred_p.shape == pred_q.shape == (4,)
    assert torch.isfinite(pred_p).all() and torch.isfinite(pred_q).all()


def test_heads_read_reconstructed_spectrum():
    model = MlpV2Plain(input_size=1, hidden=8, num_blocks=2, num_bins=20).cpu()
    model.eval()
    x = torch.randn(4, 20, 1)
    with torch.no_grad():
        pred_p, pred_q, recon = model(x, return_recon=True)
        from_recon_p = model.head_p(recon).squeeze(-1)
        from_recon_q = model.head_q(recon).squeeze(-1)
    assert recon.shape == (4, 20)
    assert torch.allclose(pred_p, from_recon_p)
    assert torch.allclose(pred_q, from_recon_q)


def _save_run(path, *, spectrum, final_spectrum, duration=None, final_P=None, final_Q=None):
    n, bins = spectrum.shape
    if duration is None:
        duration = np.zeros(n, dtype=np.float64)
    if final_P is None:
        final_P = np.array([0.11, 0.22, 0.33, 0.44])
    if final_Q is None:
        final_Q = np.array([0.55, 0.66, 0.77, 0.88])
    np.savez(
        path,
        spectrum_plus=np.zeros((n, bins), dtype=np.float32),
        spectrum_minus=np.ones((n, bins), dtype=np.float32),
        powers=np.zeros((n, bins), dtype=np.float32),
        manipulation=np.array(['states_only'] * n),
        duration=duration,
        p0=np.linspace(0.2, 0.5, n),
        spectrum=spectrum,
        final_spectrum=final_spectrum,
        state_P=np.array([0.21, 0.22, 0.23, 0.24]),
        state_Q=np.array([0.31, 0.32, 0.33, 0.34]),
        final_P=final_P,
        final_Q=final_Q,
    )


def test_nonfinite_final_fields_use_state_targets(tmp_path):
    n, bins = 4, 8
    spectrum = np.arange(n * bins, dtype=np.float32).reshape(n, bins)
    path = tmp_path / 'pq_45_55.npz'
    _save_run(
        path,
        spectrum=spectrum,
        final_spectrum=np.full((n, bins), np.nan, dtype=np.float32),
        final_P=np.full(n, np.nan),
        final_Q=np.full(n, np.nan),
    )
    arrays = load_spectra_npz(path)
    assert np.isfinite(arrays['spectra']).all()
    assert np.allclose(arrays['spectra'][:, 0, :], spectrum)
    assert np.allclose(arrays['P_total'], [0.21, 0.22, 0.23, 0.24])
    assert np.allclose(arrays['Q_total'], [0.31, 0.32, 0.33, 0.34])


def test_pq_thermal_uses_spectrum_not_final_spectrum(tmp_path):
    n, bins = 4, 8
    spectrum = np.arange(n * bins, dtype=np.float32).reshape(n, bins)
    final = np.full((n, bins), np.nan, dtype=np.float32)
    thermal = tmp_path / 'pq_thermal.npz'
    other = tmp_path / 'run5.npz'
    _save_run(thermal, spectrum=spectrum, final_spectrum=final)
    _save_run(other, spectrum=np.zeros_like(spectrum), final_spectrum=spectrum)
    thermal_arrays = load_spectra_npz(thermal)
    other_arrays = load_spectra_npz(other)
    assert np.allclose(thermal_arrays['spectra'][:, 0, :], spectrum)
    assert np.allclose(thermal_arrays['P_total'], [0.21, 0.22, 0.23, 0.24])
    assert np.allclose(thermal_arrays['Q_total'], [0.31, 0.32, 0.33, 0.34])
    assert np.allclose(other_arrays['spectra'][:, 0, :], spectrum)
    assert np.allclose(other_arrays['P_total'], [0.11, 0.22, 0.33, 0.44])
    assert np.allclose(other_arrays['Q_total'], [0.55, 0.66, 0.77, 0.88])


def test_thermal_nan_duration_is_zero_steps(tmp_path):
    n, bins = 4, 8
    spectrum = np.arange(n * bins, dtype=np.float32).reshape(n, bins)
    thermal = tmp_path / 'pq_thermal.npz'
    _save_run(
        thermal,
        spectrum=spectrum,
        final_spectrum=np.full((n, bins), np.nan, dtype=np.float32),
        duration=np.full(n, np.nan, dtype=np.float64),
    )
    arrays = load_spectra_npz(thermal)
    assert np.array_equal(arrays['n_steps'], np.zeros(n))


def test_attach_clean_targets_repeats_with_noise_copies():
    n, copies, bins = 3, 4, 8
    x = torch.zeros(n * copies, bins, 1)
    y = torch.zeros(n * copies)
    dataset = data.TensorDataset(x, y, y)
    arrays = {'spectra': np.arange(n * bins, dtype=np.float32).reshape(n, 1, bins)}
    stats = {'ps_mean': 0.0, 'ps_std': 1.0}
    wrapped = attach_clean_targets(dataset, arrays, np.arange(n), stats, copies=copies)
    assert len(wrapped) == n * copies
    clean = wrapped.tensors[3].numpy()
    assert clean.shape == (n * copies, bins)
    assert np.allclose(clean[0], arrays['spectra'][0, 0])
    assert np.allclose(clean[copies], arrays['spectra'][1, 0])
