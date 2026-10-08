"""
Transformer encoder: manipulated Ps spectrum → scalar P_total and Q_total.

Same task and data as ``ml/lstm.py``. Frequency bins are a sequence of

  [Ps, power_profile, n_steps]  (n_steps is broadcast to every bin)

A local convolutional token embedding, sinusoidal positional encoding, and
a pre-norm encoder stack read the spectrum. Adaptive average and max pools
then keep a coarse frequency grid for the P and Q heads. The Voigt burn is
about one bin wide (500 bins across R in [-6, 6]), so a global mean would
divide that spike by the sequence length before the head ever sees it.

Targets:
  P_total, Q_total from ``spectra.npz`` (population n+−n− / n+−2n0+n−)

Usage:
  python ml/transformer.py --spectra ml/data/spectra_v7 --epochs 30
  python ml/transformer.py --test-only --checkpoint ml/results/transformer/transformer_pq_results_v1/transformer_pq_best.pth
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils.constants import (
    BATCH_SIZE, DEFAULT_SPECTRA_PATH, DEVICE, LEARNING_RATE, ML_DIR,
    N_EXAMPLE_PLOTS, NOISE_STD, NUM_EPOCHS, SEED, STATS_KEYS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lstm as lstm_pq

DEFAULT_OUTPUT_DIR = ML_DIR / 'results' / 'transformer' / 'transformer_pq_results_v1'
D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 2
DIM_FEEDFORWARD = 512
DROPOUT = 0.0
PATCH_KERNEL = 7
PATCH_STRIDE = 1
POOL_BINS = 32
MAX_POS_LEN = 8192


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoids so attention can tell frequency bins apart."""

    def __init__(self, d_model, max_len=MAX_POS_LEN):
        super().__init__()
        if int(d_model) < 1:
            raise ValueError(f'd_model must be positive, got {d_model}')
        if int(max_len) < 1:
            raise ValueError(f'max_len must be positive, got {max_len}')
        self.max_len = int(max_len)
        position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, int(d_model), 2, dtype=torch.float32) * (-math.log(10000.0) / int(d_model)))
        pe = torch.zeros(self.max_len, int(d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        length = x.size(1)
        if length > self.max_len:
            raise ValueError(f'sequence length {length} exceeds positional encoding max_len {self.max_len}')
        return x + self.pe[:, :length]


class TransformerPQModel(nn.Module):
    """Encoder over local spectral tokens, then a coarse bin grid → P/Q heads.

    Each token is a short convolution over neighboring bins, so a one-bin
    Voigt burn is a local feature before attention. Residuals keep that
    feature in the stream. Average and max pools over a coarse frequency
    grid keep where the burn sits; a global mean would shrink it by the
    number of bins.
    """

    def __init__(
        self,
        input_size=3,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        patch_kernel=PATCH_KERNEL,
        patch_stride=PATCH_STRIDE,
        pool_bins=POOL_BINS,
    ):
        super().__init__()
        if int(input_size) < 1:
            raise ValueError(f'input_size must be positive, got {input_size}')
        if int(d_model) < 1:
            raise ValueError(f'd_model must be positive, got {d_model}')
        if int(nhead) < 1 or int(d_model) % int(nhead) != 0:
            raise ValueError(f'nhead must be a positive divisor of d_model={d_model}, got {nhead}')
        if int(num_layers) < 1:
            raise ValueError(f'num_layers must be positive, got {num_layers}')
        if int(dim_feedforward) < 1:
            raise ValueError(f'dim_feedforward must be positive, got {dim_feedforward}')
        if int(patch_kernel) < 1 or int(patch_kernel) % 2 == 0:
            raise ValueError(f'patch_kernel must be a positive odd integer, got {patch_kernel}')
        if int(patch_stride) < 1:
            raise ValueError(f'patch_stride must be positive, got {patch_stride}')
        if int(pool_bins) < 1:
            raise ValueError(f'pool_bins must be positive, got {pool_bins}')
        self.input_size = int(input_size)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_p = float(dropout)
        self.patch_kernel = int(patch_kernel)
        self.patch_stride = int(patch_stride)
        self.pool_bins = int(pool_bins)
        self.token_embed = nn.Conv1d(
            self.input_size,
            self.d_model,
            kernel_size=self.patch_kernel,
            stride=self.patch_stride,
            padding=self.patch_kernel // 2,
        )
        self.pos_encoding = SinusoidalPositionalEncoding(self.d_model)
        self.input_dropout = nn.Dropout(self.dropout_p)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_p,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.d_model),
            enable_nested_tensor=False,
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(self.pool_bins)
        self.max_pool = nn.AdaptiveMaxPool1d(self.pool_bins)
        head_in = self.d_model * 2 * self.pool_bins
        self.head_p = nn.Linear(head_in, 1)
        self.head_q = nn.Linear(head_in, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        hidden = self.token_embed(x.transpose(1, 2)).transpose(1, 2)
        hidden = self.input_dropout(self.pos_encoding(hidden))
        encoded = self.encoder(hidden)
        features = encoded.transpose(1, 2)
        pooled = torch.cat((self.avg_pool(features), self.max_pool(features)), dim=1)
        ctx = pooled.flatten(1)
        return (self.head_p(ctx).squeeze(-1), self.head_q(ctx).squeeze(-1))


def save_checkpoint(
    path,
    *,
    model,
    stats,
    best_val_loss,
    best_epoch,
    d_model,
    nhead,
    num_layers,
    dim_feedforward,
    dropout,
    patch_kernel,
    patch_stride,
    pool_bins,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': lstm_pq.clone_state_dict(model),
            'stats': lstm_pq._stats_for_checkpoint(stats),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'd_model': int(d_model),
            'nhead': int(nhead),
            'num_layers': int(num_layers),
            'dim_feedforward': int(dim_feedforward),
            'dropout': dropout,
            'patch_kernel': int(patch_kernel),
            'patch_stride': int(patch_stride),
            'pool_bins': int(pool_bins),
            'input_size': stats['input_size'],
        },
        path,
    )


def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TransformerPQModel(
        input_size=ckpt.get('input_size', ckpt['stats']['input_size']),
        d_model=ckpt.get('d_model', D_MODEL),
        nhead=ckpt.get('nhead', NHEAD),
        num_layers=ckpt.get('num_layers', NUM_LAYERS),
        dim_feedforward=ckpt.get('dim_feedforward', DIM_FEEDFORWARD),
        dropout=ckpt.get('dropout', DROPOUT),
        patch_kernel=ckpt.get('patch_kernel', PATCH_KERNEL),
        patch_stride=ckpt.get('patch_stride', PATCH_STRIDE),
        pool_bins=ckpt.get('pool_bins', POOL_BINS),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)


def train_model(train_dataset, val_dataset, stats, *, d_model, nhead, num_layers, dim_feedforward, dropout, patch_kernel, patch_stride, pool_bins, num_epochs, batch_size, learning_rate, checkpoint_path=None, device=DEVICE):
    model = TransformerPQModel(
        input_size=stats['input_size'], d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_feedforward, dropout=dropout, patch_kernel=patch_kernel,
        patch_stride=patch_stride, pool_bins=pool_bins,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Transformer d_model={d_model} nhead={nhead} layers={num_layers} ff={dim_feedforward} patch={patch_kernel}/{patch_stride} pool={pool_bins} | trainable parameters: {n_trainable:,}', flush=True)

    def on_best(trained, best_val, best_epoch):
        save_checkpoint(
            checkpoint_path, model=trained, stats=stats, best_val_loss=best_val, best_epoch=best_epoch,
            d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, patch_kernel=patch_kernel, patch_stride=patch_stride, pool_bins=pool_bins,
        )

    return lstm_pq.fit_pq_model(
        model, train_dataset, val_dataset, stats,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        on_best=on_best if checkpoint_path is not None else None,
        load_best=lambda: lstm_pq.reload_checkpoint(load_checkpoint, checkpoint_path, device),
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate transformer encoder: Ps spectrum + per-bin power profile → P_total, Q_total.')
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz or memmap directory')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--d-model', type=int, default=D_MODEL, help='Transformer width (default: 128)')
    parser.add_argument('--nhead', type=int, default=NHEAD, help='Attention heads (default: 8)')
    parser.add_argument('--num-layers', type=int, default=NUM_LAYERS, help='Encoder layers (default: 2)')
    parser.add_argument('--ff-dim', type=int, default=DIM_FEEDFORWARD, help='Feed-forward width inside each layer (default: 512)')
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--patch-kernel', type=int, default=PATCH_KERNEL, help='Odd conv width of each token (default: 7)')
    parser.add_argument('--patch-stride', type=int, default=PATCH_STRIDE, help='Token stride along frequency (default: 1)')
    parser.add_argument('--pool-bins', type=int, default=POOL_BINS, help='Coarse frequency grid kept for the heads (default: 32)')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional subsample size; omit to train on the full dataset')
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--n-examples', type=int, default=N_EXAMPLE_PLOTS, help='Number of per-sample example plots to save')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint path (default: <out-dir>/transformer_pq_best.pth)')
    parser.add_argument('--test-only', action='store_true', help='Skip training; load --checkpoint and evaluate on a fresh split')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out_dir / 'transformer_pq_best.pth'
    spectra_path = args.spectra or DEFAULT_SPECTRA_PATH
    print(f'Loading spectra from {spectra_path} ...', flush=True)
    arrays = lstm_pq.load_lstm_npz(spectra_path)
    print('Preparing datasets...', flush=True)
    (train_ds, val_ds, test_ds, stats) = lstm_pq.prepare_datasets(arrays, max_samples=args.max_samples, noise_std=args.noise_std)
    dataset_stats = {k: stats[k] for k in STATS_KEYS if k in stats}
    print(f"Train={stats['n_train']} Val={stats['n_val']} Test={stats['n_test']} bins={stats['num_bins']}", flush=True)
    history_path = args.out_dir / 'history.json'
    if args.test_only:
        print(f'Loading checkpoint {checkpoint_path} ...', flush=True)
        (model, ckpt) = load_checkpoint(checkpoint_path)
        stats = {**stats, **ckpt['stats']}
        stats['test_idx'] = dataset_stats['test_idx']
        best_val = ckpt.get('best_val_loss', float('nan'))
        loaded = lstm_pq.load_history_json(history_path)
        history = loaded or {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
        total_params = sum((p.numel() for p in model.parameters()))
        trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
        print(f'Total Params: {total_params:,}', flush=True)
        print(f'Trainable Params: {trainable_params:,}', flush=True)
        print(f'Non-Trainable Params: {total_params - trainable_params:,}', flush=True)
        if loaded:
            print(f'Loaded training history from {history_path}', flush=True)
    else:
        print('Training transformer model...', flush=True)
        (model, best_val, history) = train_model(
            train_ds,
            val_ds,
            stats,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.ff_dim,
            dropout=args.dropout,
            patch_kernel=args.patch_kernel,
            patch_stride=args.patch_stride,
            pool_bins=args.pool_bins,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_path=checkpoint_path,
        )
        lstm_pq.save_history_json(history, history_path)
    lstm_pq.write_pq_report(
        model, test_ds, stats, dataset_stats, arrays, history, best_val, args.out_dir, checkpoint_path,
        batch_size=args.batch_size, n_examples=args.n_examples, seed=SEED,
    )


if __name__ == '__main__':
    main()
