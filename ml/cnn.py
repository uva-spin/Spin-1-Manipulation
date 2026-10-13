"""
1D CNN: manipulated Ps spectrum → scalar P_total and Q_total.

Same task and data as ``ml/lstm.py``. A stack of residual conv blocks reads
frequency bins as a 1D axis, then adaptive average and max pools keep a
coarse frequency grid for the P and Q heads. Per-bin channels:

  [Ps, power_profile, n_steps]  (n_steps is broadcast to every bin)

Targets:
  P_total, Q_total from ``spectra.npz`` (population n+−n− / n+−2n0+n−)

Usage:
  python ml/cnn.py --spectra ml/data/spectra_v7 --epochs 30
  python ml/cnn.py --test-only --checkpoint ml/results/cnn/cnn_pq_results_v2/cnn_pq_best.pth
"""
import argparse
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

DEFAULT_OUTPUT_DIR = ML_DIR / 'results' / 'cnn' / 'cnn_pq_results_v3'
CHANNELS = (64, 128, 128, 192)
KERNEL_SIZES = (7, 5, 3, 3)
POOL_BINS = 32
DROPOUT = 0.0


class ResidualConv1d(nn.Module):
    """Two same-length convs plus a skip, so later stages can stay deep.

    The skip is identity when the width is unchanged and a 1x1 conv otherwise.
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        hidden = self.relu(self.bn1(self.conv1(x)))
        hidden = self.bn2(self.conv2(hidden))
        return self.relu(hidden + residual)


class CnnPQModel(nn.Module):
    """Residual conv stack over frequency bins, then a coarse bin grid → P/Q heads.

    Adaptive average and max pools keep where a burn sits along the spectrum.
    """

    def __init__(self, input_size=3, channels=CHANNELS, kernel_sizes=KERNEL_SIZES, dropout=DROPOUT, pool_bins=POOL_BINS):
        super().__init__()
        channels = tuple(channels)
        kernel_sizes = tuple(kernel_sizes)
        if not channels:
            raise ValueError('channels must be a non-empty sequence')
        if len(kernel_sizes) != len(channels):
            raise ValueError(f'kernel_sizes length {len(kernel_sizes)} != channels length {len(channels)}')
        if any((k < 1 or k % 2 == 0 for k in kernel_sizes)):
            raise ValueError(f'kernel_sizes must be positive odd integers, got {kernel_sizes}')
        if int(pool_bins) < 1:
            raise ValueError(f'pool_bins must be positive, got {pool_bins}')
        self.input_size = int(input_size)
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dropout_p = float(dropout)
        self.pool_bins = int(pool_bins)
        layers = []
        prev = self.input_size
        for (ch, k) in zip(channels, kernel_sizes):
            layers.append(ResidualConv1d(prev, ch, k))
            prev = ch
        self.encoder = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(self.pool_bins)
        self.max_pool = nn.AdaptiveMaxPool1d(self.pool_bins)
        self.dropout = nn.Dropout(self.dropout_p)
        head_in = prev * 2 * self.pool_bins
        self.head_p = nn.Linear(head_in, 1)
        self.head_q = nn.Linear(head_in, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2))
        pooled = torch.cat((self.avg_pool(h), self.max_pool(h)), dim=1)
        ctx = self.dropout(pooled.flatten(1))
        return (self.head_p(ctx).squeeze(-1), self.head_q(ctx).squeeze(-1))


def save_checkpoint(path, *, model, stats, best_val_loss, best_epoch, channels, kernel_sizes, dropout, pool_bins):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': lstm_pq.clone_state_dict(model),
            'stats': lstm_pq._stats_for_checkpoint(stats),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'channels': tuple(channels),
            'kernel_sizes': tuple(kernel_sizes),
            'dropout': dropout,
            'pool_bins': int(pool_bins),
            'input_size': stats['input_size'],
        },
        path,
    )


def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CnnPQModel(
        input_size=ckpt.get('input_size', ckpt['stats']['input_size']),
        channels=ckpt.get('channels', CHANNELS),
        kernel_sizes=ckpt.get('kernel_sizes', KERNEL_SIZES),
        dropout=ckpt.get('dropout', DROPOUT),
        pool_bins=ckpt.get('pool_bins', POOL_BINS),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)


def train_model(train_dataset, val_dataset, stats, *, channels, kernel_sizes, dropout, pool_bins, num_epochs, batch_size, learning_rate, checkpoint_path=None, device=DEVICE):
    model = CnnPQModel(input_size=stats['input_size'], channels=channels, kernel_sizes=kernel_sizes, dropout=dropout, pool_bins=pool_bins).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)

    def on_best(trained, best_val, best_epoch):
        save_checkpoint(checkpoint_path, model=trained, stats=stats, best_val_loss=best_val, best_epoch=best_epoch, channels=channels, kernel_sizes=kernel_sizes, dropout=dropout, pool_bins=pool_bins)

    return lstm_pq.fit_pq_model(
        model, train_dataset, val_dataset, stats,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        on_best=on_best if checkpoint_path is not None else None,
        load_best=lambda: lstm_pq.reload_checkpoint(load_checkpoint, checkpoint_path, device),
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate 1D CNN: Ps spectrum + per-bin power profile → P_total, Q_total.')
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz or memmap directory')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--channels', type=lstm_pq.parse_int_tuple, default=CHANNELS, help='Residual-block channel widths, comma-separated (default: 64,128,128,192)')
    parser.add_argument('--kernel-sizes', type=lstm_pq.parse_int_tuple, default=KERNEL_SIZES, help='Odd conv kernel sizes, one per residual block (default: 7,5,3,3)')
    parser.add_argument('--pool-bins', type=int, default=POOL_BINS, help='Coarse frequency bins kept after conv (default: 32)')
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--max-samples', type=int, default=None, help='Optional subsample size; omit to train on the full dataset')
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--n-examples', type=int, default=N_EXAMPLE_PLOTS, help='Number of per-sample example plots to save')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint path (default: <out-dir>/cnn_pq_best.pth)')
    parser.add_argument('--test-only', action='store_true', help='Skip training; load --checkpoint and evaluate on a fresh split')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out_dir / 'cnn_pq_best.pth'
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
        if loaded:
            print(f'Loaded training history from {history_path}', flush=True)
    else:
        print('Training CNN model...', flush=True)
        (model, best_val, history) = train_model(
            train_ds,
            val_ds,
            stats,
            channels=args.channels,
            kernel_sizes=args.kernel_sizes,
            dropout=args.dropout,
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
