"""
1D Inception: manipulated Ps spectrum → scalar P_total and Q_total.

Same task and data as ``ml/cnn.py``. Each Inception module (InceptionTime)
reads the frequency axis at several kernel widths in parallel, concatenates
those branches, and a residual shortcut is added every three modules.
Adaptive average and max pools then keep a coarse frequency grid for the
P and Q heads. Per-bin channels:

  [Ps, power_profile, n_steps]  (n_steps is broadcast to every bin)

Targets:
  P_total, Q_total from ``spectra.npz`` (population n+−n− / n+−2n0+n−)

Usage:
  python ml/inception.py --spectra ml/data/spectra_v7 --epochs 30
  python ml/inception.py --test-only --checkpoint ml/results/inception/inception_pq_results_v1/inception_pq_best.pth
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lstm as lstm_pq

DEFAULT_SPECTRA_PATH = lstm_pq.DEFAULT_SPECTRA_PATH
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'results' / 'inception' / 'inception_pq_results_v1'
SEED = lstm_pq.SEED
DEVICE = lstm_pq.DEVICE
NUM_EPOCHS = lstm_pq.NUM_EPOCHS
BATCH_SIZE = lstm_pq.BATCH_SIZE
LEARNING_RATE = lstm_pq.LEARNING_RATE
N_FILTERS = 64
KERNEL_SIZES = (1, 3, 5, 7)
BOTTLENECK_CHANNELS = 32
DEPTH = 2
RESIDUAL_EVERY = 3
POOL_BINS = 32
DROPOUT = 0.0
NOISE_STD = lstm_pq.NOISE_STD
N_EXAMPLE_PLOTS = lstm_pq.N_EXAMPLE_PLOTS

class InceptionModule(nn.Module):
    """Parallel same-length convs at several kernel widths, then concatenate.

    A 1x1 bottleneck feeds the wide kernels. A max-pool branch keeps the
    raw input scale. Batch norm and ReLU follow the concatenation.
    """

    def __init__(self, in_channels, n_filters, kernel_sizes, bottleneck_channels):
        super().__init__()
        kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if any(k < 1 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError(f'kernel_sizes must be positive odd integers, got {kernel_sizes}')
        self.kernel_sizes = kernel_sizes
        self.use_bottleneck = int(in_channels) > 1 and int(bottleneck_channels) > 0
        bottleneck_in = int(bottleneck_channels) if self.use_bottleneck else int(in_channels)
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_in, kernel_size=1, bias=False)
        else:
            self.bottleneck = nn.Identity()
        self.convs = nn.ModuleList(
            nn.Conv1d(bottleneck_in, n_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)
        out_channels = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        hidden = self.bottleneck(x)
        branches = [conv(hidden) for conv in self.convs]
        branches.append(self.pool_conv(self.pool(x)))
        return self.relu(self.bn(torch.cat(branches, dim=1)))


class InceptionPQModel(nn.Module):
    """InceptionTime stack over frequency bins, then a coarse bin grid → P/Q heads.

    A residual shortcut is added after every ``residual_every`` modules so
    the stack can stay deep. Adaptive average and max pools keep where a
    burn sits along the spectrum.
    """

    def __init__(
        self,
        input_size=3,
        n_filters=N_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        bottleneck_channels=BOTTLENECK_CHANNELS,
        depth=DEPTH,
        residual_every=RESIDUAL_EVERY,
        dropout=DROPOUT,
        pool_bins=POOL_BINS,
    ):
        super().__init__()
        kernel_sizes = tuple(int(k) for k in kernel_sizes)
        if int(n_filters) < 1:
            raise ValueError(f'n_filters must be positive, got {n_filters}')
        if int(depth) < 1:
            raise ValueError(f'depth must be positive, got {depth}')
        if int(residual_every) < 1:
            raise ValueError(f'residual_every must be positive, got {residual_every}')
        if int(pool_bins) < 1:
            raise ValueError(f'pool_bins must be positive, got {pool_bins}')
        if int(input_size) < 1:
            raise ValueError(f'input_size must be positive, got {input_size}')
        if not kernel_sizes or any(k < 1 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError(f'kernel_sizes must be positive odd integers, got {kernel_sizes}')
        self.input_size = int(input_size)
        self.n_filters = int(n_filters)
        self.kernel_sizes = kernel_sizes
        self.bottleneck_channels = int(bottleneck_channels)
        self.depth = int(depth)
        self.residual_every = int(residual_every)
        self.dropout_p = float(dropout)
        self.pool_bins = int(pool_bins)
        modules = []
        shortcuts = nn.ModuleDict()
        in_channels = self.input_size
        for index in range(self.depth):
            module = InceptionModule(
                in_channels,
                self.n_filters,
                self.kernel_sizes,
                self.bottleneck_channels,
            )
            modules.append(module)
            out_channels = module.out_channels
            if (index + 1) % self.residual_every == 0:
                if index + 1 == self.residual_every:
                    block_in = self.input_size
                else:
                    block_in = modules[index - self.residual_every].out_channels
                if block_in == out_channels:
                    shortcuts[str(index)] = nn.Identity()
                else:
                    shortcuts[str(index)] = nn.Sequential(
                        nn.Conv1d(block_in, out_channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(out_channels),
                    )
            in_channels = out_channels
        self.modules_list = nn.ModuleList(modules)
        self.shortcuts = shortcuts
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool1d(self.pool_bins)
        self.max_pool = nn.AdaptiveMaxPool1d(self.pool_bins)
        self.dropout = nn.Dropout(self.dropout_p)
        head_in = in_channels * 2 * self.pool_bins
        self.head_p = nn.Linear(head_in, 1)
        self.head_q = nn.Linear(head_in, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        hidden = x.transpose(1, 2)
        residual_in = hidden
        for index, module in enumerate(self.modules_list):
            hidden = module(hidden)
            key = str(index)
            if key in self.shortcuts:
                shortcut = self.shortcuts[key]
                hidden = self.relu(hidden + shortcut(residual_in))
                residual_in = hidden
        pooled = torch.cat((self.avg_pool(hidden), self.max_pool(hidden)), dim=1)
        ctx = self.dropout(pooled.flatten(1))
        return (self.head_p(ctx).squeeze(-1), self.head_q(ctx).squeeze(-1))


def save_checkpoint(
    path,
    *,
    model,
    stats,
    best_val_loss,
    best_epoch,
    n_filters,
    kernel_sizes,
    bottleneck_channels,
    depth,
    residual_every,
    dropout,
    pool_bins,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': lstm_pq.clone_state_dict(model),
            'stats': lstm_pq._stats_for_checkpoint(stats),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'n_filters': int(n_filters),
            'kernel_sizes': tuple(kernel_sizes),
            'bottleneck_channels': int(bottleneck_channels),
            'depth': int(depth),
            'residual_every': int(residual_every),
            'dropout': dropout,
            'pool_bins': int(pool_bins),
            'input_size': stats['input_size'],
        },
        path,
    )


def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = InceptionPQModel(
        input_size=ckpt.get('input_size', ckpt['stats']['input_size']),
        n_filters=ckpt.get('n_filters', N_FILTERS),
        kernel_sizes=ckpt.get('kernel_sizes', KERNEL_SIZES),
        bottleneck_channels=ckpt.get('bottleneck_channels', BOTTLENECK_CHANNELS),
        depth=ckpt.get('depth', DEPTH),
        residual_every=ckpt.get('residual_every', RESIDUAL_EVERY),
        dropout=ckpt.get('dropout', DROPOUT),
        pool_bins=ckpt.get('pool_bins', POOL_BINS),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)


def train_model(train_dataset, val_dataset, stats, *, n_filters, kernel_sizes, bottleneck_channels, depth, residual_every, dropout, pool_bins, num_epochs, batch_size, learning_rate, checkpoint_path=None, device=DEVICE):
    model = InceptionPQModel(
        input_size=stats['input_size'], n_filters=n_filters, kernel_sizes=kernel_sizes,
        bottleneck_channels=bottleneck_channels, depth=depth, residual_every=residual_every,
        dropout=dropout, pool_bins=pool_bins,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)

    def on_best(trained, best_val, best_epoch):
        save_checkpoint(
            checkpoint_path, model=trained, stats=stats, best_val_loss=best_val, best_epoch=best_epoch,
            n_filters=n_filters, kernel_sizes=kernel_sizes, bottleneck_channels=bottleneck_channels,
            depth=depth, residual_every=residual_every, dropout=dropout, pool_bins=pool_bins,
        )

    return lstm_pq.fit_pq_model(
        model, train_dataset, val_dataset, stats,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        on_best=on_best if checkpoint_path is not None else None,
        load_best=lambda: lstm_pq.reload_checkpoint(load_checkpoint, checkpoint_path, device),
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Train/evaluate 1D Inception: Ps spectrum + per-bin power profile → P_total, Q_total.',
    )
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz or memmap directory')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--n-filters', type=int, default=N_FILTERS, help='Filters per Inception branch (default: 64)')
    parser.add_argument(
        '--kernel-sizes',
        type=lstm_pq.parse_int_tuple,
        default=KERNEL_SIZES,
        help='Odd kernel widths of the parallel conv branches (default: 1,3,5,7)',
    )
    parser.add_argument(
        '--bottleneck-channels',
        type=int,
        default=BOTTLENECK_CHANNELS,
        help='1x1 bottleneck width before the wide kernels (default: 32)',
    )
    parser.add_argument('--depth', type=int, default=DEPTH, help='Number of Inception modules (default: 2)')
    parser.add_argument(
        '--residual-every',
        type=int,
        default=RESIDUAL_EVERY,
        help='Add a residual shortcut after this many modules (default: 3)',
    )
    parser.add_argument('--pool-bins', type=int, default=POOL_BINS, help='Coarse frequency bins kept after conv (default: 32)')
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--max-samples', type=int, default=None, help='Optional subsample size; omit to train on the full dataset')
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--n-examples', type=int, default=N_EXAMPLE_PLOTS, help='Number of per-sample example plots to save')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint path (default: <out-dir>/inception_pq_best.pth)')
    parser.add_argument('--test-only', action='store_true', help='Skip training; load --checkpoint and evaluate on a fresh split')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out_dir / 'inception_pq_best.pth'
    spectra_path = args.spectra or DEFAULT_SPECTRA_PATH
    print(f'Loading spectra from {spectra_path} ...', flush=True)
    arrays = lstm_pq.load_lstm_npz(spectra_path)
    print('Preparing datasets...', flush=True)
    (train_ds, val_ds, test_ds, stats) = lstm_pq.prepare_datasets(
        arrays, max_samples=args.max_samples, noise_std=args.noise_std,
    )
    dataset_stats = {k: stats[k] for k in lstm_pq.STATS_KEYS if k in stats}
    print(
        f"Train={stats['n_train']} Val={stats['n_val']} Test={stats['n_test']} bins={stats['num_bins']}",
        flush=True,
    )
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
        print('Training Inception model...', flush=True)
        (model, best_val, history) = train_model(
            train_ds,
            val_ds,
            stats,
            n_filters=args.n_filters,
            kernel_sizes=args.kernel_sizes,
            bottleneck_channels=args.bottleneck_channels,
            depth=args.depth,
            residual_every=args.residual_every,
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
