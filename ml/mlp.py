"""
Residual MLP: manipulated Ps spectrum → scalar P_total and Q_total.

Same task and data as ``ml/cnn.py``. Frequency bins are flattened and read
by residual linear blocks, then separate heads predict P and Q. The default
widths are sized to about the same trainable-parameter count as
``CnnPQModel`` on the 500-bin spectra.

Per-bin channels:
  [Ps, power_profile, n_steps]  (n_steps is broadcast to every bin)

Targets:
  P_total, Q_total from ``spectra.npz`` (population n+−n− / n+−2n0+n−)

Usage:
  python ml/mlp.py --spectra ml/data/spectra_v7 --epochs 30
  python ml/mlp.py --test-only --checkpoint ml/results/mlp/mlp_pq_results_v1/mlp_pq_best.pth
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'results' / 'mlp' / 'mlp_pq_results_v3'
SEED = lstm_pq.SEED
DEVICE = lstm_pq.DEVICE
NUM_EPOCHS = lstm_pq.NUM_EPOCHS
BATCH_SIZE = lstm_pq.BATCH_SIZE
LEARNING_RATE = lstm_pq.LEARNING_RATE
NUM_BINS = 500
HIDDEN_DIMS = (144, 144)
DROPOUT = 0.0
NOISE_STD = lstm_pq.NOISE_STD
N_EXAMPLE_PLOTS = lstm_pq.N_EXAMPLE_PLOTS


class ResidualMLP(nn.Module):
    """Two linears plus a skip. The skip is identity when the width is unchanged."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        if in_dim == out_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        residual = self.skip(x)
        hidden = self.relu(self.bn1(self.fc1(x)))
        hidden = self.bn2(self.fc2(hidden))
        return self.relu(hidden + residual)


class MlpPQModel(nn.Module):
    """Flattened spectrum → residual MLP → P and Q heads.

    Inputs whose bin count differs from ``num_bins`` are average-pooled
    along frequency first, so a checkpoint still runs.
    """

    def __init__(self, input_size=3, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT, num_bins=NUM_BINS):
        super().__init__()
        hidden_dims = tuple(int(h) for h in hidden_dims)
        if not hidden_dims or any(h < 1 for h in hidden_dims):
            raise ValueError(f'hidden_dims must be positive integers, got {hidden_dims}')
        if int(num_bins) < 1:
            raise ValueError(f'num_bins must be positive, got {num_bins}')
        if int(input_size) < 1:
            raise ValueError(f'input_size must be positive, got {input_size}')
        self.input_size = int(input_size)
        self.hidden_dims = hidden_dims
        self.dropout_p = float(dropout)
        self.num_bins = int(num_bins)
        self.pool = nn.AdaptiveAvgPool1d(self.num_bins)
        layers = []
        prev = self.input_size * self.num_bins
        for width in hidden_dims:
            layers.append(ResidualMLP(prev, width))
            prev = width
        self.encoder = nn.Sequential(*layers)
        self.dropout = nn.Dropout(self.dropout_p)
        self.head_p = nn.Linear(prev, 1)
        self.head_q = nn.Linear(prev, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        for head in (self.head_p, self.head_q):
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, 0.0)

    def forward(self, x):
        if x.shape[1] != self.num_bins:
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(self.encoder(x.flatten(1)))
        return (self.head_p(hidden).squeeze(-1), self.head_q(hidden).squeeze(-1))


def save_checkpoint(path, *, model, stats, best_val_loss, best_epoch, hidden_dims, dropout, num_bins):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': lstm_pq.clone_state_dict(model),
            'stats': lstm_pq._stats_for_checkpoint(stats),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'hidden_dims': tuple(hidden_dims),
            'dropout': dropout,
            'num_bins': int(num_bins),
            'input_size': stats['input_size'],
        },
        path,
    )


def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MlpPQModel(
        input_size=ckpt.get('input_size', ckpt['stats']['input_size']),
        hidden_dims=ckpt.get('hidden_dims', HIDDEN_DIMS),
        dropout=ckpt.get('dropout', DROPOUT),
        num_bins=ckpt.get('num_bins', NUM_BINS),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)


def train_model(train_dataset, val_dataset, stats, *, hidden_dims, dropout, num_bins, num_epochs, batch_size, learning_rate, checkpoint_path=None, device=DEVICE):
    model = MlpPQModel(input_size=stats['input_size'], hidden_dims=hidden_dims, dropout=dropout, num_bins=num_bins).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)

    def on_best(trained, best_val, best_epoch):
        save_checkpoint(checkpoint_path, model=trained, stats=stats, best_val_loss=best_val, best_epoch=best_epoch, hidden_dims=hidden_dims, dropout=dropout, num_bins=num_bins)

    return lstm_pq.fit_pq_model(
        model, train_dataset, val_dataset, stats,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        on_best=on_best if checkpoint_path is not None else None,
        load_best=lambda: lstm_pq.reload_checkpoint(load_checkpoint, checkpoint_path, device),
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Train/evaluate residual MLP: Ps spectrum + per-bin power profile → P_total, Q_total.',
    )
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz or memmap directory')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument(
        '--hidden-dims',
        type=lstm_pq.parse_int_tuple,
        default=HIDDEN_DIMS,
        help='Residual-block widths, comma-separated (default: 144,144)',
    )
    parser.add_argument(
        '--num-bins',
        type=int,
        default=None,
        help='Frequency bins the MLP flattens (default: the spectra bin count)',
    )
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--max-samples', type=int, default=None, help='Optional subsample size; omit to train on the full dataset')
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--n-examples', type=int, default=N_EXAMPLE_PLOTS, help='Number of per-sample example plots to save')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint path (default: <out-dir>/mlp_pq_best.pth)')
    parser.add_argument('--test-only', action='store_true', help='Skip training; load --checkpoint and evaluate on a fresh split')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out_dir / 'mlp_pq_best.pth'
    spectra_path = args.spectra or DEFAULT_SPECTRA_PATH
    print(f'Loading spectra from {spectra_path} ...', flush=True)
    arrays = lstm_pq.load_lstm_npz(spectra_path)
    print('Preparing datasets...', flush=True)
    (train_ds, val_ds, test_ds, stats) = lstm_pq.prepare_datasets(
        arrays, max_samples=args.max_samples, noise_std=args.noise_std,
    )
    dataset_stats = {k: stats[k] for k in lstm_pq.STATS_KEYS if k in stats}
    num_bins = int(stats['num_bins'] if args.num_bins is None else args.num_bins)
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
        print('Training MLP model...', flush=True)
        (model, best_val, history) = train_model(
            train_ds,
            val_ds,
            stats,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            num_bins=num_bins,
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
