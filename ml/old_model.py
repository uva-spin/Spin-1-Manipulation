"""
Inception + Residual + SE CNN: manipulated Ps spectrum → scalar P_total and Q_total.

Same task and data as ``ml/cnn.py`` / ``ml/inception.py``. Architecture:

  1. Inception block (parallel Conv1D at kernel sizes 1, 3, 5, and max-pool+1x1)
  2. Residual blocks (two Conv1D layers each)
  3. Optional Squeeze-and-Excitation block
  4. Global average pooling → FC + ReLU → P and Q heads

Per-bin channels:

  [Ps, power_profile, n_steps]  (n_steps is broadcast to every bin)

Targets:
  P_total, Q_total from ``spectra.npz`` (population n+−n− / n+−2n0+n−)

Usage:
  python ml/old_model.py --spectra ml/data/spectra_v7 --epochs 30
  python ml/old_model.py --test-only --checkpoint ml/results/old_model/old_model_pq_results_v1/old_model_pq_best.pth
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lstm as lstm_pq

DEFAULT_SPECTRA_PATH = lstm_pq.DEFAULT_SPECTRA_PATH
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'results' / 'old_model' / 'old_model_pq_results_v1'
SEED = lstm_pq.SEED
DEVICE = lstm_pq.DEVICE
NUM_EPOCHS = lstm_pq.NUM_EPOCHS
BATCH_SIZE = lstm_pq.BATCH_SIZE
LEARNING_RATE = lstm_pq.LEARNING_RATE
NUM_RESIDUAL_BLOCKS = 3
USE_SE_BLOCK = True
FC_HIDDEN = 32
DROPOUT = 0.0
NOISE_STD = lstm_pq.NOISE_STD
N_EXAMPLE_PLOTS = lstm_pq.N_EXAMPLE_PLOTS


class InceptionBlock(nn.Module):
    """Four parallel Conv1D branches (1, 3, 5, and max-pool+1x1) → concatenate."""

    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=1),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, c4, kernel_size=1),
            nn.ReLU(),
        )
        self.out_channels = c1 + c2[1] + c3[1] + c4

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )


class ResidualBlock(nn.Module):
    """Two Conv1D layers with batch norm and a residual skip."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.conv1(x))
        out = self.bn(self.conv2(out))
        return out + residual


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel reweighting."""

    def __init__(self, channels, reduction=2):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x):
        y = self.global_pool(x).squeeze(-1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).unsqueeze(-1)
        return x * y


class CNNArchitectureModel(nn.Module):
    """Inception → residual stack → optional SE → GAP → P/Q heads."""

    def __init__(
        self,
        input_size=3,
        num_residual_blocks=NUM_RESIDUAL_BLOCKS,
        use_se_block=USE_SE_BLOCK,
        fc_hidden=FC_HIDDEN,
        dropout=DROPOUT,
    ):
        super().__init__()
        if int(input_size) < 1:
            raise ValueError(f'input_size must be positive, got {input_size}')
        if int(num_residual_blocks) < 0:
            raise ValueError(f'num_residual_blocks must be >= 0, got {num_residual_blocks}')
        if int(fc_hidden) < 1:
            raise ValueError(f'fc_hidden must be positive, got {fc_hidden}')
        self.input_size = int(input_size)
        self.num_residual_blocks = int(num_residual_blocks)
        self.use_se_block = bool(use_se_block)
        self.fc_hidden = int(fc_hidden)
        self.dropout_p = float(dropout)

        c1 = 64
        c2 = (32, 32 * 3)  # 32 → 96
        c3 = (32, 32 * 5)  # 32 → 160
        c4 = 32
        channels = c1 + c2[1] + c3[1] + c4  # 352

        self.inception_block = InceptionBlock(self.input_size, c1, c2, c3, c4)
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(channels, channels) for _ in range(self.num_residual_blocks)
        )
        self.se_block = SEBlock(channels, reduction=2) if self.use_se_block else None
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.fc = nn.Linear(channels, self.fc_hidden)
        self.head_p = nn.Linear(self.fc_hidden, 1)
        self.head_q = nn.Linear(self.fc_hidden, 1)
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
        # x: (batch, length, channels) → (batch, channels, length)
        hidden = x.transpose(1, 2)
        hidden = self.inception_block(hidden)
        for residual_block in self.residual_blocks:
            hidden = residual_block(hidden)
        if self.se_block is not None:
            hidden = self.se_block(hidden)
        hidden = self.global_pool(hidden).flatten(1)
        hidden = self.dropout(F.relu(self.fc(hidden)))
        return (self.head_p(hidden).squeeze(-1), self.head_q(hidden).squeeze(-1))


def save_checkpoint(
    path,
    *,
    model,
    stats,
    best_val_loss,
    best_epoch,
    num_residual_blocks,
    use_se_block,
    fc_hidden,
    dropout,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': lstm_pq.clone_state_dict(model),
            'stats': lstm_pq._stats_for_checkpoint(stats),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'num_residual_blocks': int(num_residual_blocks),
            'use_se_block': bool(use_se_block),
            'fc_hidden': int(fc_hidden),
            'dropout': dropout,
            'input_size': stats['input_size'],
        },
        path,
    )


def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CNNArchitectureModel(
        input_size=ckpt.get('input_size', ckpt['stats']['input_size']),
        num_residual_blocks=ckpt.get('num_residual_blocks', NUM_RESIDUAL_BLOCKS),
        use_se_block=ckpt.get('use_se_block', USE_SE_BLOCK),
        fc_hidden=ckpt.get('fc_hidden', FC_HIDDEN),
        dropout=ckpt.get('dropout', DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)


def train_model(train_dataset, val_dataset, stats, *, num_residual_blocks, use_se_block, fc_hidden, dropout, num_epochs, batch_size, learning_rate, checkpoint_path=None, device=DEVICE):
    model = CNNArchitectureModel(
        input_size=stats['input_size'], num_residual_blocks=num_residual_blocks,
        use_se_block=use_se_block, fc_hidden=fc_hidden, dropout=dropout,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)

    def on_best(trained, best_val, best_epoch):
        save_checkpoint(
            checkpoint_path, model=trained, stats=stats, best_val_loss=best_val, best_epoch=best_epoch,
            num_residual_blocks=num_residual_blocks, use_se_block=use_se_block,
            fc_hidden=fc_hidden, dropout=dropout,
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
        description='Train/evaluate Inception+Residual+SE CNN: Ps spectrum → P_total, Q_total.',
    )
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz or memmap directory')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument(
        '--num-residual-blocks',
        type=int,
        default=NUM_RESIDUAL_BLOCKS,
        help='Number of residual blocks after Inception (default: 3)',
    )
    parser.add_argument(
        '--use-se-block',
        action=argparse.BooleanOptionalAction,
        default=USE_SE_BLOCK,
        help='Enable Squeeze-and-Excitation block (default: on)',
    )
    parser.add_argument('--fc-hidden', type=int, default=FC_HIDDEN, help='Hidden size before P/Q heads')
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--max-samples', type=int, default=None, help='Optional subsample size; omit to train on the full dataset')
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--n-examples', type=int, default=N_EXAMPLE_PLOTS)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help='Checkpoint path (default: <out-dir>/old_model_pq_best.pth)',
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Skip training; load --checkpoint and evaluate on a fresh split',
    )
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out_dir / 'old_model_pq_best.pth'
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
        print('Training Inception+Residual+SE model...', flush=True)
        (model, best_val, history) = train_model(
            train_ds,
            val_ds,
            stats,
            num_residual_blocks=args.num_residual_blocks,
            use_se_block=args.use_se_block,
            fc_hidden=args.fc_hidden,
            dropout=args.dropout,
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
