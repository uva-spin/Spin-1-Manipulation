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
  python ml/mlp.py --spectra ml/data/spectra_v5.npz --max-samples 20000 --epochs 30
  python ml/mlp.py --test-only --checkpoint ml/mlp_pq_results_v1/mlp_pq_best.pth
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lstm as lstm_pq

DEFAULT_SPECTRA_PATH = SCRIPT_DIR / 'data' / 'spectra_v6.npz'
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / 'mlp_pq_results_v3'
SEED = lstm_pq.SEED
DEVICE = lstm_pq.DEVICE
NUM_EPOCHS = lstm_pq.NUM_EPOCHS
BATCH_SIZE = lstm_pq.BATCH_SIZE
LEARNING_RATE = lstm_pq.LEARNING_RATE
WEIGHT_DECAY = lstm_pq.WEIGHT_DECAY
PATIENCE = lstm_pq.PATIENCE
MIN_DELTA = lstm_pq.MIN_DELTA
T_0 = lstm_pq.T_0
T_MULT = lstm_pq.T_MULT
LR_MIN = lstm_pq.LR_MIN
MAX_GRAD_NORM = lstm_pq.MAX_GRAD_NORM
NUM_BINS = 500
HIDDEN_DIMS = (144, 144)
DROPOUT = 0.0
NOISE_STD = lstm_pq.NOISE_STD
N_EXAMPLE_PLOTS = lstm_pq.N_EXAMPLE_PLOTS
SOURCE_PROFILE = lstm_pq.SOURCE_PROFILE
STATS_KEYS = (
    'ps_mean',
    'ps_std',
    'pwr_mean',
    'pwr_std',
    'steps_mean',
    'steps_std',
    'P_mean',
    'P_std',
    'Q_mean',
    'Q_std',
    'noise_std',
    'test_idx',
    'n_test',
)


def parse_int_tuple(text):
    values = tuple(int(part.strip()) for part in str(text).split(',') if part.strip())
    if not values:
        raise argparse.ArgumentTypeError('expected a comma-separated list of integers')
    return values


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


def train_model(
    train_dataset,
    val_dataset,
    stats,
    *,
    hidden_dims,
    dropout,
    num_bins,
    num_epochs,
    batch_size,
    learning_rate,
    patience,
    checkpoint_path=None,
    device=DEVICE,
):
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    p_mean = stats['P_mean']
    p_std = stats['P_std']
    q_mean = stats['Q_mean']
    q_std = stats['Q_std']
    model = MlpPQModel(
        input_size=stats['input_size'],
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_bins=num_bins,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_MULT, eta_min=LR_MIN,
    )
    history = {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
    best_val = float('inf')
    best_epoch = -1
    best_state = None
    stale = 0
    for epoch in range(num_epochs):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for (x_b, y_p, y_q) in train_loader:
            x_b = x_b.to(device)
            y_p = y_p.to(device)
            y_q = y_q.to(device)
            (pred_p, pred_q) = model(x_b)
            loss = lstm_pq.relative_weighted_loss(pred_p, y_p, mean=p_mean, std=p_std)
            loss = loss + lstm_pq.relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            train_sum += loss.item()
            train_batches += 1
        avg_train = train_sum / max(train_batches, 1)
        model.eval()
        val_sum = 0.0
        val_batches = 0
        rpe_p_sum = 0.0
        rpe_q_sum = 0.0
        with torch.no_grad():
            for (x_v, y_p, y_q) in val_loader:
                x_v = x_v.to(device)
                y_p = y_p.to(device)
                y_q = y_q.to(device)
                (pred_p, pred_q) = model(x_v)
                loss_p = lstm_pq.relative_weighted_loss(pred_p, y_p, mean=p_mean, std=p_std)
                loss_q = lstm_pq.relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
                val_sum += (loss_p + loss_q).item()
                rpe_p_sum += loss_p.item()
                rpe_q_sum += loss_q.item()
                val_batches += 1
        avg_val = val_sum / max(val_batches, 1)
        rpe_p = rpe_p_sum / max(val_batches, 1) * 100.0
        rpe_q = rpe_q_sum / max(val_batches, 1) * 100.0
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_p_rpe'].append(rpe_p)
        history['val_q_rpe'].append(rpe_q)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(
            f'epoch {epoch + 1:03d}/{num_epochs} | train {avg_train:.6f} | val {avg_val:.6f} | '
            f'RPE% P={rpe_p:.3f} Q={rpe_q:.3f} | lr {lr:.2e}',
            flush=True,
        )
        if avg_val < best_val - MIN_DELTA:
            best_val = avg_val
            best_epoch = epoch + 1
            best_state = lstm_pq.clone_state_dict(model)
            stale = 0
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    stats=stats,
                    best_val_loss=best_val,
                    best_epoch=best_epoch,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    num_bins=num_bins,
                )
        # else:
        #     stale += 1
        #     if stale >= patience:
        #         print(f'Early stopping at epoch {epoch + 1}', flush=True)
        #         break
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        (model, ckpt) = load_checkpoint(checkpoint_path, device=device)
        best_val = ckpt.get('best_val_loss', best_val)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return (model, best_val, history)


def main():
    parser = argparse.ArgumentParser(
        description='Train/evaluate residual MLP: Ps spectrum + per-bin power profile → P_total, Q_total.',
    )
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument(
        '--hidden-dims',
        type=parse_int_tuple,
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
    parser.add_argument('--max-samples', type=int, default=None, help='Subsample this many events (recommended for first runs)')
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
    dataset_stats = {k: stats[k] for k in STATS_KEYS if k in stats}
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
            patience=args.patience,
            checkpoint_path=checkpoint_path,
        )
        lstm_pq.save_history_json(history, history_path)
    print('Evaluating on test set...', flush=True)
    metrics = lstm_pq.evaluate_model(model, test_ds, stats, batch_size=args.batch_size, label_stats=dataset_stats)
    print(f"Test RPE% median  P={metrics['P_rpe_median']:.3f}  Q={metrics['Q_rpe_median']:.3f}", flush=True)
    print(
        f"Test MAE  P={metrics['P_mae']:.6f}  Q={metrics['Q_mae']:.6f}  "
        f"R²  P={metrics['P_r2']:.4f}  Q={metrics['Q_r2']:.4f}",
        flush=True,
    )
    lstm_pq.print_snr_stats(metrics)
    lstm_pq.print_range_stats_table(metrics['range_stats_by_P'], title='Performance by |P_total| Range')
    if metrics['range_stats_by_p0']:
        lstm_pq.print_range_stats_table(metrics['range_stats_by_p0'], title='Performance by |p0| Range')
    lstm_pq.save_range_stats_csv(metrics['range_stats_by_P'], args.out_dir / 'range_stats_by_P.csv')
    if metrics['range_stats_by_p0']:
        lstm_pq.save_range_stats_csv(metrics['range_stats_by_p0'], args.out_dir / 'range_stats_by_p0.csv')
    test_idx = np.asarray(dataset_stats.get('test_idx', stats.get('test_idx')), dtype=np.int64)
    pred_csv = args.out_dir / 'test_predictions.csv'
    lstm_pq.save_predictions_csv(
        metrics,
        pred_csv,
        test_idx=test_idx,
        p0=np.asarray(arrays['p0']).reshape(-1)[test_idx] if 'p0' in arrays else None,
        source=np.asarray(arrays['source']).reshape(-1)[test_idx] if 'source' in arrays else None,
        n_steps=np.asarray(arrays['n_steps']).reshape(-1)[test_idx],
        applied_power=np.asarray(arrays['applied_power']).reshape(-1)[test_idx],
    )
    print(f'Saved test predictions -> {pred_csv}', flush=True)
    json_metrics = {k: v for (k, v) in metrics.items() if isinstance(v, (float, int, str, list))}
    json_metrics['best_val_loss'] = best_val
    json_metrics['checkpoint'] = str(checkpoint_path)
    with (args.out_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(lstm_pq.json_ready(json_metrics), f, indent=2)
    print('Generating plots...', flush=True)
    lstm_pq.save_plots(history, metrics, args.out_dir, best_val_loss=best_val)
    lstm_pq.save_example_signal_plots(
        arrays, stats, metrics, args.out_dir, test_dataset=test_ds, input_stats=dataset_stats, n_examples=args.n_examples,
    )
    lstm_pq.save_example_signal_plots(
        arrays,
        stats,
        metrics,
        args.out_dir,
        test_dataset=test_ds,
        input_stats=dataset_stats,
        n_examples=args.n_examples,
        source_filter=SOURCE_PROFILE,
        examples_subdir='examples_optimal',
        summary_name='examples_optimal_summary.png',
        seed=SEED + 1,
    )
    print(f'Done. Results in {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
