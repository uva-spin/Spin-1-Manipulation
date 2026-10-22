"""
Plain Bi-LSTM like ``ml/lstm.py``, but Ps spectrum only (no RF profile / n_steps).

Same architecture as ``LstmModel``: Bi-LSTM over frequency bins, mean pool,
linear P/Q heads. Input is per-bin ``Ps`` only (``input_size=1``), z-scored
by ``prepare_datasets`` then sliced with ``keep_ps_channel``.

Usage:
  python ml/lstm_v4.py --spectra ml/data/spectra_9.npz
  python ml/lstm_v4.py --test-only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from utils.constants import (
    BATCH_SIZE, DEVICE, DROPOUT, LEARNING_RATE,
    ML_DIR, NOISE_STD, N_EXAMPLE_PLOTS, NUM_EPOCHS, NUM_LAYERS, SEED,
    STATS_KEYS,
)
from utils.helpers import keep_ps_channel

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lstm as lstm_pq


DEFAULT_OUTPUT_DIR = ML_DIR / 'results' / 'lstm_v4' / 'lstm_v4_v1'
DEFAULT_SPECTRA_PATH = ML_DIR / 'data' / 'spectra_varied.npz'
# Match ml/lstm.py hidden size.
HIDDEN_SIZE = 128


def save_checkpoint(path, *, model, stats, best_val_loss, best_epoch, hidden_size, num_layers, dropout):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': lstm_pq.clone_state_dict(model),
            'stats': lstm_pq._stats_for_checkpoint(stats),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'input_size': stats['input_size'],
        },
        path,
    )


def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = lstm_pq.LstmModel(
        input_size=ckpt.get('input_size', ckpt['stats']['input_size']),
        hidden_size=ckpt['hidden_size'],
        num_layers=ckpt['num_layers'],
        dropout=ckpt.get('dropout', DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)


def train_model(train_dataset, val_dataset, stats, *, hidden_size, num_layers, dropout, num_epochs, batch_size, learning_rate, checkpoint_path=None, device=DEVICE):
    model = lstm_pq.LstmModel(
        input_size=stats['input_size'],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)

    def on_best(trained, best_val, best_epoch):
        save_checkpoint(
            checkpoint_path,
            model=trained,
            stats=stats,
            best_val_loss=best_val,
            best_epoch=best_epoch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    return lstm_pq.fit_pq_model(
        model, train_dataset, val_dataset, stats,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        on_best=on_best if checkpoint_path is not None else None,
        load_best=lambda: lstm_pq.reload_checkpoint(load_checkpoint, checkpoint_path, device),
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate Ps-only Bi-LSTM (v4): Ps → P_total, Q_total.')
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--hidden-size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--num-layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--max-samples', type=int, default=None, help='Optional subsample size; omit to train on the full dataset')
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--n-examples', type=int, default=N_EXAMPLE_PLOTS, help='Number of per-sample example plots to save')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint path (default: <out-dir>/lstm_v4_best.pth)')
    parser.add_argument('--test-only', action='store_true', help='Skip training; load --checkpoint and evaluate on a fresh split')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out_dir / 'lstm_v4_best.pth'
    spectra_path = args.spectra or DEFAULT_SPECTRA_PATH
    print(f'Loading spectra from {spectra_path} ...', flush=True)
    arrays = lstm_pq.load_npz(spectra_path)
    print('Preparing datasets (Ps channel only)...', flush=True)
    (train_ds, val_ds, test_ds, stats) = lstm_pq.prepare_datasets(arrays, max_samples=args.max_samples, noise_std=args.noise_std)
    (train_ds, val_ds, test_ds, stats) = keep_ps_channel(train_ds, val_ds, test_ds, stats)
    dataset_stats = {k: stats[k] for k in STATS_KEYS if k in stats}
    print(f"Train={stats['n_train']} Val={stats['n_val']} Test={stats['n_test']} bins={stats['num_bins']} input_size={stats['input_size']}", flush=True)
    history_path = args.out_dir / 'history.json'
    if args.test_only:
        print(f'Loading checkpoint {checkpoint_path} ...', flush=True)
        (model, ckpt) = load_checkpoint(checkpoint_path)
        stats = {**stats, **ckpt['stats']}
        stats['test_idx'] = dataset_stats['test_idx']
        stats['input_size'] = 1
        best_val = ckpt.get('best_val_loss', float('nan'))
        loaded = lstm_pq.load_history_json(history_path)
        history = loaded or {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
        if loaded:
            print(f'Loaded training history from {history_path}', flush=True)
    else:
        print('Training Ps-only lstm model...', flush=True)
        (model, best_val, history) = train_model(
            train_ds,
            val_ds,
            stats,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
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
