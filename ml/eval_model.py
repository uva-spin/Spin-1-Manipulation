"""
Evaluate a trained spectrum→P/Q model on the full spectra store with fresh noise.

Uses input/label normalization from the checkpoint (training stats), the full
dataset (no ``--max-samples`` subsampling), and a separate noise seed so injected
noise is independent from training.

Usage:
  python ml/eval_model.py --model lstm \\
    --checkpoint ml/results/lstm/lstm_result_v9/lstm_best.pth \\
    --spectra ml/data/spectra_v6.npz \\
    --out-dir ml/results/lstm/lstm_eval_v6_noise43

  python ml/eval_model.py --model cnn \\
    --checkpoint ml/results/cnn/cnn_pq_results_v2/cnn_pq_best.pth \\
    --spectra ml/data/spectra_v7 \\
    --noise-seed 123 --split-seed 42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cnn
import inception
import lstm as lstm_pq
import mlp
import old_model
import transformer as transformer_mod

MODEL_MODULES = {
    'lstm': lstm_pq,
    'cnn': cnn,
    'mlp': mlp,
    'transformer': transformer_mod,
    'inception': inception,
    'old_model': old_model,
}

NORM_KEYS = (
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
)

DATASET_STAT_KEYS = NORM_KEYS + (
    'noise_std',
    'test_idx',
    'n_test',
    'input_size',
    'num_bins',
)


def norm_stats_from_checkpoint(ckpt):
    stats = ckpt['stats']
    missing = [k for k in NORM_KEYS if k not in stats]
    if missing:
        raise KeyError(f'Checkpoint stats missing normalization keys: {missing}')
    return {k: stats[k] for k in NORM_KEYS}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a P/Q checkpoint on the full spectra store with a fresh noise seed.',
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=sorted(MODEL_MODULES),
        help='Model family (must match checkpoint architecture)',
    )
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to .pth checkpoint')
    parser.add_argument(
        '--spectra',
        type=Path,
        default=lstm_pq.DEFAULT_SPECTRA_PATH,
        help='Path to spectra.npz or memmap directory (full dataset, no subsampling)',
    )
    parser.add_argument('--out-dir', type=Path, required=True, help='Directory for metrics, CSV, and plots')
    parser.add_argument('--batch-size', type=int, default=lstm_pq.BATCH_SIZE)
    parser.add_argument(
        '--split-seed',
        type=int,
        default=lstm_pq.SEED,
        help='Seed for train/val/test split (default: same as training, %d)' % lstm_pq.SEED,
    )
    parser.add_argument(
        '--noise-seed',
        type=int,
        default=lstm_pq.SEED + 1,
        help='Seed for per-event Gaussian noise on Ps (default: training seed + 1)',
    )
    parser.add_argument(
        '--noise-std',
        type=float,
        default=None,
        help='Noise std on Ps (default: from checkpoint, else training default)',
    )
    parser.add_argument('--val-frac', type=float, default=lstm_pq.VAL_FRAC)
    parser.add_argument('--test-frac', type=float, default=lstm_pq.TEST_FRAC)
    parser.add_argument('--n-examples', type=int, default=lstm_pq.N_EXAMPLE_PLOTS)
    parser.add_argument('--no-plots', action='store_true', help='Skip PNG plots and example spectra')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    mod = MODEL_MODULES[args.model]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading checkpoint {args.checkpoint} ({args.model}) ...', flush=True)
    model, ckpt = mod.load_checkpoint(args.checkpoint)
    model.eval()

    noise_std = args.noise_std
    if noise_std is None:
        noise_std = float(ckpt.get('stats', {}).get('noise_std', lstm_pq.NOISE_STD))

    norm_stats = norm_stats_from_checkpoint(ckpt)
    print(f'Loading spectra from {args.spectra} (full dataset) ...', flush=True)
    arrays = lstm_pq.load_lstm_npz(args.spectra)
    n_events = int(arrays['spectra'].shape[0])
    print(
        f'Preparing datasets (split_seed={args.split_seed}, noise_seed={args.noise_seed}, '
        f'noise_std={noise_std}) ...',
        flush=True,
    )
    _train_ds, _val_ds, test_ds, stats = lstm_pq.prepare_datasets(
        arrays,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.split_seed,
        noise_seed=args.noise_seed,
        max_samples=None,
        noise_std=noise_std,
        norm_stats=norm_stats,
    )
    del _train_ds, _val_ds

    test_idx = np.asarray(stats['test_idx'], dtype=np.int64)
    stats = {**stats, **ckpt['stats']}
    stats['test_idx'] = test_idx
    stats['noise_std'] = noise_std
    stats['noise_seed'] = args.noise_seed
    stats['split_seed'] = args.split_seed
    stats['p0_test'] = (
        np.asarray(arrays['p0']).reshape(-1)[test_idx]
        if 'p0' in arrays
        else np.asarray(arrays['P_total']).reshape(-1)[test_idx]
    )

    dataset_stats = {k: stats[k] for k in DATASET_STAT_KEYS if k in stats}

    print(
        f'N={n_events:,}  Test={stats["n_test"]:,}  bins={stats["num_bins"]}  '
        f'(lazy={stats.get("lazy_dataset", False)})',
        flush=True,
    )
    print('Evaluating on test set ...', flush=True)
    metrics = lstm_pq.evaluate_model(
        model,
        test_ds,
        stats,
        batch_size=args.batch_size,
        label_stats=dataset_stats,
    )
    print(
        f"Test RPE% median  P={metrics['P_rpe_median']:.3f}  Q={metrics['Q_rpe_median']:.3f}",
        flush=True,
    )
    print(
        f"Test MAE  P={metrics['P_mae']:.6f}  Q={metrics['Q_mae']:.6f}  "
        f"R²  P={metrics['P_r2']:.4f}  Q={metrics['Q_r2']:.4f}",
        flush=True,
    )
    lstm_pq.print_snr_stats(metrics)
    lstm_pq.print_range_stats_table(metrics['range_stats_by_P'], title='Performance by |P_total| Range')
    if metrics['range_stats_by_p0']:
        lstm_pq.print_range_stats_table(metrics['range_stats_by_p0'], title='Performance by |p0| Range')

    test_idx = np.asarray(dataset_stats['test_idx'], dtype=np.int64)
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

    json_metrics = {k: v for k, v in metrics.items() if isinstance(v, (float, int, str, list))}
    json_metrics.update(
        {
            'checkpoint': str(args.checkpoint),
            'model': args.model,
            'spectra': str(args.spectra),
            'n_events': n_events,
            'split_seed': args.split_seed,
            'noise_seed': args.noise_seed,
            'noise_std': noise_std,
            'best_val_loss': ckpt.get('best_val_loss'),
            'best_epoch': ckpt.get('best_epoch'),
        }
    )
    with (args.out_dir / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(lstm_pq.json_ready(json_metrics), f, indent=2)

    lstm_pq.save_range_stats_csv(metrics['range_stats_by_P'], args.out_dir / 'range_stats_by_P.csv')
    if metrics['range_stats_by_p0']:
        lstm_pq.save_range_stats_csv(metrics['range_stats_by_p0'], args.out_dir / 'range_stats_by_p0.csv')

    if not args.no_plots:
        print('Generating plots ...', flush=True)
        empty_history = {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
        lstm_pq.save_plots(empty_history, metrics, args.out_dir, best_val_loss=ckpt.get('best_val_loss'))
        lstm_pq.save_example_signal_plots(
            arrays,
            stats,
            metrics,
            args.out_dir,
            test_dataset=test_ds,
            input_stats=dataset_stats,
            n_examples=args.n_examples,
            seed=args.noise_seed,
        )

    print(f'Done. Results in {args.out_dir}', flush=True)


if __name__ == '__main__':
    torch.manual_seed(lstm_pq.SEED)
    np.random.seed(lstm_pq.SEED)
    main()
