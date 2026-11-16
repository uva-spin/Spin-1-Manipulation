"""
Ps-only residual MLP. Same data path as ``mlp_v2``.

Flattened noisy Ps goes through residual Linear-LayerNorm-GELU blocks, a
linear decoder reconstructs the clean spectrum, and P/Q heads read that
reconstruction. Training adds an MSE recon term; the logged val metric is
still relative P+Q error.

State-only files (``pq_thermal.npz``, or any file whose ``final_*`` fields are
not finite) are loaded from ``spectrum``, ``state_P``, and ``state_Q``.
Other files use ``final_spectrum``, ``final_P``, and ``final_Q``.

Usage:
  python ml/mlp_simple.py --spectra ml/data/run5.npz
  python ml/mlp_simple.py --test-only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from utils.constants import (
    BATCH_SIZE, DEVICE, LEARNING_RATE, LR_MIN, MAX_GRAD_NORM, MIN_DELTA, ML_DIR,
    N_EXAMPLE_PLOTS, SEED, STATS_KEYS,
)
from utils.helpers import (
    keep_ps_channel, clone_state_dict, _stats_for_checkpoint, make_pq_loaders,
    relative_weighted_loss, reload_checkpoint, load_npz, prepare_datasets,
    load_history_json, save_history_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analysis.pq_report import write_pq_report

NOISE_STD = 5e-4
NOISE_COPIES = 20
HIDDEN = 16
NUM_BLOCKS = 2
NUM_BINS = 500
WEIGHT_DECAY = 1e-5
RECON_WEIGHT = 1.0
NUM_EPOCHS = 200

DEFAULT_OUTPUT_DIR = ML_DIR / 'results' / 'mlp_simple' / 'mlp_v2_unmanip_45_55_v5'
DEFAULT_SPECTRA_PATH = ML_DIR / 'data' / 'pq_45_55.npz'


class ResidualBlock(nn.Module):
    """Two Linear-LayerNorm-GELU layers plus an identity skip. The last linear starts at zero so the block is identity at init."""

    def __init__(self, dim):
        super().__init__()
        if int(dim) < 1:
            raise ValueError(f'dim must be positive, got {dim}')
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        hidden = self.act(self.ln1(self.fc1(x)))
        return x + self.ln2(self.fc2(hidden))


class MlpV2Plain(nn.Module):
    """Flattened Ps → residual trunk → clean-spectrum recon → P and Q heads."""

    def __init__(self, input_size=1, hidden=HIDDEN, num_blocks=NUM_BLOCKS, num_bins=NUM_BINS):
        super().__init__()
        if int(input_size) < 1:
            raise ValueError(f'input_size must be positive, got {input_size}')
        if int(hidden) < 1:
            raise ValueError(f'hidden must be positive, got {hidden}')
        if int(num_blocks) < 1:
            raise ValueError(f'num_blocks must be positive, got {num_blocks}')
        if int(num_bins) < 1:
            raise ValueError(f'num_bins must be positive, got {num_bins}')
        self.input_size = int(input_size)
        self.hidden = int(hidden)
        self.num_blocks = int(num_blocks)
        self.num_bins = int(num_bins)
        in_dim = self.input_size * self.num_bins
        self.stem = nn.Sequential(nn.Linear(in_dim, self.hidden), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(self.hidden) for _ in range(self.num_blocks)])
        self.decoder = nn.Linear(self.hidden, self.num_bins)
        self.head_p = nn.Linear(self.num_bins, 1)
        self.head_q = nn.Linear(self.num_bins, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in (self.stem, self.blocks):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(layer.bias)
        for block in self.blocks:
            nn.init.zeros_(block.fc2.weight)
            nn.init.zeros_(block.fc2.bias)
        for layer in (self.decoder, self.head_p, self.head_q):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, return_recon=False):
        if x.ndim != 3 or x.shape[-1] != self.input_size or x.shape[1] != self.num_bins:
            raise ValueError(
                f'MlpV2Plain expects input [B, {self.num_bins}, {self.input_size}], got shape {tuple(x.shape)}'
            )
        recon = self.decoder(self.blocks(self.stem(x.flatten(1))))
        pred_p = self.head_p(recon).squeeze(-1)
        pred_q = self.head_q(recon).squeeze(-1)
        if return_recon:
            return (pred_p, pred_q, recon)
        return (pred_p, pred_q)


def attach_clean_targets(dataset, arrays, indices, stats, copies=1):
    """Append z-scored clean Ps, repeating each spectrum ``copies`` times."""
    copies = int(copies)
    if copies < 1:
        raise ValueError(f'copies must be positive, got {copies}')
    ps = np.asarray(arrays['spectra'][indices, 0, :], dtype=np.float32)
    if ps.ndim != 2:
        raise ValueError(f'clean Ps must be (N, bins), got shape {tuple(ps.shape)}')
    clean = (ps - float(stats['ps_mean'])) / float(stats['ps_std'])
    if copies > 1:
        clean = np.repeat(clean, copies, axis=0)
    (x, y_p, y_q) = dataset.tensors
    if clean.shape[0] != int(x.shape[0]) or clean.shape[1] != int(x.shape[1]):
        raise ValueError(
            f'clean Ps shape {tuple(clean.shape)} does not match dataset spectra {(int(x.shape[0]), int(x.shape[1]))}'
        )
    return data.TensorDataset(x, y_p, y_q, torch.from_numpy(np.ascontiguousarray(clean)))


def save_checkpoint(path, *, model, stats, best_val_loss, best_epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': clone_state_dict(model),
            'stats': _stats_for_checkpoint(stats),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'hidden': model.hidden,
            'num_blocks': model.num_blocks,
            'num_bins': model.num_bins,
            'input_size': stats['input_size'],
        },
        path,
    )


def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MlpV2Plain(
        input_size=ckpt.get('input_size', ckpt['stats']['input_size']),
        hidden=ckpt.get('hidden', HIDDEN),
        num_blocks=ckpt.get('num_blocks', NUM_BLOCKS),
        num_bins=ckpt.get('num_bins', ckpt['stats']['num_bins']),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)


def train_model(train_dataset, val_dataset, stats, *, num_bins, num_epochs, batch_size, learning_rate, checkpoint_path=None, device=DEVICE):
    if len(train_dataset[0]) != 4:
        raise ValueError('train_dataset must include clean z-scored Ps as a 4th tensor')
    model = MlpV2Plain(input_size=stats['input_size'], num_bins=num_bins).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)
    (train_loader, val_loader) = make_pq_loaders(train_dataset, val_dataset, batch_size)
    p_mean, p_std = stats['P_mean'], stats['P_std']
    q_mean, q_std = stats['Q_mean'], stats['Q_std']
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, threshold=MIN_DELTA, threshold_mode='abs', min_lr=LR_MIN,
    )
    history = {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
    best_val = float('inf')
    best_epoch = -1
    for epoch in range(num_epochs):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for (x_b, y_p, y_q, y_clean) in train_loader:
            x_b = x_b.to(device)
            y_p = y_p.to(device)
            y_q = y_q.to(device)
            y_clean = y_clean.to(device)
            (pred_p, pred_q, recon) = model(x_b, return_recon=True)
            pq_loss = (
                relative_weighted_loss(pred_p, y_p, mean=p_mean, std=p_std)
                + relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
            )
            loss = pq_loss + RECON_WEIGHT * F.mse_loss(recon, y_clean)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            train_sum += pq_loss.item()
            train_batches += 1
        avg_train = train_sum / max(train_batches, 1)
        model.eval()
        val_sum = 0.0
        val_batches = 0
        rpe_p_sum = 0.0
        rpe_q_sum = 0.0
        with torch.no_grad():
            for (x_v, y_p, y_q) in val_loader:
                x_v, y_p, y_q = x_v.to(device), y_p.to(device), y_q.to(device)
                (pred_p, pred_q) = model(x_v)
                loss_p = relative_weighted_loss(pred_p, y_p, mean=p_mean, std=p_std)
                loss_q = relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
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
        scheduler.step(avg_val)
        lr = optimizer.param_groups[0]['lr']
        print(
            f'epoch {epoch + 1:03d}/{num_epochs} | train {avg_train:.6f} | val {avg_val:.6f} '
            f'| RPE% P={rpe_p:.3f} Q={rpe_q:.3f} | lr {lr:.2e}',
            flush=True,
        )
        if avg_val < best_val - MIN_DELTA:
            best_val = avg_val
            best_epoch = epoch + 1
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path, model=model, stats=stats, best_val_loss=best_val, best_epoch=best_epoch,
                )
    if checkpoint_path is not None:
        loaded = reload_checkpoint(load_checkpoint, checkpoint_path, device)
        if loaded is not None:
            (model, loaded_val) = loaded
            if loaded_val is not None:
                best_val = loaded_val
    return (model, best_val, history)


def load_spectra_npz(path):
    """Load training arrays. Ps and the P/Q targets are the post-burn values.

    ``final_spectrum`` is already Ps. Channel 1 is zero so the I+ + I- sum
    used downstream equals that array. ``pq_thermal.npz`` and any file whose
    post-burn fields are not finite use ``spectrum``, ``state_P``, and ``state_Q``.
    """
    arrays = load_npz(path)

    def _finite(raw, key, dtype):
        if key not in raw.files:
            return None
        arr = np.asarray(raw[key], dtype=dtype)
        if not np.isfinite(arr).all():
            return None
        return arr

    with np.load(path, allow_pickle=False) as raw:
        use_final = Path(path).name != 'pq_thermal.npz'
        ps = _finite(raw, 'final_spectrum', np.float32) if use_final else None
        target_p = _finite(raw, 'final_P', np.float64) if use_final else None
        target_q = _finite(raw, 'final_Q', np.float64) if use_final else None
        if ps is None or target_p is None or target_q is None:
            ps = _finite(raw, 'spectrum', np.float32)
            target_p = _finite(raw, 'state_P', np.float64)
            target_q = _finite(raw, 'state_Q', np.float64)
        if ps is None or target_p is None or target_q is None:
            raise ValueError(f'{path} has no finite Ps spectrum and P/Q targets')
        target_p = np.asarray(target_p, dtype=np.float64).reshape(-1)
        target_q = np.asarray(target_q, dtype=np.float64).reshape(-1)
    if ps.ndim != 2:
        raise ValueError(f'Ps spectrum must be (N, bins), got shape {tuple(ps.shape)}')
    n = int(ps.shape[0])
    if target_p.shape != (n,) or target_q.shape != (n,) or int(arrays['spectra'].shape[0]) != n:
        raise ValueError(f'Ps spectrum, P, and Q must each have length {n}')
    arrays['spectra'] = np.stack((ps, np.zeros_like(ps)), axis=1)
    arrays['P_total'] = target_p
    arrays['Q_total'] = target_q
    return arrays


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate Ps-only residual MLP: Ps → P_total, Q_total.')
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--max-samples', type=int, default=None, help='Optional subsample size; omit to train on the full dataset')
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--noise-copies', type=int, default=NOISE_COPIES, help='Independent noise draws per training spectrum')
    parser.add_argument('--n-examples', type=int, default=N_EXAMPLE_PLOTS, help='Number of per-sample example plots to save')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint path (default: <out-dir>/mlp_simple_best.pth)')
    parser.add_argument('--test-only', action='store_true', help='Skip training; load --checkpoint and evaluate on a fresh split')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out_dir / 'mlp_simple_best.pth'
    spectra_path = args.spectra or DEFAULT_SPECTRA_PATH
    print(f'Loading spectra from {spectra_path} ...', flush=True)
    arrays = load_spectra_npz(spectra_path)
    print('Preparing datasets (Ps channel only)...', flush=True)
    (train_ds, val_ds, test_ds, stats) = prepare_datasets(
        arrays, max_samples=args.max_samples, noise_std=args.noise_std, noise_copies=args.noise_copies,
    )
    (train_ds, val_ds, test_ds, stats) = keep_ps_channel(train_ds, val_ds, test_ds, stats)
    dataset_stats = {k: stats[k] for k in STATS_KEYS if k in stats}
    num_bins = int(stats['num_bins'])
    print(
        f"Train={stats['n_train']} ({stats['train_idx'].size} spectra x {stats['noise_copies']} noise draws) "
        f"Val={stats['n_val']} Test={stats['n_test']} "
        f"({stats['noise_copies']} noise draws) bins={stats['num_bins']} input_size={stats['input_size']}",
        flush=True,
    )
    history_path = args.out_dir / 'history.json'
    if args.test_only:
        print(f'Loading checkpoint {checkpoint_path} ...', flush=True)
        (model, ckpt) = load_checkpoint(checkpoint_path)
        stats = {**stats, **ckpt['stats']}
        stats['test_idx'] = dataset_stats['test_idx']
        stats['input_size'] = 1
        best_val = ckpt.get('best_val_loss', float('nan'))
        loaded = load_history_json(history_path)
        history = loaded or {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
        if loaded:
            print(f'Loaded training history from {history_path}', flush=True)
    else:
        train_ds = attach_clean_targets(
            train_ds, arrays, stats['train_idx'], stats, copies=stats['noise_copies'],
        )
        print('Training Ps-only residual MLP...', flush=True)
        (model, best_val, history) = train_model(
            train_ds,
            val_ds,
            stats,
            num_bins=num_bins,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_path=checkpoint_path,
        )
        save_history_json(history, history_path)
    write_pq_report(
        model, test_ds, stats, dataset_stats, arrays, history, best_val, args.out_dir, checkpoint_path,
        batch_size=args.batch_size, n_examples=args.n_examples, seed=SEED,
    )


if __name__ == '__main__':
    main()
