"""
Bi-LSTM with RF profile shape/strength decoupling (recommendation A).

Same task, inputs, targets, mean pooling, and linear P/Q heads as
``ml/lstm.py``. The only change is what happens before the Bi-LSTM:

  * the per-bin ``power_profile`` channel is split into a gamma-invariant
    *shape* (where the burn is) and scalar *strength* features (how strong
    it is: log peak and log integral, embedded by a small MLP and
    broadcast to every bin);
  * log-compressed (``asinh``) profile/dose terms and explicit
    ``Ps x shape`` / ``Ps x strength`` / ``shape x n_steps`` products are
    added as per-bin features, so the LSTM no longer has to discover the
    multiplicative dose interaction through saturated gates;
  * the expanded features pass through ``Linear -> LayerNorm -> GELU``
    before the LSTM.

Input is unchanged: ``[Ps, power_profile, n_steps]`` per frequency bin,
z-scored by ``prepare_datasets``. Because that z-score is affine, the
per-sample median of the profile channel recovers the off-burn baseline,
so shape is recovered without touching the input pipeline. AFP and
unmanipulated spectra have a constant (zero before z-scoring) profile,
which maps to exactly zero shape and zero strength.

Usage:
  python ml/lstm_v2.py --spectra ml/data/spectra_9.npz
  python ml/lstm_v2.py --test-only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils.constants import (
    BATCH_SIZE, DEFAULT_SPECTRA_PATH, DEVICE, DROPOUT, LEARNING_RATE,
    ML_DIR, NOISE_STD, N_EXAMPLE_PLOTS, NUM_EPOCHS, NUM_LAYERS, SEED,
    STATS_KEYS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lstm as lstm_pq


DEFAULT_OUTPUT_DIR = ML_DIR / 'results' / 'lstm_v2' / 'lstm_v2_v1'
DEFAULT_SPECTRA_PATH = ML_DIR / 'data' / 'spectra_varied.npz'
# Match ml/lstm.py (not the smaller shared constant) so A is the only change.
HIDDEN_SIZE = 128
PROJ_DIM = 32
STRENGTH_DIM = 16
_EPS = 1e-6


def profile_shape_and_strength(power_profile):
    """Split a (z-scored) power profile into burn shape and strength.

    Args:
        power_profile: ``[B, L]`` or ``[B, L, 1]`` profile channel.

    Returns:
        ``(shape, strength)`` where ``shape`` is ``[B, L]`` with peak 1 at
        the burn bin (all zeros when there is no burn) and ``strength`` is
        ``[B, 2]``: ``log1p`` of the baseline-subtracted peak and signed
        ``log1p`` of the baseline-subtracted integral. Rescaling the raw
        profile by any positive gamma (an affine change after z-scoring)
        leaves ``shape`` unchanged and moves only ``strength``.
    """
    prof = power_profile
    if prof.ndim == 3 and prof.shape[-1] == 1:
        prof = prof.squeeze(-1)
    if prof.ndim != 2:
        raise ValueError(f'power_profile must be [B, L] or [B, L, 1], got shape {tuple(power_profile.shape)}')
    # Most bins sit off-burn, so the median is the z-scored zero level.
    baseline = prof.median(dim=1, keepdim=True).values
    centered = prof - baseline
    peak_abs = centered.abs().amax(dim=1, keepdim=True)
    has_burn = peak_abs > _EPS
    shape = torch.where(has_burn, centered / peak_abs.clamp_min(_EPS), torch.zeros_like(centered))
    integral = centered.sum(dim=1)
    strength = torch.stack(
        [
            torch.log1p(peak_abs.squeeze(1).clamp_min(0.0)),
            torch.sign(integral) * torch.log1p(integral.abs()),
        ],
        dim=-1,
    )
    return shape, strength


class DecoupledLstmModel(nn.Module):
    """Bi-LSTM on shape/strength-decoupled features; baseline trunk/heads.

    Per-bin features built in ``forward`` from the unchanged 3 channels:
    ``[Ps, shape, asinh(centered_profile), n_steps, asinh(shape*n_steps),
    Ps*shape, Ps*asinh(centered_profile)]``, concatenated with the
    broadcast strength embedding (profile peak/integral + ``n_steps``).
    Pooling (mean) and heads (single Linear each) intentionally match
    ``LstmModel`` so this script isolates recommendation A.
    """

    # [Ps, shape, compressed profile, steps, compressed dose,
    #  Ps*shape, Ps*compressed profile]
    N_TOKEN_FEATURES = 7
    # [log peak, signed-log integral, n_steps]
    N_STRENGTH_INPUTS = 3

    def __init__(self, input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, proj_dim=PROJ_DIM, strength_dim=STRENGTH_DIM):
        super().__init__()
        if int(input_size) != 3:
            raise ValueError(f'DecoupledLstmModel expects input_size=3 ([Ps, power_profile, n_steps]), got {input_size}')
        if int(proj_dim) < 1:
            raise ValueError(f'proj_dim must be positive, got {proj_dim}')
        if int(strength_dim) < 1:
            raise ValueError(f'strength_dim must be positive, got {strength_dim}')
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.proj_dim = int(proj_dim)
        self.strength_dim = int(strength_dim)
        self.strength_mlp = nn.Sequential(
            nn.Linear(self.N_STRENGTH_INPUTS, self.strength_dim),
            nn.GELU(),
            nn.Linear(self.strength_dim, self.strength_dim),
            nn.GELU(),
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.N_TOKEN_FEATURES + self.strength_dim, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.GELU(),
        )
        enc_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_size=self.proj_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=enc_dropout, device=DEVICE)
        trunk_dim = 2 * self.hidden_size
        self.head_p = nn.Linear(trunk_dim, 1)
        self.head_q = nn.Linear(trunk_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(f'DecoupledLstmModel expects input_size=3 input [B, L, 3], got shape {tuple(x.shape)}')
        ps = x[..., 0:1]
        steps = x[..., 2:3]
        shape, strength = profile_shape_and_strength(x[..., 1])
        shape = shape.unsqueeze(-1)
        # Baseline-subtracted profile, amplitude-compressed: retains the
        # burn location at unit scale while gamma only moves it slowly.
        centered = x[..., 1:2] - x[..., 1].median(dim=1, keepdim=True).values.unsqueeze(-1)
        compressed_profile = torch.asinh(centered)
        dose = shape * steps

        strength_in = torch.cat([strength, steps.mean(dim=1)], dim=-1)
        strength_emb = self.strength_mlp(strength_in).unsqueeze(1).expand(-1, x.shape[1], -1)

        features = torch.cat(
            [
                ps,
                shape,
                compressed_profile,
                steps,
                torch.asinh(dose),
                ps * shape,
                ps * compressed_profile,
                strength_emb,
            ],
            dim=-1,
        )
        (enc_out, _) = self.encoder(self.input_proj(features))
        ctx = enc_out.mean(dim=1)
        return (self.head_p(ctx).squeeze(-1), self.head_q(ctx).squeeze(-1))


def save_checkpoint(path, *, model, stats, best_val_loss, best_epoch, hidden_size, num_layers, dropout, proj_dim, strength_dim):
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
            'proj_dim': proj_dim,
            'strength_dim': strength_dim,
            'input_size': stats['input_size'],
        },
        path,
    )


def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DecoupledLstmModel(
        input_size=ckpt.get('input_size', ckpt['stats']['input_size']),
        hidden_size=ckpt['hidden_size'],
        num_layers=ckpt['num_layers'],
        dropout=ckpt.get('dropout', DROPOUT),
        proj_dim=ckpt.get('proj_dim', PROJ_DIM),
        strength_dim=ckpt.get('strength_dim', STRENGTH_DIM),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)


def train_model(train_dataset, val_dataset, stats, *, hidden_size, num_layers, dropout, proj_dim, strength_dim, num_epochs, batch_size, learning_rate, checkpoint_path=None, device=DEVICE):
    model = DecoupledLstmModel(input_size=stats['input_size'], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, proj_dim=proj_dim, strength_dim=strength_dim).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_trainable:,}', flush=True)

    def on_best(trained, best_val, best_epoch):
        save_checkpoint(checkpoint_path, model=trained, stats=stats, best_val_loss=best_val, best_epoch=best_epoch, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, proj_dim=proj_dim, strength_dim=strength_dim)

    return lstm_pq.fit_pq_model(
        model, train_dataset, val_dataset, stats,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
        on_best=on_best if checkpoint_path is not None else None,
        load_best=lambda: lstm_pq.reload_checkpoint(load_checkpoint, checkpoint_path, device),
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate decoupled Bi-LSTM (A): Ps spectrum + per-bin power profile → P_total, Q_total.')
    parser.add_argument('--spectra', type=Path, default=DEFAULT_SPECTRA_PATH, help='Path to spectra.npz')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--hidden-size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--num-layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--proj-dim', type=int, default=PROJ_DIM, help='Per-bin projection width before the LSTM')
    parser.add_argument('--strength-dim', type=int, default=STRENGTH_DIM, help='Profile strength embedding width')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional subsample size; omit to train on the full dataset')
    parser.add_argument('--noise-std', type=float, default=NOISE_STD)
    parser.add_argument('--n-examples', type=int, default=N_EXAMPLE_PLOTS, help='Number of per-sample example plots to save')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Checkpoint path (default: <out-dir>/lstm_v2_best.pth)')
    parser.add_argument('--test-only', action='store_true', help='Skip training; load --checkpoint and evaluate on a fresh split')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out_dir / 'lstm_v2_best.pth'
    spectra_path = args.spectra or DEFAULT_SPECTRA_PATH
    print(f'Loading spectra from {spectra_path} ...', flush=True)
    arrays = lstm_pq.load_npz(spectra_path)
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
        print('Training decoupled lstm model...', flush=True)
        (model, best_val, history) = train_model(
            train_ds,
            val_ds,
            stats,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            proj_dim=args.proj_dim,
            strength_dim=args.strength_dim,
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
