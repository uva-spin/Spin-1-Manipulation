"""
LSTM sequence model: manipulated Ps spectrum → scalar P_total and Q_total.

Input sequence (over frequency bins):
  [Ps, power_profile, n_steps]  (n_steps is broadcast to every bin)

``power_profile`` is the per-bin RF envelope from the NPZ (ssRF Voigt,
AFP zeros, optimal U(R)). Scalar ``applied_power`` is kept for diagnostics
and as a fallback if ``power_profile`` is missing.

Targets:
  P_total, Q_total from ``spectra.npz`` (population n+−n− / n+−2n0+n−)

Usage:
  python ml/lstm.py
  python ml/lstm.py --test-only
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.constants import (
    BATCH_SIZE, ML_DIR, DEFAULT_SPECTRA_PATH, DEVICE, DROPOUT,
    HIDDEN_SIZE, LEARNING_RATE, LR_MIN, MAX_GRAD_NORM, MIN_DELTA, NOISE_STD,
    NUM_EPOCHS, NUM_LAYERS, N_EXAMPLE_PLOTS, REL_LOSS_EPS, RESTART_LR_DECAY,
    RESTART_WARMUP_EPOCHS, SEED, SOURCE_PROFILE, SPECTRUM_R_MAX,
    SPECTRUM_R_MIN, SOURCE_NAME, STATS_KEYS, TEST_FRAC, T_0, T_MULT, VAL_FRAC,
    WEIGHT_DECAY,
)
from utils.helpers import (
    _stats_for_checkpoint, clone_state_dict, json_ready, load_history_json,
    load_lstm_npz, make_pq_loaders, parse_int_tuple, prepare_datasets,
    reload_checkpoint, resolve_power_profile, save_history_json,
)
from analysis.pq_report import (
    compute_rpe, evaluate_model, print_range_stats_table, print_snr_stats,
    save_example_signal_plots, save_plots, save_predictions_csv,
    save_range_stats_csv, test_input_ps, write_pq_report,
)

DEFAULT_OUTPUT_DIR = ML_DIR / 'results' / 'lstm' / 'lstm_result_ssrf_afp_combined_v4'

class LstmModel(nn.Module):

    def __init__(self, input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        enc_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True, dropout=enc_dropout, device=DEVICE)
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
        (enc_out, _) = self.encoder(x)
        ctx = enc_out.mean(dim=1)
        return (self.head_p(ctx).squeeze(-1), self.head_q(ctx).squeeze(-1))

def relative_weighted_loss(pred_z, true_z, *, mean, std, eps=REL_LOSS_EPS):
    pred = pred_z * std + mean
    true = true_z * std + mean
    return torch.mean(torch.abs(pred - true) / (torch.abs(true) + eps))

def save_checkpoint(path, *, model, stats, best_val_loss, best_epoch, hidden_size, num_layers, dropout):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state_dict': clone_state_dict(model), 'stats': _stats_for_checkpoint(stats), 'best_val_loss': best_val_loss, 'best_epoch': best_epoch, 'hidden_size': hidden_size, 'num_layers': num_layers, 'dropout': dropout, 'input_size': stats['input_size']}, path)

def load_checkpoint(path, *, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = LstmModel(input_size=ckpt.get('input_size', ckpt['stats']['input_size']), hidden_size=ckpt['hidden_size'], num_layers=ckpt['num_layers'], dropout=ckpt.get('dropout', DROPOUT)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return (model, ckpt)

def _step_cosine_restart_scheduler(scheduler, optimizer, *, init_lr, cycle_state, decay=RESTART_LR_DECAY, warmup_epochs=RESTART_WARMUP_EPOCHS, eta_min=LR_MIN):
    """Step CosineAnnealingWarmRestarts with restart LR decay + warmup + Adam reset.

    - Detects a warm restart via ``scheduler.T_cur == 0`` right after ``step()``.
    - Decays the cycle max LR as ``init_lr * decay**cycle`` so late restarts
      (e.g. epoch 310) kick far less than early ones.
    - Clears Adam/AdamW moments at each restart (stale 2nd moments + fresh
      full LR = overshoot into Inf/NaN).
    - Linearly warms LR back up over ``warmup_epochs`` instead of jumping
      straight to max on the first step of a new cycle.
    - Returns ``(lr, restarted)``. ``cycle_state`` is a ``dict(cycle=int,
      base=float)`` mutated in place so train/eval loops stay stateless.
    """
    scheduler.step()
    restarted = bool(scheduler.T_cur == 0)
    if restarted:
        cycle_state['cycle'] += 1
        cycle_state['base'] = float(init_lr * (decay ** cycle_state['cycle']))
        scheduler.base_lrs = [cycle_state['base']] * len(scheduler.base_lrs)
        for pg in optimizer.param_groups:
            pg['lr'] = cycle_state['base']
        optimizer.state.clear()
    if warmup_epochs and warmup_epochs > 0 and cycle_state['cycle'] > 0 and scheduler.T_cur < warmup_epochs:
        warm_lr = float(eta_min + (cycle_state['base'] - eta_min) * (scheduler.T_cur + 1) / warmup_epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = warm_lr
    return optimizer.param_groups[0]['lr'], restarted


def fit_pq_model(model, train_dataset, val_dataset, stats, *, num_epochs, batch_size, learning_rate, on_best=None, load_best=None, device=DEVICE, restart_decay=RESTART_LR_DECAY, warmup_epochs=RESTART_WARMUP_EPOCHS):
    """Shared P/Q loop used by the other model scripts (no non-finite guards; see train_model)."""
    (train_loader, val_loader) = make_pq_loaders(train_dataset, val_dataset, batch_size)
    p_mean, p_std = stats['P_mean'], stats['P_std']
    q_mean, q_std = stats['Q_mean'], stats['Q_std']
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=LR_MIN)
    cycle_state = {'cycle': 0, 'base': float(learning_rate)}
    history = {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
    best_val = float('inf')
    best_epoch = -1
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for (x_b, y_p, y_q) in train_loader:
            x_b, y_p, y_q = x_b.to(device), y_p.to(device), y_q.to(device)
            (pred_p, pred_q) = model(x_b)
            loss = relative_weighted_loss(pred_p, y_p, mean=p_mean, std=p_std) + relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
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
        lr, restarted = _step_cosine_restart_scheduler(scheduler, optimizer, init_lr=learning_rate, cycle_state=cycle_state, decay=restart_decay, warmup_epochs=warmup_epochs)
        restart_note = ' | restart' if restarted else ''
        print(f'epoch {epoch + 1:03d}/{num_epochs} | train {avg_train:.6f} | val {avg_val:.6f} | RPE% P={rpe_p:.3f} Q={rpe_q:.3f} | lr {lr:.2e}{restart_note}', flush=True)
        if avg_val < best_val - MIN_DELTA:
            best_val = avg_val
            best_epoch = epoch + 1
            best_state = clone_state_dict(model)
            if on_best is not None:
                on_best(model, best_val, best_epoch)
    if load_best is not None:
        loaded = load_best()
        if loaded is not None:
            model, loaded_val = loaded
            if loaded_val is not None:
                best_val = loaded_val
        elif best_state is not None:
            model.load_state_dict(best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return (model, best_val, history)


def _tensors_finite(tensors):
    for tensor in tensors:
        if tensor is not None and not torch.isfinite(tensor).all():
            return False
    return True

def _restore_training_state(model, optimizer, state):
    """Roll back to finite weights and drop Adam moments computed on the bad ones."""
    model.load_state_dict(state)
    optimizer.state.clear()

def train_model(train_dataset, val_dataset, stats, *, hidden_size, num_layers, dropout, num_epochs, batch_size, learning_rate, checkpoint_path=None, device=DEVICE, restart_decay=RESTART_LR_DECAY, warmup_epochs=RESTART_WARMUP_EPOCHS):
    (train_loader, val_loader) = make_pq_loaders(train_dataset, val_dataset, batch_size)
    p_mean = stats['P_mean']
    p_std = stats['P_std']
    q_mean = stats['Q_mean']
    q_std = stats['Q_std']
    model = LstmModel(input_size=stats['input_size'], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_MULT, eta_min=LR_MIN)
    cycle_state = {'cycle': 0, 'base': float(learning_rate)}
    history = {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
    best_val = float('inf')
    best_epoch = -1
    best_state = None
    init_state = clone_state_dict(model)
    for epoch in range(num_epochs):
        model.train()
        train_sum = 0.0
        train_batches = 0
        skipped = 0
        diverged = False
        for (x_b, y_p, y_q) in train_loader:
            x_b = x_b.to(device)
            y_p = y_p.to(device)
            y_q = y_q.to(device)
            (pred_p, pred_q) = model(x_b)
            loss = relative_weighted_loss(pred_p, y_p, mean=p_mean, std=p_std) + relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_ turns Inf into NaN (Inf * 0). Skip the step instead.
            if not _tensors_finite((p.grad for p in model.parameters())):
                optimizer.zero_grad()
                skipped += 1
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            if not _tensors_finite((p.data for p in model.parameters())):
                diverged = True
                break
            train_sum += loss.item()
            train_batches += 1
        if diverged or train_batches == 0:
            _restore_training_state(model, optimizer, best_state if best_state is not None else init_state)
            print(f'Stopped at epoch {epoch + 1}: non-finite loss or weights; restored last good checkpoint.', flush=True)
            break
        avg_train = train_sum / train_batches
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
                loss_p = relative_weighted_loss(pred_p, y_p, mean=p_mean, std=p_std)
                loss_q = relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
                if not (torch.isfinite(loss_p) and torch.isfinite(loss_q)):
                    diverged = True
                    break
                val_sum += (loss_p + loss_q).item()
                rpe_p_sum += loss_p.item()
                rpe_q_sum += loss_q.item()
                val_batches += 1
        if diverged or val_batches == 0:
            _restore_training_state(model, optimizer, best_state if best_state is not None else init_state)
            print(f'Stopped at epoch {epoch + 1}: non-finite validation loss; restored last good checkpoint.', flush=True)
            break
        avg_val = val_sum / val_batches
        rpe_p = rpe_p_sum / val_batches * 100.0
        rpe_q = rpe_q_sum / val_batches * 100.0
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_p_rpe'].append(rpe_p)
        history['val_q_rpe'].append(rpe_q)
        lr, restarted = _step_cosine_restart_scheduler(scheduler, optimizer, init_lr=learning_rate, cycle_state=cycle_state, decay=restart_decay, warmup_epochs=warmup_epochs)
        skip_note = f' | skipped {skipped}' if skipped else ''
        restart_note = ' | restart' if restarted else ''
        print(f'epoch {epoch + 1:03d}/{num_epochs} | train {avg_train:.6f} | val {avg_val:.6f} | RPE% P={rpe_p:.3f} Q={rpe_q:.3f} | lr {lr:.2e}{skip_note}{restart_note}', flush=True)
        if avg_val < best_val - MIN_DELTA:
            best_val = avg_val
            best_epoch = epoch + 1
            best_state = clone_state_dict(model)
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model=model, stats=stats, best_val_loss=best_val, best_epoch=best_epoch, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        (model, ckpt) = load_checkpoint(checkpoint_path, device=device)
        best_val = ckpt.get('best_val_loss', best_val)
    elif best_state is not None:
        model.load_state_dict(best_state)
    return (model, best_val, history)

def main():
    parser = argparse.ArgumentParser(description='Train/evaluate LSTM seq model: Ps spectrum + per-bin power profile → P_total, Q_total.')
    parser.add_argument('--test-only', action='store_true', help='Skip training; load checkpoint and evaluate on a fresh split')
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / 'lstm_best.pth'
    spectra_path = DEFAULT_SPECTRA_PATH
    print(f'Loading spectra from {spectra_path} ...', flush=True)
    arrays = load_lstm_npz(spectra_path)
    print('Preparing datasets...', flush=True)
    (train_ds, val_ds, test_ds, stats) = prepare_datasets(arrays, max_samples=None, noise_std=NOISE_STD)
    dataset_stats = {k: stats[k] for k in STATS_KEYS if k in stats}
    print(f"Train={stats['n_train']} Val={stats['n_val']} Test={stats['n_test']} bins={stats['num_bins']}", flush=True)
    history_path = out_dir / 'history.json'
    if args.test_only:
        print(f'Loading checkpoint {checkpoint_path} ...', flush=True)
        (model, ckpt) = load_checkpoint(checkpoint_path)
        stats = {**stats, **ckpt['stats']}
        stats['test_idx'] = dataset_stats['test_idx']
        best_val = ckpt.get('best_val_loss', float('nan'))
        loaded = load_history_json(history_path)
        history = loaded or {'train_loss': [], 'val_loss': [], 'val_p_rpe': [], 'val_q_rpe': []}
        n_param = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total Params: {n_param:,}', flush=True)
        print(f'Trainable Params: {n_train:,}', flush=True)
        print(f'Non-Trainable Params: {n_param - n_train:,}', flush=True)
        if loaded:
            print(f'Loaded training history from {history_path}', flush=True)
    else:
        print('Training lstm model...', flush=True)
        (model, best_val, history) = train_model(
            train_ds, val_ds, stats, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
            dropout=DROPOUT, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE, checkpoint_path=checkpoint_path,
            restart_decay=RESTART_LR_DECAY, warmup_epochs=RESTART_WARMUP_EPOCHS,
        )
        save_history_json(history, history_path)
    write_pq_report(
        model, test_ds, stats, dataset_stats, arrays, history, best_val, out_dir, checkpoint_path,
        batch_size=BATCH_SIZE, n_examples=N_EXAMPLE_PLOTS,
    )


if __name__ == '__main__':
    main()
