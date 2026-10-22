"""
LabVIEW Python Node API: load LSTM once, then predict P_total / Q_total.

Call once after Open Python Session:
  InitModel(checkpoint_path)

Call repeatedly:
  P, Q = main(ps, power_profile, n_steps)

Standalone smoke:
  python ml/lstm_infer_labview.py --checkpoint ml/results/lstm/lstm_v11/lstm_best.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lstm as lstm_pq
from utils.constants import DEVICE

model = None
stats = None
device = DEVICE


def InitModel(checkpoint_path: str) -> None:
    """Load checkpoint into module globals (call once per LabVIEW Python session)."""
    global model, stats, device
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    device = DEVICE
    loaded, ckpt = lstm_pq.load_checkpoint(path, device=device)
    loaded.eval()
    model = loaded
    stats = dict(ckpt['stats'])


def main(ps, power_profile, n_steps):
    """Normalize inputs with checkpoint stats and return denormalized (P, Q)."""
    if model is None or stats is None:
        raise RuntimeError('Call InitModel(checkpoint_path) before main(...)')

    ps_arr = np.asarray(ps, dtype=np.float32).reshape(-1)
    pwr_arr = np.asarray(power_profile, dtype=np.float32).reshape(-1)
    if ps_arr.size != pwr_arr.size:
        raise ValueError(
            f'ps and power_profile length mismatch: {ps_arr.size} vs {pwr_arr.size}'
        )
    n_bins = int(ps_arr.size)
    expected = stats.get('num_bins')
    if expected is not None and n_bins != int(expected):
        raise ValueError(f'expected {int(expected)} bins, got {n_bins}')

    ps_n = (ps_arr - float(stats['ps_mean'])) / float(stats['ps_std'])
    pwr_n = (pwr_arr - float(stats['pwr_mean'])) / float(stats['pwr_std'])
    steps_n = (float(n_steps) - float(stats['steps_mean'])) / float(stats['steps_std'])
    steps_seq = np.full(n_bins, steps_n, dtype=np.float32)
    x = np.stack([ps_n, pwr_n, steps_seq], axis=-1).astype(np.float32)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_p, pred_q = model(x_t)

    p = float(pred_p.item() * float(stats['P_std']) + float(stats['P_mean']))
    q = float(pred_q.item() * float(stats['Q_std']) + float(stats['Q_mean']))
    return (p, q)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smoke-test LabVIEW LSTM infer globals.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    InitModel(str(args.checkpoint))
    n = int(stats['num_bins'])
    p, q = main(np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32), 1.0)
    print(f'P={p:.8g} Q={q:.8g}')
