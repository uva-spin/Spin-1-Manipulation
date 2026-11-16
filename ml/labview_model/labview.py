import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import bilstm
from utils.constants import DEVICE

model = None
stats = None
device = DEVICE


def InitModel(model_path):
    """Load checkpoint into module globals (call once per LabVIEW Python session)."""
    global model, stats, device
    device = DEVICE
    loaded, ckpt = bilstm.load_checkpoint(model_path, device=device)
    loaded.eval()
    model = loaded
    stats = dict(ckpt['stats'])


def main(ps):
    """Run inference: normalize Ps, forward model, return denormalized (P, Q)."""
    ps_arr = np.asarray(ps, dtype=np.float32).reshape(-1)
    ps_n = (ps_arr - float(stats['ps_mean'])) / float(stats['ps_std'])
    x_t = torch.from_numpy(ps_n.reshape(1, -1, 1)).to(device)

    with torch.no_grad():
        pred_p, pred_q = model(x_t)

    p = float(pred_p.item() * float(stats['P_std']) + float(stats['P_mean']))
    q = float(pred_q.item() * float(stats['Q_std']) + float(stats['Q_mean']))
    return (p, q)
