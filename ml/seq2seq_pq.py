"""
LSTM sequence model: manipulated Ps spectrum → scalar P_total and Q_total.

Input sequence (over frequency bins):
  [Ps, applied_power, n_steps]  (power / steps are broadcast to every bin)

Targets (same as ``spectrum_pq.py``):
  P_total, Q_total from ``spectra.npz``

Usage:
  python ml/seq2seq_pq.py --spectra ml/data/spectra.npz --max-samples 20000 --epochs 30
  python ml/seq2seq_pq.py --test-only --checkpoint ml/seq2seq_pq_results/seq2seq_pq_best.pth
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SPECTRA_CANDIDATES = (
    SCRIPT_DIR / "data" / "spectra.npz",
    REPO_ROOT / "Data_Creation" / "dae_voigt_burn_spectra" / "spectra.npz",
    SCRIPT_DIR / "dae_voigt_burn_spectra" / "spectra.npz",
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "seq2seq_pq_results"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 5
PATIENCE = 15
MIN_DELTA = 1e-6
LR_FACTOR = 0.5
LR_MIN = 1e-7
MAX_GRAD_NORM = 1.0
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
VAL_FRAC = 0.15
TEST_FRAC = 0.15
REL_LOSS_EPS = 1e-4
RPE_ABS_EPS = 1e-10
NOISE_STD = 1e-4


class Seq2SeqPQModel(nn.Module):
    """BiLSTM encoder over the Ps sequence → scalar P_total and Q_total."""

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        enc_dropout = float(dropout) if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=enc_dropout,
        )
        trunk_dim = 2 * self.hidden_size
        self.head_p = nn.Linear(trunk_dim, 1)
        self.head_q = nn.Linear(trunk_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C_in) → pred_p, pred_q: (B,)
        enc_out, _ = self.encoder(x)
        ctx = enc_out.mean(dim=1)
        return self.head_p(ctx).squeeze(-1), self.head_q(ctx).squeeze(-1)


def resolve_spectra_path(spectra: Path | None) -> Path:
    if spectra is not None:
        path = Path(spectra)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Spectra NPZ not found: {path}")
    for candidate in DEFAULT_SPECTRA_CANDIDATES:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(p) for p in DEFAULT_SPECTRA_CANDIDATES)
    raise FileNotFoundError(
        "Spectra NPZ not found. Pass --spectra PATH. Tried: " + tried
    )


def load_seq2seq_npz(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    required = ("spectra", "applied_power", "n_steps", "P_total", "Q_total")
    with np.load(path, allow_pickle=False) as raw:
        missing = [k for k in required if k not in raw.files]
        if missing:
            raise KeyError(f"{path}: missing fields {missing}; found {raw.files}")
        spectra = np.asarray(raw["spectra"], dtype=np.float32)
        applied_power = np.asarray(raw["applied_power"], dtype=np.float32).reshape(-1)
        n_steps = np.asarray(raw["n_steps"], dtype=np.float32).reshape(-1)
        p_total = np.asarray(raw["P_total"], dtype=np.float32).reshape(-1)
        q_total = np.asarray(raw["Q_total"], dtype=np.float32).reshape(-1)
        optional: dict[str, np.ndarray] = {}
        for key in ("p0", "center_bin", "source"):
            if key in raw.files:
                optional[key] = np.asarray(raw[key]).reshape(-1)

    if spectra.ndim != 3 or spectra.shape[1] != 2:
        raise ValueError(
            f"{path}: expected spectra shape (N, 2, num_bins), got {spectra.shape}"
        )
    n = int(spectra.shape[0])
    num_bins = int(spectra.shape[2])
    for name, arr in (
        ("applied_power", applied_power),
        ("n_steps", n_steps),
        ("P_total", p_total),
        ("Q_total", q_total),
    ):
        if int(arr.shape[0]) != n:
            raise ValueError(f"{path}: {name} length {arr.shape[0]} != N={n}")

    out: dict[str, np.ndarray] = {
        "spectra": spectra.astype(np.float32, copy=False),
        "P_total": p_total,
        "Q_total": q_total,
        "applied_power": applied_power,
        "n_steps": n_steps,
        "num_bins": np.asarray(num_bins, dtype=np.int32),
    }
    out.update(optional)
    return out


def prepare_datasets(
    arrays: dict[str, np.ndarray],
    *,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
    max_samples: int | None = None,
    noise_std: float = NOISE_STD,
) -> tuple[data.TensorDataset, data.TensorDataset, data.TensorDataset, dict[str, Any]]:
    spectra = np.asarray(arrays["spectra"], dtype=np.float32)
    ps = spectra[:, 0, :] + spectra[:, 1, :]
    applied_power = np.asarray(arrays["applied_power"], dtype=np.float32).reshape(-1)
    n_steps = np.asarray(arrays["n_steps"], dtype=np.float32).reshape(-1)
    y_p = np.asarray(arrays["P_total"], dtype=np.float32).reshape(-1)
    y_q = np.asarray(arrays["Q_total"], dtype=np.float32).reshape(-1)
    p0 = arrays.get("p0")

    n = int(ps.shape[0])
    rng = np.random.default_rng(int(seed))
    if max_samples is not None and int(max_samples) < n:
        keep = rng.choice(n, size=int(max_samples), replace=False)
        ps = ps[keep]
        applied_power = applied_power[keep]
        n_steps = n_steps[keep]
        y_p = y_p[keep]
        y_q = y_q[keep]
        if p0 is not None:
            p0 = np.asarray(p0).reshape(-1)[keep]
        n = int(ps.shape[0])

    if float(noise_std) > 0.0:
        ps = ps + rng.normal(0.0, float(noise_std), size=ps.shape).astype(np.float32)

    perm = rng.permutation(n)
    n_test = max(1, int(round(n * float(test_frac))))
    n_val = max(1, int(round(n * float(val_frac))))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    ps_mean = float(ps[train_idx].mean())
    ps_std = float(ps[train_idx].std())
    ps_std = ps_std if ps_std > 1e-8 else 1.0

    pwr_mean = float(applied_power[train_idx].mean())
    pwr_std = float(applied_power[train_idx].std())
    pwr_std = pwr_std if pwr_std > 1e-8 else 1.0

    steps_mean = float(n_steps[train_idx].mean())
    steps_std = float(n_steps[train_idx].std())
    steps_std = steps_std if steps_std > 1e-8 else 1.0

    p_mean = float(y_p[train_idx].mean())
    p_std = float(y_p[train_idx].std())
    p_std = p_std if p_std > 1e-8 else 1.0

    q_mean = float(y_q[train_idx].mean())
    q_std = float(y_q[train_idx].std())
    q_std = q_std if q_std > 1e-8 else 1.0

    def _pack(indices: np.ndarray) -> data.TensorDataset:
        t = int(ps.shape[1])
        ps_n = (ps[indices] - ps_mean) / ps_std
        pwr_n = (applied_power[indices] - pwr_mean) / pwr_std
        steps_n = (n_steps[indices] - steps_mean) / steps_std
        pwr_seq = np.broadcast_to(pwr_n.reshape(-1, 1), (indices.size, t))
        steps_seq = np.broadcast_to(steps_n.reshape(-1, 1), (indices.size, t))
        x = np.stack([ps_n, pwr_seq, steps_seq], axis=-1).astype(np.float32)
        yp = ((y_p[indices] - p_mean) / p_std).astype(np.float32)
        yq = ((y_q[indices] - q_mean) / q_std).astype(np.float32)
        return data.TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(yp),
            torch.from_numpy(yq),
        )

    stats: dict[str, Any] = {
        "ps_mean": ps_mean,
        "ps_std": ps_std,
        "pwr_mean": pwr_mean,
        "pwr_std": pwr_std,
        "steps_mean": steps_mean,
        "steps_std": steps_std,
        "P_mean": p_mean,
        "P_std": p_std,
        "Q_mean": q_mean,
        "Q_std": q_std,
        "input_size": 3,
        "num_bins": int(ps.shape[1]),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "p0_test": (p0 if p0 is not None else y_p)[test_idx],
        "noise_std": float(noise_std),
    }
    return _pack(train_idx), _pack(val_idx), _pack(test_idx), stats


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _stats_for_checkpoint(stats: dict[str, Any]) -> dict[str, Any]:
    skip = {"train_idx", "val_idx", "test_idx", "p0_test"}
    return {k: v for k, v in stats.items() if k not in skip}


def relative_weighted_loss(
    pred_z: torch.Tensor,
    true_z: torch.Tensor,
    *,
    mean: float,
    std: float,
    eps: float = REL_LOSS_EPS,
) -> torch.Tensor:
    pred = pred_z * float(std) + float(mean)
    true = true_z * float(std) + float(mean)
    return torch.mean(torch.abs(pred - true) / (torch.abs(true) + float(eps)))


def compute_rpe(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred_a = np.asarray(pred, dtype=np.float64).reshape(-1)
    true_a = np.asarray(true, dtype=np.float64).reshape(-1)
    rpe = np.full_like(true_a, np.nan, dtype=np.float64)
    valid = np.abs(true_a) > RPE_ABS_EPS
    rpe[valid] = np.abs(pred_a[valid] - true_a[valid]) / np.abs(true_a[valid]) * 100.0
    return rpe


def save_checkpoint(
    path: Path,
    *,
    model: Seq2SeqPQModel,
    stats: dict[str, Any],
    best_val_loss: float,
    best_epoch: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": clone_state_dict(model),
            "stats": _stats_for_checkpoint(stats),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "input_size": int(stats["input_size"]),
        },
        path,
    )


def load_checkpoint(
    path: Path, *, device: torch.device = DEVICE
) -> tuple[Seq2SeqPQModel, dict[str, Any]]:
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Seq2SeqPQModel(
        input_size=int(ckpt.get("input_size", ckpt["stats"]["input_size"])),
        hidden_size=int(ckpt["hidden_size"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt.get("dropout", DROPOUT)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def train_model(
    train_dataset: data.TensorDataset,
    val_dataset: data.TensorDataset,
    stats: dict[str, Any],
    *,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    checkpoint_path: Path | None = None,
    device: torch.device = DEVICE,
) -> tuple[Seq2SeqPQModel, float, dict[str, list[float]]]:
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    p_mean = float(stats["P_mean"])
    p_std = float(stats["P_std"])
    q_mean = float(stats["Q_mean"])
    q_std = float(stats["Q_std"])

    model = Seq2SeqPQModel(
        input_size=int(stats["input_size"]),
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        dropout=float(dropout),
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_p_rpe": [],
        "val_q_rpe": [],
    }
    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(int(num_epochs)):
        model.train()
        train_sum = 0.0
        train_batches = 0
        for x_b, y_p, y_q in train_loader:
            x_b = x_b.to(device)
            y_p = y_p.to(device)
            y_q = y_q.to(device)
            pred_p, pred_q = model(x_b)
            loss = relative_weighted_loss(
                pred_p, y_p, mean=p_mean, std=p_std
            ) + relative_weighted_loss(pred_q, y_q, mean=q_mean, std=q_std)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            train_sum += float(loss.item())
            train_batches += 1
        avg_train = train_sum / max(train_batches, 1)

        model.eval()
        val_sum = 0.0
        val_batches = 0
        rpe_p_sum = 0.0
        rpe_q_sum = 0.0
        with torch.no_grad():
            for x_v, y_p, y_q in val_loader:
                x_v = x_v.to(device)
                y_p = y_p.to(device)
                y_q = y_q.to(device)
                pred_p, pred_q = model(x_v)
                loss_p = relative_weighted_loss(
                    pred_p, y_p, mean=p_mean, std=p_std
                )
                loss_q = relative_weighted_loss(
                    pred_q, y_q, mean=q_mean, std=q_std
                )
                val_sum += float((loss_p + loss_q).item())
                rpe_p_sum += float(loss_p.item())
                rpe_q_sum += float(loss_q.item())
                val_batches += 1

        avg_val = val_sum / max(val_batches, 1)
        rpe_p = (rpe_p_sum / max(val_batches, 1)) * 100.0
        rpe_q = (rpe_q_sum / max(val_batches, 1)) * 100.0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_p_rpe"].append(rpe_p)
        history["val_q_rpe"].append(rpe_q)

        scheduler.step(avg_val)
        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"epoch {epoch + 1:03d}/{num_epochs} | "
            f"train {avg_train:.6f} | val {avg_val:.6f} | "
            f"RPE% P={rpe_p:.3f} Q={rpe_q:.3f} | lr {lr:.2e}",
            flush=True,
        )

        if avg_val < best_val - MIN_DELTA:
            best_val = avg_val
            best_epoch = epoch + 1
            best_state = clone_state_dict(model)
            stale = 0
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    stats=stats,
                    best_val_loss=best_val,
                    best_epoch=best_epoch,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
        else:
            stale += 1
            if stale >= int(patience):
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
                break

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        model, ckpt = load_checkpoint(checkpoint_path, device=device)
        best_val = float(ckpt.get("best_val_loss", best_val))
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, history


@torch.no_grad()
def predict_denormalized(
    model: Seq2SeqPQModel,
    dataset: data.TensorDataset,
    stats: dict[str, Any],
    *,
    batch_size: int = BATCH_SIZE,
    device: torch.device = DEVICE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    p_mean = float(stats["P_mean"])
    p_std = float(stats["P_std"])
    q_mean = float(stats["Q_mean"])
    q_std = float(stats["Q_std"])
    pred_p_all: list[np.ndarray] = []
    pred_q_all: list[np.ndarray] = []
    true_p_all: list[np.ndarray] = []
    true_q_all: list[np.ndarray] = []
    for x_b, y_p, y_q in loader:
        pred_p, pred_q = model(x_b.to(device))
        pred_p_all.append(pred_p.cpu().numpy() * p_std + p_mean)
        pred_q_all.append(pred_q.cpu().numpy() * q_std + q_mean)
        true_p_all.append(y_p.numpy() * p_std + p_mean)
        true_q_all.append(y_q.numpy() * q_std + q_mean)
    return (
        np.concatenate(pred_p_all),
        np.concatenate(pred_q_all),
        np.concatenate(true_p_all),
        np.concatenate(true_q_all),
    )


@torch.no_grad()
def evaluate_model(
    model: Seq2SeqPQModel,
    dataset: data.TensorDataset,
    stats: dict[str, Any],
    *,
    batch_size: int = BATCH_SIZE,
    device: torch.device = DEVICE,
) -> dict[str, Any]:
    pred_p_arr, pred_q_arr, true_p_arr, true_q_arr = predict_denormalized(
        model, dataset, stats, batch_size=batch_size, device=device
    )

    def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
        err = pred - true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((true - np.mean(true)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else float("nan")
        rpe = compute_rpe(pred, true)
        finite = rpe[np.isfinite(rpe)]
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "rpe_mean": float(np.mean(finite)) if finite.size else float("nan"),
            "rpe_median": float(np.median(finite)) if finite.size else float("nan"),
            "rpe_std": float(np.std(finite)) if finite.size else float("nan"),
        }

    p_m = _metrics(pred_p_arr, true_p_arr)
    q_m = _metrics(pred_q_arr, true_q_arr)

    return {
        "P_mae": p_m["mae"],
        "P_rmse": p_m["rmse"],
        "P_r2": p_m["r2"],
        "P_rpe_mean": p_m["rpe_mean"],
        "P_rpe_median": p_m["rpe_median"],
        "P_rpe_std": p_m["rpe_std"],
        "Q_mae": q_m["mae"],
        "Q_rmse": q_m["rmse"],
        "Q_r2": q_m["r2"],
        "Q_rpe_mean": q_m["rpe_mean"],
        "Q_rpe_median": q_m["rpe_median"],
        "Q_rpe_std": q_m["rpe_std"],
        "pred_P": pred_p_arr,
        "pred_Q": pred_q_arr,
        "true_P": true_p_arr,
        "true_Q": true_q_arr,
    }


def save_plots(
    history: dict[str, list[float]],
    metrics: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if history.get("train_loss"):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(history["train_loss"], label="train")
        ax.plot(history["val_loss"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("rel loss P + Q")
        ax.set_title("Seq2seq → P_total / Q_total training loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curves.png", dpi=140)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, label, true_key, pred_key, mae_key, r2_key in (
        (axes[0], "P_total", "true_P", "pred_P", "P_mae", "P_r2"),
        (axes[1], "Q_total", "true_Q", "pred_Q", "Q_mae", "Q_r2"),
    ):
        true = np.asarray(metrics[true_key], dtype=float)
        pred = np.asarray(metrics[pred_key], dtype=float)
        ax.scatter(true, pred, s=12, alpha=0.55, edgecolors="none")
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        pad = 0.05 * (hi - lo + 1e-8)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0)
        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"pred {label}")
        ax.set_title(
            f"{label}: MAE={float(metrics[mae_key]):.4g}  "
            f"R²={float(metrics[r2_key]):.4f}"
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_true.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate LSTM seq model: Ps spectrum → P_total, Q_total."
    )
    parser.add_argument("--spectra", type=Path, default=None, help="Path to spectra.npz")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Subsample this many events (recommended for first runs)",
    )
    parser.add_argument("--noise-std", type=float, default=NOISE_STD)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: <out-dir>/seq2seq_pq_best.pth)",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training; load --checkpoint and evaluate on a fresh split",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (args.out_dir / "seq2seq_pq_best.pth")

    spectra_path = resolve_spectra_path(args.spectra)
    print(f"Loading {spectra_path} ...", flush=True)
    arrays = load_seq2seq_npz(spectra_path)

    print("Preparing datasets...", flush=True)
    train_ds, val_ds, test_ds, stats = prepare_datasets(
        arrays,
        max_samples=args.max_samples,
        noise_std=args.noise_std,
    )
    print(
        f"Train={stats['n_train']} Val={stats['n_val']} Test={stats['n_test']} "
        f"bins={stats['num_bins']}",
        flush=True,
    )

    history: dict[str, list[float]]
    if args.test_only:
        print(f"Loading checkpoint {checkpoint_path} ...", flush=True)
        model, ckpt = load_checkpoint(checkpoint_path)
        stats = {**stats, **ckpt["stats"]}
        best_val = float(ckpt.get("best_val_loss", float("nan")))
        history = {"train_loss": [], "val_loss": [], "val_p_rpe": [], "val_q_rpe": []}
    else:
        print("Training seq2seq model...", flush=True)
        model, best_val, history = train_model(
            train_ds,
            val_ds,
            stats,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            checkpoint_path=checkpoint_path,
        )

    print("Evaluating on test set...", flush=True)
    metrics = evaluate_model(model, test_ds, stats, batch_size=args.batch_size)

    print(
        f"Test RPE% median  P={metrics['P_rpe_median']:.3f}  "
        f"Q={metrics['Q_rpe_median']:.3f}",
        flush=True,
    )
    print(
        f"Test MAE  P={metrics['P_mae']:.6f}  Q={metrics['Q_mae']:.6f}  "
        f"R²  P={metrics['P_r2']:.4f}  Q={metrics['Q_r2']:.4f}",
        flush=True,
    )

    json_metrics = {
        k: v for k, v in metrics.items() if isinstance(v, (float, int, str, list))
    }
    json_metrics["best_val_loss"] = float(best_val)
    json_metrics["checkpoint"] = str(checkpoint_path)
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=2)

    save_plots(history, metrics, args.out_dir)
    print(f"Done. Results in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
