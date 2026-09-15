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
from matplotlib.axes import Axes
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "seq2seq_pq_results_v2"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 500
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
N_EXAMPLE_PLOTS = 24

POL_ABS_BANDS: tuple[tuple[float, float], ...] = tuple(
    (lo / 100.0, (lo + 5) / 100.0) for lo in range(5, 95, 5)
)


class Seq2SeqPQModel(nn.Module):
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
    orig_idx = np.arange(n, dtype=np.int64)
    if max_samples is not None and int(max_samples) < n:
        keep = rng.choice(n, size=int(max_samples), replace=False)
        ps = ps[keep]
        applied_power = applied_power[keep]
        n_steps = n_steps[keep]
        y_p = y_p[keep]
        y_q = y_q[keep]
        if p0 is not None:
            p0 = np.asarray(p0).reshape(-1)[keep]
        orig_idx = orig_idx[keep]
        n = int(ps.shape[0])

    if float(noise_std) > 0.0:
        ps = ps + rng.normal(0.0, float(noise_std), size=ps.shape).astype(np.float32)

    perm = rng.permutation(n)
    n_test = max(1, int(round(n * float(test_frac))))
    n_val = max(1, int(round(n * float(val_frac))))
    local_test = perm[:n_test]
    local_val = perm[n_test : n_test + n_val]
    local_train = perm[n_test + n_val :]
    test_idx = orig_idx[local_test]
    val_idx = orig_idx[local_val]
    train_idx = orig_idx[local_train]

    ps_mean = float(ps[local_train].mean())
    ps_std = float(ps[local_train].std())
    ps_std = ps_std if ps_std > 1e-8 else 1.0

    pwr_mean = float(applied_power[local_train].mean())
    pwr_std = float(applied_power[local_train].std())
    pwr_std = pwr_std if pwr_std > 1e-8 else 1.0

    steps_mean = float(n_steps[local_train].mean())
    steps_std = float(n_steps[local_train].std())
    steps_std = steps_std if steps_std > 1e-8 else 1.0

    p_mean = float(y_p[local_train].mean())
    p_std = float(y_p[local_train].std())
    p_std = p_std if p_std > 1e-8 else 1.0

    q_mean = float(y_q[local_train].mean())
    q_std = float(y_q[local_train].std())
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
        "n_train": int(local_train.size),
        "n_val": int(local_val.size),
        "n_test": int(local_test.size),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "p0_test": (p0 if p0 is not None else y_p)[local_test],
        "noise_std": float(noise_std),
    }
    return _pack(local_train), _pack(local_val), _pack(local_test), stats


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


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
        }
    return {
        "n": float(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
    }


def polarization_range_stats(
    pred_p: np.ndarray,
    true_p: np.ndarray,
    pred_q: np.ndarray,
    true_q: np.ndarray,
    *,
    pol_ref: np.ndarray,
    bands: tuple[tuple[float, float], ...] = POL_ABS_BANDS,
    ref_name: str = "abs_P",
) -> list[dict[str, Any]]:
    pred_p = np.asarray(pred_p, dtype=np.float64).reshape(-1)
    true_p = np.asarray(true_p, dtype=np.float64).reshape(-1)
    pred_q = np.asarray(pred_q, dtype=np.float64).reshape(-1)
    true_q = np.asarray(true_q, dtype=np.float64).reshape(-1)
    ref = np.abs(np.asarray(pol_ref, dtype=np.float64).reshape(-1))
    rpe_p = compute_rpe(pred_p, true_p)
    rpe_q = compute_rpe(pred_q, true_q)
    res_p = pred_p - true_p
    res_q = pred_q - true_q

    rows: list[dict[str, Any]] = []
    for lo, hi in bands:
        mask = (ref >= lo) & (ref < hi) if hi < 0.999 else (ref >= lo) & (ref <= hi)
        label = f"{int(round(lo * 100))}-{int(round(hi * 100))}%"
        p_rpe = _summary_stats(rpe_p[mask])
        q_rpe = _summary_stats(rpe_q[mask])
        p_res = _summary_stats(res_p[mask])
        q_res = _summary_stats(res_q[mask])
        rows.append(
            {
                "ref": ref_name,
                "range": label,
                "lo": float(lo),
                "hi": float(hi),
                "n": int(np.sum(mask)),
                "P_rpe_mean": p_rpe["mean"],
                "P_rpe_median": p_rpe["median"],
                "P_rpe_std": p_rpe["std"],
                "P_rpe_p25": p_rpe["p25"],
                "P_rpe_p75": p_rpe["p75"],
                "P_residual_mean": p_res["mean"],
                "P_residual_median": p_res["median"],
                "P_residual_std": p_res["std"],
                "Q_rpe_mean": q_rpe["mean"],
                "Q_rpe_median": q_rpe["median"],
                "Q_rpe_std": q_rpe["std"],
                "Q_rpe_p25": q_rpe["p25"],
                "Q_rpe_p75": q_rpe["p75"],
                "Q_residual_mean": q_res["mean"],
                "Q_residual_median": q_res["median"],
                "Q_residual_std": q_res["std"],
            }
        )
    return rows


def print_range_stats_table(rows: list[dict[str, Any]], *, title: str) -> None:
    print(f"\n===== {title} =====", flush=True)
    header = (
        f"{'range':>10} {'n':>5} "
        f"{'P_RPE_med':>10} {'P_RPE_mean':>10} {'P_RPE_std':>10} "
        f"{'P_res_mean':>11} {'P_res_std':>10} "
        f"{'Q_RPE_med':>10} {'Q_RPE_mean':>10} {'Q_RPE_std':>10} "
        f"{'Q_res_mean':>11} {'Q_res_std':>10}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for row in rows:
        if int(row["n"]) == 0:
            continue
        print(
            f"{row['range']:>10} {int(row['n']):5d} "
            f"{row['P_rpe_median']:10.3f} {row['P_rpe_mean']:10.3f} {row['P_rpe_std']:10.3f} "
            f"{row['P_residual_mean']:11.5f} {row['P_residual_std']:10.5f} "
            f"{row['Q_rpe_median']:10.3f} {row['Q_rpe_mean']:10.3f} {row['Q_rpe_std']:10.3f} "
            f"{row['Q_residual_mean']:11.5f} {row['Q_residual_std']:10.5f}",
            flush=True,
        )


def save_range_stats_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(
                ",".join(
                    ""
                    if row[k] is None
                    or (isinstance(row[k], float) and not np.isfinite(row[k]))
                    else str(row[k])
                    for k in keys
                )
                + "\n"
            )


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
        rpe_s = _summary_stats(compute_rpe(pred, true))
        res_s = _summary_stats(err)
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "rpe_mean": rpe_s["mean"],
            "rpe_median": rpe_s["median"],
            "rpe_std": rpe_s["std"],
            "residual_mean": res_s["mean"],
            "residual_median": res_s["median"],
            "residual_std": res_s["std"],
        }

    p_m = _metrics(pred_p_arr, true_p_arr)
    q_m = _metrics(pred_q_arr, true_q_arr)

    range_by_p = polarization_range_stats(
        pred_p_arr,
        true_p_arr,
        pred_q_arr,
        true_q_arr,
        pol_ref=true_p_arr,
        ref_name="abs_P_total",
    )
    p0_test = stats.get("p0_test")
    range_by_p0: list[dict[str, Any]] = []
    if p0_test is not None and int(np.asarray(p0_test).size) == int(true_p_arr.size):
        range_by_p0 = polarization_range_stats(
            pred_p_arr,
            true_p_arr,
            pred_q_arr,
            true_q_arr,
            pol_ref=np.asarray(p0_test, dtype=np.float64),
            ref_name="abs_p0",
        )

    return {
        "P_mae": p_m["mae"],
        "P_rmse": p_m["rmse"],
        "P_r2": p_m["r2"],
        "P_rpe_mean": p_m["rpe_mean"],
        "P_rpe_median": p_m["rpe_median"],
        "P_rpe_std": p_m["rpe_std"],
        "P_residual_mean": p_m["residual_mean"],
        "P_residual_median": p_m["residual_median"],
        "P_residual_std": p_m["residual_std"],
        "Q_mae": q_m["mae"],
        "Q_rmse": q_m["rmse"],
        "Q_r2": q_m["r2"],
        "Q_rpe_mean": q_m["rpe_mean"],
        "Q_rpe_median": q_m["rpe_median"],
        "Q_rpe_std": q_m["rpe_std"],
        "Q_residual_mean": q_m["residual_mean"],
        "Q_residual_median": q_m["residual_median"],
        "Q_residual_std": q_m["residual_std"],
        "pred_P": pred_p_arr,
        "pred_Q": pred_q_arr,
        "true_P": true_p_arr,
        "true_Q": true_q_arr,
        "range_stats_by_P": range_by_p,
        "range_stats_by_p0": range_by_p0,
    }


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 1.15,
        }
    )


def _apply_axes_style(ax: Axes) -> None:
    ax.set_facecolor("#f7f8fa")
    ax.grid(True, which="major", color="white", linewidth=1.2, alpha=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#1f2933")
        spine.set_linewidth(1.15)
    ax.tick_params(colors="#4a5560", labelsize=9)


def _fmt_stat(value: float, *, percent: bool = False) -> str:
    if not np.isfinite(value):
        return r"n/a"
    if percent:
        return rf"${value:.3f}\%$"
    av = abs(float(value))
    if av == 0.0:
        return r"$0$"
    if av >= 1e-2:
        return rf"${value:.4g}$"
    return rf"${value:.3e}$"


def _annotate_stats_box(
    ax: Axes,
    lines: list[str],
    *,
    loc: str = "upper right",
) -> None:
    text = "\n".join(lines)
    anchors = {
        "upper right": (0.98, 0.97, "right", "top"),
        "upper left": (0.02, 0.97, "left", "top"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "lower left": (0.02, 0.03, "left", "bottom"),
    }
    x, y, ha, va = anchors[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.5,
        color="#24303a",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d0d7de",
            "alpha": 0.92,
        },
    )


def save_history_json(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: [float(x) for x in v] for k, v in history.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_history_json(path: Path) -> dict[str, list[float]] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "train_loss" not in raw:
        return None
    return {
        key: [float(x) for x in values]
        for key, values in raw.items()
        if isinstance(values, list)
    }


def _plot_loss_curves(
    history: dict[str, list[float]],
    plots_dir: Path,
    *,
    best_val_loss: float | None = None,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    color_train = "#2f6fed"
    color_val = "#c45c26"
    color_best = "#0f766e"

    train = np.asarray(history["train_loss"], dtype=np.float64)
    val = np.asarray(history.get("val_loss", []), dtype=np.float64)
    n = int(train.size)
    epochs = np.arange(1, n + 1)
    val = val[:n] if val.size else np.full(n, np.nan)
    has_val = bool(np.isfinite(val).any())
    best_i = int(np.nanargmin(val)) if has_val else -1
    best_ep = int(epochs[best_i]) if best_i >= 0 else -1
    best_v = float(val[best_i]) if best_i >= 0 else float("nan")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(epochs, train, color=color_train, lw=2.2, label="Train")
    if has_val:
        ax.plot(epochs, val, color=color_val, lw=2.0, label="Val")
        ax.axvline(best_ep, color=color_best, lw=1.1, ls="--", alpha=0.85)
        ax.scatter(
            [best_ep],
            [best_v],
            s=54,
            color=color_best,
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
            label=rf"Best val @ ${best_ep}$",
        )

    positive_parts = [train[train > 0]]
    if has_val:
        positive_parts.append(val[np.isfinite(val) & (val > 0)])
    positive = (
        np.concatenate(positive_parts)
        if any(p.size for p in positive_parts)
        else np.array([])
    )
    if positive.size:
        ax.set_yscale("log")
        ax.set_ylim(float(np.min(positive)) * 0.7, float(np.max(positive)) * 1.25)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Relative loss ($P + Q$)")
    ax.set_title("Training and validation loss")
    x_pad = max(2.0, 0.04 * max(n, 1))
    ax.set_xlim(1.0 - 0.25 * x_pad, max(n, 1) + x_pad)
    _apply_axes_style(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    final_train = float(train[-1]) if train.size else float("nan")
    final_val = float(val[-1]) if has_val else float("nan")
    shown_best = (
        float(best_val_loss)
        if best_val_loss is not None and np.isfinite(best_val_loss)
        else best_v
    )
    _annotate_stats_box(
        ax,
        [
            rf"epochs ${n}$",
            rf"best val {_fmt_stat(shown_best)}",
            rf"final train {_fmt_stat(final_train)}",
            rf"final val {_fmt_stat(final_val)}",
        ],
        loc="lower left",
    )

    fig.tight_layout()
    path = plots_dir / "loss_curves.png"
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return path


def save_plots(
    history: dict[str, list[float]] | None,
    metrics: dict[str, Any],
    plots_dir: Path,
    *,
    best_val_loss: float | None = None,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    _configure_plot_style()

    color_p = "#2f6fed"
    color_q = "#c45c26"
    color_zero = "#1f2933"
    color_mean = "#0f766e"
    color_median = "#b45309"
    color_sigma = "#64748b"
    label_mean_pm = r"mean $\pm 1\sigma$"

    if history is not None and history.get("train_loss"):
        path = _plot_loss_curves(
            history, plots_dir, best_val_loss=best_val_loss
        )
        saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    for ax, label, true_key, pred_key, mae_key, r2_key, color in (
        (
            axes[0],
            r"$P_{\mathrm{total}}$",
            "true_P",
            "pred_P",
            "P_mae",
            "P_r2",
            color_p,
        ),
        (
            axes[1],
            r"$Q_{\mathrm{total}}$",
            "true_Q",
            "pred_Q",
            "Q_mae",
            "Q_r2",
            color_q,
        ),
    ):
        true = np.asarray(metrics[true_key], dtype=float)
        pred = np.asarray(metrics[pred_key], dtype=float)
        ax.scatter(
            true,
            pred,
            s=14,
            alpha=0.35,
            c=color,
            edgecolors="none",
            rasterized=True,
        )
        lo = float(min(true.min(), pred.min()))
        hi = float(max(true.max(), pred.max()))
        pad = 0.05 * (hi - lo + 1e-8)
        lims = [lo - pad, hi + pad]
        ax.plot(lims, lims, color=color_zero, ls="--", lw=1.2, label="ideal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(rf"True {label}")
        ax.set_ylabel(rf"Predicted {label}")
        ax.set_title(rf"{label}: predicted vs true")
        ax.set_aspect("equal", adjustable="box")
        _apply_axes_style(ax)
        _annotate_stats_box(
            ax,
            [
                rf"MAE {_fmt_stat(float(metrics[mae_key]))}",
                rf"$R^2$ {_fmt_stat(float(metrics[r2_key]))}",
            ],
            loc="lower right",
        )
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    path = plots_dir / "pred_vs_true.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    saved.append(path)

    true_p = np.asarray(metrics["true_P"], dtype=np.float64)
    pred_p = np.asarray(metrics["pred_P"], dtype=np.float64)
    true_q = np.asarray(metrics["true_Q"], dtype=np.float64)
    pred_q = np.asarray(metrics["pred_Q"], dtype=np.float64)
    res_p = pred_p - true_p
    res_q = pred_q - true_q
    rpe_p = compute_rpe(pred_p, true_p)
    rpe_q = compute_rpe(pred_q, true_q)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, res, label, color, mean_key, med_key, std_key in (
        (
            axes[0],
            res_p,
            r"$P_{\mathrm{total}}$",
            color_p,
            "P_residual_mean",
            "P_residual_median",
            "P_residual_std",
        ),
        (
            axes[1],
            res_q,
            r"$Q_{\mathrm{total}}$",
            color_q,
            "Q_residual_mean",
            "Q_residual_median",
            "Q_residual_std",
        ),
    ):
        finite = res[np.isfinite(res)]
        mean_v = float(metrics[mean_key])
        med_v = float(metrics[med_key])
        std_v = float(metrics[std_key])
        if finite.size:
            ax.hist(
                finite,
                bins=55,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.88,
            )
            ax.axvline(0.0, color=color_zero, lw=1.1, ls="--", label="zero")
            ax.axvline(mean_v, color=color_mean, lw=1.4, label="mean")
            ax.axvline(med_v, color=color_median, lw=1.4, ls="-.", label="median")
            if np.isfinite(std_v) and std_v > 0:
                ax.axvspan(
                    mean_v - std_v,
                    mean_v + std_v,
                    color=color_sigma,
                    alpha=0.18,
                    label=label_mean_pm,
                    zorder=0,
                )
                ax.axvline(mean_v - std_v, color=color_sigma, lw=1.0, ls=":")
                ax.axvline(mean_v + std_v, color=color_sigma, lw=1.0, ls=":")
        ax.set_xlabel(rf"Residual (pred $-$ true) {label}")
        ax.set_ylabel("Count")
        ax.set_title(rf"{label} residual distribution")
        _apply_axes_style(ax)
        _annotate_stats_box(
            ax,
            [
                rf"mean {_fmt_stat(mean_v)}",
                rf"median {_fmt_stat(med_v)}",
                rf"std {_fmt_stat(std_v)}",
            ],
            loc="upper right",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = plots_dir / "residual_histograms.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, rpe, label, color, mean_key, med_key, std_key in (
        (
            axes[0],
            rpe_p,
            r"$P_{\mathrm{total}}$",
            color_p,
            "P_rpe_mean",
            "P_rpe_median",
            "P_rpe_std",
        ),
        (
            axes[1],
            rpe_q,
            r"$Q_{\mathrm{total}}$",
            color_q,
            "Q_rpe_mean",
            "Q_rpe_median",
            "Q_rpe_std",
        ),
    ):
        finite = rpe[np.isfinite(rpe)]
        mean_v = float(metrics[mean_key])
        med_v = float(metrics[med_key])
        std_v = float(metrics[std_key])
        if finite.size:
            ax.hist(
                finite,
                bins=55,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.88,
            )
            ax.axvline(mean_v, color=color_mean, lw=1.4, label="mean")
            ax.axvline(med_v, color=color_median, lw=1.4, ls="-.", label="median")
            if np.isfinite(std_v) and std_v > 0:
                ax.axvspan(
                    max(0.0, mean_v - std_v),
                    mean_v + std_v,
                    color=color_sigma,
                    alpha=0.18,
                    label=label_mean_pm,
                    zorder=0,
                )
                ax.axvline(
                    max(0.0, mean_v - std_v), color=color_sigma, lw=1.0, ls=":"
                )
                ax.axvline(mean_v + std_v, color=color_sigma, lw=1.0, ls=":")
            p99 = float(np.percentile(finite, 99))
            x_hi = max(p99 * 1.05, mean_v + 1.2 * std_v if np.isfinite(std_v) else p99)
            if np.isfinite(x_hi) and x_hi > 0:
                ax.set_xlim(0.0, x_hi)
        ax.set_xlabel(rf"RPE (\%)  {label}")
        ax.set_ylabel("Count")
        ax.set_title(rf"{label} relative percent error")
        _apply_axes_style(ax)
        _annotate_stats_box(
            ax,
            [
                rf"mean {_fmt_stat(mean_v, percent=True)}",
                rf"median {_fmt_stat(med_v, percent=True)}",
                rf"std {_fmt_stat(std_v, percent=True)}",
            ],
            loc="upper right",
        )
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    path = plots_dir / "rpe_histograms.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, true, res, label, color, mean_key, std_key in (
        (
            axes[0],
            true_p,
            res_p,
            r"$P_{\mathrm{total}}$",
            color_p,
            "P_residual_mean",
            "P_residual_std",
        ),
        (
            axes[1],
            true_q,
            res_q,
            r"$Q_{\mathrm{total}}$",
            color_q,
            "Q_residual_mean",
            "Q_residual_std",
        ),
    ):
        mean_v = float(metrics[mean_key])
        std_v = float(metrics[std_key])
        ax.scatter(
            true,
            res,
            s=12,
            alpha=0.28,
            c=color,
            edgecolors="none",
            rasterized=True,
        )
        ax.axhline(0.0, color=color_zero, lw=1.1, ls="--", label="zero")
        ax.axhline(mean_v, color=color_mean, lw=1.3, label="mean")
        if np.isfinite(std_v) and std_v > 0:
            ax.axhspan(
                mean_v - std_v,
                mean_v + std_v,
                color=color_sigma,
                alpha=0.16,
                label=label_mean_pm,
                zorder=0,
            )
            ax.axhline(mean_v - std_v, color=color_sigma, lw=1.0, ls=":")
            ax.axhline(mean_v + std_v, color=color_sigma, lw=1.0, ls=":")
        ax.set_xlabel(rf"True {label}")
        ax.set_ylabel(rf"Residual {label}")
        ax.set_title(rf"{label}: residual vs true")
        _apply_axes_style(ax)
        _annotate_stats_box(
            ax,
            [
                rf"mean {_fmt_stat(mean_v)}",
                rf"std {_fmt_stat(std_v)}",
            ],
            loc="upper right",
        )
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    path = plots_dir / "residuals_vs_true.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    saved.append(path)

    range_rows = list(metrics.get("range_stats_by_P") or [])
    nonempty = [r for r in range_rows if int(r["n"]) > 0]
    if nonempty:
        labels = [r["range"] for r in nonempty]
        x = np.arange(len(labels))
        n_bands = sum(int(r["n"]) for r in nonempty)
        fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
        for ax, prefix, ylabel, title, color in (
            (
                axes[0],
                "P",
                r"$P$ RPE (\%)",
                r"$P$ RPE by $|P_{\mathrm{total}}|$ band",
                color_p,
            ),
            (
                axes[1],
                "Q",
                r"$Q$ RPE (\%)",
                r"$Q$ RPE by $|P_{\mathrm{total}}|$ band",
                color_q,
            ),
        ):
            med = np.asarray([r[f"{prefix}_rpe_median"] for r in nonempty], dtype=float)
            mean = np.asarray([r[f"{prefix}_rpe_mean"] for r in nonempty], dtype=float)
            std = np.asarray([r[f"{prefix}_rpe_std"] for r in nonempty], dtype=float)
            ax.bar(
                x - 0.18,
                med,
                width=0.32,
                color="#94a3b8",
                edgecolor="white",
                linewidth=0.6,
                label="median",
            )
            ax.bar(
                x + 0.18,
                mean,
                width=0.32,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label="mean",
                yerr=std,
                error_kw={
                    "ecolor": color_zero,
                    "elinewidth": 1.1,
                    "capsize": 3.0,
                    "capthick": 1.0,
                    "alpha": 0.85,
                },
            )
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            _apply_axes_style(ax)
            ax.legend(frameon=False, fontsize=8, loc="upper right")
            _annotate_stats_box(
                ax,
                [
                    r"error bars: $\pm 1\sigma$ of RPE",
                    rf"bands $n={n_bands}$",
                ],
                loc="upper left",
            )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=40, ha="right")
        axes[1].set_xlabel(r"$|P_{\mathrm{total}}|$ band")
        fig.tight_layout()
        path = plots_dir / "rpe_by_polarization_range.png"
        fig.savefig(path, dpi=160, facecolor="white")
        plt.close(fig)
        saved.append(path)

        fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True)
        for ax, prefix, ylabel, title, color in (
            (
                axes[0],
                "P",
                r"Mean residual $P$",
                r"$P$ residuals by $|P_{\mathrm{total}}|$ band",
                color_p,
            ),
            (
                axes[1],
                "Q",
                r"Mean residual $Q$",
                r"$Q$ residuals by $|P_{\mathrm{total}}|$ band",
                color_q,
            ),
        ):
            mean = np.asarray(
                [r[f"{prefix}_residual_mean"] for r in nonempty], dtype=float
            )
            std = np.asarray(
                [r[f"{prefix}_residual_std"] for r in nonempty], dtype=float
            )
            ax.bar(
                x,
                mean,
                width=0.62,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                yerr=std,
                error_kw={
                    "ecolor": color_zero,
                    "elinewidth": 1.1,
                    "capsize": 3.0,
                    "capthick": 1.0,
                    "alpha": 0.85,
                },
            )
            ax.axhline(0.0, color=color_zero, lw=1.0, ls="--")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            _apply_axes_style(ax)
            _annotate_stats_box(
                ax,
                [
                    r"error bars: $\pm 1\sigma$ of residual",
                    rf"bands $n={n_bands}$",
                ],
                loc="upper left",
            )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=40, ha="right")
        axes[1].set_xlabel(r"$|P_{\mathrm{total}}|$ band")
        fig.tight_layout()
        path = plots_dir / "residuals_by_polarization_range.png"
        fig.savefig(path, dpi=160, facecolor="white")
        plt.close(fig)
        saved.append(path)

    return saved


def _select_example_indices(
    n_test: int,
    n_examples: int,
    *,
    true_p: np.ndarray,
    pred_p: np.ndarray,
    true_q: np.ndarray,
    pred_q: np.ndarray,
    seed: int = SEED,
) -> np.ndarray:
    n = int(n_test)
    k = min(max(1, int(n_examples)), n)
    if k >= n:
        return np.arange(n, dtype=int)

    err = np.abs(pred_p - true_p) + np.abs(pred_q - true_q)
    order_err = np.argsort(err)
    err_picks = order_err[np.round(np.linspace(0, n - 1, max(1, k // 2))).astype(int)]
    p_order = np.argsort(np.abs(true_p))
    p_picks = p_order[
        np.round(np.linspace(0, n - 1, max(1, k - err_picks.size))).astype(int)
    ]
    picks = np.unique(np.concatenate([err_picks, p_picks]))
    if picks.size < k:
        rng = np.random.default_rng(int(seed))
        extra = rng.choice(
            np.setdiff1d(np.arange(n), picks, assume_unique=False),
            size=k - picks.size,
            replace=False,
        )
        picks = np.concatenate([picks, extra])
    return np.sort(picks[:k].astype(int))


def save_example_signal_plots(
    arrays: dict[str, np.ndarray],
    stats: dict[str, Any],
    metrics: dict[str, Any],
    plots_dir: Path,
    *,
    n_examples: int = N_EXAMPLE_PLOTS,
    seed: int = SEED,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = plots_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    _configure_plot_style()

    test_idx = np.asarray(stats["test_idx"], dtype=int)
    spectra = np.asarray(arrays["spectra"], dtype=np.float64)
    num_bins = int(spectra.shape[2])
    freq = np.linspace(-6.0, 6.0, num_bins)

    pred_p = np.asarray(metrics["pred_P"], dtype=np.float64).reshape(-1)
    true_p = np.asarray(metrics["true_P"], dtype=np.float64).reshape(-1)
    pred_q = np.asarray(metrics["pred_Q"], dtype=np.float64).reshape(-1)
    true_q = np.asarray(metrics["true_Q"], dtype=np.float64).reshape(-1)
    if pred_p.size != test_idx.size:
        raise ValueError(
            f"metrics length {pred_p.size} != test_idx length {test_idx.size}"
        )

    pick = _select_example_indices(
        int(test_idx.size),
        int(n_examples),
        true_p=true_p,
        pred_p=pred_p,
        true_q=true_q,
        pred_q=pred_q,
        seed=int(seed),
    )

    applied = np.asarray(arrays["applied_power"], dtype=np.float64).reshape(-1)
    n_steps_arr = np.asarray(arrays["n_steps"], dtype=np.float64).reshape(-1)
    p0_arr = (
        np.asarray(arrays["p0"], dtype=np.float64).reshape(-1)
        if "p0" in arrays
        else None
    )
    source = (
        np.asarray(arrays["source"], dtype=np.int64).reshape(-1)
        if "source" in arrays
        else None
    )
    source_name = {0: "ssRF", 1: "AFP"}

    saved: list[Path] = []
    summary_rows = min(4, int(pick.size))
    if summary_rows > 0:
        fig, axes = plt.subplots(
            summary_rows,
            2,
            figsize=(11.5, 3.2 * summary_rows),
            squeeze=False,
        )
    else:
        fig = None
        axes = None

    for panel_i, local_i in enumerate(pick):
        gi = int(test_idx[int(local_i)])
        ip = spectra[gi, 0]
        im = spectra[gi, 1]
        ps = ip + im
        q_spec = ip - im
        tp = float(true_p[local_i])
        pp = float(pred_p[local_i])
        tq = float(true_q[local_i])
        pq = float(pred_q[local_i])
        rpe_p = float(compute_rpe(np.array([pp]), np.array([tp]))[0])
        rpe_q = float(compute_rpe(np.array([pq]), np.array([tq]))[0])
        p0_v = float(p0_arr[gi]) if p0_arr is not None else float("nan")
        power = float(applied[gi])
        steps = int(n_steps_arr[gi])
        src = (
            source_name.get(int(source[gi]), str(int(source[gi])))
            if source is not None
            else "?"
        )

        fig_e, (ax_s, ax_pq) = plt.subplots(
            1,
            2,
            figsize=(11.5, 4.2),
            gridspec_kw={"width_ratios": [2.2, 1.0]},
        )
        ax_s.plot(freq, ps, color="#1f2933", label=r"$P_s = I_+ + I_-$", lw=1.8)
        ax_s.plot(freq, ip, color="#2f6fed", label=r"$I_+$", alpha=0.75, ls="--", lw=1.2)
        ax_s.plot(freq, im, color="#c45c26", label=r"$I_-$", alpha=0.75, ls="--", lw=1.2)
        ax_s.plot(
            freq,
            q_spec,
            color="#0f766e",
            label=r"$Q = I_+ - I_-$",
            alpha=0.8,
            lw=1.2,
        )
        ax_s.set_xlabel("Frequency offset")
        ax_s.set_ylabel("Amplitude")
        ax_s.set_title(
            rf"Test #{gi} ({src})  ·  "
            rf"$p_0={p0_v:.3f}$  ·  P={power:.1e}  ·  steps={steps}"
        )
        _apply_axes_style(ax_s)
        ax_s.legend(loc="upper right", fontsize=8, frameon=False)

        x_bars = np.arange(2)
        ax_pq.bar(
            x_bars - 0.2,
            [tp, tq],
            width=0.38,
            color="#94a3b8",
            edgecolor="white",
            label="True",
        )
        ax_pq.bar(
            x_bars + 0.2,
            [pp, pq],
            width=0.38,
            color="#2f6fed",
            edgecolor="white",
            label="Pred",
        )
        ax_pq.set_xticks(x_bars)
        ax_pq.set_xticklabels([r"$P_{\mathrm{total}}$", r"$Q_{\mathrm{total}}$"])
        ax_pq.set_title(
            rf"$P$ RPE ${rpe_p:.2f}\%$  ·  $Q$ RPE ${rpe_q:.2f}\%$"
        )
        _apply_axes_style(ax_pq)
        ax_pq.legend(loc="upper right", fontsize=8, frameon=False)

        fig_e.tight_layout()
        path_e = examples_dir / f"test_example_{gi:05d}.png"
        fig_e.savefig(path_e, dpi=150, facecolor="white")
        plt.close(fig_e)
        saved.append(path_e)

        if axes is not None and panel_i < summary_rows:
            ax_sum_s = axes[panel_i, 0]
            ax_sum_pq = axes[panel_i, 1]
            ax_sum_s.plot(freq, ps, color="#1f2933", label=r"$P_s$", lw=1.6)
            ax_sum_s.plot(
                freq,
                q_spec,
                color="#0f766e",
                label=r"$Q$",
                alpha=0.8,
                ls="--",
                lw=1.1,
            )
            ax_sum_s.set_title(rf"#{gi} ({src})  ·  $p_0={p0_v:.3f}$")
            _apply_axes_style(ax_sum_s)
            if panel_i == 0:
                ax_sum_s.legend(fontsize=8, frameon=False)

            ax_sum_pq.bar(
                x_bars - 0.2,
                [tp, tq],
                width=0.38,
                color="#94a3b8",
                edgecolor="white",
                label="True",
            )
            ax_sum_pq.bar(
                x_bars + 0.2,
                [pp, pq],
                width=0.38,
                color="#2f6fed",
                edgecolor="white",
                label="Pred",
            )
            ax_sum_pq.set_xticks(x_bars)
            ax_sum_pq.set_xticklabels([r"$P$", r"$Q$"])
            ax_sum_pq.set_title(rf"$P$ ${rpe_p:.2f}\%$  ·  $Q$ ${rpe_q:.2f}\%$")
            _apply_axes_style(ax_sum_pq)
            if panel_i == 0:
                ax_sum_pq.legend(fontsize=8, frameon=False)

    if fig is not None:
        fig.tight_layout()
        path_sum = plots_dir / "examples_summary.png"
        fig.savefig(path_sum, dpi=160, facecolor="white")
        plt.close(fig)
        saved.append(path_sum)

    return saved


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
        "--n-examples",
        type=int,
        default=N_EXAMPLE_PLOTS,
        help="Number of per-sample example plots to save",
    )
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
    history_path = args.out_dir / "history.json"
    if args.test_only:
        print(f"Loading checkpoint {checkpoint_path} ...", flush=True)
        model, ckpt = load_checkpoint(checkpoint_path)
        stats = {**stats, **ckpt["stats"]}
        best_val = float(ckpt.get("best_val_loss", float("nan")))
        loaded = load_history_json(history_path)
        history = loaded or {
            "train_loss": [],
            "val_loss": [],
            "val_p_rpe": [],
            "val_q_rpe": [],
        }
        if loaded:
            print(f"Loaded training history from {history_path}", flush=True)
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
        save_history_json(history, history_path)

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

    print_range_stats_table(
        metrics["range_stats_by_P"], title="Performance by |P_total| Range"
    )
    if metrics["range_stats_by_p0"]:
        print_range_stats_table(
            metrics["range_stats_by_p0"], title="Performance by |p0| Range"
        )

    save_range_stats_csv(
        metrics["range_stats_by_P"], args.out_dir / "range_stats_by_P.csv"
    )
    if metrics["range_stats_by_p0"]:
        save_range_stats_csv(
            metrics["range_stats_by_p0"], args.out_dir / "range_stats_by_p0.csv"
        )

    json_metrics = {
        k: v for k, v in metrics.items() if isinstance(v, (float, int, str, list))
    }
    json_metrics["best_val_loss"] = float(best_val)
    json_metrics["checkpoint"] = str(checkpoint_path)
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=2)

    print("Generating plots...", flush=True)
    save_plots(history, metrics, args.out_dir, best_val_loss=float(best_val))
    save_example_signal_plots(
        arrays,
        stats,
        metrics,
        args.out_dir,
        n_examples=args.n_examples,
    )
    print(f"Done. Results in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
