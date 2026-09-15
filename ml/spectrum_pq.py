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
DEFAULT_SPECTRA_PATH = REPO_ROOT / "Data_Creation" / "dae_voigt_burn_spectra" / "spectra.npz"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "spectrum_pq_results_v3"

NUM_BINS = 500
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 8
PATIENCE = 200
MIN_DELTA = 1e-6
LR_FACTOR = 0.5
LR_MIN = 1e-8
MAX_GRAD_NORM = 1.0
HIDDEN_DIMS = (256, 256)
REL_LOSS_EPS = 1e-4
VAL_FRAC = 0.15
TEST_FRAC = 0.15

POL_ABS_BANDS: tuple[tuple[float, float], ...] = tuple(
    (lo / 100.0, (lo + 5) / 100.0) for lo in range(5, 95, 5)
)
RPE_ABS_EPS = 1e-10


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
                    "" if row[k] is None or (isinstance(row[k], float) and not np.isfinite(row[k]))
                    else str(row[k])
                    for k in keys
                )
                + "\n"
            )


class SpectrumPQModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, int(h)),
                    nn.ReLU(),
                    nn.BatchNorm1d(int(h)),
                ]
            )
            prev = int(h)
        self.trunk = nn.Sequential(*layers)
        self.head_p = nn.Linear(prev, 1)
        self.head_q = nn.Linear(prev, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.head_p(hidden).squeeze(-1), self.head_q(hidden).squeeze(-1)


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


def resolve_spectra_path(spectra: Path | None) -> Path:
    if spectra is not None:
        path = Path(spectra)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Spectra NPZ not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Spectra NPZ not found: {path}")
    return path


def load_spectrum_pq_npz(path: Path) -> dict[str, np.ndarray]:
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

    ps = spectra[:, 0, :] + spectra[:, 1, :]
    features = np.concatenate(
        [
            ps.reshape(n, num_bins),
            applied_power.reshape(n, 1),
            n_steps.reshape(n, 1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    out = {
        "spectra": spectra.astype(np.float32, copy=False),
        "features": features,
        "P_total": p_total,
        "Q_total": q_total,
        "applied_power": applied_power,
        "n_steps": n_steps,
        "num_bins": np.asarray(num_bins, dtype=np.int32),
    }
    out.update(optional)
    return out


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def _apply_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32, copy=False)


def prepare_datasets(
    arrays: dict[str, np.ndarray],
    *,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
) -> tuple[data.TensorDataset, data.TensorDataset, data.TensorDataset, dict[str, Any]]:
    spectra = np.asarray(arrays["spectra"], dtype=np.float32)
    ps = spectra[:, 0, :] + spectra[:, 1, :]
    ps += np.random.normal(0, 1e-4, size=ps.shape)

    applied_power = np.asarray(arrays["applied_power"], dtype=np.float32).reshape(-1, 1)
    n_steps = np.asarray(arrays["n_steps"], dtype=np.float32).reshape(-1, 1)

    y_p = np.asarray(arrays["P_total"], dtype=np.float32).reshape(-1)
    y_q = np.asarray(arrays["Q_total"], dtype=np.float32).reshape(-1)
    
    n = int(ps.shape[0])
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * float(test_frac))))
    n_val = max(1, int(round(n * float(val_frac))))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    ps_train = ps[train_idx]
    ps_mean = float(ps_train.mean())
    ps_std = float(ps_train.std())
    ps_std = ps_std if ps_std > 1e-8 else 1.0

    pwr_train = applied_power[train_idx]
    pwr_mean = float(pwr_train.mean())
    pwr_std = float(pwr_train.std())
    pwr_std = pwr_std if pwr_std > 1e-8 else 1.0

    steps_train = n_steps[train_idx]
    steps_mean = float(steps_train.mean())
    steps_std = float(steps_train.std())
    steps_std = steps_std if steps_std > 1e-8 else 1.0

    p_m, p_s = _standardize_fit(y_p[train_idx].reshape(-1, 1))
    q_m, q_s = _standardize_fit(y_q[train_idx].reshape(-1, 1))
    p_mean_s, p_std_s = float(p_m[0]), float(p_s[0])
    q_mean_s, q_std_s = float(q_m[0]), float(q_s[0])

    def _pack(indices: np.ndarray) -> data.TensorDataset:
        ps_norm = (ps[indices] - ps_mean) / ps_std
        pwr_norm = (applied_power[indices] - pwr_mean) / pwr_std
        steps_norm = (n_steps[indices] - steps_mean) / steps_std
        
        features_combined = np.concatenate([ps_norm, pwr_norm, steps_norm], axis=1).astype(np.float32)
        yp = ((y_p[indices] - p_mean_s) / p_std_s).astype(np.float32)
        yq = ((y_q[indices] - q_mean_s) / q_std_s).astype(np.float32)
        
        return data.TensorDataset(
            torch.from_numpy(features_combined),
            torch.from_numpy(yp),
            torch.from_numpy(yq),
        )

    stats = {
        "ps_mean": ps_mean, "ps_std": ps_std,
        "pwr_mean": pwr_mean, "pwr_std": pwr_std,
        "steps_mean": steps_mean, "steps_std": steps_std,
        "P_mean": p_mean_s, "P_std": p_std_s,
        "Q_mean": q_mean_s, "Q_std": q_std_s,
        "input_dim": int(ps.shape[1] + 2),
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "p0_test": arrays.get("p0", y_p)[test_idx],
    }
    return _pack(train_idx), _pack(val_idx), _pack(test_idx), stats


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _stats_for_checkpoint(stats: dict[str, Any]) -> dict[str, Any]:
    skip = {"train_idx", "val_idx", "test_idx", "p0_test"}
    return {k: v for k, v in stats.items() if k not in skip}


def save_best_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    stats: dict[str, Any],
    hidden_dims: tuple[int, ...],
    best_val_loss: float,
    best_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": clone_state_dict(model),
            "stats": _stats_for_checkpoint(stats),
            "hidden_dims": tuple(int(h) for h in hidden_dims),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "input_dim": int(stats["input_dim"]),
        },
        path,
    )


def load_best_checkpoint(
    path: Path,
    *,
    device: torch.device = DEVICE,
) -> tuple[SpectrumPQModel, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hidden_dims = tuple(int(h) for h in ckpt["hidden_dims"])
    input_dim = int(ckpt.get("input_dim", ckpt["stats"]["input_dim"]))
    model = SpectrumPQModel(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def train_model(
    train_dataset: data.TensorDataset,
    val_dataset: data.TensorDataset,
    stats: dict[str, Any],
    *,
    hidden_dims: tuple[int, ...],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    checkpoint_path: Path | None = None,
    device: torch.device = DEVICE,
) -> tuple[SpectrumPQModel, float, dict[str, list[float]]]:
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    p_mean = float(stats["P_mean"])
    p_std = float(stats["P_std"])
    q_mean = float(stats["Q_mean"])
    q_std = float(stats["Q_std"])

    model = SpectrumPQModel(
        input_dim=int(train_dataset.tensors[0].shape[1]),
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=WEIGHT_DECAY,
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
        n_val = 0
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
                val_batches += 1
                pred_p_n = pred_p * p_std + p_mean
                true_p_n = y_p * p_std + p_mean
                pred_q_n = pred_q * q_std + q_mean
                true_q_n = y_q * q_std + q_mean
                rpe_p_sum += float(
                    (
                        torch.abs(pred_p_n - true_p_n)
                        / (torch.abs(true_p_n) + REL_LOSS_EPS)
                    )
                    .sum()
                    .item()
                )
                rpe_q_sum += float(
                    (
                        torch.abs(pred_q_n - true_q_n)
                        / (torch.abs(true_q_n) + REL_LOSS_EPS)
                    )
                    .sum()
                    .item()
                )
                n_val += int(y_p.numel())
        avg_val = val_sum / max(val_batches, 1)
        rpe_p = (rpe_p_sum / max(n_val, 1)) * 100.0
        rpe_q = (rpe_q_sum / max(n_val, 1)) * 100.0
        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_p_rpe"].append(rpe_p)
        history["val_q_rpe"].append(rpe_q)

        if best_state is None or avg_val < best_val - MIN_DELTA:
            best_val = avg_val
            best_epoch = int(epoch)
            best_state = clone_state_dict(model)
            stale = 0
            if checkpoint_path is not None:
                save_best_checkpoint(
                    checkpoint_path,
                    model=model,
                    stats=stats,
                    hidden_dims=hidden_dims,
                    best_val_loss=best_val,
                    best_epoch=best_epoch,
                )
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == int(num_epochs) - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch {epoch:04d} | train {avg_train:.6f} | val {avg_val:.6f} | "
                f"val RPE% P={rpe_p:.3f} Q={rpe_q:.3f} | lr {lr:.2e}",
                flush=True,
            )

        if stale >= int(patience):
            print(
                f"Early stop at epoch {epoch} (best val {best_val:.6f})",
                flush=True,
            )
            break

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        model, ckpt = load_best_checkpoint(checkpoint_path, device=device)
        best_val = float(ckpt.get("best_val_loss", best_val))
    elif best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, history


@torch.no_grad()
def predict_denormalized(
    model: SpectrumPQModel,
    dataset: data.TensorDataset,
    stats: dict[str, Any],
    *,
    batch_size: int = 1024,
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
    model: SpectrumPQModel,
    dataset: data.TensorDataset,
    stats: dict[str, Any],
    *,
    batch_size: int = 1024,
    device: torch.device = DEVICE,
) -> dict[str, float | np.ndarray | list]:
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
        rpe_s = _summary_stats(rpe)
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


def save_plots(
    history: dict[str, list[float]] | None,
    metrics: dict[str, float | np.ndarray | list],
    plots_dir: Path,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    if history is not None and history.get("train_loss"):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(history["train_loss"], label="train")
        ax.plot(history["val_loss"], label="val")
        ax.set_xlabel("epoch")
        ax.set_ylabel("rel loss P + Q")
        ax.set_title("Spectrum → P/Q training loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = plots_dir / "loss_curves.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

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
    path = plots_dir / "pred_vs_true.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    saved.append(path)

    range_rows = list(metrics.get("range_stats_by_P") or [])
    nonempty = [r for r in range_rows if int(r["n"]) > 0]
    if nonempty:
        labels = [r["range"] for r in nonempty]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)
        axes[0].bar(x - 0.18, [r["P_rpe_median"] for r in nonempty], width=0.35, label="median")
        axes[0].bar(x + 0.18, [r["P_rpe_mean"] for r in nonempty], width=0.35, label="mean")
        axes[0].set_ylabel("P RPE (%)")
        axes[0].set_title("P RPE by |P_total| band")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[1].bar(x - 0.18, [r["Q_rpe_median"] for r in nonempty], width=0.35, label="median")
        axes[1].bar(x + 0.18, [r["Q_rpe_mean"] for r in nonempty], width=0.35, label="mean")
        axes[1].set_ylabel("Q RPE (%)")
        axes[1].set_title("Q RPE by |P_total| band")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "rpe_by_polarization_range.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

        fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)
        axes[0].bar(x, [r["P_residual_mean"] for r in nonempty], width=0.6)
        axes[0].axhline(0.0, color="k", lw=0.8)
        axes[0].set_ylabel("mean residual P")
        axes[0].set_title("P residuals by |P_total| band")
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[1].bar(x, [r["Q_residual_mean"] for r in nonempty], width=0.6)
        axes[1].axhline(0.0, color="k", lw=0.8)
        axes[1].set_ylabel("mean residual Q")
        axes[1].set_title("Q residuals by |P_total| band")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = plots_dir / "residuals_by_polarization_range.png"
        fig.savefig(path, dpi=140)
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
    p_picks = p_order[np.round(np.linspace(0, n - 1, max(1, k - err_picks.size))).astype(int)]
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
    metrics: dict[str, float | np.ndarray | list],
    plots_dir: Path,
    *,
    n_examples: int = 6,
    seed: int = SEED,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = plots_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

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
    has_p0 = "p0" in arrays
    p0_arr = (
        np.asarray(arrays["p0"], dtype=np.float64).reshape(-1)
        if has_p0
        else None
    )
    center = (
        np.asarray(arrays["center_bin"], dtype=np.int64).reshape(-1)
        if "center_bin" in arrays
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
        bin_v = int(center[gi]) if center is not None else -1
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
        ax_s.plot(freq, ps, color="black", label="Ps (I+ + I-)", lw=1.5)
        ax_s.plot(freq, ip, color="blue", label="I+", alpha=0.6, ls="--")
        ax_s.plot(freq, im, color="red", label="I-", alpha=0.6, ls="--")
        ax_s.plot(freq, q_spec, color="green", label="Q spec (I+ - I-)", alpha=0.6)
        
        ax_s.set_xlabel("Frequency Bin / Offset")
        ax_s.set_ylabel("Signal Amplitude")
        ax_s.set_title(f"Test Idx {gi} ({src}): p0={p0_v:.3f}, Power={power:.1e}, Steps={steps}")
        ax_s.legend(loc="upper right", fontsize=8)
        ax_s.grid(True, alpha=0.3)

        x_bars = np.arange(2)
        ax_pq.bar(x_bars - 0.2, [tp, tq], width=0.4, color="gray", label="True")
        ax_pq.bar(x_bars + 0.2, [pp, pq], width=0.4, color="orange", label="Pred")
        ax_pq.set_xticks(x_bars)
        ax_pq.set_xticklabels(["P_total", "Q_total"])
        ax_pq.set_title(f"P RPE: {rpe_p:.1f}% | Q RPE: {rpe_q:.1f}%")
        ax_pq.legend(loc="upper right", fontsize=8)
        ax_pq.grid(True, axis="y", alpha=0.3)

        fig_e.tight_layout()
        path_e = examples_dir / f"test_example_{gi:05d}.png"
        fig_e.savefig(path_e, dpi=120)
        plt.close(fig_e)
        saved.append(path_e)

        if axes is not None and panel_i < summary_rows:
            ax_sum_s = axes[panel_i, 0]
            ax_sum_pq = axes[panel_i, 1]
            
            ax_sum_s.plot(freq, ps, color="black", label="Ps", lw=1.5)
            ax_sum_s.plot(freq, q_spec, color="green", label="Q_spec", alpha=0.7, ls="--")
            ax_sum_s.set_title(f"Idx {gi} ({src}): p0={p0_v:.3f}")
            ax_sum_s.grid(True, alpha=0.3)
            if panel_i == 0:
                ax_sum_s.legend(fontsize=8)

            ax_sum_pq.bar(x_bars - 0.2, [tp, tq], width=0.4, color="gray", label="True")
            ax_sum_pq.bar(x_bars + 0.2, [pp, pq], width=0.4, color="orange", label="Pred")
            ax_sum_pq.set_xticks(x_bars)
            ax_sum_pq.set_xticklabels(["P", "Q"])
            ax_sum_pq.set_title(f"P RPE: {rpe_p:.1f}% | Q RPE: {rpe_q:.1f}%")
            ax_sum_pq.grid(True, axis="y", alpha=0.3)
            if panel_i == 0:
                ax_sum_pq.legend(fontsize=8)

    if fig is not None:
        fig.tight_layout()
        path_sum = plots_dir / "examples_summary.png"
        fig.savefig(path_sum, dpi=140)
        plt.close(fig)
        saved.append(path_sum)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Spectrum P/Q Model.")
    parser.add_argument("--spectra", type=Path, default=DEFAULT_SPECTRA_PATH, help="Path to spectra.npz")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Best-model checkpoint path (default: <out-dir>/spectrum_pq_best.pth)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (args.out_dir / "spectrum_pq_best.pth")

    spectra_path = args.spectra or DEFAULT_SPECTRA_PATH
    print(f"Loading spectra from {spectra_path} ...", flush=True)
    arrays = load_spectrum_pq_npz(spectra_path)

    print("Preparing datasets...", flush=True)
    train_ds, val_ds, test_ds, stats = prepare_datasets(arrays)

    print(f"Train size: {stats['n_train']}, Val size: {stats['n_val']}, Test size: {stats['n_test']}")
    
    print("Training model...", flush=True)
    model, best_val, history = train_model(
        train_ds,
        val_ds,
        stats,
        hidden_dims=HIDDEN_DIMS,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )

    print("Evaluating model...", flush=True)
    metrics = evaluate_model(model, test_ds, stats)

    print_range_stats_table(metrics["range_stats_by_P"], title="Performance by |P_total| Range")
    if metrics["range_stats_by_p0"]:
        print_range_stats_table(metrics["range_stats_by_p0"], title="Performance by |p0| Range")

    print("Saving metrics...", flush=True)
    save_range_stats_csv(metrics["range_stats_by_P"], args.out_dir / "range_stats_by_P.csv")
    if metrics["range_stats_by_p0"]:
        save_range_stats_csv(metrics["range_stats_by_p0"], args.out_dir / "range_stats_by_p0.csv")

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json_metrics = {k: v for k, v in metrics.items() if isinstance(v, (float, int, str, list))}
        json_metrics["best_val_loss"] = float(best_val)
        json_metrics["checkpoint"] = str(checkpoint_path)
        json.dump(json_metrics, f, indent=2)

    print("Generating plots...", flush=True)
    save_plots(history, metrics, args.out_dir)
    save_example_signal_plots(arrays, stats, metrics, args.out_dir)

    print(f"All done! Results and plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()