"""
Train one per-bin model from per-bin NPZ data.

Features: p0 (initial polarization) and ps at the observation bin.
Targets: CC-calibrated P and Q stored in the NPZ (from the data-generation pipeline).

Required NPZ arrays: ps, p0, iplus, iminus, P, Q
Optional: q, amp, source, center_bin, meta_json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

### Training parameters ###

TRAIN_POLARIZATION_FRACTION = 0.8
FEATURE_SET = "p0_ps"
TARGET_MODE = "pq"
NUM_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 5
PATIENCE = 50
MIN_DELTA = 1e-6
LR_FACTOR = 0.5
LR_MIN = 1e-8
MAX_GRAD_NORM = 1.0
HIDDEN_DIM = 256
HEAD_LAYOUT = "split"

DEFAULT_DATA_DIR = Path("combined_train_all")
DEFAULT_OUTPUT_DIR = Path("single_bin_models")


def to_column(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).float().reshape(-1, 1)


def to_matrix(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).float()


class BinModel(nn.Module):
    """Shared trunk with separate P and Q output heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        self.head_p = nn.Linear(hidden_dim, 1, bias=True)
        self.head_q = nn.Linear(hidden_dim, 1, bias=True)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        return self.head_p(hidden).squeeze(-1), self.head_q(hidden).squeeze(-1)


def _is_split_head_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return "head_p.weight" in state_dict


def _legacy_fused_to_split_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Upgrade checkpoints from a single Linear(hidden, 2) output layer."""
    if _is_split_head_state_dict(state_dict):
        return state_dict
    if "net.8.weight" not in state_dict:
        raise KeyError(
            "Unrecognized checkpoint layout: expected split heads (head_p/head_q) "
            "or legacy fused net.8 output"
        )
    out: Dict[str, torch.Tensor] = {}
    for layer_idx in (0, 2, 4, 6):
        out[f"trunk.{layer_idx}.weight"] = state_dict[f"net.{layer_idx}.weight"]
        out[f"trunk.{layer_idx}.bias"] = state_dict[f"net.{layer_idx}.bias"]
    fused_weight = state_dict["net.8.weight"]
    fused_bias = state_dict["net.8.bias"]
    out["head_p.weight"] = fused_weight[0:1].clone()
    out["head_p.bias"] = fused_bias[0:1].clone()
    out["head_q.weight"] = fused_weight[1:2].clone()
    out["head_q.bias"] = fused_bias[1:2].clone()
    return out


def load_bin_model_state_dict(
    model: BinModel,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Load split-head or legacy fused-head per-bin checkpoint weights."""
    model.load_state_dict(_legacy_fused_to_split_state_dict(state_dict), strict=True)


def resolve_bin_npz(data_dir: Path, bin_idx: int) -> Path:
    """Prefer combined train_bin_*, then ssRF / AFP organized names."""
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"train_bin_{int(bin_idx):04d}.npz",
        data_dir / f"ssrf_train_bin_{int(bin_idx):04d}.npz",
        data_dir / f"afp_train_bin_{int(bin_idx):04d}.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No training NPZ for bin {bin_idx} under {data_dir}; tried: "
        + ", ".join(p.name for p in candidates)
    )


def load_bin_npz(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        ps = np.asarray(data["ps"], dtype=np.float32)
        p0 = np.asarray(data["p0"], dtype=np.float32)
        if "P" not in data.files or "Q" not in data.files:
            raise KeyError(
                f"{path}: missing calibrated 'P' and/or 'Q'; "
                "regenerate train NPZs with combine_all_train.py or "
                "combine_spectrum_train_bins.py"
            )
        out: Dict[str, np.ndarray] = {
            "ps": ps,
            "p0": p0,
            "P": np.asarray(data["P"], dtype=np.float32),
            "Q": np.asarray(data["Q"], dtype=np.float32),
            "amp": (
                np.asarray(data["amp"], dtype=np.float32)
                if "amp" in data.files
                else np.abs(ps)
            ),
        }
        if "q" in data.files:
            out["q"] = np.asarray(data["q"], dtype=np.float32)
        if "iplus" in data.files:
            out["iplus"] = np.asarray(data["iplus"], dtype=np.float32)
        if "iminus" in data.files:
            out["iminus"] = np.asarray(data["iminus"], dtype=np.float32)
        if "is_mirror" in data.files:
            out["is_mirror"] = np.asarray(data["is_mirror"], dtype=np.float32)
        if "source" in data.files:
            out["source"] = np.asarray(data["source"], dtype=np.float32)
        if "center_bin" in data.files:
            out["center_bin"] = np.asarray(data["center_bin"], dtype=np.float32)
        elif "burn_bin" in data.files:
            out["center_bin"] = np.asarray(data["burn_bin"], dtype=np.float32)
        if "step" in data.files:
            out["step"] = np.asarray(data["step"], dtype=np.float32)
        if "meta_json" in data.files:
            out["meta_json"] = np.asarray(data["meta_json"])
    n = int(ps.size)
    for key, value in list(out.items()):
        if key == "meta_json":
            continue
        if int(np.asarray(value).size) != n:
            raise ValueError(
                f"{path}: field {key!r} length {np.asarray(value).size} != ps length {n}"
            )
    return out


def build_features(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], int]:
    p0 = arrays["p0"].reshape(-1, 1).astype(np.float32, copy=False)
    ps = arrays["ps"].reshape(-1, 1).astype(np.float32, copy=False)
    features = np.concatenate([p0, ps], axis=1).astype(np.float32, copy=False)
    return features, ["p0", "ps"], 1


def clip_features_z(features: np.ndarray, clip_z: float) -> np.ndarray:
    if clip_z is None or float(clip_z) <= 0.0:
        return features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    z = (features - mean) / std
    z = np.clip(z, -float(clip_z), float(clip_z))
    return (z * std + mean).astype(np.float32, copy=False)


def load_bin_arrays(
    data_path: Path,
    train_polarization_fraction: float,
    feature_clip_z: float = 0.0,
) -> Dict[str, Any]:
    raw = load_bin_npz(data_path)
    features, feature_names, ps_col = build_features(raw)
    features = clip_features_z(features, feature_clip_z)

    p_target = np.asarray(raw["P"], dtype=np.float32)
    q_target = np.asarray(raw["Q"], dtype=np.float32)
    polarizations = raw["p0"]

    unique_p = np.unique(polarizations)
    if unique_p.size < 2:
        raise ValueError(
            f"{data_path}: need >= 2 distinct p0 values for train/holdout split, "
            f"got {unique_p.size}"
        )

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(unique_p)
    n_train_p = int(round(unique_p.size * float(train_polarization_fraction)))
    n_train_p = max(1, min(n_train_p, unique_p.size - 1))
    train_p = set(shuffled[:n_train_p].tolist())
    holdout_p = set(shuffled[n_train_p:].tolist())

    train_mask = np.isin(polarizations, list(train_p))
    holdout_mask = np.isin(polarizations, list(holdout_p))
    if not np.any(train_mask) or not np.any(holdout_mask):
        raise RuntimeError(f"{data_path}: empty train or holdout after p0 split")

    return {
        "x_train": to_matrix(features[train_mask]),
        "P_train": to_column(p_target[train_mask]),
        "Q_train": to_column(q_target[train_mask]),
        "x_holdout": to_matrix(features[holdout_mask]),
        "P_holdout": to_column(p_target[holdout_mask]),
        "Q_holdout": to_column(q_target[holdout_mask]),
        "feature_names": feature_names,
        "ps_col": int(ps_col),
        "n_samples": int(features.shape[0]),
        "n_train": int(train_mask.sum()),
        "n_holdout": int(holdout_mask.sum()),
        "train_p0": sorted(train_p),
        "holdout_p0": sorted(holdout_p),
        "data_path": str(data_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single binning model for one bin index from per-bin NPZ."
    )
    parser.add_argument("--bin-idx", type=int, required=True, help="Target bin index")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with train_bin_XXXX.npz (or ssrf_/afp_train_bin_XXXX.npz)",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Optional explicit NPZ path (overrides --data-dir / --bin-idx lookup)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints and metrics",
    )
    parser.add_argument(
        "--feature-clip-z",
        type=float,
        default=0.0,
        help="Optional |z|-clip on ps before split (0 disables)",
    )
    parser.add_argument(
        "--train-polarization-fraction",
        type=float,
        default=TRAIN_POLARIZATION_FRACTION,
        help="Fraction of distinct p0 values used for training",
    )
    return parser.parse_args()


def build_bin_datasets(
    arrays: Dict[str, Any],
    validation_fraction: float,
) -> Tuple[data.TensorDataset, data.TensorDataset, data.TensorDataset, Dict[str, torch.Tensor]]:
    x_train = arrays["x_train"]
    x_holdout = arrays["x_holdout"]
    ps_col = int(arrays["ps_col"])

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-12)
    x_train_norm = (x_train - x_mean) / x_std
    x_holdout_norm = (x_holdout - x_mean) / x_std

    p_train = arrays["P_train"]
    q_train = arrays["Q_train"]
    p_holdout = arrays["P_holdout"]
    q_holdout = arrays["Q_holdout"]

    p_mean = p_train.mean()
    p_std = p_train.std().clamp_min(1e-12)
    q_mean = q_train.mean()
    q_std = q_train.std().clamp_min(1e-12)

    stats: Dict[str, torch.Tensor] = {
        "x_mean": x_mean.detach().cpu(),
        "x_std": x_std.detach().cpu(),
        "ps_mean": x_mean[0, ps_col].detach().cpu(),
        "ps_std": x_std[0, ps_col].detach().cpu(),
        "ps_col": ps_col,
        "P_mean": p_mean.detach().cpu(),
        "P_std": p_std.detach().cpu(),
        "Q_mean": q_mean.detach().cpu(),
        "Q_std": q_std.detach().cpu(),
    }

    p_train_norm = (p_train - p_mean) / p_std
    q_train_norm = (q_train - q_mean) / q_std
    p_holdout_norm = (p_holdout - p_mean) / p_std
    q_holdout_norm = (q_holdout - q_mean) / q_std

    train_dataset = data.TensorDataset(x_train_norm, p_train_norm, q_train_norm)
    split_source = data.TensorDataset(
        x_holdout_norm,
        p_holdout_norm,
        q_holdout_norm,
    )

    val_count = int(round(len(split_source) * validation_fraction))
    val_count = max(1, min(val_count, len(split_source) - 1))
    test_count = len(split_source) - val_count
    val_dataset, test_dataset = data.random_split(
        split_source,
        [val_count, test_count],
        generator=torch.Generator().manual_seed(SEED),
    )

    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    val_bin_dataset = data.TensorDataset(
        split_source.tensors[0][val_indices],
        split_source.tensors[1][val_indices],
        split_source.tensors[2][val_indices],
    )
    test_bin_dataset = data.TensorDataset(
        split_source.tensors[0][test_indices],
        split_source.tensors[1][test_indices],
        split_source.tensors[2][test_indices],
    )

    return train_dataset, val_bin_dataset, test_bin_dataset, stats


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_model(
    train_dataset: data.TensorDataset,
    val_dataset: data.TensorDataset,
    args: argparse.Namespace,
    stats: Dict[str, torch.Tensor],
    device: torch.device = DEVICE,
) -> Tuple[nn.Module, float]:
    train_loader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = BinModel(
        input_dim=train_dataset.tensors[0].shape[1],
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    best_val_loss = float("inf")
    best_model_state = None

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )
    loss_fn = nn.L1Loss()
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for x_batch, y_p, y_q in train_loader:
            x_batch = x_batch.to(device)
            y_p = y_p.squeeze(-1).to(device)
            y_q = y_q.squeeze(-1).to(device)
            pred_p, pred_q = model(x_batch)
            loss = loss_fn(pred_p, y_p) + loss_fn(pred_q, y_q)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_val, y_p_val, y_q_val in val_loader:
                x_val = x_val.to(device)
                y_p_val = y_p_val.squeeze(-1).to(device)
                y_q_val = y_q_val.squeeze(-1).to(device)
                pred_p_val, pred_q_val = model(x_val)
                val_loss = loss_fn(pred_p_val, y_p_val) + loss_fn(pred_q_val, y_q_val)
                val_loss_sum += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step(avg_val_loss)

        if best_model_state is None or avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            best_model_state = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 50 == 0 or epoch == NUM_EPOCHS - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Bin {args.bin_idx} | epoch {epoch:04d} | "
                f"train {avg_train_loss:.6f} | val {avg_val_loss:.6f} | "
                f"lr {current_lr:.2e}",
                flush=True,
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"Stopping early at epoch {epoch} (best val {best_val_loss:.6f})",
                flush=True,
            )
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_loss


def evaluate_model(
    model: nn.Module,
    test_dataset: data.TensorDataset,
    stats: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    test_loader = data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
    loss_fn = nn.L1Loss()

    p_mean = float(stats["P_mean"].item())
    p_std = float(stats["P_std"].item())
    q_mean = float(stats["Q_mean"].item())
    q_std = float(stats["Q_std"].item())

    test_loss_sum = 0.0
    test_batches = 0
    pred_p_batches: List[torch.Tensor] = []
    pred_q_batches: List[torch.Tensor] = []
    true_p_batches: List[torch.Tensor] = []
    true_q_batches: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for x_test, y_p, y_q in test_loader:
            x_test = x_test.to(device)
            y_p = y_p.squeeze(-1).to(device)
            y_q = y_q.squeeze(-1).to(device)
            pred_p, pred_q = model(x_test)
            test_loss = loss_fn(pred_p, y_p) + loss_fn(pred_q, y_q)
            pred_p_batches.append((pred_p * p_std + p_mean).cpu())
            pred_q_batches.append((pred_q * q_std + q_mean).cpu())
            true_p_batches.append((y_p * p_std + p_mean).cpu())
            true_q_batches.append((y_q * q_std + q_mean).cpu())
            test_loss_sum += test_loss.item()
            test_batches += 1

    pred_p = torch.cat(pred_p_batches).numpy()
    pred_q = torch.cat(pred_q_batches).numpy()
    true_p = torch.cat(true_p_batches).numpy()
    true_q = torch.cat(true_q_batches).numpy()

    ss_res_p = np.sum((true_p - pred_p) ** 2)
    ss_tot_p = np.sum((true_p - np.mean(true_p)) ** 2)
    ss_res_q = np.sum((true_q - pred_q) ** 2)
    ss_tot_q = np.sum((true_q - np.mean(true_q)) ** 2)
    rpe_p = np.zeros_like(true_p)
    rpe_q = np.zeros_like(true_q)
    mask_p = np.abs(true_p) > 1e-10
    mask_q = np.abs(true_q) > 1e-10
    rpe_p[mask_p] = np.abs(pred_p[mask_p] - true_p[mask_p]) / np.abs(true_p[mask_p]) * 100.0
    rpe_q[mask_q] = np.abs(pred_q[mask_q] - true_q[mask_q]) / np.abs(true_q[mask_q]) * 100.0

    return {
        "test_l1_loss": test_loss_sum / max(test_batches, 1),
        "l1_P": float(np.mean(np.abs(true_p - pred_p))),
        "l1_Q": float(np.mean(np.abs(true_q - pred_q))),
        "r2_P": 1.0 - float(ss_res_p / (ss_tot_p + 1e-12)),
        "r2_Q": 1.0 - float(ss_res_q / (ss_tot_q + 1e-12)),
        "median_rpe_P": float(np.median(rpe_p[mask_p])) if np.any(mask_p) else 0.0,
        "median_rpe_Q": float(np.median(rpe_q[mask_q])) if np.any(mask_q) else 0.0,
    }


def _validate_saved_stats(
    stats: Dict[str, torch.Tensor],
    arrays: Dict[str, Any],
    *,
    bin_idx: int,
) -> None:
    """Sanity-check normalization stats before writing a checkpoint."""
    x_train = arrays.get("x_train")
    if x_train is None:
        return
    x_mean = stats["x_mean"].reshape(-1)
    train_p0 = np.asarray(x_train[:, 0].numpy(), dtype=np.float64)
    if train_p0.size == 0:
        return
    expected_p0 = float(np.mean(train_p0))
    saved_p0 = float(x_mean[0].item())
    if abs(saved_p0 - expected_p0) > 1e-4:
        raise RuntimeError(
            f"bin {bin_idx}: X_mean[p0]={saved_p0:.6f} != train p0 mean "
            f"{expected_p0:.6f}; refusing to save a corrupt checkpoint"
        )


def save_outputs(
    args: argparse.Namespace,
    model: nn.Module,
    best_val_loss: float,
    stats: Dict[str, torch.Tensor],
    metrics: Dict[str, float],
    feature_names: List[str],
    model_path: Path,
    metrics_path: Path,
    *,
    arrays: Dict[str, Any] | None = None,
) -> None:
    _validate_saved_stats(stats, arrays or {}, bin_idx=int(args.bin_idx))
    x_mean = stats["x_mean"].numpy().reshape(-1)
    x_std = stats["x_std"].numpy().reshape(-1)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "X_mean": x_mean,
        "X_std": x_std,
        "Ps_mean": float(stats["ps_mean"].item()),
        "Ps_std": float(stats["ps_std"].item()),
        "target_mode": TARGET_MODE,
        "head_layout": HEAD_LAYOUT,
        "ps_col": int(stats["ps_col"]),
        "feature_names": list(feature_names),
        "input_dim": int(len(feature_names)),
        "use_hidden": True,
        "bin_idx": args.bin_idx,
        "hidden_dim": HIDDEN_DIM,
        "metrics": metrics,
        "P_mean": float(stats["P_mean"].item()),
        "P_std": float(stats["P_std"].item()),
        "Q_mean": float(stats["Q_mean"].item()),
        "Q_std": float(stats["Q_std"].item()),
        "targets_precalibrated": True,
        "pq_target_scope": "per_bin",
        "pq_post_correct": True,
        "args": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "feature_set": FEATURE_SET,
        },
    }
    torch.save(payload, model_path)

    metrics_payload = {
        "bin_idx": args.bin_idx,
        "model": str(model_path),
        "best_val_loss": best_val_loss,
        "feature_names": list(feature_names),
        **metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()

    model_path = args.output_dir / f"binning_model_bin_{args.bin_idx}.pth"
    metrics_path = args.output_dir / f"binning_model_bin_{args.bin_idx}_metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_path = (
        Path(args.data_file)
        if args.data_file is not None
        else resolve_bin_npz(args.data_dir, args.bin_idx)
    )
    print(
        f"bin_idx={args.bin_idx}  data={data_path}  features={FEATURE_SET}  "
        f"targets=P,Q (oracle p0 input)  device={DEVICE}",
        flush=True,
    )

    arrays = load_bin_arrays(
        data_path=data_path,
        train_polarization_fraction=args.train_polarization_fraction,
        feature_clip_z=args.feature_clip_z,
    )
    print(
        f"samples={arrays['n_samples']}  train={arrays['n_train']}  "
        f"holdout={arrays['n_holdout']}  features={arrays['feature_names']}",
        flush=True,
    )

    train_dataset, val_dataset, test_dataset, stats = build_bin_datasets(
        arrays=arrays,
        validation_fraction=0.5,
    )

    model, best_val_loss = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args,
        stats=stats,
        device=DEVICE,
    )
    metrics = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        stats=stats,
        device=DEVICE,
    )
    save_outputs(
        args=args,
        model=model,
        best_val_loss=best_val_loss,
        stats=stats,
        metrics=metrics,
        feature_names=arrays["feature_names"],
        model_path=model_path,
        metrics_path=metrics_path,
        arrays=arrays,
    )

    print(f"Saved model to {model_path}", flush=True)
    print(f"Saved metrics to {metrics_path}", flush=True)
    print(
        " | ".join(
            [
                f"best_val={best_val_loss:.6f}",
                f"test_l1={metrics['test_l1_loss']:.6f}",
                f"r2_P={metrics['r2_P']:.6f}",
                f"r2_Q={metrics['r2_Q']:.6f}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
