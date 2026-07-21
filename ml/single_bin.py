"""
Train one per-bin model: (ps, …) -> (iplus, iminus) at that spectral bin.

Expects a per-bin training NPZ from the rate_eqs_test organize/combine pipeline:
  combined_train/train_bin_XXXX.npz
  ssrf_train/ssrf_train_bin_XXXX.npz
  afp_train/afp_train_bin_XXXX.npz

Required arrays: ps, iplus, iminus, p0
Optional: amp, is_mirror, source, center_bin / burn_bin
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
FEATURE_SET = "ps_p0"
NUM_BINS = 500
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

DEFAULT_DATA_DIR = Path("physics/lineshape/rate_eqs_test/combined_train_all")
DEFAULT_OUTPUT_DIR = Path("TensorStudies/single_bin_results_v2")


def to_column(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).float().reshape(-1, 1)


def to_matrix(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).float()


class BinModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2, bias=True),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, 0], out[:, 1]


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
        iplus = np.asarray(data["iplus"], dtype=np.float32)
        iminus = np.asarray(data["iminus"], dtype=np.float32)
        p0 = np.asarray(data["p0"], dtype=np.float32)
        out: Dict[str, np.ndarray] = {
            "ps": ps,
            "iplus": iplus,
            "iminus": iminus,
            "p0": p0,
            "amp": (
                np.asarray(data["amp"], dtype=np.float32)
                if "amp" in data.files
                else np.abs(ps)
            ),
        }
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


def build_features(
    arrays: Dict[str, np.ndarray],
    feature_set: str,
) -> Tuple[np.ndarray, List[str]]:
    """Build model input matrix and ordered feature names."""
    name = str(feature_set).strip().lower()
    # Legacy alias from the pickle-lookup era.
    if name in ("burn_context", "ps_p0"):
        cols = [arrays["ps"], arrays["p0"]]
        names = ["ps", "p0"]
    elif name in ("ps", "ps_only"):
        cols = [arrays["ps"]]
        names = ["ps"]
    elif name in ("amp_p0",):
        cols = [arrays["amp"], arrays["p0"]]
        names = ["amp", "p0"]
    elif name in ("ps_p0_source", "full"):
        if "source" not in arrays:
            raise KeyError(
                f"feature_set={feature_set!r} needs 'source' "
                "(use combined_train/train_bin_*.npz)"
            )
        cols = [arrays["ps"], arrays["p0"], arrays["source"]]
        names = ["ps", "p0", "source"]
        if "is_mirror" in arrays:
            cols.append(arrays["is_mirror"])
            names.append("is_mirror")
    else:
        raise ValueError(
            f"Unknown feature_set={feature_set!r}; "
            "expected one of: ps_p0, burn_context, ps, amp_p0, ps_p0_source, full"
        )
    features = np.column_stack(cols).astype(np.float32, copy=False)
    return features, names


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
    feature_set: str,
    feature_clip_z: float = 0.0,
) -> Dict[str, Any]:
    raw = load_bin_npz(data_path)
    features, feature_names = build_features(raw, feature_set)
    features = clip_features_z(features, feature_clip_z)

    iplus = raw["iplus"]
    iminus = raw["iminus"]
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
        "iplus_train": to_column(iplus[train_mask]),
        "iminus_train": to_column(iminus[train_mask]),
        "x_holdout": to_matrix(features[holdout_mask]),
        "iplus_holdout": to_column(iplus[holdout_mask]),
        "iminus_holdout": to_column(iminus[holdout_mask]),
        "feature_names": feature_names,
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
        "--feature-set",
        type=str,
        default=FEATURE_SET,
        help="Feature set: ps_p0|burn_context|ps|amp_p0|ps_p0_source|full",
    )
    parser.add_argument(
        "--feature-clip-z",
        type=float,
        default=0.0,
        help="Optional |z|-clip on features before split (0 disables)",
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
    iplus_train = arrays["iplus_train"]
    iminus_train = arrays["iminus_train"]
    x_holdout = arrays["x_holdout"]
    iplus_holdout = arrays["iplus_holdout"]
    iminus_holdout = arrays["iminus_holdout"]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-12)

    iplus_mean = iplus_train.mean()
    iplus_std = iplus_train.std().clamp_min(1e-12)

    iminus_mean = iminus_train.mean()
    iminus_std = iminus_train.std().clamp_min(1e-12)

    x_train_norm = (x_train - x_mean) / x_std
    iplus_train_norm = (iplus_train - iplus_mean) / iplus_std
    iminus_train_norm = (iminus_train - iminus_mean) / iminus_std

    x_holdout_norm = (x_holdout - x_mean) / x_std
    iplus_holdout_norm = (iplus_holdout - iplus_mean) / iplus_std
    iminus_holdout_norm = (iminus_holdout - iminus_mean) / iminus_std

    split_source = data.TensorDataset(
        x_holdout_norm, iplus_holdout_norm, iminus_holdout_norm
    )
    val_count = int(round(len(split_source) * validation_fraction))
    val_count = max(1, min(val_count, len(split_source) - 1))
    test_count = len(split_source) - val_count
    val_dataset, test_dataset = data.random_split(
        split_source,
        [val_count, test_count],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_dataset = data.TensorDataset(
        x_train_norm,
        iplus_train_norm,
        iminus_train_norm,
    )

    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    val_bin_dataset = data.TensorDataset(
        x_holdout_norm[val_indices],
        iplus_holdout_norm[val_indices],
        iminus_holdout_norm[val_indices],
    )
    test_bin_dataset = data.TensorDataset(
        x_holdout_norm[test_indices],
        iplus_holdout_norm[test_indices],
        iminus_holdout_norm[test_indices],
    )

    # Ps_* are the first feature column (ps or amp) for combined-model tooling.
    stats = {
        "x_mean": x_mean.detach().cpu(),
        "x_std": x_std.detach().cpu(),
        "ps_mean": x_mean[0, 0].detach().cpu(),
        "ps_std": x_std[0, 0].detach().cpu(),
        "iplus_mean": iplus_mean.detach().cpu(),
        "iplus_std": iplus_std.detach().cpu(),
        "iminus_mean": iminus_mean.detach().cpu(),
        "iminus_std": iminus_std.detach().cpu(),
    }
    return train_dataset, val_bin_dataset, test_bin_dataset, stats


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_model(
    train_dataset: data.TensorDataset,
    val_dataset: data.TensorDataset,
    args: argparse.Namespace,
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
        for x_batch, y_iplus, y_iminus in train_loader:
            x_batch = x_batch.to(device)
            y_iplus = y_iplus.squeeze(-1).to(device)
            y_iminus = y_iminus.squeeze(-1).to(device)

            pred_iplus, pred_iminus = model(x_batch)
            loss = loss_fn(pred_iplus, y_iplus) + loss_fn(pred_iminus, y_iminus)

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
            for x_val, y_iplus_val, y_iminus_val in val_loader:
                x_val = x_val.to(device)
                y_iplus_val = y_iplus_val.squeeze(-1).to(device)
                y_iminus_val = y_iminus_val.squeeze(-1).to(device)

                pred_iplus_val, pred_iminus_val = model(x_val)
                val_loss = loss_fn(pred_iplus_val, y_iplus_val) + loss_fn(
                    pred_iminus_val, y_iminus_val
                )
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
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    test_loader = data.DataLoader(test_dataset, batch_size=1024, shuffle=False)
    loss_fn = nn.L1Loss()

    pred_iplus_batches = []
    pred_iminus_batches = []
    true_iplus_batches = []
    true_iminus_batches = []
    test_loss_sum = 0.0
    test_batches = 0

    model.eval()
    with torch.no_grad():
        for x_test, y_iplus, y_iminus in test_loader:
            x_test = x_test.to(device)
            y_iplus = y_iplus.squeeze(-1).to(device)
            y_iminus = y_iminus.squeeze(-1).to(device)

            pred_iplus, pred_iminus = model(x_test)
            test_loss = loss_fn(pred_iplus, y_iplus) + loss_fn(pred_iminus, y_iminus)

            pred_iplus_batches.append(pred_iplus.cpu())
            pred_iminus_batches.append(pred_iminus.cpu())
            true_iplus_batches.append(y_iplus.cpu())
            true_iminus_batches.append(y_iminus.cpu())
            test_loss_sum += test_loss.item()
            test_batches += 1

    pred_iplus_norm = torch.cat(pred_iplus_batches).numpy()
    pred_iminus_norm = torch.cat(pred_iminus_batches).numpy()
    true_iplus_norm = torch.cat(true_iplus_batches).numpy()
    true_iminus_norm = torch.cat(true_iminus_batches).numpy()

    iplus_mean = float(stats["iplus_mean"].item())
    iplus_std = float(stats["iplus_std"].item())
    iminus_mean = float(stats["iminus_mean"].item())
    iminus_std = float(stats["iminus_std"].item())

    pred_iplus = pred_iplus_norm * iplus_std + iplus_mean
    pred_iminus = pred_iminus_norm * iminus_std + iminus_mean
    true_iplus = true_iplus_norm * iplus_std + iplus_mean
    true_iminus = true_iminus_norm * iminus_std + iminus_mean

    ss_res_iplus = np.sum((true_iplus - pred_iplus) ** 2)
    ss_tot_iplus = np.sum((true_iplus - np.mean(true_iplus)) ** 2)
    ss_res_iminus = np.sum((true_iminus - pred_iminus) ** 2)
    ss_tot_iminus = np.sum((true_iminus - np.mean(true_iminus)) ** 2)

    rpe_iplus = np.zeros_like(true_iplus)
    rpe_iminus = np.zeros_like(true_iminus)
    mask_iplus = np.abs(true_iplus) > 1e-10
    mask_iminus = np.abs(true_iminus) > 1e-10
    rpe_iplus[mask_iplus] = (
        np.abs(pred_iplus[mask_iplus] - true_iplus[mask_iplus])
        / np.abs(true_iplus[mask_iplus])
        * 100.0
    )
    rpe_iminus[mask_iminus] = (
        np.abs(pred_iminus[mask_iminus] - true_iminus[mask_iminus])
        / np.abs(true_iminus[mask_iminus])
        * 100.0
    )

    return {
        "test_l1_loss": test_loss_sum / max(test_batches, 1),
        "r2_iplus": 1.0 - float(ss_res_iplus / (ss_tot_iplus + 1e-12)),
        "r2_iminus": 1.0 - float(ss_res_iminus / (ss_tot_iminus + 1e-12)),
        "median_rpe_iplus": float(np.median(rpe_iplus[mask_iplus]))
        if np.any(mask_iplus)
        else 0.0,
        "median_rpe_iminus": float(np.median(rpe_iminus[mask_iminus]))
        if np.any(mask_iminus)
        else 0.0,
    }


def save_outputs(
    args: argparse.Namespace,
    model: nn.Module,
    best_val_loss: float,
    stats: Dict[str, torch.Tensor],
    metrics: Dict[str, float],
    feature_names: List[str],
    model_path: Path,
    metrics_path: Path,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "X_mean": stats["x_mean"].numpy(),
        "X_std": stats["x_std"].numpy(),
        "Ps_mean": float(stats["ps_mean"].item()),
        "Ps_std": float(stats["ps_std"].item()),
        "Iplus_mean": float(stats["iplus_mean"].item()),
        "Iplus_std": float(stats["iplus_std"].item()),
        "Iminus_mean": float(stats["iminus_mean"].item()),
        "Iminus_std": float(stats["iminus_std"].item()),
        "feature_names": list(feature_names),
        "input_dim": int(len(feature_names)),
        "use_hidden": True,
        "bin_idx": args.bin_idx,
        "hidden_dim": HIDDEN_DIM,
        "metrics": metrics,
        "args": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
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
        f"bin_idx={args.bin_idx}  data={data_path}  "
        f"feature_set={args.feature_set}  device={DEVICE}",
        flush=True,
    )

    arrays = load_bin_arrays(
        data_path=data_path,
        train_polarization_fraction=args.train_polarization_fraction,
        feature_set=args.feature_set,
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
        device=DEVICE,
    )
    metrics = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        stats=stats,
        args=args,
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
    )

    print(f"Saved model to {model_path}", flush=True)
    print(f"Saved metrics to {metrics_path}", flush=True)
    print(
        " | ".join(
            [
                f"best_val={best_val_loss:.6f}",
                f"test_l1={metrics['test_l1_loss']:.6f}",
                f"r2_iplus={metrics['r2_iplus']:.6f}",
                f"r2_iminus={metrics['r2_iminus']:.6f}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
