"""
Train a single binning model for one frequency bin.

This script is designed for Slurm job arrays so each task can train one bin
independently and save its own checkpoint/metrics.

Example:
  python TensorStudies/train_single_bin_model.py --bin-idx 202
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class LinearBinModel(nn.Module):
    """Map one Ps value to Iplus and Iminus for a single bin."""

    def __init__(self, use_hidden: bool = True, hidden_dim: int = 128):
        super().__init__()
        if use_hidden:
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2, bias=True),
            )
        else:
            self.net = nn.Linear(1, 2, bias=True)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single binning model for one bin index."
    )
    parser.add_argument("--bin-idx", type=int, required=True, help="Target bin index")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing binning_training.pkl and binning_testing.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("TensorStudies/single_bin_results"),
        help="Directory for checkpoints and metrics",
    )
    parser.add_argument("--num-bins", type=int, default=500)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device selection",
    )
    parser.add_argument(
        "--no-hidden",
        action="store_false",
        dest="use_hidden",
        help="Use a purely linear model",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Warm-start from an existing checkpoint if one exists",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Exit successfully if the output checkpoint already exists",
    )
    parser.set_defaults(use_hidden=True)
    return parser.parse_args()


def choose_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_arrays(data_dir: Path, num_bins: int) -> Dict[str, torch.Tensor]:
    train_path = data_dir / "binning_training.pkl"
    test_path = data_dir / "binning_testing.pkl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing testing data: {test_path}")

    print(f"Loading {train_path}", flush=True)
    df_train = pd.read_pickle(train_path)
    print(f"Loading {test_path}", flush=True)
    df_test = pd.read_pickle(test_path)

    ps_train = torch.from_numpy(
        np.stack(df_train["Ps"].values).reshape(-1, num_bins)
    ).float()
    iplus_train = torch.from_numpy(
        np.stack(df_train["Iplus"].values).reshape(-1, num_bins)
    ).float()
    iminus_train = torch.from_numpy(
        np.stack(df_train["Iminus"].values).reshape(-1, num_bins)
    ).float()

    ps_test = torch.from_numpy(
        np.stack(df_test["Ps"].values).reshape(-1, num_bins)
    ).float()
    iplus_test = torch.from_numpy(
        np.stack(df_test["Iplus"].values).reshape(-1, num_bins)
    ).float()
    iminus_test = torch.from_numpy(
        np.stack(df_test["Iminus"].values).reshape(-1, num_bins)
    ).float()

    return {
        "ps_train": ps_train,
        "iplus_train": iplus_train,
        "iminus_train": iminus_train,
        "ps_test": ps_test,
        "iplus_test": iplus_test,
        "iminus_test": iminus_test,
    }


def build_bin_datasets(
    arrays: Dict[str, torch.Tensor], target_bin: int, seed: int
) -> Tuple[data.TensorDataset, data.TensorDataset, data.TensorDataset, Dict[str, torch.Tensor]]:
    ps_train = arrays["ps_train"]
    iplus_train = arrays["iplus_train"]
    iminus_train = arrays["iminus_train"]
    ps_test = arrays["ps_test"]
    iplus_test = arrays["iplus_test"]
    iminus_test = arrays["iminus_test"]

    ps_mean = ps_train.mean(dim=0)
    ps_std = ps_train.std(dim=0)
    ps_std[ps_std == 0] = 1.0

    iplus_mean = iplus_train.mean(dim=0)
    iplus_std = iplus_train.std(dim=0)
    iplus_std[iplus_std == 0] = 1.0

    iminus_mean = iminus_train.mean(dim=0)
    iminus_std = iminus_train.std(dim=0)
    iminus_std[iminus_std == 0] = 1.0

    ps_train_norm = (ps_train - ps_mean) / ps_std
    iplus_train_norm = (iplus_train - iplus_mean) / iplus_std
    iminus_train_norm = (iminus_train - iminus_mean) / iminus_std

    ps_test_norm = (ps_test - ps_mean) / ps_std
    iplus_test_norm = (iplus_test - iplus_mean) / iplus_std
    iminus_test_norm = (iminus_test - iminus_mean) / iminus_std

    split_source = data.TensorDataset(ps_test_norm, iplus_test_norm, iminus_test_norm)
    split_generator = torch.Generator().manual_seed(seed)
    val_dataset, test_dataset = data.random_split(
        split_source,
        [len(split_source) // 2, len(split_source) - len(split_source) // 2],
        generator=split_generator,
    )

    train_dataset = data.TensorDataset(
        ps_train_norm[:, target_bin : target_bin + 1],
        iplus_train_norm[:, target_bin : target_bin + 1],
        iminus_train_norm[:, target_bin : target_bin + 1],
    )

    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    val_bin_dataset = data.TensorDataset(
        ps_test_norm[val_indices, target_bin : target_bin + 1],
        iplus_test_norm[val_indices, target_bin : target_bin + 1],
        iminus_test_norm[val_indices, target_bin : target_bin + 1],
    )
    test_bin_dataset = data.TensorDataset(
        ps_test_norm[test_indices, target_bin : target_bin + 1],
        iplus_test_norm[test_indices, target_bin : target_bin + 1],
        iminus_test_norm[test_indices, target_bin : target_bin + 1],
    )

    stats = {
        "ps_mean": ps_mean[target_bin].detach().cpu(),
        "ps_std": ps_std[target_bin].detach().cpu(),
        "iplus_mean": iplus_mean[target_bin].detach().cpu(),
        "iplus_std": iplus_std[target_bin].detach().cpu(),
        "iminus_mean": iminus_mean[target_bin].detach().cpu(),
        "iminus_std": iminus_std[target_bin].detach().cpu(),
    }
    return train_dataset, val_bin_dataset, test_bin_dataset, stats


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_model(
    args: argparse.Namespace,
    train_dataset: data.TensorDataset,
    val_dataset: data.TensorDataset,
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, float]:
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = LinearBinModel(
        use_hidden=args.use_hidden,
        hidden_dim=args.hidden_dim,
    ).to(device)

    best_val_loss = float("inf")
    best_model_state = None
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        best_model_state = clone_state_dict(model)
        print(f"Resumed from {checkpoint_path}", flush=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )
    loss_fn = nn.L1Loss()
    epochs_without_improvement = 0

    for epoch in range(args.num_epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
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

        if best_model_state is None or avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            best_model_state = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % args.log_every == 0 or epoch == args.num_epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Bin {args.bin_idx} | epoch {epoch:04d} | "
                f"train {avg_train_loss:.6f} | val {avg_val_loss:.6f} | "
                f"lr {current_lr:.2e}",
                flush=True,
            )

        if epochs_without_improvement >= args.patience:
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
        "median_rpe_iplus": float(np.median(rpe_iplus[mask_iplus])) if np.any(mask_iplus) else 0.0,
        "median_rpe_iminus": float(np.median(rpe_iminus[mask_iminus])) if np.any(mask_iminus) else 0.0,
    }


def save_outputs(
    args: argparse.Namespace,
    model: nn.Module,
    best_val_loss: float,
    stats: Dict[str, torch.Tensor],
    metrics: Dict[str, float],
    checkpoint_path: Path,
    metrics_path: Path,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "Ps_mean": float(stats["ps_mean"].item()),
        "Ps_std": float(stats["ps_std"].item()),
        "Iplus_mean": float(stats["iplus_mean"].item()),
        "Iplus_std": float(stats["iplus_std"].item()),
        "Iminus_mean": float(stats["iminus_mean"].item()),
        "Iminus_std": float(stats["iminus_std"].item()),
        "bin_idx": args.bin_idx,
        "use_hidden": args.use_hidden,
        "hidden_dim": args.hidden_dim,
        "metrics": metrics,
        "args": vars(args),
    }
    torch.save(payload, checkpoint_path)

    metrics_payload = {
        "bin_idx": args.bin_idx,
        "checkpoint": str(checkpoint_path),
        "best_val_loss": best_val_loss,
        **metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.bin_idx < 0 or args.bin_idx >= args.num_bins:
        raise ValueError(f"--bin-idx must be in [0, {args.num_bins - 1}]")

    checkpoint_path = args.output_dir / f"binning_model_bin_{args.bin_idx}.pth"
    metrics_path = args.output_dir / f"binning_model_bin_{args.bin_idx}_metrics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_if_exists and checkpoint_path.exists():
        print(f"Checkpoint already exists, skipping: {checkpoint_path}", flush=True)
        return

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)
    print(f"Training bin {args.bin_idx}", flush=True)

    arrays = load_arrays(args.data_dir, args.num_bins)
    train_dataset, val_dataset, test_dataset, stats = build_bin_datasets(
        arrays=arrays,
        target_bin=args.bin_idx,
        seed=args.seed,
    )

    print(
        f"Samples | train {len(train_dataset)} | val {len(val_dataset)} | test {len(test_dataset)}",
        flush=True,
    )

    model, best_val_loss = train_model(
        args=args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    metrics = evaluate_model(
        model=model,
        test_dataset=test_dataset,
        stats=stats,
        device=device,
    )
    save_outputs(
        args=args,
        model=model,
        best_val_loss=best_val_loss,
        stats=stats,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
    )

    print(f"Saved checkpoint to {checkpoint_path}", flush=True)
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
