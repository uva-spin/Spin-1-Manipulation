import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class LinearBinModel(nn.Module):

    def __init__(self, use_hidden: bool = False, hidden_dim: int = 1):
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, 0], out[:, 1]

class Combined500BinModel(nn.Module):
    def __init__(
        self,
        bin_models: List[nn.Module],
        stats: Dict[str, np.ndarray],
    ):
        super().__init__()
        self.num_bins = len(bin_models)
        self.bin_models = nn.ModuleList(bin_models)
        self.stats = stats

        self._Ps_mean = torch.from_numpy(stats["Ps_mean"]).float()
        self._Ps_std = torch.from_numpy(stats["Ps_std"]).float()
        self._Iplus_mean = torch.from_numpy(stats["Iplus_mean"]).float()
        self._Iplus_std = torch.from_numpy(stats["Iplus_std"]).float()
        self._Iminus_mean = torch.from_numpy(stats["Iminus_mean"]).float()
        self._Iminus_std = torch.from_numpy(stats["Iminus_std"]).float()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        batch_size = x.shape[0]

        Ps_mean = self._Ps_mean.to(device)
        Ps_std = self._Ps_std.to(device)
        Iplus_mean = self._Iplus_mean.to(device)
        Iplus_std = self._Iplus_std.to(device)
        Iminus_mean = self._Iminus_mean.to(device)
        Iminus_std = self._Iminus_std.to(device)

        pred_iplus = torch.zeros(batch_size, self.num_bins, device=device)
        pred_iminus = torch.zeros(batch_size, self.num_bins, device=device)

        for bin_idx in range(self.num_bins):
            x_bin = x[:, bin_idx : bin_idx + 1]
            x_norm = (x_bin - Ps_mean[bin_idx]) / (Ps_std[bin_idx] + 1e-12)

            iplus_norm, iminus_norm = self.bin_models[bin_idx](x_norm)

            pred_iplus[:, bin_idx] = (
                iplus_norm * Iplus_std[bin_idx] + Iplus_mean[bin_idx]
            )
            pred_iminus[:, bin_idx] = (
                iminus_norm * Iminus_std[bin_idx] + Iminus_mean[bin_idx]
            )

        return pred_iplus, pred_iminus


def train_single_bin(
    target_bin: int,
    Ps_train_norm: torch.Tensor,
    Iplus_train_norm: torch.Tensor,
    Iminus_train_norm: torch.Tensor,
    val_Ps_bin: torch.Tensor,
    val_Iplus_bin: torch.Tensor,
    val_Iminus_bin: torch.Tensor,
    Ps_mean: torch.Tensor,
    Ps_std: torch.Tensor,
    Iplus_mean: torch.Tensor,
    Iplus_std: torch.Tensor,
    Iminus_mean: torch.Tensor,
    Iminus_std: torch.Tensor,
    device: torch.device,
    args,
    resume_checkpoint: Optional[str] = None,
) -> Tuple[nn.Module, Dict]:
    """Train one bin model and return the model plus checkpoint metadata."""
    train_Ps_bin = Ps_train_norm[:, target_bin : target_bin + 1]
    train_Iplus_bin = Iplus_train_norm[:, target_bin : target_bin + 1]
    train_Iminus_bin = Iminus_train_norm[:, target_bin : target_bin + 1]

    bin_train_dataset = data.TensorDataset(
        train_Ps_bin, train_Iplus_bin, train_Iminus_bin
    )
    bin_val_dataset = data.TensorDataset(
        val_Ps_bin, val_Iplus_bin, val_Iminus_bin
    )

    bin_train_loader = data.DataLoader(
        bin_train_dataset, batch_size=args.batch_size, shuffle=True
    )
    bin_val_loader = data.DataLoader(
        bin_val_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = LinearBinModel(
        use_hidden=args.use_hidden, hidden_dim=args.hidden_dim
    ).to(device)
    if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Bin {target_bin}: resumed from {resume_checkpoint}")

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
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(args.num_epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for X_batch, y_iplus, y_iminus in bin_train_loader:
            X_batch = X_batch.to(device)
            y_iplus = y_iplus.squeeze().to(device)
            y_iminus = y_iminus.squeeze().to(device)

            pred_iplus, pred_iminus = model(X_batch)
            loss = loss_fn(pred_iplus, y_iplus) + loss_fn(pred_iminus, y_iminus)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for X_val, y_iplus_val, y_iminus_val in bin_val_loader:
                X_val = X_val.to(device)
                y_iplus_val = y_iplus_val.squeeze().to(device)
                y_iminus_val = y_iminus_val.squeeze().to(device)

                pred_iplus, pred_iminus = model(X_val)
                val_loss = loss_fn(pred_iplus, y_iplus_val) + loss_fn(
                    pred_iminus, y_iminus_val
                )
                val_loss_sum += val_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else float("inf")
        scheduler.step(avg_val_loss)
        if best_model_state is None or avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epoch % 100 == 0 or epoch == args.num_epochs - 1:
            print(
                f"  Bin {target_bin} Epoch {epoch}: "
                f"train={avg_train_loss:.6f} val={avg_val_loss:.6f}"
            )
        if epochs_without_improvement >= args.patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "best_val_loss": best_val_loss,
        "Ps_mean": Ps_mean[target_bin].item(),
        "Ps_std": Ps_std[target_bin].item(),
        "Iplus_mean": Iplus_mean[target_bin].item(),
        "Iplus_std": Iplus_std[target_bin].item(),
        "Iminus_mean": Iminus_mean[target_bin].item(),
        "Iminus_std": Iminus_std[target_bin].item(),
        "bin_idx": target_bin,
        "use_hidden": args.use_hidden,
        "hidden_dim": args.hidden_dim,
    }
    return model, ckpt


# =============================================================================
# Main
# =============================================================================


def augment_training_data_with_gap_fill(
    Ps_train: np.ndarray,
    Iplus_train: np.ndarray,
    Iminus_train: np.ndarray,
    lookup_path: str,
    num_bins: int = 500,
    num_fill_per_bin: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill training-data gaps using lookup-table interpolation.

    At center-frequency bins the clean lineshape signal never passes through
    zero, creating a symmetric gap in Ps around zero.  Burns push Ps into this
    gap, causing the model to extrapolate.  We fill each bin's gap with
    uniformly-spaced Ps values whose Iplus/Iminus come from linear
    interpolation of the sorted lookup table -- exactly the same mapping
    ``map_signal_fast`` uses at inference time.
    """
    lookup = pd.read_pickle(lookup_path)
    lookup_Ps = np.stack(lookup["Ps"].values)
    lookup_Ip = np.stack(lookup["Iplus"].values)
    lookup_Im = np.stack(lookup["Iminus"].values)

    aug_Ps = np.copy(Ps_train)
    aug_Ip = np.copy(Iplus_train)
    aug_Im = np.copy(Iminus_train)

    n_orig = Ps_train.shape[0]
    fill_Ps = np.zeros((num_fill_per_bin, num_bins))
    fill_Ip = np.zeros((num_fill_per_bin, num_bins))
    fill_Im = np.zeros((num_fill_per_bin, num_bins))

    bins_filled = 0
    for b in range(num_bins):
        ps_sorted_idx = np.argsort(lookup_Ps[:, b])
        ps_sorted = lookup_Ps[ps_sorted_idx, b]
        ip_sorted = lookup_Ip[ps_sorted_idx, b]
        im_sorted = lookup_Im[ps_sorted_idx, b]

        train_ps = np.sort(Ps_train[:, b])
        gaps = np.diff(train_ps)
        max_gap_idx = np.argmax(gaps)
        gap_lo = train_ps[max_gap_idx]
        gap_hi = train_ps[max_gap_idx + 1]
        gap_size = gap_hi - gap_lo

        ps_range = train_ps[-1] - train_ps[0]
        if ps_range == 0 or gap_size < 0.05 * ps_range:
            fill_Ps[:, b] = np.random.uniform(train_ps[0], train_ps[-1], num_fill_per_bin)
            fill_Ip[:, b] = np.interp(fill_Ps[:, b], ps_sorted, ip_sorted)
            fill_Im[:, b] = np.interp(fill_Ps[:, b], ps_sorted, im_sorted)
            continue

        bins_filled += 1
        fill_vals = np.linspace(gap_lo, gap_hi, num_fill_per_bin)
        fill_Ps[:, b] = fill_vals
        fill_Ip[:, b] = np.interp(fill_vals, ps_sorted, ip_sorted)
        fill_Im[:, b] = np.interp(fill_vals, ps_sorted, im_sorted)

    aug_Ps = np.concatenate([aug_Ps, fill_Ps], axis=0)
    aug_Ip = np.concatenate([aug_Ip, fill_Ip], axis=0)
    aug_Im = np.concatenate([aug_Im, fill_Im], axis=0)

    print(
        f"Gap-fill augmentation: {bins_filled}/{num_bins} bins had significant gaps. "
        f"Added {num_fill_per_bin} synthetic samples -> total {aug_Ps.shape[0]}"
    )
    return aug_Ps, aug_Ip, aug_Im


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 500 bin models and combine into a single model."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="combined_500_results",
        help="Directory for combined model and outputs",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="single_bin_results",
        help="Directory with existing binning_model_bin_{idx}.pth checkpoints to resume/load",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing binning_training.pkl and binning_testing.pkl",
    )
    parser.add_argument("--num-bins", type=int, default=500)
    parser.add_argument(
        "--max-bins",
        type=int,
        default=None,
        help="Max bins to train (for testing). Default: all 500",
    )
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_bins = args.num_bins
    max_bins = args.max_bins if args.max_bins is not None else num_bins
    max_bins = min(max_bins, num_bins)
    # Load data: prefer binning data (Ps -> Iplus, Iminus) from ssRFData
    train_path = os.path.join(args.data_dir, "binning_training.pkl")
    test_path = os.path.join(args.data_dir, "binning_testing.pkl")

    print("Loading data...")
    df_train = pd.read_pickle(train_path)
    df_test = pd.read_pickle(test_path)

    Ps_train_full = np.stack(df_train["Ps"].values).reshape(-1, num_bins)
    Iminus_train_full = np.stack(df_train["Iminus"].values).reshape(-1, num_bins)
    Iplus_train_full = np.stack(df_train["Iplus"].values).reshape(-1, num_bins)


    Ps_test_full = np.stack(df_test["Ps"].values).reshape(-1, num_bins)
    Iminus_test_full = np.stack(df_test["Iminus"].values).reshape(-1, num_bins)
    Iplus_test_full = np.stack(df_test["Iplus"].values).reshape(-1, num_bins)

    Ps_train_full = torch.from_numpy(Ps_train_full).float()
    Iminus_train_full = torch.from_numpy(Iminus_train_full).float()
    Iplus_train_full = torch.from_numpy(Iplus_train_full).float()

    Ps_test_full = torch.from_numpy(Ps_test_full).float()
    Iminus_test_full = torch.from_numpy(Iminus_test_full).float()
    Iplus_test_full = torch.from_numpy(Iplus_test_full).float()

    test_dataset_full = data.TensorDataset(
        Ps_test_full, Iminus_test_full, Iplus_test_full
    )
    val_dataset, test_dataset = data.random_split(
        test_dataset_full, [0.5, 0.5]
    )

    Ps_mean = Ps_train_full.mean(dim=0)
    Ps_std = Ps_train_full.std(dim=0)
    Ps_std[Ps_std == 0] = 1.0

    Iminus_mean = Iminus_train_full.mean(dim=0)
    Iminus_std = Iminus_train_full.std(dim=0)
    Iminus_std[Iminus_std == 0] = 1.0

    Iplus_mean = Iplus_train_full.mean(dim=0)
    Iplus_std = Iplus_train_full.std(dim=0)
    Iplus_std[Iplus_std == 0] = 1.0

    Ps_train_norm = (Ps_train_full - Ps_mean) / Ps_std
    Iminus_train_norm = (Iminus_train_full - Iminus_mean) / Iminus_std
    Iplus_train_norm = (Iplus_train_full - Iplus_mean) / Iplus_std

    Ps_test_norm = (Ps_test_full - Ps_mean) / Ps_std
    Iminus_test_norm = (Iminus_test_full - Iminus_mean) / Iminus_std
    Iplus_test_norm = (Iplus_test_full - Iplus_mean) / Iplus_std

    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    val_Ps = Ps_test_norm[val_indices]
    val_Iplus = Iplus_test_norm[val_indices]
    val_Iminus = Iminus_test_norm[val_indices]

    test_Ps = Ps_test_norm[test_indices]
    test_Iplus = Iplus_test_norm[test_indices]
    test_Iminus = Iminus_test_norm[test_indices]

    bin_models = []
    stats = {
        "Ps_mean": Ps_mean.numpy(),
        "Ps_std": Ps_std.numpy(),
        "Iplus_mean": Iplus_mean.numpy(),
        "Iplus_std": Iplus_std.numpy(),
        "Iminus_mean": Iminus_mean.numpy(),
        "Iminus_std": Iminus_std.numpy(),
    }

    print(f"Training {max_bins} bin models...")
    for target_bin in range(max_bins):
        val_Ps_bin = val_Ps[:, target_bin : target_bin + 1]
        val_Iplus_bin = val_Iplus[:, target_bin : target_bin + 1]
        val_Iminus_bin = val_Iminus[:, target_bin : target_bin + 1]

        resume_from = None
        resume_candidates = [
            os.path.join(args.output_dir, f"binning_model_bin_{target_bin}.pth"),
            os.path.join(args.model_dir, f"binning_model_bin_{target_bin}.pth"),
        ]
        for candidate in resume_candidates:
            if os.path.exists(candidate):
                resume_from = candidate
                break

        model, _ = train_single_bin(
            target_bin=target_bin,
            Ps_train_norm=Ps_train_norm,
            Iplus_train_norm=Iplus_train_norm,
            Iminus_train_norm=Iminus_train_norm,
            val_Ps_bin=val_Ps_bin,
            val_Iplus_bin=val_Iplus_bin,
            val_Iminus_bin=val_Iminus_bin,
            Ps_mean=Ps_mean,
            Ps_std=Ps_std,
            Iplus_mean=Iplus_mean,
            Iplus_std=Iplus_std,
            Iminus_mean=Iminus_mean,
            Iminus_std=Iminus_std,
            device=device,
            args=args,
            resume_checkpoint=resume_from,
        )
        bin_models.append(model)

    # Save individual bin checkpoints
    for bin_idx, model in enumerate(bin_models[:max_bins]):
        out_path = os.path.join(
            args.output_dir, f"binning_model_bin_{bin_idx}.pth"
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "Ps_mean": stats["Ps_mean"][bin_idx],
                "Ps_std": stats["Ps_std"][bin_idx],
                "Iplus_mean": stats["Iplus_mean"][bin_idx],
                "Iplus_std": stats["Iplus_std"][bin_idx],
                "Iminus_mean": stats["Iminus_mean"][bin_idx],
                "Iminus_std": stats["Iminus_std"][bin_idx],
                "bin_idx": bin_idx,
                "use_hidden": args.use_hidden,
                "hidden_dim": args.hidden_dim,
            },
            out_path,
        )
    print(f"Saved {max_bins} individual bin checkpoints to {args.output_dir}")

    # Build combined model
    combined = Combined500BinModel(bin_models=bin_models, stats=stats)
    combined = combined.to(device)

    # Save combined model
    combined_path = os.path.join(args.output_dir, "combined_500_bin_model.pth")
    payload = {
        "num_bins": num_bins,
        "bin_state_dicts": [m.state_dict() for m in bin_models],
        "stats": stats,
        "use_hidden": getattr(args, "use_hidden", True),
        "hidden_dim": getattr(args, "hidden_dim", 32),
    }
    torch.save(payload, combined_path)
    print(f"Saved combined model to {combined_path}")

    print("Skipped scaling stats export (no-scaling training mode).")

    # Evaluate on test set if we have it
    test_path = os.path.join(args.data_dir, "binning_testing.pkl")
    if os.path.exists(test_path):
        df_test = pd.read_pickle(test_path)
        Ps_test = np.stack(df_test["Ps"].values).reshape(-1, num_bins)
        Iplus_test = np.stack(df_test["Iplus"].values).reshape(-1, num_bins)
        Iminus_test = np.stack(df_test["Iminus"].values).reshape(-1, num_bins)

        X = torch.from_numpy(Ps_test).float().to(device)
        with torch.no_grad():
            pred_iplus, pred_iminus = combined(X)

        pred_iplus = pred_iplus.cpu().numpy()
        pred_iminus = pred_iminus.cpu().numpy()

        loss_iplus = np.mean(np.abs(pred_iplus - Iplus_test))
        loss_iminus = np.mean(np.abs(pred_iminus - Iminus_test))
        print(f"\nTest L1 loss (I+): {loss_iplus:.6f}")
        print(f"Test L1 loss (I-): {loss_iminus:.6f}")

        # RPE
        rpe_iplus = np.zeros_like(Iplus_test)
        rpe_iminus = np.zeros_like(Iminus_test)
        mask_iplus = np.abs(Iplus_test) > 1e-10
        mask_iminus = np.abs(Iminus_test) > 1e-10
        rpe_iplus[mask_iplus] = (
            np.abs(pred_iplus[mask_iplus] - Iplus_test[mask_iplus])
            / np.abs(Iplus_test[mask_iplus])
            * 100.0
        )
        rpe_iminus[mask_iminus] = (
            np.abs(pred_iminus[mask_iminus] - Iminus_test[mask_iminus])
            / np.abs(Iminus_test[mask_iminus])
            * 100.0
        )
        print(
            f"Median RPE I+ (across all): {np.median(rpe_iplus[mask_iplus]):.4f}%"
        )
        print(
            f"Median RPE I- (across all): {np.median(rpe_iminus[mask_iminus]):.4f}%"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
