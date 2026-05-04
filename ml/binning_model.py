"""
Train 500 separate bin models (one per frequency bin) and combine them into
a single model for inference.

Usage:
  # Train all 500 bins and save combined model (run from project root):
  python TensorStudies/binning_model_500_bins_combined.py

  # Or train only a subset for testing:
  python TensorStudies/binning_model_500_bins_combined.py --max-bins 10

Outputs:
  - combined_500_bin_model.pth: full model with weights and scaling stats
  - scaling_stats.npz: standalone scaling values for use with real data

Load and scale real data:
  import numpy as np
  data = np.load("combined_500_results/scaling_stats.npz")
  Ps_mean, Ps_std = data["Ps_mean"], data["Ps_std"]
  Ps_scaled = (Ps_real - Ps_mean) / (Ps_std + 1e-12)  # shape (batch, 500)
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


# =============================================================================
# LinearBinModel - same as binning_model_single_bin_job.py
# =============================================================================


class LinearBinModel(nn.Module):
    """
    Linear model with optional hidden layer: 1 input -> 2 outputs (Iplus, Iminus)
    """

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


# =============================================================================
# Combined500BinModel - wraps 500 LinearBinModels into one inference model
# =============================================================================


class Combined500BinModel(nn.Module):
    """
    Wraps 500 LinearBinModels into a single model.
    Input:  (batch_size, 500) - Ps values per bin
    Output: (batch_size, 500) Iplus, (batch_size, 500) Iminus
    """

    def __init__(
        self,
        bin_models: List[nn.Module],
        stats: Dict[str, np.ndarray],
    ):
        super().__init__()
        self.num_bins = len(bin_models)
        self.bin_models = nn.ModuleList(bin_models)
        self.stats = stats

        # Register stats as buffers for device portability (optional, we use numpy)
        self._Ps_mean = torch.from_numpy(stats["Ps_mean"]).float()
        self._Ps_std = torch.from_numpy(stats["Ps_std"]).float()
        self._Iplus_mean = torch.from_numpy(stats["Iplus_mean"]).float()
        self._Iplus_std = torch.from_numpy(stats["Iplus_std"]).float()
        self._Iminus_mean = torch.from_numpy(stats["Iminus_mean"]).float()
        self._Iminus_std = torch.from_numpy(stats["Iminus_std"]).float()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch_size, 500) - raw Ps values
        Returns: (iplus, iminus) each (batch_size, 500) in original scale
        """
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


# =============================================================================
# Training logic for a single bin
# =============================================================================


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
    parser.add_argument(
        "--no-hidden",
        action="store_false",
        dest="use_hidden",
        help="Use a purely linear model for each bin",
    )
    parser.set_defaults(use_hidden=True)
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
        "use_hidden": args.use_hidden,
        "hidden_dim": args.hidden_dim,
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
