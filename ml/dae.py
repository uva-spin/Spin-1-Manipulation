"""
Denoising Autoencoder (DAE) for 500-point NMR signals.

Uses Ps (Iplus + Iminus) at each bin as the training target. Loads binning_training.pkl /
binning_testing.pkl, applies noise to the 500-point Ps signals BEFORE normalizing,
then trains a simple DAE to denoise.

For use with the DAE+combined pipeline (dae_combined_pipeline.py), train with
--no-scale-01 so the DAE output is in raw Ps scale compatible with the combined model.

Usage:
  python TensorStudies/dae.py
  python TensorStudies/dae.py --data-dir . --noise-std 0.1
  python TensorStudies/dae.py --test-only --checkpoint dae_denoise_results/dae_denoise_500.pth
"""

import argparse
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


# =============================================================================
# Denoising Autoencoder
# =============================================================================


class DenoisingAutoencoder(nn.Module):
    """
    Simple DAE: 500 -> bottleneck -> 500.
    Input: noisy 500-point signal (normalized)
    Output: clean 500-point signal (normalized)
    """

    def __init__(self, input_dim: int = 500, hidden_dims: tuple = (256, 128, 64)):
        super().__init__()
        self.input_dim = input_dim

        # Encoder
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
            ])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.bottleneck_dim = hidden_dims[-1]

        # Decoder (mirror)
        layers = []
        for h in reversed(hidden_dims[:-1]):
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
            ])
            prev = h
        layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# =============================================================================
# Data loading and noise
# =============================================================================


def add_noise(signal: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to signal. Applied before normalizing."""
    noise = rng.normal(0, noise_std, signal.shape)
    return signal + noise


def select_examples_by_polarization(
    P: np.ndarray,
    scales: List[Tuple[float, float, float]],
    rng: np.random.Generator,
) -> List[Tuple[int, str]]:
    """
    Select one example index per polarization scale band.
    scales: list of (target_pct, low, high) e.g. (5, 0.03, 0.08) for ±5%
    Returns: list of (index, label) e.g. [(idx, "±5%"), ...]
    """
    result: List[Tuple[int, str]] = []
    P_abs = np.abs(P)
    for target_pct, low, high in scales:
        mask = (P_abs >= low) & (P_abs <= high)
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            dist = np.abs(P_abs - target_pct / 100.0)
            idx = int(np.argmin(dist))
            result.append((idx, f"±{target_pct}% (closest)"))
        else:
            idx = int(rng.choice(candidates))
            result.append((idx, f"±{target_pct}%"))
    return result


def select_examples_by_burn_intensity(
    clean: np.ndarray,
    rng: np.random.Generator,
    n: int = 4,
    sigma: float = 15.0,
) -> List[Tuple[int, str]]:
    """
    Select examples with strongest hole-burning artifacts.
    Burns create local deviations from the smooth baseline. We smooth the signal
    and pick samples with largest max |residual| (burn intensity).
    Returns: list of (index, label) e.g. [(idx, "burn strong"), ...]
    """
    burn_scores = np.zeros(len(clean))
    for i in range(len(clean)):
        smoothed = gaussian_filter1d(clean[i].astype(np.float64), sigma=sigma, mode="nearest")
        residual = np.abs(clean[i] - smoothed)
        burn_scores[i] = np.max(residual)

    # Pick top n by burn intensity, with some randomness among top candidates
    top_k = min(n * 3, len(clean))
    top_indices = np.argsort(burn_scores)[::-1][:top_k]
    chosen = rng.choice(top_indices, size=min(n, len(top_indices)), replace=False)
    return [(int(idx), f"burn (intensity={burn_scores[idx]:.3f})") for idx in chosen]


def _minmax_scale_01(x: np.ndarray) -> np.ndarray:
    """Scale each sample to [0, 1] per row."""
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    span = x_max - x_min
    span[span < 1e-12] = 1.0
    return (x - x_min) / span


def load_and_prepare_data(
    data_dir: str,
    num_points: int = 500,
    noise_std: float = 0.1,
    val_frac: float = 0.2,
    seed: int = 42,
    scale_01: bool = True,
    override_stats: dict = None,
):
    """
    Load binning data, add noise to signals before normalizing.
    If scale_01=True (default), each spectrum is min-max scaled to [0,1] per sample
    first, then noise is added. Signals stay in a 0-1 range (noise may push slightly outside).
    If override_stats is provided (e.g. from a checkpoint), use its Ps_mean/Ps_std for
    normalization instead of computing from train (ensures test-only matches training).
    Returns: train/val loaders, stats for denormalization.
    """
    train_path = os.path.join(data_dir, "training_data.pkl")
    test_path = os.path.join(data_dir, "testing_data.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found: {train_path}. "
            "Run from project root or set --data-dir."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")

    rng = np.random.default_rng(seed)

    df_train = pd.read_pickle(train_path)
    df_test = pd.read_pickle(test_path)
    print(f"Loaded burned signal data: {len(df_train)} train, {len(df_test)} test (Ps_burned -> denoise)")

    # Ps = Iplus + Iminus (500-point signal per sample)
    Ps_train = np.stack(df_train["Ps"].values).reshape(-1, num_points)
    Ps_test = np.stack(df_test["Ps"].values).reshape(-1, num_points)
    if "P" in df_test.columns:
        P_test = df_test["P"].values.astype(np.float64)
    else:
        # Fallback for data created before P was added to ssRFData.py
        P_test = np.zeros(len(df_test), dtype=np.float64)

    if scale_01:
        # Min-max to [0,1] per sample first so signals are in 0-1 range
        Ps_train = _minmax_scale_01(Ps_train)
        Ps_test = _minmax_scale_01(Ps_test)
        # Add noise to [0,1]-scaled signal (noise_std is in [0,1] units, e.g. 0.05 = 5%)
        Ps_train_noisy = add_noise(Ps_train, noise_std, rng)
        Ps_test_noisy = add_noise(Ps_test, noise_std, rng)
        # Identity stats for denormalization (data already in [0,1])
        stats = {"Ps_mean": np.zeros(num_points), "Ps_std": np.ones(num_points), "scale_01": True}
        Ps_train_norm = Ps_train
        Ps_train_noisy_norm = Ps_train_noisy
        Ps_test_norm = Ps_test
        Ps_test_noisy_norm = Ps_test_noisy
    else:
        # Z-score: add noise before normalizing (noise proportional to signal std)
        sig_scale = np.std(Ps_train)
        if sig_scale < 1e-10:
            sig_scale = 1.0
        effective_noise_std = noise_std * sig_scale
        Ps_train_noisy = add_noise(Ps_train, effective_noise_std, rng)
        Ps_test_noisy = add_noise(Ps_test, effective_noise_std, rng)
        if override_stats is not None:
            Ps_mean = override_stats["Ps_mean"]
            Ps_std = override_stats["Ps_std"]
            stats = dict(override_stats)
        else:
            Ps_mean = Ps_train.mean(axis=0)
            Ps_std = Ps_train.std(axis=0)
            Ps_std[Ps_std < 1e-12] = 1.0
            stats = {"Ps_mean": Ps_mean, "Ps_std": Ps_std, "scale_01": False}
        Ps_train_norm = (Ps_train - Ps_mean) / Ps_std
        Ps_train_noisy_norm = (Ps_train_noisy - Ps_mean) / Ps_std
        Ps_test_norm = (Ps_test - Ps_mean) / Ps_std
        Ps_test_noisy_norm = (Ps_test_noisy - Ps_mean) / Ps_std

    # Train/val split on training set
    n_train = len(Ps_train)
    perm = rng.permutation(n_train)
    n_val = int(n_train * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_dataset = data.TensorDataset(
        torch.from_numpy(Ps_train_noisy_norm[train_idx]).float(),
        torch.from_numpy(Ps_train_norm[train_idx]).float(),
    )
    val_dataset = data.TensorDataset(
        torch.from_numpy(Ps_train_noisy_norm[val_idx]).float(),
        torch.from_numpy(Ps_train_norm[val_idx]).float(),
    )
    test_dataset = data.TensorDataset(
        torch.from_numpy(Ps_test_noisy_norm).float(),
        torch.from_numpy(Ps_test_norm).float(),
    )

    return train_dataset, val_dataset, test_dataset, stats, P_test


# =============================================================================
# Training
# =============================================================================


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        optimizer.zero_grad()
        pred = model(noisy)
        loss = loss_fn(pred, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = model(noisy)
            loss = loss_fn(pred, clean)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches if n_batches > 0 else 0.0


# =============================================================================
# Testing: statistics and plots
# =============================================================================


def _publication_style():
    """Configure matplotlib for publication-ready figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.2,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })


def run_test(
    model,
    test_dataset,
    stats,
    device,
    output_dir,
    noise_std,
    P_test,
    seed=42,
    n_examples=5,
    max_heatmap_samples=200,
):
    """
    Run full test evaluation: collect predictions, compute statistics,
    save JSON and plots.
    """
    model.eval()
    Ps_mean = stats["Ps_mean"]
    Ps_std = stats["Ps_std"]
    scale_01 = stats.get("scale_01", False)

    # Collect all predictions
    all_noisy = []
    all_clean = []
    all_pred = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            noisy, clean = test_dataset[i]
            noisy_b = noisy.unsqueeze(0).to(device)
            pred = model(noisy_b).squeeze(0).cpu().numpy()
            all_noisy.append(noisy.numpy())
            all_clean.append(clean.numpy())
            all_pred.append(pred)

    noisy_np = np.stack(all_noisy)
    clean_np = np.stack(all_clean)
    pred_np = np.stack(all_pred)

    # Denormalize for raw-scale metrics
    noisy_denorm = noisy_np * Ps_std + Ps_mean
    clean_denorm = clean_np * Ps_std + Ps_mean
    pred_denorm = pred_np * Ps_std + Ps_mean

    n_samples, n_bins = pred_np.shape

    # ---- Statistics ----
    mse_norm = np.mean((pred_np - clean_np) ** 2)
    mse_raw = np.mean((pred_denorm - clean_denorm) ** 2)
    mae_norm = np.mean(np.abs(pred_np - clean_np))
    mae_raw = np.mean(np.abs(pred_denorm - clean_denorm))
    rmse_raw = np.sqrt(mse_raw)

    # Noisy vs clean (baseline)
    mse_noisy_norm = np.mean((noisy_np - clean_np) ** 2)
    mse_noisy_raw = np.mean((noisy_denorm - clean_denorm) ** 2)
    mae_noisy_raw = np.mean(np.abs(noisy_denorm - clean_denorm))

    mse_improvement_pct = (1 - mse_raw / mse_noisy_raw) * 100 if mse_noisy_raw > 0 else 0
    mae_improvement_pct = (1 - mae_raw / mae_noisy_raw) * 100 if mae_noisy_raw > 0 else 0

    residuals = pred_denorm - clean_denorm
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    # Per-bin median absolute error
    per_bin_mae = np.median(np.abs(residuals), axis=0)

    stats_dict = {
        "n_samples": int(n_samples),
        "n_bins": int(n_bins),
        "noise_std": float(noise_std),
        "MSE_normalized": float(mse_norm),
        "MSE_raw": float(mse_raw),
        "MAE_normalized": float(mae_norm),
        "MAE_raw": float(mae_raw),
        "RMSE_raw": float(rmse_raw),
        "MSE_noisy_vs_clean_raw": float(mse_noisy_raw),
        "MAE_noisy_vs_clean_raw": float(mae_noisy_raw),
        "MSE_improvement_pct": float(mse_improvement_pct),
        "MAE_improvement_pct": float(mae_improvement_pct),
        "residual_mean": float(residual_mean),
        "residual_std": float(residual_std),
        "per_bin_MAE": {
            "median": float(np.median(per_bin_mae)),
            "mean": float(np.mean(per_bin_mae)),
            "min": float(np.min(per_bin_mae)),
            "max": float(np.max(per_bin_mae)),
        },
    }

    stats_path = os.path.join(output_dir, "test_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    # Print summary
    print("\n===== Test Results =====")
    print(f"MSE (normalized): {mse_norm:.6f}")
    print(f"MSE (raw): {mse_raw:.6e}")
    print(f"MAE (raw): {mae_raw:.6e}")
    print(f"RMSE (raw): {rmse_raw:.6e}")
    print(f"Noisy vs clean MSE: {mse_noisy_raw:.6e}")
    print(f"MSE improvement: {mse_improvement_pct:.2f}%")
    print(f"MAE improvement: {mae_improvement_pct:.2f}%")
    print(f"Residual mean: {residual_mean:.6e}, std: {residual_std:.6e}")

    # ---- Plots ----
    _publication_style()
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    x = np.linspace(-3, 3, n_bins)

    # Select examples: polarization + hole-burning (prioritize burn-heavy)
    polarization_scales: List[Tuple[float, float, float]] = [
        (5, 0.03, 0.08),
        (10, 0.08, 0.12),
        (20, 0.15, 0.25),
        (30, 0.25, 0.35),
        (40, 0.35, 0.45),
        (50, 0.45, 0.55),
    ]
    rng = np.random.default_rng(seed)
    pol_selections = select_examples_by_polarization(P_test, polarization_scales, rng)
    n_burn = min(10, max(6, n_examples - len(polarization_scales)))  # prioritize burn examples
    burn_selections = select_examples_by_burn_intensity(
        clean_denorm, rng, n=n_burn, sigma=5.0  # smaller sigma = detect localized burns
    )
    # Combine: burn-heavy FIRST (so burns are visible), then polarization (avoid duplicates)
    seen: set = set()
    example_selections: List[Tuple[int, str]] = []
    for idx, label in burn_selections:
        if idx not in seen:
            seen.add(idx)
            example_selections.append((idx, label))
    for idx, label in pol_selections:
        if idx not in seen:
            seen.add(idx)
            example_selections.append((idx, f"P {label}"))
    example_indices = [idx for idx, _ in example_selections]
    example_labels = [label for _, label in example_selections]
    num_ex = len(example_indices)

    # 0. Individual publication-ready plots for each example
    for i in range(num_ex):
        idx = example_indices[i]
        label = example_labels[i]
        # Sanitize label for filename (replace spaces, parentheses)
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(x, clean_denorm[idx], color="#1f77b4", lw=2, label="Clean")
        ax.plot(x, noisy_denorm[idx], color="#7f7f7f", alpha=0.7, lw=1, label="Noisy")
        ax.plot(x, pred_denorm[idx], color="#d62728", linestyle="--", lw=1.5, label="DAE output")
        ax.set_xlabel("R")
        ax.set_ylabel("Signal")
        ax.set_title(f"{label} (P = {P_test[idx]:.3f})")
        if scale_01:
            ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        path_base = os.path.join(plots_dir, f"dae_denoise_example_{i:02d}_{safe_label}")
        for fmt in ("png", "pdf"):
            fig.savefig(f"{path_base}.{fmt}", dpi=600 if fmt == "png" else None, bbox_inches="tight")
        plt.close()
        print(f"  Saved individual plot: {path_base}.png, .pdf")

    # 1. Multiple example denoising (2-column grid when many examples)
    n_cols = 2 if num_ex >= 4 else 1
    n_rows = (num_ex + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 2.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    for i in range(num_ex):
        ax = axes_flat[i]
        idx = example_indices[i]
        ax.plot(x, clean_denorm[idx], "b-", lw=2, label="Clean")
        ax.plot(x, noisy_denorm[idx], "gray", alpha=0.7, lw=1, label="Noisy")
        ax.plot(x, pred_denorm[idx], "r--", lw=1.5, label="DAE output")
        ax.set_xlabel("R")
        ax.set_ylabel("Signal")
        ax.set_title(f"{example_labels[i]} (P={P_test[idx]:.3f}) - Denoising (noise_std={noise_std})")
        if scale_01:
            ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
    for j in range(num_ex, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(
            os.path.join(plots_dir, f"lineshape_examples.{fmt}"),
            dpi=600 if fmt == "png" else None,
            bbox_inches="tight",
        )
    plt.close()

    # 1b. Dedicated hole-burning examples (Ps_burned) - ensures burns are visible
    burn_only = [(idx, lbl) for idx, lbl in example_selections if "burn" in lbl.lower()]
    if burn_only:
        n_burn_plot = len(burn_only)
        n_cols_b = 2 if n_burn_plot >= 4 else 1
        n_rows_b = (n_burn_plot + n_cols_b - 1) // n_cols_b
        fig_b, axes_b = plt.subplots(n_rows_b, n_cols_b, figsize=(6 * n_cols_b, 2.5 * n_rows_b), squeeze=False)
        axes_b_flat = axes_b.flatten()
        for i, (idx, lbl) in enumerate(burn_only):
            ax = axes_b_flat[i]
            ax.plot(x, clean_denorm[idx], "b-", lw=2, label="Ps (burned)")
            ax.plot(x, noisy_denorm[idx], "gray", alpha=0.7, lw=1, label="Noisy")
            ax.plot(x, pred_denorm[idx], "r--", lw=1.5, label="DAE output")
            ax.set_xlabel("R")
            ax.set_ylabel("Signal")
            ax.set_title(f"Hole-burning: {lbl} (P={P_test[idx]:.3f})")
            if scale_01:
                ax.set_ylim(-0.05, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3)
        for j in range(n_burn_plot, len(axes_b_flat)):
            axes_b_flat[j].set_visible(False)
        fig_b.suptitle("Ps_burned examples (hole-burning) - trained on burned signal data", y=1.02)
        plt.tight_layout()
        for fmt in ("png", "pdf"):
            fig_b.savefig(
                os.path.join(plots_dir, f"lineshape_examples_burns.{fmt}"),
                dpi=600 if fmt == "png" else None,
                bbox_inches="tight",
            )
        plt.close()

    # 2. Residual heatmap
    heatmap_n = min(max_heatmap_samples, n_samples)
    fig2 = plt.figure(figsize=(8, 4))
    plt.imshow(
        residuals[:heatmap_n],
        aspect="auto",
        cmap="coolwarm",
    )
    plt.colorbar(label="Residual (pred - clean)")
    plt.xlabel("Bin index")
    plt.ylabel("Sample index")
    plt.title("Residuals heatmap (DAE pred - clean)")
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig2.savefig(
            os.path.join(plots_dir, f"residuals_heatmap.{fmt}"),
            dpi=600 if fmt == "png" else None,
            bbox_inches="tight",
        )
    plt.close()

    # 3. Residual histogram
    fig3 = plt.figure(figsize=(6, 4))
    plt.hist(residuals.flatten(), bins=80, color="steelblue", alpha=0.8, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", lw=2)
    plt.xlabel("Residual (pred - clean)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig3.savefig(
            os.path.join(plots_dir, f"residuals_hist.{fmt}"),
            dpi=600 if fmt == "png" else None,
            bbox_inches="tight",
        )
    plt.close()

    # 4. Per-bin median absolute error
    fig4 = plt.figure(figsize=(8, 4))
    plt.plot(np.arange(n_bins), per_bin_mae, "b-", lw=1.5)
    plt.xlabel("Bin index")
    plt.ylabel("Median absolute error")
    plt.title("Per-bin median absolute error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig4.savefig(
            os.path.join(plots_dir, f"per_bin_mae.{fmt}"),
            dpi=600 if fmt == "png" else None,
            bbox_inches="tight",
        )
    plt.close()

    # 5. Pred vs clean scatter (flattened)
    fig5 = plt.figure(figsize=(5, 5))
    # Subsample for large datasets (use rng for reproducibility)
    n_scatter = min(50000, n_samples * n_bins)
    scatter_idx = rng.choice(n_samples * n_bins, n_scatter, replace=False)
    flat_clean = clean_denorm.flatten()
    flat_pred = pred_denorm.flatten()
    plt.scatter(flat_clean[scatter_idx], flat_pred[scatter_idx], alpha=0.3, s=8)
    lims = [
        min(flat_clean.min(), flat_pred.min()),
        max(flat_clean.max(), flat_pred.max()),
    ]
    plt.plot(lims, lims, "r--", lw=2, label="y=x")
    plt.xlabel("Clean (true)")
    plt.ylabel("DAE prediction")
    plt.title("Pred vs clean (scatter)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig5.savefig(
            os.path.join(plots_dir, f"pred_vs_clean.{fmt}"),
            dpi=600 if fmt == "png" else None,
            bbox_inches="tight",
        )
    plt.close()

    # 6. Single example (original style) - use first polarization-scale example
    ex0 = example_indices[min(9, num_ex - 1)]
    # ex0 = 
    fig6, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(x, clean_denorm[ex0], color="#1f77b4", lw=2, label="Clean")
    ax.plot(x, noisy_denorm[ex0], color="#7f7f7f", alpha=0.7, lw=1, label="Noisy")
    ax.plot(x, pred_denorm[ex0], color="#d62728", linestyle="--", lw=1.5, label="DAE output")
    ax.set_xlabel("R")
    ax.set_ylabel("Signal (a.u.)")
    # ax.set_title(f"DAE denoising (P = {P_test[ex0]:.3f}, noise_std = {noise_std})")
    if scale_01:
        ax.set_ylim(-0.05, 1.05)
    # ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig6.savefig(
            os.path.join(plots_dir, f"dae_example.{fmt}"),
            dpi=600 if fmt == "png" else None,
            bbox_inches="tight",
        )
    plt.close()

    print(f"\nPlots saved to {plots_dir}/ (individual examples + summary figures)")


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DAE to denoise 500-point NMR signals."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing training_data.pkl and testing_data.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dae_denoise_results",
        help="Directory for model and plots",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.3,
        help="Noise level (fraction of signal std). Applied before scaling.",
    )
    parser.add_argument(
        "--no-scale-01",
        action="store_true",
        help="Disable min-max scaling to [0,1]. Use z-score normalization instead.",
    )
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--examples",
        type=int,
        default=12,
        help="Max example plots (polarization + hole-burning)",
    )
    parser.add_argument(
        "--max-heatmap-samples",
        type=int,
        default=200,
        help="Max samples for residual heatmap",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training; load checkpoint and run evaluation/plots only.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pth checkpoint (default: <output-dir>/dae_denoise_500.pth). Required for --test-only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.test_only:
        # Load checkpoint first so we use the same preprocessing as at training time
        ckpt_path = args.checkpoint or os.path.join(args.output_dir, "dae_denoise_500.pth")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Train first or set --checkpoint."
            )
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_stats = ckpt["stats"]
        noise_std_ckpt = ckpt.get("noise_std", args.noise_std)
        scale_01_ckpt = ckpt_stats.get("scale_01", True)
        print(f"Noise std (from checkpoint): {noise_std_ckpt}")
        print(f"Scale to [0,1] (from checkpoint): {scale_01_ckpt}")
        # Prepare test data exactly as at training: same noise_std, scale_01, and normalization stats
        train_dataset, val_dataset, test_dataset, stats, P_test = load_and_prepare_data(
            data_dir=args.data_dir,
            noise_std=noise_std_ckpt,
            seed=args.seed,
            scale_01=scale_01_ckpt,
            override_stats=ckpt_stats if not scale_01_ckpt else None,
        )
        hidden_dims = tuple(ckpt["hidden_dims"])
        model = DenoisingAutoencoder(
            input_dim=500,
            hidden_dims=hidden_dims,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
        run_test(
            model=model,
            test_dataset=test_dataset,
            stats=stats,
            device=device,
            output_dir=args.output_dir,
            noise_std=noise_std_ckpt,
            P_test=P_test,
            seed=args.seed,
            n_examples=args.examples,
            max_heatmap_samples=args.max_heatmap_samples,
        )
        print("Done.")
        return

    print(f"Noise std (fraction of signal): {args.noise_std}")
    print(f"Scale to [0,1]: {not args.no_scale_01}")
    # Load data: noise applied before normalizing; scale to [0,1] by default
    train_dataset, val_dataset, test_dataset, stats, P_test = load_and_prepare_data(
        data_dir=args.data_dir,
        noise_std=args.noise_std,
        seed=args.seed,
        scale_01=not args.no_scale_01,
    )

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = DenoisingAutoencoder(
        input_dim=500,
        hidden_dims=tuple(args.hidden_dims),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: train={train_loss:.6f} val={val_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"\nTest MSE (normalized): {test_loss:.6f}")

    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stats": stats,
            "hidden_dims": args.hidden_dims,
            "noise_std": args.noise_std,
        },
        os.path.join(args.output_dir, "dae_denoise_500.pth"),
    )
    print(f"Saved model to {args.output_dir}/dae_denoise_500.pth")

    # Run full test: statistics and plots
    run_test(
        model=model,
        test_dataset=test_dataset,
        stats=stats,
        device=device,
        output_dir=args.output_dir,
        noise_std=args.noise_std,
        P_test=P_test,
        seed=args.seed,
        n_examples=args.examples,
        max_heatmap_samples=args.max_heatmap_samples,
    )

    print("Done.")


if __name__ == "__main__":
    main()
