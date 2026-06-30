"""
Evaluate the combined per-bin burn-context model on test lineshapes.

Run:
  python ml/test-binning.py
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = "models/combined_bin_model.pth"
TEST_FILE = "data/random_burn_lineshapes_1000.pkl"
OUTPUT_DIR = "results/test_binning"
SCALING_FILE = None

DEVICE = "cuda"
NUM_BINS = 500
FEATURE_CLIP_Z = 8.0
EXAMPLES = 12
EXAMPLE_SELECTION = "stratified"  # stratified | sequential | spread
MAX_HEATMAP_SAMPLES = 200

BURN_CONTEXT_FEATURES = (
    "ps_at_burn_bin",
    "P",
    "burn_step_norm",
    "ps_ratio",
    "burn_progress",
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


class _NumpyCompatUnpickler(pickle.Unpickler):
    _MODULE_REMAP = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
    }

    def find_class(self, module: str, name: str):
        return super().find_class(self._MODULE_REMAP.get(module, module), name)


def read_pickle_compat(path: str) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        print("Retrying pickle load with NumPy compatibility shim...")
        with open(path, "rb") as f:
            return _NumpyCompatUnpickler(f).load()


# ---------------------------------------------------------------------------
# Lineshape events
# ---------------------------------------------------------------------------


@dataclass
class LineshapeEvent:
    polarization: float
    frequency: np.ndarray
    ps: np.ndarray
    iplus: np.ndarray
    iminus: np.ndarray
    burn_bin_idx: Optional[int] = None
    burn_step_norm: float = 0.0
    ps_ratio: float = 1.0
    burn_progress: float = 0.0

    def __post_init__(self) -> None:
        self.frequency = np.asarray(self.frequency, dtype=np.float32)
        self.ps = np.asarray(self.ps, dtype=np.float32)
        self.iplus = np.asarray(self.iplus, dtype=np.float32)
        self.iminus = np.asarray(self.iminus, dtype=np.float32)
        if self.burn_bin_idx is not None:
            self.burn_bin_idx = int(self.burn_bin_idx)

    @property
    def num_bins(self) -> int:
        return int(self.ps.shape[0])

    def feature_matrix(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Row j: Ps[j], burn context at j only if j == burn_bin_idx."""
        n = self.num_bins
        burn_step_norm = np.zeros(n, dtype=np.float32)
        ps_ratio = np.ones(n, dtype=np.float32)
        burn_progress = np.zeros(n, dtype=np.float32)
        if self.burn_bin_idx is not None and 0 <= self.burn_bin_idx < n:
            b = self.burn_bin_idx
            burn_step_norm[b] = np.float32(self.burn_step_norm)
            ps_ratio[b] = np.float32(self.ps_ratio)
            burn_progress[b] = np.float32(self.burn_progress)

        columns = {
            "ps_at_burn_bin": self.ps,
            "P": np.full(n, self.polarization, dtype=np.float32),
            "burn_step_norm": burn_step_norm,
            "ps_ratio": ps_ratio,
            "burn_progress": burn_progress,
        }
        names = feature_names or list(BURN_CONTEXT_FEATURES)
        return np.column_stack([columns[name] for name in names]).astype(np.float32)


def event_from_row(row: pd.Series, n_bins: int) -> LineshapeEvent:
    if "P_initial" in row.index:
        p = float(row["P_initial"])
    elif "P" in row.index:
        p = float(row["P"])
    else:
        raise KeyError("Row missing P_initial or P.")

    burn_bin_idx = row.get("burn_bin_idx")
    if burn_bin_idx is not None and pd.isna(burn_bin_idx):
        burn_bin_idx = None
    elif burn_bin_idx is not None:
        burn_bin_idx = int(burn_bin_idx)

    freq = row["frequency"] if "frequency" in row.index else np.arange(n_bins)
    return LineshapeEvent(
        polarization=p,
        frequency=np.asarray(freq, dtype=np.float32),
        ps=np.asarray(row["Ps"], dtype=np.float32),
        iplus=np.asarray(row["Iplus"], dtype=np.float32),
        iminus=np.asarray(row["Iminus"], dtype=np.float32),
        burn_bin_idx=burn_bin_idx,
        burn_step_norm=float(row.get("burn_step_norm", 0.0)),
        ps_ratio=float(row.get("ps_ratio", 1.0)),
        burn_progress=float(row.get("burn_progress", 0.0)),
    )


def load_test_events(path: str) -> Tuple[List[LineshapeEvent], pd.DataFrame]:
    df = read_pickle_compat(path)
    missing = {"Ps", "Iplus", "Iminus"} - set(df.columns)
    if missing:
        raise KeyError(f"Test file missing columns: {sorted(missing)}")
    n_bins = int(np.asarray(df["Ps"].iloc[0]).shape[0])
    return [event_from_row(row, n_bins) for _, row in df.iterrows()], df


# ---------------------------------------------------------------------------
# Model (must match ml/single_bin.py + ml/combine_single_bin_models.py)
# ---------------------------------------------------------------------------


class LinearBinModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, 0], out[:, 1]


def _resolve_input_stats(
    stats: Dict[str, np.ndarray], num_models: int, input_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    if "X_mean" in stats:
        x_mean = np.asarray(stats["X_mean"], dtype=np.float32)
        x_std = np.asarray(stats["X_std"], dtype=np.float32)
    elif "Ps_mean" in stats:
        ps_mean = np.asarray(stats["Ps_mean"], dtype=np.float32)
        ps_std = np.asarray(stats["Ps_std"], dtype=np.float32)
        if ps_mean.ndim == 0:
            raise ValueError("Ps_mean must be per-bin, not a global scalar.")
        x_mean = ps_mean[:, None] if ps_mean.ndim == 1 else ps_mean
        x_std = ps_std[:, None] if ps_std.ndim == 1 else ps_std
    else:
        raise KeyError("Stats need X_mean/X_std or Ps_mean/Ps_std.")
    if x_mean.shape != (num_models, input_dim):
        raise ValueError(f"Expected X_mean shape ({num_models}, {input_dim}), got {x_mean.shape}.")
    return x_mean, x_std


def _resolve_output_stats(
    stats: Dict[str, np.ndarray], num_models: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    out = {}
    for key in ("Iplus_mean", "Iplus_std", "Iminus_mean", "Iminus_std"):
        arr = np.asarray(stats[key], dtype=np.float32)
        if arr.shape != (num_models,):
            raise ValueError(f"Expected per-bin {key} length {num_models}, got {arr.shape}.")
        out[key] = arr
    return out["Iplus_mean"], out["Iplus_std"], out["Iminus_mean"], out["Iminus_std"]


class Combined500BinModel(nn.Module):
    """One bin model per spectrum index; per-bin normalization only."""

    def __init__(
        self,
        bin_models: List[nn.Module],
        stats: Dict[str, np.ndarray],
        feature_names: List[str],
        feature_clip_z: float = FEATURE_CLIP_Z,
        loaded_bin_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_models = len(bin_models)
        self.bin_models = nn.ModuleList(bin_models)
        self.feature_names = list(feature_names)
        self.input_dim = len(self.feature_names)
        self.feature_clip_z = feature_clip_z
        self.loaded_bin_indices = (
            list(range(self.num_models))
            if loaded_bin_indices is None
            else [int(i) for i in loaded_bin_indices]
        )

        x_mean, x_std = _resolve_input_stats(stats, self.num_models, self.input_dim)
        ip_m, ip_s, im_m, im_s = _resolve_output_stats(stats, self.num_models)
        self._X_mean = torch.from_numpy(x_mean).float()
        self._X_std = torch.from_numpy(x_std).float()
        self._Iplus_mean = torch.from_numpy(ip_m).float()
        self._Iplus_std = torch.from_numpy(ip_s).float()
        self._Iminus_mean = torch.from_numpy(im_m).float()
        self._Iminus_std = torch.from_numpy(im_s).float()

    def _predict_bin(self, model_idx: int, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self._X_mean[model_idx].to(x.device)
        std = self._X_std[model_idx].to(x.device)
        x_norm = (x - mean) / (std + 1e-12)
        if self.feature_clip_z > 0:
            x_norm = torch.clamp(x_norm, -self.feature_clip_z, self.feature_clip_z)
        ip_n, im_n = self.bin_models[model_idx](x_norm)
        ip = ip_n * self._Iplus_std[model_idx].to(x.device) + self._Iplus_mean[model_idx].to(x.device)
        im = im_n * self._Iminus_std[model_idx].to(x.device) + self._Iminus_mean[model_idx].to(x.device)
        return ip, im

    def forward(
        self, features: torch.Tensor, spectrum_bins: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, n_feat, dim = features.shape
        if dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {dim}.")
        n_out = n_feat if spectrum_bins is None else spectrum_bins
        pred_ip = torch.full((batch, n_out), float("nan"), device=features.device)
        pred_im = torch.full((batch, n_out), float("nan"), device=features.device)
        for model_idx, bin_idx in enumerate(self.loaded_bin_indices):
            if bin_idx >= n_feat or bin_idx >= n_out:
                continue
            ip, im = self._predict_bin(model_idx, features[:, bin_idx, :])
            pred_ip[:, bin_idx] = ip
            pred_im[:, bin_idx] = im
        return pred_ip, pred_im

    def predict_events(
        self, events: List[LineshapeEvent], spectrum_bins: int = NUM_BINS
    ) -> Tuple[np.ndarray, np.ndarray]:
        feats = torch.from_numpy(
            np.stack([e.feature_matrix(self.feature_names) for e in events], axis=0)
        ).float().to(next(self.parameters()).device)
        with torch.no_grad():
            ip, im = self(feats, spectrum_bins=spectrum_bins)
        return ip.cpu().numpy(), im.cpu().numpy()


def load_combined_model(
    model_path: str,
    device: torch.device,
    scaling_path: Optional[str] = None,
) -> Combined500BinModel:
    payload = torch.load(model_path, map_location=device, weights_only=False)
    num_bins = payload["num_bins"]
    feature_names = list(payload.get("feature_names", ["ps_at_burn_bin"]))
    input_dim = int(payload.get("input_dim", len(feature_names)))
    hidden_dim = int(payload.get("hidden_dim", 256))
    stats = payload.get("stats")

    if stats is None:
        path = scaling_path or os.path.join(os.path.dirname(model_path), "scaling_stats.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No embedded stats and no file at {path}")
        data = np.load(path)
        stats = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}

    _resolve_input_stats(stats, num_bins, input_dim)
    _resolve_output_stats(stats, num_bins)

    models = []
    for i in range(num_bins):
        m = LinearBinModel(input_dim, hidden_dim)
        m.load_state_dict(payload["bin_state_dicts"][i])
        m.eval()
        models.append(m)

    combined = Combined500BinModel(
        models,
        stats,
        feature_names,
        loaded_bin_indices=payload.get("loaded_bin_indices"),
    )
    return combined.to(device).eval()


# ---------------------------------------------------------------------------
# Metrics & plots
# ---------------------------------------------------------------------------


def integrated_polarization(iplus: np.ndarray, iminus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.nansum(iplus + iminus, axis=1), np.nansum(iplus - iminus, axis=1)


def compute_rpe(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rpe = np.full_like(true, np.nan, dtype=np.float64)
    valid = mask & (np.abs(true) > 1e-10)
    rpe[valid] = (pred[valid] - true[valid]) / true[valid] * 100.0
    return rpe


def print_results_table(stats: Dict[str, float]) -> None:
    """Print evaluation metrics as aligned tables."""
    sections = [
        ("Lineshape decomposition", [
            ("L1 I+", stats["L1_Iplus"], ""),
            ("L1 I-", stats["L1_Iminus"], ""),
            ("Median RPE I+", stats["median_RPE_Iplus"], "%"),
            ("Median RPE I-", stats["median_RPE_Iminus"], "%"),
        ]),
        ("Vector polarization P", [
            ("Mean RPE", stats["mean_RPE_P"], "%"),
            ("Std RPE", stats["std_RPE_P"], "%"),
            ("Mean residual", stats["mean_residual_P"], ""),
            ("Std residual", stats["std_residual_P"], ""),
        ]),
        ("Tensor polarization Q", [
            ("Mean RPE", stats["mean_RPE_Q"], "%"),
            ("Std RPE", stats["std_RPE_Q"], "%"),
            ("Mean residual", stats["mean_residual_Q"], ""),
            ("Std residual", stats["std_residual_Q"], ""),
        ]),
    ]
    label_width = max(len(name) for _, rows in sections for name, _, _ in rows)
    print("\n===== Test Results =====")
    print(f"{'Metric':<{label_width}}  {'Value':>12}")
    print("-" * (label_width + 15))
    for i, (title, rows) in enumerate(sections):
        if i > 0:
            print()
        print(title)
        for name, value, unit in rows:
            print(f"  {name:<{label_width - 2}}  {value:12.4f}{unit}")
    print(f"\nSamples: {stats['n_samples']}  |  Bins: {stats['n_bins']}")


def select_example_indices(
    n_test: int, n_examples: int, residuals: np.ndarray, mode: str
) -> np.ndarray:
    n = min(n_examples, n_test)
    if n == 0:
        return np.array([], dtype=int)
    if mode == "sequential":
        return np.arange(n, dtype=int)
    if mode == "spread":
        return np.array([n_test // 2], dtype=int) if n == 1 else np.linspace(0, n_test - 1, n, dtype=int)
    err = np.nanmean(np.abs(residuals), axis=1)
    order = np.argsort(err)
    ranks = np.linspace(0, n_test - 1, n)
    return order[np.round(ranks).astype(int)]


def plot_lineshape_examples(
    out_path: str,
    indices: np.ndarray,
    ps: np.ndarray,
    ip_true: np.ndarray,
    im_true: np.ndarray,
    ip_pred: np.ndarray,
    im_pred: np.ndarray,
    title_prefix: str = "Sample",
) -> None:
    if indices.size == 0:
        return
    x = np.arange(ps.shape[1])
    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 3 * len(indices)), squeeze=False)
    for ax, idx in zip(axes[:, 0], indices):
        ax.plot(x, ps[idx], "k-", lw=1.5, label="Ps")
        ax.plot(x, ip_true[idx], color="#d55e00", alpha=0.35, lw=2, label="True I+")
        ax.plot(x, im_true[idx], color="#0072b2", alpha=0.35, lw=2, label="True I-")
        ax.plot(x, ip_pred[idx], color="#d55e00", ls="--", lw=1.3, label="Pred I+")
        ax.plot(x, im_pred[idx], color="#0072b2", ls="--", lw=1.3, label="Pred I-")
        ax.set_title(f"{title_prefix} {idx}")
        ax.set_xlabel("Bin index")
        ax.set_ylabel("Signal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(out_path: str, data: np.ndarray, title: str, label: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.imshow(data, aspect="auto", cmap="coolwarm")
    plt.colorbar(label=label)
    plt.xlabel("Bin index")
    plt.ylabel("Sample index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    model = load_combined_model(MODEL_PATH, device, SCALING_FILE)
    events, df = load_test_events(TEST_FILE)

    ps = np.stack([e.ps for e in events])
    ip_true = np.stack([e.iplus for e in events])
    im_true = np.stack([e.iminus for e in events])
    n_bins = ps.shape[1]

    ip_pred, im_pred = model.predict_events(events, spectrum_bins=n_bins)
    pred_mask = np.isfinite(ip_pred) & np.isfinite(im_pred)
    if not pred_mask.any():
        raise ValueError("No finite predictions produced.")

    print(
        f"Loaded {model.num_models} bin models "
        f"({model.loaded_bin_indices[0]}..{model.loaded_bin_indices[-1]}), per-bin normalization."
    )

    if "true_P" in df.columns:
        p_true, q_true = df["true_P"].to_numpy(float), df["true_Q"].to_numpy(float)
    else:
        p_true, q_true = integrated_polarization(ip_true, im_true)
    p_pred, q_pred = integrated_polarization(ip_pred, im_pred)

    # sample_mask = (p_true <= -5.0) | (p_true >= 5.0)
    sample_mask = np.ones(p_true.shape, dtype=bool)
    if not sample_mask.any():
        raise ValueError("No samples with |true P| >= 5.")

    ps, ip_true, im_true = ps[sample_mask], ip_true[sample_mask], im_true[sample_mask]
    ip_pred, im_pred = ip_pred[sample_mask], im_pred[sample_mask]
    pred_mask = pred_mask[sample_mask]
    p_true, q_true = p_true[sample_mask], q_true[sample_mask]
    p_pred, q_pred = p_pred[sample_mask], q_pred[sample_mask]

    ip_rpe = compute_rpe(ip_pred, ip_true, pred_mask)
    im_rpe = compute_rpe(im_pred, im_true, pred_mask)
    ip_mask = pred_mask & (np.abs(ip_true) > 1e-10)
    im_mask = pred_mask & (np.abs(im_true) > 1e-10)
    res_ip = np.where(pred_mask, ip_pred - ip_true, np.nan)
    res_im = np.where(pred_mask, im_pred - im_true, np.nan)

    p_mask, q_mask = np.abs(p_true) >= 5.0, np.abs(q_true) >= 5.0
    p_rpe = compute_rpe(p_pred, p_true, p_mask)
    q_rpe = compute_rpe(q_pred, q_true, q_mask)
    res_p = p_pred - p_true
    res_q = q_pred - q_true

    med_ip_bin = np.nanmedian(ip_rpe, axis=0)
    med_im_bin = np.nanmedian(im_rpe, axis=0)

    stats = {
        "n_samples": int(ps.shape[0]),
        "n_bins": int(n_bins),
        "L1_Iplus": float(np.mean(np.abs(ip_pred[pred_mask] - ip_true[pred_mask]))),
        "L1_Iminus": float(np.mean(np.abs(im_pred[pred_mask] - im_true[pred_mask]))),
        "median_RPE_Iplus": float(np.nanmedian(ip_rpe[ip_mask])),
        "median_RPE_Iminus": float(np.nanmedian(im_rpe[im_mask])),
        "mean_RPE_P": float(np.nanmean(p_rpe[p_mask])),
        "std_RPE_P": float(np.nanstd(p_rpe[p_mask])),
        "mean_RPE_Q": float(np.nanmean(q_rpe[q_mask])),
        "std_RPE_Q": float(np.nanstd(q_rpe[q_mask])),
        "mean_residual_P": float(np.mean(res_p)),
        "std_residual_P": float(np.std(res_p)),
        "mean_residual_Q": float(np.mean(res_q)),
        "std_residual_Q": float(np.std(res_q)),
    }
    with open(os.path.join(OUTPUT_DIR, "test_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    pd.DataFrame(
        {"bin_idx": np.arange(n_bins), "median_rpe_iplus": med_ip_bin, "median_rpe_iminus": med_im_bin}
    ).to_csv(os.path.join(OUTPUT_DIR, "median_rpe_per_bin.csv"), index=False)

    print_results_table(stats)

    n_show = min(MAX_HEATMAP_SAMPLES, ps.shape[0])
    plot_heatmap(
        os.path.join(OUTPUT_DIR, "residuals_heatmap_iplus.png"),
        res_ip[:n_show],
        "I+ residuals",
        "Residual",
    )
    plot_heatmap(
        os.path.join(OUTPUT_DIR, "residuals_heatmap_iminus.png"),
        res_im[:n_show],
        "I- residuals",
        "Residual",
    )

    example_idx = select_example_indices(
        ps.shape[0], EXAMPLES, res_ip + res_im, EXAMPLE_SELECTION
    )
    plot_lineshape_examples(
        os.path.join(OUTPUT_DIR, "lineshape_examples.png"),
        example_idx,
        ps,
        ip_true,
        im_true,
        ip_pred,
        im_pred,
        title_prefix=f"Sample ({EXAMPLE_SELECTION})",
    )

    print(f"\nOutputs written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
