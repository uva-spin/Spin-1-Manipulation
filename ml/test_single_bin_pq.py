"""
Evaluate single_bin models against integrated P/Q.

Two data modes:

1) Spectra NPZ (same file as ``ml/spectrum_pq.py``) — default comparison path.
   Held-out split uses the same seed / val / test fractions as spectrum_pq.
   Writes spectrum_pq-style metrics and plots:
     test_metrics.json, rpe_by_P_total_range.csv, rpe_by_p0_range.csv
     plots/pred_vs_true.png
     plots/rpe_by_polarization_range.png
     plots/residuals_by_polarization_range.png
     plots/example_manipulated_signals_pq.png
     plots/examples/*.png

2) Sample test events from ``create_sample_single_bin_data.py`` (``--sample-dir``).
   Per-bin P/Q spectra plots under ``<sample-dir>/test_pq_plots/``.

Supports:
  - Combined model (default): combined_bin_model.pth
  - Per-bin checkpoints: binning_model_bin_XXXX.pth directory (sample-dir mode only)

Examples (from repo root)::

  python ml/test_single_bin_pq.py \\
      --spectra Data_Creation/dae_voigt_burn_spectra/spectra.npz \\
      --combined-model ml/models/combined_bin_model.pth \\
      --output-dir ml/single_bin_pq_results

  python ml/test_single_bin_pq.py \\
      --sample-dir Data_Creation/rivanna/sample_single_bin \\
      --combined-model single_bin_models/combined_bin_model.pth
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "ml"
RIVANNA_DIR = REPO_ROOT / "Data_Creation" / "rivanna"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(RIVANNA_DIR) not in sys.path:
    sys.path.insert(0, str(RIVANNA_DIR))

import spectrum_pq as spq
from pq_calibration import calibrated_pq_spectrum, load_pq_calibration
from single_bin import (
    BinModel,
    build_feature_row,
    event_manipulation_features,
    load_bin_model_state_dict,
)

DEFAULT_SAMPLE_DIR = (
    REPO_ROOT / "Data_Creation" / "rivanna" / "sample_single_bin"
)
DEFAULT_SPECTRA_OUTPUT_DIR = ML_DIR / "single_bin_pq_results"
DEFAULT_SEED = spq.SEED
DEFAULT_VAL_FRAC = spq.VAL_FRAC
DEFAULT_TEST_FRAC = spq.TEST_FRAC


def _load_test_binning_module():
    path = REPO_ROOT / "ml" / "test-binning.py"
    spec = importlib.util.spec_from_file_location("test_binning", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["test_binning"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_bin_model(model_path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(model_path, map_location=device, weights_only=False)
    feature_names = list(payload.get("feature_names", ["p0", "ps"]))
    input_dim = int(payload.get("input_dim", len(feature_names)))
    hidden_dim = int(payload.get("hidden_dim", 256))
    model = BinModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    load_bin_model_state_dict(model, payload["model_state_dict"])
    model.eval()
    x_mean = torch.as_tensor(payload["X_mean"], dtype=torch.float32, device=device)
    x_std = torch.as_tensor(payload["X_std"], dtype=torch.float32, device=device).clamp_min(
        1e-12
    )
    return {
        "model": model,
        "x_mean": x_mean,
        "x_std": x_std,
        "feature_names": feature_names,
        "P_mean": float(payload["P_mean"]),
        "P_std": float(payload["P_std"]),
        "Q_mean": float(payload["Q_mean"]),
        "Q_std": float(payload["Q_std"]),
    }


def _integrated_totals(
    pred_p: np.ndarray,
    pred_q: np.ndarray,
    true_p: np.ndarray,
    true_q: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Integrated P/Q from mean of CC-calibrated per-bin spectra (test-binning convention)."""
    true_p_total = float(np.mean(true_p[mask]))
    true_q_total = float(np.mean(true_q[mask]))
    pred_p_total = float(np.mean(pred_p[mask]))
    pred_q_total = float(np.mean(pred_q[mask]))
    out = {
        "true_P_total": true_p_total,
        "true_Q_total": true_q_total,
        "pred_P_total": pred_p_total,
        "pred_Q_total": pred_q_total,
        "P_total_residual": pred_p_total - true_p_total,
        "Q_total_residual": pred_q_total - true_q_total,
    }
    if abs(true_p_total) > 1e-10:
        out["P_total_rpe_pct"] = float(
            abs(out["P_total_residual"] / true_p_total) * 100.0
        )
    if abs(true_q_total) > 1e-10:
        out["Q_total_rpe_pct"] = float(
            abs(out["Q_total_residual"] / true_q_total) * 100.0
        )
    return out


def _integrated_truth_from_event(
    p_bins: np.ndarray,
    q_bins: np.ndarray,
    event_path: Path,
) -> tuple[float, float]:
    """Prefer stored P_total/Q_total; fall back to mean of per-bin calibrated spectra."""
    with np.load(event_path, allow_pickle=False) as data:
        if "P_total" in data.files and "Q_total" in data.files:
            return (
                float(np.asarray(data["P_total"]).reshape(())),
                float(np.asarray(data["Q_total"]).reshape(())),
            )
    return float(np.mean(p_bins)), float(np.mean(q_bins))


def _metrics(
    pred_p: np.ndarray,
    pred_q: np.ndarray,
    true_p: np.ndarray,
    true_q: np.ndarray,
    *,
    true_p_total: float | None = None,
    true_q_total: float | None = None,
) -> Dict[str, float]:
    mask = np.isfinite(pred_p) & np.isfinite(pred_q)
    if not np.any(mask):
        raise RuntimeError("No finite predictions produced")
    err_p = pred_p[mask] - true_p[mask]
    err_q = pred_q[mask] - true_q[mask]
    integrated = _integrated_totals(pred_p, pred_q, true_p, true_q, mask)
    if true_p_total is not None:
        integrated["true_P_total"] = float(true_p_total)
        integrated["P_total_residual"] = integrated["pred_P_total"] - float(true_p_total)
        if abs(float(true_p_total)) > 1e-10:
            integrated["P_total_rpe_pct"] = float(
                abs(integrated["P_total_residual"] / float(true_p_total)) * 100.0
            )
        elif "P_total_rpe_pct" in integrated:
            del integrated["P_total_rpe_pct"]
    if true_q_total is not None:
        integrated["true_Q_total"] = float(true_q_total)
        integrated["Q_total_residual"] = integrated["pred_Q_total"] - float(true_q_total)
        if abs(float(true_q_total)) > 1e-10:
            integrated["Q_total_rpe_pct"] = float(
                abs(integrated["Q_total_residual"] / float(true_q_total)) * 100.0
            )
        elif "Q_total_rpe_pct" in integrated:
            del integrated["Q_total_rpe_pct"]
    return {
        "n_bins_modeled": int(mask.sum()),
        "l1_P": float(np.mean(np.abs(err_p))),
        "l1_Q": float(np.mean(np.abs(err_q))),
        "r2_P": float(
            1.0 - np.sum(err_p**2) / (np.sum((true_p[mask] - np.mean(true_p[mask])) ** 2) + 1e-12)
        ),
        "r2_Q": float(
            1.0 - np.sum(err_q**2) / (np.sum((true_q[mask] - np.mean(true_q[mask])) ** 2) + 1e-12)
        ),
        **integrated,
    }


def event_from_npz(event_path: Path, tb_mod) -> tuple[Any, np.ndarray, np.ndarray, float, dict]:
    with np.load(event_path, allow_pickle=False) as data:
        ps = np.asarray(data["ps"], dtype=np.float32)
        iplus = np.asarray(data["iplus"], dtype=np.float32)
        iminus = np.asarray(data["iminus"], dtype=np.float32)
        frequency = np.asarray(data["frequency"], dtype=np.float32)
        p_bins = np.asarray(data["P_bins"], dtype=np.float32)
        q_bins = np.asarray(data["Q_bins"], dtype=np.float32)
        p0 = float(np.asarray(data["p0"]).reshape(()))
        center_bin = int(np.asarray(data["center_bin"]).reshape(()))
        step = int(np.asarray(data["step"]).reshape(()))
        meta = json.loads(str(np.asarray(data["meta_json"]).reshape(())))

    burn_bin = center_bin if meta.get("manipulation_mode") == "ssrf" else None
    source = 0 if meta.get("manipulation_mode") == "ssrf" else (1 if meta.get("manipulation_mode") == "afp" else 2)
    gamma_rf_val = float(np.asarray(data["gamma_rf"]).reshape(())) if "gamma_rf" in data.files else None
    n_steps_val = float(step) if step > 0 else None
    gamma_rf, n_steps = event_manipulation_features(
        source=source,
        gamma_rf=gamma_rf_val,
        n_steps=n_steps_val,
    )
    event = tb_mod.LineshapeEvent(
        polarization=p0,
        frequency=frequency,
        ps=ps,
        iplus=iplus,
        iminus=iminus,
        burn_bin_idx=burn_bin,
        gamma_rf=gamma_rf,
        n_steps=n_steps,
        burn_step_norm=float(step) / 100.0 if step > 0 else 0.0,
    )
    return event, p_bins, q_bins, p0, meta


def predict_event_combined(
    event_path: Path,
    combined_model,
    tb_mod,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    event, p_bins, q_bins, p0, meta = event_from_npz(event_path, tb_mod)
    n_bins = int(event.num_bins)
    with torch.no_grad():
        p_pred, q_pred = combined_model.predict_events([event], spectrum_bins=n_bins)
    pred_p = np.asarray(p_pred[0], dtype=np.float64)
    pred_q = np.asarray(q_pred[0], dtype=np.float64)
    true_p_total, true_q_total = _integrated_truth_from_event(
        p_bins, q_bins, event_path
    )
    stats = _metrics(
        pred_p,
        pred_q,
        p_bins,
        q_bins,
        true_p_total=true_p_total,
        true_q_total=true_q_total,
    )
    return {
        "event": event_path.name,
        "manipulation_mode": meta.get("manipulation_mode"),
        "p0": p0,
        **stats,
        "pred_P": pred_p,
        "pred_Q": pred_q,
        "true_P": p_bins,
        "true_Q": q_bins,
        "ps": np.asarray(event.ps, dtype=np.float64),
        "frequency": np.asarray(event.frequency, dtype=np.float64),
    }


def predict_event_per_bin(
    event_path: Path,
    model_dir: Path,
    *,
    device: torch.device,
    bin_start: int = 0,
    bin_end: int | None = None,
) -> Dict[str, Any]:
    with np.load(event_path, allow_pickle=False) as data:
        ps = np.asarray(data["ps"], dtype=np.float32)
        frequency = np.asarray(data["frequency"], dtype=np.float32)
        p_bins = np.asarray(data["P_bins"], dtype=np.float32)
        q_bins = np.asarray(data["Q_bins"], dtype=np.float32)
        p0 = float(np.asarray(data["p0"]).reshape(()))
        meta = json.loads(str(np.asarray(data["meta_json"]).reshape(())))
        source = 0 if meta.get("manipulation_mode") == "ssrf" else (
            1 if meta.get("manipulation_mode") == "afp" else 2
        )
        gamma_rf_val = (
            float(np.asarray(data["gamma_rf"]).reshape(()))
            if "gamma_rf" in data.files
            else None
        )
        step = int(np.asarray(data["step"]).reshape(())) if "step" in data.files else 0
        gamma_rf, n_steps = event_manipulation_features(
            source=source,
            gamma_rf=gamma_rf_val,
            n_steps=float(step) if step > 0 else None,
        )

    n_bins = int(ps.size)
    end = int(bin_end) if bin_end is not None else n_bins
    pred_p = np.full(n_bins, np.nan, dtype=np.float64)
    pred_q = np.full(n_bins, np.nan, dtype=np.float64)

    for j in range(max(0, int(bin_start)), min(end, n_bins)):
        model_path = model_dir / f"binning_model_bin_{j}.pth"
        if not model_path.is_file():
            continue
        bundle = load_bin_model(model_path, device)
        x_raw = build_feature_row(
            bundle["feature_names"],
            gamma_rf=gamma_rf,
            n_steps=n_steps,
            ps=float(ps[j]),
            p0=p0,
        )
        x = torch.as_tensor(x_raw, dtype=torch.float32, device=device).reshape(1, -1)
        x_norm = (x - bundle["x_mean"]) / bundle["x_std"]
        with torch.no_grad():
            p_hat, q_hat = bundle["model"](x_norm)
        pred_p[j] = float(p_hat.item() * bundle["P_std"] + bundle["P_mean"])
        pred_q[j] = float(q_hat.item() * bundle["Q_std"] + bundle["Q_mean"])

    true_p_total, true_q_total = _integrated_truth_from_event(
        p_bins, q_bins, event_path
    )
    stats = _metrics(
        pred_p,
        pred_q,
        p_bins,
        q_bins,
        true_p_total=true_p_total,
        true_q_total=true_q_total,
    )
    return {
        "event": event_path.name,
        "manipulation_mode": meta.get("manipulation_mode"),
        "p0": p0,
        **stats,
        "pred_P": pred_p,
        "pred_Q": pred_q,
        "true_P": p_bins,
        "true_Q": q_bins,
        "ps": np.asarray(ps, dtype=np.float64),
        "frequency": np.asarray(frequency, dtype=np.float64),
    }


def _x_axis(result: Dict[str, Any]) -> np.ndarray:
    freq = np.asarray(result.get("frequency", []), dtype=np.float64)
    n = int(np.asarray(result["true_P"]).size)
    if freq.size == n:
        return freq
    return np.arange(n, dtype=np.float64)


def plot_pq_spectrum_examples(
    out_path: Path,
    results: Sequence[Dict[str, Any]],
    *,
    title_prefix: str = "Event",
) -> None:
    """True vs predicted P and Q spectra (test-binning lineshape style)."""
    if not results:
        return
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3.2 * len(results)), squeeze=False)
    xlabel = "Frequency R"
    for ax, result in zip(axes[:, 0], results):
        x = _x_axis(result)
        if not (x.size > 1 and np.max(np.abs(x)) <= 6.0):
            xlabel = "Bin index"
        true_p = np.asarray(result["true_P"], dtype=np.float64)
        true_q = np.asarray(result["true_Q"], dtype=np.float64)
        pred_p = np.asarray(result["pred_P"], dtype=np.float64)
        pred_q = np.asarray(result["pred_Q"], dtype=np.float64)
        ps = np.asarray(result.get("ps", []), dtype=np.float64)

        if ps.size == x.size:
            ax.plot(x, ps, "k-", lw=1.0, alpha=0.45, label="Ps")

        ax.plot(x, true_p, color="#d55e00", alpha=0.45, lw=2.0, label="True P")
        ax.plot(x, pred_p, color="#d55e00", ls="--", lw=1.4, label="Pred P")
        ax.plot(x, true_q, color="#0072b2", alpha=0.45, lw=2.0, label="True Q")
        ax.plot(x, pred_q, color="#0072b2", ls="--", lw=1.4, label="Pred Q")

        mode = result.get("manipulation_mode", "?")
        p0 = float(result.get("p0", float("nan")))
        name = str(result.get("event", "")).replace(".npz", "")
        p_tot = float(result.get("pred_P_total", float("nan")))
        p_true_tot = float(result.get("true_P_total", float("nan")))
        q_tot = float(result.get("pred_Q_total", float("nan")))
        q_true_tot = float(result.get("true_Q_total", float("nan")))
        ax.set_title(
            f"{title_prefix}: {name}  ({mode}, p0={p0:.3f})\n"
            f"P_tot pred/true={p_tot:.4f}/{p_true_tot:.4f}  "
            f"Q_tot pred/true={q_tot:.4f}/{q_true_tot:.4f}"
        )
        ax.set_ylabel("Polarization")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)

    axes[-1, 0].set_xlabel(xlabel)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pq_spectrum_panels(out_path: Path, result: Dict[str, Any]) -> None:
    """Separate stacked P and Q panels for one event."""
    x = _x_axis(result)
    xlabel = "Frequency R" if x.size > 1 and np.max(np.abs(x)) <= 6.0 else "Bin index"
    true_p = np.asarray(result["true_P"], dtype=np.float64)
    true_q = np.asarray(result["true_Q"], dtype=np.float64)
    pred_p = np.asarray(result["pred_P"], dtype=np.float64)
    pred_q = np.asarray(result["pred_Q"], dtype=np.float64)
    mode = result.get("manipulation_mode", "?")
    p0 = float(result.get("p0", float("nan")))
    name = str(result.get("event", "")).replace(".npz", "")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, true_y, pred_y, label, color in (
        (axes[0], true_p, pred_p, "P", "#d55e00"),
        (axes[1], true_q, pred_q, "Q", "#0072b2"),
    ):
        ax.plot(x, true_y, color=color, alpha=0.55, lw=2.0, label=f"True {label}")
        ax.plot(x, pred_y, color=color, ls="--", lw=1.4, label=f"Pred {label}")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    p_tot = float(result.get("pred_P_total", float("nan")))
    p_true_tot = float(result.get("true_P_total", float("nan")))
    q_tot = float(result.get("pred_Q_total", float("nan")))
    q_true_tot = float(result.get("true_Q_total", float("nan")))
    p_rpe = result.get("P_total_rpe_pct")
    q_rpe = result.get("Q_total_rpe_pct")
    p_rpe_s = f"  RPE={float(p_rpe):.2f}%" if p_rpe is not None else ""
    q_rpe_s = f"  RPE={float(q_rpe):.2f}%" if q_rpe is not None else ""
    axes[0].set_title(
        f"{name}  ({mode}, p0={p0:.3f})\n"
        f"P_tot pred={p_tot:.4f}  true={p_true_tot:.4f}{p_rpe_s}  |  "
        f"Q_tot pred={q_tot:.4f}  true={q_true_tot:.4f}{q_rpe_s}"
    )
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pq_heatmap(out_path: Path, data: np.ndarray, title: str, label: str) -> None:
    plt.figure(figsize=(12, max(3.0, 0.35 * data.shape[0])))
    plt.imshow(data, aspect="auto", cmap="coolwarm")
    plt.colorbar(label=label)
    plt.xlabel("Bin index")
    plt.ylabel("Event index")
    plt.title(title)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def select_example_indices(
    n_events: int,
    n_examples: int,
    residuals: np.ndarray,
    mode: str,
) -> np.ndarray:
    n = min(n_examples, n_events)
    if n <= 0:
        return np.array([], dtype=int)
    if mode == "sequential":
        return np.arange(n, dtype=int)
    if mode == "spread":
        if n == 1:
            return np.array([n_events // 2], dtype=int)
        return np.linspace(0, n_events - 1, n, dtype=int).astype(int)
    err = np.nanmean(np.abs(residuals), axis=1)
    order = np.argsort(err)
    ranks = np.linspace(0, n_events - 1, n)
    return order[np.round(ranks).astype(int)]


def write_pq_plots(
    output_dir: Path,
    results: Sequence[Dict[str, Any]],
    *,
    examples: int,
    example_selection: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_p = np.stack([np.asarray(r["pred_P"], dtype=np.float64) for r in results], axis=0)
    pred_q = np.stack([np.asarray(r["pred_Q"], dtype=np.float64) for r in results], axis=0)
    true_p = np.stack([np.asarray(r["true_P"], dtype=np.float64) for r in results], axis=0)
    true_q = np.stack([np.asarray(r["true_Q"], dtype=np.float64) for r in results], axis=0)
    mask = np.isfinite(pred_p) & np.isfinite(pred_q)
    res_p = np.where(mask, pred_p - true_p, np.nan)
    res_q = np.where(mask, pred_q - true_q, np.nan)

    plot_pq_heatmap(output_dir / "residuals_heatmap_P.png", res_p, "P residuals", "P residual")
    plot_pq_heatmap(output_dir / "residuals_heatmap_Q.png", res_q, "Q residuals", "Q residual")

    example_idx = select_example_indices(
        len(results),
        int(examples),
        res_p + res_q,
        str(example_selection),
    )
    selected = [results[int(i)] for i in example_idx]
    plot_pq_spectrum_examples(
        output_dir / "pq_spectrum_examples.png",
        selected,
        title_prefix=f"Event ({example_selection})",
    )

    per_event_dir = output_dir / "events"
    for result in results:
        stem = str(result["event"]).replace(".npz", "")
        plot_pq_spectrum_panels(per_event_dir / f"{stem}_pq.png", result)


def _split_indices(
    n: int,
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match ``spectrum_pq.prepare_datasets`` index splitting exactly."""
    if n < 6:
        raise ValueError(f"Need at least 6 samples for train/val/test split, got {n}")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * float(test_frac))))
    n_val = max(1, int(round(n * float(val_frac))))
    if n_test + n_val >= n - 1:
        n_test = max(1, n // 5)
        n_val = max(1, n // 5)
    test_idx = perm[:n_test].astype(np.int64)
    val_idx = perm[n_test : n_test + n_val].astype(np.int64)
    train_idx = perm[n_test + n_val :].astype(np.int64)
    if train_idx.size < 2:
        raise RuntimeError("Train split too small; lower --val-frac / --test-frac")
    return train_idx, val_idx, test_idx


def event_from_spectrum_row(
    arrays: dict[str, np.ndarray],
    index: int,
    tb_mod,
    *,
    frequency: np.ndarray,
) -> Any:
    """Build a LineshapeEvent from one row of spectra.npz."""
    i = int(index)
    ip = np.asarray(arrays["spectra"][i, 0], dtype=np.float32)
    im = np.asarray(arrays["spectra"][i, 1], dtype=np.float32)
    p0 = float(np.asarray(arrays["p0"][i]).reshape(())) if "p0" in arrays else 0.0
    source = (
        int(np.asarray(arrays["source"][i]).reshape(()))
        if "source" in arrays
        else 0
    )
    applied = (
        float(np.asarray(arrays["applied_power"][i]).reshape(()))
        if "applied_power" in arrays
        else None
    )
    n_steps_raw = float(np.asarray(arrays["n_steps"][i]).reshape(()))
    gamma_rf, n_steps = event_manipulation_features(
        source=source,
        applied_power=applied,
        n_steps=n_steps_raw,
    )
    center = (
        int(np.asarray(arrays["center_bin"][i]).reshape(()))
        if "center_bin" in arrays
        else None
    )
    return tb_mod.LineshapeEvent(
        polarization=p0,
        frequency=np.asarray(frequency, dtype=np.float32),
        ps=(ip + im).astype(np.float32, copy=False),
        iplus=ip,
        iminus=im,
        burn_bin_idx=center,
        gamma_rf=gamma_rf,
        n_steps=n_steps,
        burn_step_norm=float(n_steps) / 100.0 if n_steps > 0 else 0.0,
    )


def _integrated_from_bin_preds(
    pred0: np.ndarray,
    pred1: np.ndarray,
    *,
    p0: float,
    target_mode: str,
    calibration: dict[str, Any] | None,
) -> tuple[float, float]:
    """Convert per-bin model outputs to integrated P_total / Q_total."""
    out0 = np.asarray(pred0, dtype=np.float64).reshape(-1)
    out1 = np.asarray(pred1, dtype=np.float64).reshape(-1)
    mask = np.isfinite(out0) & np.isfinite(out1)
    if not np.any(mask):
        return float("nan"), float("nan")
    if target_mode == "pq":
        return float(np.mean(out0[mask])), float(np.mean(out1[mask]))
    ip = out0[mask]
    im = out1[mask]
    ps = ip + im
    p_bins, q_bins = calibrated_pq_spectrum(
        ps,
        ip,
        im,
        float(p0),
        calibration=calibration,
        num_bins=int(out0.size),
    )
    return float(np.mean(p_bins)), float(np.mean(q_bins))


def _scalar_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred_a = np.asarray(pred, dtype=np.float64).reshape(-1)
    true_a = np.asarray(true, dtype=np.float64).reshape(-1)
    err = pred_a - true_a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true_a - np.mean(true_a)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else float("nan")
    rpe = spq.compute_rpe(pred_a, true_a)
    rpe_s = spq._summary_stats(rpe)
    res_s = spq._summary_stats(err)
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


def evaluate_combined_on_spectra(
    combined_model,
    arrays: dict[str, np.ndarray],
    test_idx: np.ndarray,
    tb_mod,
    *,
    batch_size: int = 32,
    max_test: int | None = None,
) -> dict[str, Any]:
    """Predict integrated P/Q on held-out spectra.npz rows with the combined model."""
    test_idx = np.asarray(test_idx, dtype=np.int64).reshape(-1)
    if max_test is not None and int(max_test) > 0:
        test_idx = test_idx[: int(max_test)]
    n_test = int(test_idx.size)
    num_bins = int(np.asarray(arrays["num_bins"]).reshape(()))
    frequency = np.linspace(-6.0, 6.0, num_bins, dtype=np.float32)
    target_mode = str(combined_model.target_mode)
    calibration = (
        None
        if target_mode == "pq"
        else load_pq_calibration(num_bins=num_bins)
    )

    true_p = np.asarray(arrays["P_total"], dtype=np.float64).reshape(-1)[test_idx]
    true_q = np.asarray(arrays["Q_total"], dtype=np.float64).reshape(-1)[test_idx]
    p0_all = (
        np.asarray(arrays["p0"], dtype=np.float64).reshape(-1)
        if "p0" in arrays
        else true_p.copy()
    )
    p0_test = p0_all[test_idx]

    pred_p = np.empty(n_test, dtype=np.float64)
    pred_q = np.empty(n_test, dtype=np.float64)
    bs = max(1, int(batch_size))

    for start in range(0, n_test, bs):
        stop = min(start + bs, n_test)
        batch_idx = test_idx[start:stop]
        events = [
            event_from_spectrum_row(arrays, int(gi), tb_mod, frequency=frequency)
            for gi in batch_idx
        ]
        out0, out1 = combined_model.predict_events(events, spectrum_bins=num_bins)
        for local, gi in enumerate(batch_idx):
            pp, pq = _integrated_from_bin_preds(
                out0[local],
                out1[local],
                p0=float(p0_all[int(gi)]),
                target_mode=target_mode,
                calibration=calibration,
            )
            pred_p[start + local] = pp
            pred_q[start + local] = pq
        if start == 0 or stop == n_test or (stop % max(bs * 20, 1) == 0):
            print(f"  evaluated {stop}/{n_test}", flush=True)

    p_m = _scalar_metrics(pred_p, true_p)
    q_m = _scalar_metrics(pred_q, true_q)
    range_by_p = spq.polarization_range_stats(
        pred_p,
        true_p,
        pred_q,
        true_q,
        pol_ref=true_p,
        ref_name="abs_P_total",
    )
    range_by_p0 = spq.polarization_range_stats(
        pred_p,
        true_p,
        pred_q,
        true_q,
        pol_ref=p0_test,
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
        "pred_P": pred_p,
        "pred_Q": pred_q,
        "true_P": true_p,
        "true_Q": true_q,
        "range_stats_by_P": range_by_p,
        "range_stats_by_p0": range_by_p0,
        "test_idx": test_idx,
        "p0_test": p0_test.astype(np.float32),
        "target_mode": target_mode,
    }


def write_spectra_eval_outputs(
    output_dir: Path,
    *,
    spectra_path: Path,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
    stats: dict[str, Any],
    n_examples: int,
    seed: int,
    no_plots: bool,
) -> None:
    """Write the same metric/plot artifacts as ``spectrum_pq`` test mode."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    metrics_path = output_dir / "test_metrics.json"
    spq._write_metrics(
        metrics_path,
        spectra_path=spectra_path,
        best_val=None,
        metrics=metrics,
        stats=stats,
    )
    spq.save_range_stats_csv(
        list(metrics["range_stats_by_P"]),
        output_dir / "rpe_by_P_total_range.csv",
    )
    if metrics.get("range_stats_by_p0"):
        spq.save_range_stats_csv(
            list(metrics["range_stats_by_p0"]),
            output_dir / "rpe_by_p0_range.csv",
        )
    print(f"Saved metrics -> {metrics_path}", flush=True)

    if no_plots:
        return

    plot_paths = spq.save_plots(None, metrics, plots_dir)
    example_paths = spq.save_example_signal_plots(
        arrays,
        stats,
        metrics,
        plots_dir,
        n_examples=int(n_examples),
        seed=int(seed),
    )
    plot_paths.extend(example_paths)
    print(
        f"Wrote {len(plot_paths)} plots -> {plots_dir} "
        f"({len(example_paths)} example signal plots)",
        flush=True,
    )


def run_spectra_mode(args: argparse.Namespace) -> None:
    """Evaluate combined single_bin model on the same spectra.npz as spectrum_pq."""
    spectra_path = spq.resolve_spectra_path(args.spectra)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else DEFAULT_SPECTRA_OUTPUT_DIR
    )
    device = torch.device(args.device)

    combined_path = args.combined_model
    if combined_path is None:
        model_dir = (
            Path(args.model_dir)
            if args.model_dir is not None
            else ML_DIR / "models"
        )
        combined_path = model_dir / "combined_bin_model.pth"
    combined_path = Path(combined_path)
    if not combined_path.is_file():
        raise FileNotFoundError(
            f"Combined model not found at {combined_path}; "
            "pass --combined-model PATH"
        )

    print(f"Loading {spectra_path}", flush=True)
    arrays = spq.load_spectrum_pq_npz(spectra_path)
    train_idx, val_idx, test_idx = _split_indices(
        int(arrays["P_total"].shape[0]),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )

    tb = _load_test_binning_module()
    combined_model, _model_meta = tb.load_combined_model(str(combined_path), device)
    print(f"combined_model={combined_path}", flush=True)
    print(
        f"loaded_bins={combined_model.num_models}  "
        f"feature_names={combined_model.feature_names}  "
        f"target_mode={combined_model.target_mode}",
        flush=True,
    )
    print(
        f"N={arrays['P_total'].shape[0]}  "
        f"train/val/test={train_idx.size}/{val_idx.size}/{test_idx.size}  "
        f"(same-file split as spectrum_pq, seed={args.seed})  device={device}",
        flush=True,
    )

    metrics = evaluate_combined_on_spectra(
        combined_model,
        arrays,
        test_idx,
        tb,
        batch_size=int(args.batch_size),
        max_test=args.max_test,
    )
    used_test_idx = np.asarray(metrics["test_idx"], dtype=np.int64)
    stats = {
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(used_test_idx.size),
        "test_idx": used_test_idx,
        "p0_test": metrics["p0_test"],
    }

    print(
        f"Test  P: MAE={float(metrics['P_mae']):.6g}  "
        f"RMSE={float(metrics['P_rmse']):.6g}  R²={float(metrics['P_r2']):.4f}  "
        f"RPE med/mean/std="
        f"{float(metrics['P_rpe_median']):.3f}/"
        f"{float(metrics['P_rpe_mean']):.3f}/"
        f"{float(metrics['P_rpe_std']):.3f}%",
        flush=True,
    )
    print(
        f"Test  Q: MAE={float(metrics['Q_mae']):.6g}  "
        f"RMSE={float(metrics['Q_rmse']):.6g}  R²={float(metrics['Q_r2']):.4f}  "
        f"RPE med/mean/std="
        f"{float(metrics['Q_rpe_median']):.3f}/"
        f"{float(metrics['Q_rpe_mean']):.3f}/"
        f"{float(metrics['Q_rpe_std']):.3f}%",
        flush=True,
    )
    spq.print_range_stats_table(
        list(metrics["range_stats_by_P"]),
        title="RPE / residuals by |P_total| range",
    )
    if metrics.get("range_stats_by_p0"):
        spq.print_range_stats_table(
            list(metrics["range_stats_by_p0"]),
            title="RPE / residuals by |p0| range",
        )

    write_spectra_eval_outputs(
        output_dir,
        spectra_path=spectra_path,
        arrays=arrays,
        metrics=metrics,
        stats=stats,
        n_examples=int(args.n_examples),
        seed=int(args.seed),
        no_plots=bool(args.no_plots),
    )


def run_sample_dir_mode(args: argparse.Namespace) -> None:
    """Legacy path: evaluate on create_sample_single_bin_data.py test events."""
    sample_dir = Path(args.sample_dir)
    test_dir = sample_dir / "test_events"
    model_dir = (
        Path(args.model_dir) if args.model_dir is not None else sample_dir / "single_bin_models"
    )
    device = torch.device(args.device)

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {test_dir}; run create_sample_single_bin_data.py first"
        )

    events = sorted(test_dir.glob("*.npz"))
    if not events:
        raise FileNotFoundError(f"No test events under {test_dir}")

    combined_path = args.combined_model
    if combined_path is None and not args.per_bin:
        combined_path = model_dir / "combined_bin_model.pth"

    use_combined = not args.per_bin
    if use_combined:
        if combined_path is None or not Path(combined_path).is_file():
            raise FileNotFoundError(
                f"Combined model not found at {combined_path}; "
                "run ml/combine_single_bin_models.py or pass --per-bin"
            )
        tb = _load_test_binning_module()
        combined_model, _model_meta = tb.load_combined_model(str(combined_path), device)
        print(f"sample_dir={sample_dir}", flush=True)
        print(f"combined_model={combined_path}", flush=True)
        print(
            f"loaded_bins={combined_model.num_models}  "
            f"feature_names={combined_model.feature_names}  "
            f"target_mode={combined_model.target_mode}",
            flush=True,
        )
        print(f"device={device}", flush=True)
        predict_fn = lambda path: predict_event_combined(
            path, combined_model, tb, device=device
        )
    else:
        print(f"sample_dir={sample_dir}", flush=True)
        print(f"model_dir={model_dir}  (per-bin mode)", flush=True)
        print(f"device={device}", flush=True)
        predict_fn = lambda path: predict_event_per_bin(
            path,
            model_dir,
            device=device,
            bin_start=int(args.bin_start),
            bin_end=args.bin_end,
        )

    summaries: List[Dict[str, Any]] = []
    full_results: List[Dict[str, Any]] = []
    for path in events:
        result = predict_fn(path)
        full_results.append(result)
        summaries.append(
            {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        )
        p_rpe = result.get("P_total_rpe_pct")
        q_rpe = result.get("Q_total_rpe_pct")
        p_rpe_s = f"{float(p_rpe):.2f}%" if p_rpe is not None else "n/a"
        q_rpe_s = f"{float(q_rpe):.2f}%" if q_rpe is not None else "n/a"
        print(
            f"{result['event']:40s}  mode={result['manipulation_mode']!s:7s}  "
            f"p0={result['p0']:.3f}  bins={result['n_bins_modeled']:3d}  "
            f"L1_P={result['l1_P']:.5f}  L1_Q={result['l1_Q']:.5f}  "
            f"R2_P={result['r2_P']:.4f}  R2_Q={result['r2_Q']:.4f}",
            flush=True,
        )
        print(
            f"{'':40s}  P_tot pred={result['pred_P_total']:.5f}  "
            f"true={result['true_P_total']:.5f}  "
            f"res={result['P_total_residual']:+.5f}  RPE={p_rpe_s}  |  "
            f"Q_tot pred={result['pred_Q_total']:.5f}  "
            f"true={result['true_Q_total']:.5f}  "
            f"res={result['Q_total_residual']:+.5f}  RPE={q_rpe_s}",
            flush=True,
        )

    out_path = sample_dir / "test_single_bin_pq_summary.json"
    out_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    if not args.no_plots:
        plot_dir = (
            Path(args.output_dir)
            if args.output_dir is not None
            else sample_dir / "test_pq_plots"
        )
        write_pq_plots(
            plot_dir,
            full_results,
            examples=int(args.examples),
            example_selection=str(args.example_selection),
        )
        print(f"Wrote plots to {plot_dir}/", flush=True)
        print(f"  {plot_dir / 'pq_spectrum_examples.png'}", flush=True)
        print(f"  {plot_dir / 'residuals_heatmap_P.png'}", flush=True)
        print(f"  {plot_dir / 'residuals_heatmap_Q.png'}", flush=True)
        print(f"  {plot_dir / 'events'}/*.png", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate single_bin models on the same spectra.npz as spectrum_pq "
            "(default), or on sample test events (--sample-events)."
        )
    )
    p.add_argument(
        "--spectra",
        type=Path,
        default=None,
        help=(
            "Path to spectra.npz (default: same candidates as spectrum_pq.py, "
            "Data_Creation/dae_voigt_burn_spectra/spectra.npz)"
        ),
    )
    p.add_argument(
        "--sample-events",
        action="store_true",
        help="Use sample_single_bin test_events instead of spectra.npz",
    )
    p.add_argument(
        "--sample-dir",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="Directory from create_sample_single_bin_data.py (--sample-events)",
    )
    p.add_argument(
        "--combined-model",
        type=Path,
        default=None,
        help="Path to combined_bin_model.pth",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory with per-bin .pth / combined model",
    )
    p.add_argument(
        "--per-bin",
        action="store_true",
        help="Evaluate per-bin checkpoints (--sample-events only)",
    )
    p.add_argument("--bin-start", type=int, default=0)
    p.add_argument("--bin-end", type=int, default=None)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: ml/single_bin_pq_results for spectra mode, "
            "<sample-dir>/test_pq_plots for sample-events)"
        ),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Event batch size for spectra.npz evaluation",
    )
    p.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional cap on held-out test events (smoke / quick runs)",
    )
    p.add_argument(
        "--n-examples",
        type=int,
        default=6,
        help="Manipulated-signal example plots (spectra mode)",
    )
    p.add_argument(
        "--examples",
        type=int,
        default=12,
        help="Events in pq_spectrum_examples.png (--sample-events)",
    )
    p.add_argument(
        "--example-selection",
        choices=("stratified", "sequential", "spread"),
        default="stratified",
    )
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_events:
        run_sample_dir_mode(args)
        return
    if args.per_bin:
        raise SystemExit("--per-bin requires --sample-events")
    run_spectra_mode(args)


if __name__ == "__main__":
    main()
