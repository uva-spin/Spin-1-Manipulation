"""
Evaluate single_bin P/Q models on sample test events from create_sample_single_bin_data.py.

Supports:
  - Combined model (default): combined_bin_model.pth from ml/combine_single_bin_models.py
  - Per-bin checkpoints: binning_model_bin_XXXX.pth directory

Each test event NPZ contains a full spectrum plus ``P_bins`` / ``Q_bins`` ground truth.
Features: ``[p0, ps[j]]`` at each spectral bin ``j``.

Examples (from repo root):
  python Data_Creation/dulya_fit_v5/create_sample_single_bin_data.py --quick
  sbatch ml/train_single_bin_array.slurm
  sbatch ml/combine_single_bin_models.slurm

  python ml/test_single_bin_pq.py \\
      --sample-dir Data_Creation/dulya_fit_v5/sample_single_bin \\
      --combined-model single_bin_models/combined_bin_model.pth

Writes plots under ``<sample-dir>/test_pq_plots/``:
  pq_spectrum_examples.png   true vs pred P/Q (multi-event, like test-binning)
  residuals_heatmap_P.png / residuals_heatmap_Q.png
  events/<event>_pq.png      per-event stacked P and Q panels

Integrated lineshape totals use the same convention as ``ml/test-binning.py``:
``mean(P_bins)`` and ``mean(Q_bins)`` over modeled bins (CC-calibrated per-bin
targets averaged over the spectrum equals post-corrected integrated P/Q).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from single_bin import BinModel, load_bin_model_state_dict

DEFAULT_SAMPLE_DIR = (
    REPO_ROOT / "Data_Creation" / "dulya_fit_v5" / "sample_single_bin"
)


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


def build_feature_row(p0: float, ps_j: float, feature_names: List[str]) -> np.ndarray:
    columns = {
        "p0": float(p0),
        "ps": float(ps_j),
        "ps_at_burn_bin": float(ps_j),
    }
    missing = [name for name in feature_names if name not in columns]
    if missing:
        raise KeyError(f"Unsupported feature names: {missing}")
    return np.array([columns[name] for name in feature_names], dtype=np.float32)


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
    event = tb_mod.LineshapeEvent(
        polarization=p0,
        frequency=frequency,
        ps=ps,
        iplus=iplus,
        iminus=iminus,
        burn_bin_idx=burn_bin,
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

    n_bins = int(ps.size)
    end = int(bin_end) if bin_end is not None else n_bins
    pred_p = np.full(n_bins, np.nan, dtype=np.float64)
    pred_q = np.full(n_bins, np.nan, dtype=np.float64)

    for j in range(max(0, int(bin_start)), min(end, n_bins)):
        model_path = model_dir / f"binning_model_bin_{j}.pth"
        if not model_path.is_file():
            continue
        bundle = load_bin_model(model_path, device)
        x_raw = build_feature_row(p0, float(ps[j]), bundle["feature_names"])
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate single_bin P/Q models on sample test events"
    )
    p.add_argument(
        "--sample-dir",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="Directory from create_sample_single_bin_data.py",
    )
    p.add_argument(
        "--combined-model",
        type=Path,
        default=None,
        help="Path to combined_bin_model.pth (default: <model-dir>/combined_bin_model.pth)",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory with per-bin .pth files (used to locate combined model or --per-bin mode)",
    )
    p.add_argument(
        "--per-bin",
        action="store_true",
        help="Evaluate per-bin checkpoints instead of the combined model",
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
        help="Directory for plots (default: <sample-dir>/test_pq_plots)",
    )
    p.add_argument(
        "--examples",
        type=int,
        default=12,
        help="Number of events in the multi-panel pq_spectrum_examples.png",
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
    sample_dir = Path(args.sample_dir)
    test_dir = sample_dir / "test_events"
    model_dir = Path(args.model_dir) if args.model_dir is not None else sample_dir / "single_bin_models"
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
        combined_model, model_meta = tb.load_combined_model(str(combined_path), device)
        print(f"sample_dir={sample_dir}", flush=True)
        print(f"combined_model={combined_path}", flush=True)
        print(
            f"loaded_bins={combined_model.num_models}  "
            f"feature_names={combined_model.feature_names}  "
            f"target_mode={combined_model.target_mode}",
            flush=True,
        )
        print(f"device={device}", flush=True)
        predict_fn = lambda path: predict_event_combined(path, combined_model, tb, device=device)
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
        summaries.append({k: v for k, v in result.items() if not isinstance(v, np.ndarray)})
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
        plot_dir = Path(args.output_dir) if args.output_dir is not None else sample_dir / "test_pq_plots"
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


if __name__ == "__main__":
    main()
