"""
Evaluate the global SpectrumSplitNet against holdout spectra and optional 500-bin baseline.

Run:
  python ml/test_spectrum_split.py \\
      --model ml/models/spectrum_split_model.pth \\
      --data Data_Creation/dulya_fit_v2/data/spectrum_train/spectrum_train.npz

Realistic manipulation events (one NPZ per physical event):
  python Data_Creation/create_sample_manipulation_events.py --quick
  python ml/test_spectrum_split.py \\
      --model ml/models/spectrum_split_model.pth \\
      --events-dir Data_Creation/sample_manipulation_events

Compare with per-bin baseline (if model exists):
  python ml/test_spectrum_split.py \\
      --model ml/models/spectrum_split_model.pth \\
      --baseline-model models/combined_bin_model.pth \\
      --test-pickle ../data/manipulated_test_10000.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = REPO_ROOT / "ml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import torch

from spectrum_split_model import (
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
    DEVICE,
    SpectrumSplitModel,
    integrated_polarizations_calibrated,
    load_spectrum_npz,
    load_trained_model,
    polarization_train_holdout_split,
)

try:
    from bin_io import SPECTRUM_TRAIN_MANIFEST_NAME
except ImportError:
    SPECTRUM_TRAIN_MANIFEST_NAME = "spectrum_train_manifest.json"


def is_sharded_source(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() or path.name == SPECTRUM_TRAIN_MANIFEST_NAME


def load_sharded_helpers() -> tuple[Any, Any, Any]:
    try:
        from spectrum_split_model import (
            discover_spectrum_shards,
            read_light_columns,
            row_mask_by_shard,
        )
    except ImportError as exc:
        raise SystemExit(
            "Sharded --data requires an updated spectrum_split_model.py "
            "(discover_spectrum_shards and related helpers). "
            "Copy the latest ml/spectrum_split_model.py to the cluster, "
            "or pass a single spectrum_train.npz file via --data."
        ) from exc
    return discover_spectrum_shards, read_light_columns, row_mask_by_shard

DEFAULT_OUT = ML_DIR / "results" / "test_spectrum_split"
SOURCE_LABELS = {0: "ssRF", 1: "AFP", 2: "unmanip"}
MANIPULATED_SOURCES = (0, 1)

try:
    from common import F_MAX, F_MIN
except ImportError:
    F_MIN, F_MAX = -6.0, 6.0


@dataclass
class HoldoutPredictions:
    idx: np.ndarray
    ps: np.ndarray
    ip_true: np.ndarray
    im_true: np.ndarray
    ip_pred: np.ndarray
    im_pred: np.ndarray
    p0: np.ndarray
    source: np.ndarray
    event_ids: np.ndarray | None = None


def frequency_axis(num_bins: int) -> np.ndarray:
    return np.linspace(float(F_MIN), float(F_MAX), int(num_bins), dtype=float)


def compute_metrics(
    ps: np.ndarray,
    ip_pred: np.ndarray,
    im_pred: np.ndarray,
    ip_true: np.ndarray,
    im_true: np.ndarray,
) -> dict[str, float]:
    mask = np.isfinite(ip_pred) & np.isfinite(im_pred)
    l1_ip = float(np.mean(np.abs(ip_pred[mask] - ip_true[mask]))) if mask.any() else float("nan")
    l1_im = float(np.mean(np.abs(im_pred[mask] - im_true[mask]))) if mask.any() else float("nan")
    cons = float(np.max(np.abs(ip_pred + im_pred - ps))) if ps.size else 0.0
    alpha_pred = np.divide(
        ip_pred,
        np.maximum(np.abs(ps), 1e-12),
        out=np.zeros_like(ip_pred),
        where=np.abs(ps) > 1e-12,
    )
    alpha_true = np.divide(
        ip_true,
        np.maximum(np.abs(ps), 1e-12),
        out=np.zeros_like(ip_true),
        where=np.abs(ps) > 1e-12,
    )
    l1_alpha = float(np.mean(np.abs(alpha_pred[mask] - alpha_true[mask]))) if mask.any() else float("nan")
    num_bins = int(ip_true.shape[1]) if ip_true.ndim > 1 else int(ip_true.size)
    p_true, q_true = integrated_polarizations_calibrated(
        ip_true, im_true, num_bins=num_bins, post_correct=True
    )
    p_pred, q_pred = integrated_polarizations_calibrated(
        ip_pred, im_pred, num_bins=num_bins, post_correct=True
    )
    p_mask = np.abs(p_true) > 1e-10
    q_mask = np.abs(q_true) > 1e-10
    p_rpe = np.abs((p_pred - p_true) / np.where(p_mask, p_true, 1.0)) * 100.0
    q_rpe = np.abs((q_pred - q_true) / np.where(q_mask, q_true, 1.0)) * 100.0
    p_abs_err = np.abs(p_pred - p_true)
    q_abs_err = np.abs(q_pred - q_true)
    return {
        "L1_Iplus": l1_ip,
        "L1_Iminus": l1_im,
        "L1_alpha": l1_alpha,
        "max_conservation_residual": cons,
        "mean_RPE_P": float(np.nanmean(p_rpe[p_mask])) if p_mask.any() else float("nan"),
        "mean_RPE_Q": float(np.nanmean(q_rpe[q_mask])) if q_mask.any() else float("nan"),
        "mean_abs_residual_P": float(np.mean(p_abs_err[p_mask])) if p_mask.any() else float("nan"),
        "mean_abs_residual_Q": float(np.mean(q_abs_err[q_mask])) if q_mask.any() else float("nan"),
        "mean_abs_Q_true": float(np.mean(np.abs(q_true[q_mask]))) if q_mask.any() else float("nan"),
        "median_abs_Q_true": float(np.median(np.abs(q_true[q_mask]))) if q_mask.any() else float("nan"),
    }


def predict_holdout(
    model: SpectrumSplitModel,
    arrays: dict[str, np.ndarray],
    holdout_idx: np.ndarray,
    *,
    batch_size: int = 32,
) -> HoldoutPredictions | None:
    ps = np.asarray(arrays["ps"], dtype=np.float32)
    ip_true = np.asarray(arrays["iplus"], dtype=np.float32)
    im_true = np.asarray(arrays["iminus"], dtype=np.float32)
    p0 = np.asarray(arrays["p0"], dtype=np.float32)
    source = np.asarray(arrays.get("source", np.zeros(ps.shape[0], dtype=np.uint8)))

    idx = np.asarray(holdout_idx, dtype=np.int64)
    if idx.size == 0:
        return None

    ip_pred = np.zeros_like(ip_true)
    im_pred = np.zeros_like(im_true)
    for start in range(0, int(idx.size), batch_size):
        sl = idx[start : start + batch_size]
        ip_b, im_b = model.predict_batch(ps[sl], p0_batch=p0[sl], device=DEVICE)
        ip_pred[sl] = ip_b
        im_pred[sl] = im_b

    return HoldoutPredictions(
        idx=idx,
        ps=ps[idx],
        ip_true=ip_true[idx],
        im_true=im_true[idx],
        ip_pred=ip_pred[idx],
        im_pred=im_pred[idx],
        p0=p0[idx],
        source=source[idx],
    )


def metrics_from_predictions(preds: HoldoutPredictions) -> dict[str, Any]:
    overall = compute_metrics(
        preds.ps, preds.ip_pred, preds.im_pred, preds.ip_true, preds.im_true
    )
    by_source: dict[str, dict[str, float]] = {}
    for code, name in ((0, "ssrf"), (1, "afp"), (2, "unmanip")):
        m = preds.source == code
        if not np.any(m):
            continue
        by_source[name] = compute_metrics(
            preds.ps[m], preds.ip_pred[m], preds.im_pred[m], preds.ip_true[m], preds.im_true[m]
        )
    return {
        "n_samples": int(preds.ps.shape[0]),
        "overall": overall,
        "by_source": by_source,
    }


def select_plot_indices(
    preds: HoldoutPredictions,
    *,
    n_examples: int,
    selection: str,
    seed: int,
    manipulated_only: bool,
) -> np.ndarray:
    n = int(preds.ps.shape[0])
    if n == 0 or n_examples <= 0:
        return np.array([], dtype=int)

    pool = np.arange(n, dtype=int)
    if manipulated_only:
        manip_mask = np.isin(preds.source, MANIPULATED_SOURCES)
        pool = pool[manip_mask]
    if pool.size == 0:
        return np.array([], dtype=int)

    n_pick = min(int(n_examples), int(pool.size))
    if selection == "spread":
        if n_pick == 1:
            return pool[pool.size // 2 : pool.size // 2 + 1]
        ranks = np.linspace(0, pool.size - 1, n_pick)
        return pool[np.round(ranks).astype(int)]

    num_bins = int(preds.ps.shape[1])
    _, q_true = integrated_polarizations_calibrated(
        preds.ip_true, preds.im_true, num_bins=num_bins, post_correct=True
    )
    _, q_pred = integrated_polarizations_calibrated(
        preds.ip_pred, preds.im_pred, num_bins=num_bins, post_correct=True
    )
    q_rpe = np.abs((q_pred - q_true) / np.where(np.abs(q_true) > 1e-10, q_true, 1.0)) * 100.0
    q_rpe = np.where(np.abs(q_true) > 1e-10, q_rpe, np.nan)

    if selection == "worst_q":
        order = pool[np.argsort(-np.nan_to_num(q_rpe[pool], nan=-1.0))]
        return order[:n_pick]
    if selection == "random":
        rng = np.random.default_rng(seed)
        return rng.choice(pool, size=n_pick, replace=False)

    raise ValueError(f"unsupported plot selection: {selection!r}")


def _format_example_title(preds: HoldoutPredictions, row: int) -> str:
    src = SOURCE_LABELS.get(int(preds.source[row]), f"src={int(preds.source[row])}")
    num_bins = int(preds.ps.shape[1])
    _, q_true = integrated_polarizations_calibrated(
        preds.ip_true[row : row + 1],
        preds.im_true[row : row + 1],
        num_bins=num_bins,
        post_correct=True,
    )
    _, q_pred = integrated_polarizations_calibrated(
        preds.ip_pred[row : row + 1],
        preds.im_pred[row : row + 1],
        num_bins=num_bins,
        post_correct=True,
    )
    q_true = float(q_true[0])
    q_pred = float(q_pred[0])
    q_rpe = abs((q_pred - q_true) / q_true * 100.0) if abs(q_true) > 1e-10 else float("nan")
    if preds.event_ids is not None:
        row_label = str(preds.event_ids[row])
    else:
        row_label = f"holdout row {int(preds.idx[row])}"
    return (
        f"{row_label}  |  {src}  |  P0={float(preds.p0[row]):.3f}  |  "
        f"Q_true={q_true:.3g}  Q_pred={q_pred:.3g}  Q_RPE={q_rpe:.1f}%"
    )


def _draw_lineshape_panels(
    ax0: plt.Axes,
    ax1: plt.Axes,
    freq: np.ndarray,
    ps: np.ndarray,
    ip_t: np.ndarray,
    im_t: np.ndarray,
    ip_p: np.ndarray,
    im_p: np.ndarray,
    *,
    title: str | None = None,
    legend_fontsize: float = 7.0,
) -> None:
    q_t = ip_t - im_t
    q_p = ip_p - im_p
    ax0.plot(freq, ps, "k--", lw=1.0, alpha=0.8, label=r"$P_s$")
    ax0.plot(freq, ip_t, color="#d55e00", lw=1.8, alpha=0.45, label=r"true $I_+$")
    ax0.plot(freq, im_t, color="#0072b2", lw=1.8, alpha=0.45, label=r"true $I_-$")
    ax0.plot(freq, ip_p, color="#d55e00", ls="--", lw=1.4, label=r"pred $I_+$")
    ax0.plot(freq, im_p, color="#0072b2", ls="--", lw=1.4, label=r"pred $I_-$")
    ax0.axhline(0.0, color="black", ls=":", alpha=0.25, lw=0.8)
    ax0.set_ylabel("intensity")
    if title is not None:
        ax0.set_title(title, fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=legend_fontsize, loc="best")

    ax1.plot(freq, q_t, color="#7b3294", lw=1.8, alpha=0.55, label=r"true $Q$")
    ax1.plot(freq, q_p, color="#7b3294", ls="--", lw=1.4, label=r"pred $Q$")
    ax1.axhline(0.0, color="black", ls=":", alpha=0.25, lw=0.8)
    ax1.set_xlabel("R")
    ax1.set_ylabel(r"$Q = I_+ - I_-$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=legend_fontsize, loc="best")


def plot_manipulated_lineshape_examples(
    preds: HoldoutPredictions,
    output_dir: Path,
    *,
    n_examples: int = 6,
    selection: str = "spread",
    seed: int = 0,
    manipulated_only: bool = True,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = select_plot_indices(
        preds,
        n_examples=n_examples,
        selection=selection,
        seed=seed,
        manipulated_only=manipulated_only,
    )
    if indices.size == 0:
        print("No manipulated holdout samples available for lineshape plots.", flush=True)
        return []

    freq = frequency_axis(preds.ps.shape[1])
    saved: list[Path] = []

    overview_path = output_dir / "lineshape_manipulated_examples.png"
    fig, axes = plt.subplots(len(indices), 2, figsize=(12.0, 3.6 * len(indices)), squeeze=False)
    if len(indices) == 1:
        fig.set_size_inches(12.0, 5.5)

    for row_idx, ax_pair in zip(indices, axes, strict=True):
        title = _format_example_title(preds, row_idx)
        ax0, ax1 = ax_pair
        _draw_lineshape_panels(
            ax0,
            ax1,
            freq,
            preds.ps[row_idx],
            preds.ip_true[row_idx],
            preds.im_true[row_idx],
            preds.ip_pred[row_idx],
            preds.im_pred[row_idx],
            title=title,
        )

        if preds.event_ids is not None:
            slug = str(preds.event_ids[row_idx]).replace("/", "_")
        else:
            slug = f"row{int(preds.idx[row_idx]):06d}"
        single_path = output_dir / f"lineshape_manipulated_{slug}.png"
        single_fig, single_axes = plt.subplots(2, 1, figsize=(11.0, 6.5), sharex=True, layout="constrained")
        _draw_lineshape_panels(
            single_axes[0],
            single_axes[1],
            freq,
            preds.ps[row_idx],
            preds.ip_true[row_idx],
            preds.im_true[row_idx],
            preds.ip_pred[row_idx],
            preds.im_pred[row_idx],
            title=title,
            legend_fontsize=8.0,
        )
        single_fig.savefig(single_path, dpi=180, bbox_inches="tight")
        plt.close(single_fig)
        saved.append(single_path)

    axes[-1, 0].set_xlabel("R")
    axes[-1, 1].set_xlabel("R")
    fig.suptitle("Manipulated holdout lineshapes: true vs predicted", fontsize=11)
    fig.tight_layout()
    fig.savefig(overview_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.insert(0, overview_path)
    return saved


def predict_holdout_sharded(
    model: SpectrumSplitModel,
    shard_paths: list[Path],
    holdout_masks: dict[int, np.ndarray],
    *,
    batch_size: int = 32,
) -> HoldoutPredictions | None:
    idx_parts: list[np.ndarray] = []
    ps_parts: list[np.ndarray] = []
    ip_true_parts: list[np.ndarray] = []
    im_true_parts: list[np.ndarray] = []
    ip_pred_parts: list[np.ndarray] = []
    im_pred_parts: list[np.ndarray] = []
    p0_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []

    offset = 0
    for shard_id, mask in holdout_masks.items():
        with np.load(shard_paths[shard_id], allow_pickle=False) as npz:
            ps = np.asarray(npz["ps"], dtype=np.float32)[mask]
            ip_true = np.asarray(npz["iplus"], dtype=np.float32)[mask]
            im_true = np.asarray(npz["iminus"], dtype=np.float32)[mask]
            p0 = np.asarray(npz["p0"], dtype=np.float32)[mask]
            source = (
                np.asarray(npz["source"], dtype=np.uint8)[mask]
                if "source" in npz.files
                else np.zeros(ps.shape[0], dtype=np.uint8)
            )
            n_rows = int(ps.shape[0])
            row_idx = np.arange(offset, offset + n_rows, dtype=np.int64)
            offset += n_rows

        ip_pred = np.zeros_like(ip_true)
        im_pred = np.zeros_like(im_true)
        for start in range(0, ps.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            ip_b, im_b = model.predict_batch(ps[sl], p0_batch=p0[sl], device=DEVICE)
            ip_pred[sl] = ip_b
            im_pred[sl] = im_b

        idx_parts.append(row_idx)
        ps_parts.append(ps)
        ip_true_parts.append(ip_true)
        im_true_parts.append(im_true)
        ip_pred_parts.append(ip_pred)
        im_pred_parts.append(im_pred)
        p0_parts.append(p0)
        source_parts.append(source)

    if not ps_parts:
        return None

    return HoldoutPredictions(
        idx=np.concatenate(idx_parts),
        ps=np.concatenate(ps_parts),
        ip_true=np.concatenate(ip_true_parts),
        im_true=np.concatenate(im_true_parts),
        ip_pred=np.concatenate(ip_pred_parts),
        im_pred=np.concatenate(im_pred_parts),
        p0=np.concatenate(p0_parts),
        source=np.concatenate(source_parts),
    )


def evaluate_holdout(
    model: SpectrumSplitModel,
    arrays: dict[str, np.ndarray],
    holdout_idx: np.ndarray,
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    preds = predict_holdout(model, arrays, holdout_idx, batch_size=batch_size)
    if preds is None:
        return {"n_samples": 0}
    return metrics_from_predictions(preds)


def evaluate_holdout_sharded(
    model: SpectrumSplitModel,
    shard_paths: list[Path],
    holdout_masks: dict[int, np.ndarray],
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    preds = predict_holdout_sharded(model, shard_paths, holdout_masks, batch_size=batch_size)
    if preds is None:
        return {"n_samples": 0}
    return metrics_from_predictions(preds)


def predict_manipulation_events(
    model: SpectrumSplitModel,
    events_dir: Path,
    *,
    batch_size: int = 32,
) -> HoldoutPredictions:
    data_creation = REPO_ROOT / "Data_Creation"
    if str(data_creation) not in sys.path:
        sys.path.insert(0, str(data_creation))
    from manipulation_event_io import events_to_batch, load_events_directory

    events = load_events_directory(events_dir)
    batch = events_to_batch(events)
    ps = batch["ps"]
    ip_true = batch["iplus"]
    im_true = batch["iminus"]
    p0 = batch["p0"]
    source = batch["source"]
    n = int(ps.shape[0])

    ip_pred = np.zeros_like(ip_true)
    im_pred = np.zeros_like(im_true)
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        ip_b, im_b = model.predict_batch(ps[sl], p0_batch=p0[sl], device=DEVICE)
        ip_pred[sl] = ip_b
        im_pred[sl] = im_b

    return HoldoutPredictions(
        idx=np.arange(n, dtype=np.int64),
        ps=ps,
        ip_true=ip_true,
        im_true=im_true,
        ip_pred=ip_pred,
        im_pred=im_pred,
        p0=p0,
        source=source,
        event_ids=batch["event_id"],
    )


def evaluate_pickle_events(model: SpectrumSplitModel, pickle_path: Path) -> dict[str, Any] | None:
    try:
        from test_binning import load_test_events
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "test_binning", ML_DIR / "test-binning.py"
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        load_test_events = mod.load_test_events

    if not Path(pickle_path).is_file():
        return None

    events, _df = load_test_events(str(pickle_path))
    ps = np.stack([e.ps for e in events], axis=0)
    ip_true = np.stack([e.iplus for e in events], axis=0)
    im_true = np.stack([e.iminus for e in events], axis=0)
    p0 = np.asarray([e.polarization for e in events], dtype=np.float32)
    ip_pred, im_pred = model.predict_batch(ps, p0_batch=p0, device=DEVICE)
    return {
        "n_samples": int(ps.shape[0]),
        "overall": compute_metrics(ps, ip_pred, im_pred, ip_true, im_true),
        "pickle_path": str(pickle_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate SpectrumSplitNet on holdout / test pickle")
    p.add_argument("--model", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Single spectrum_train.npz, or a sharded output directory/manifest",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--test-pickle",
        type=Path,
        default=None,
        help="Optional manipulated_test pickle for external benchmark",
    )
    p.add_argument(
        "--events-dir",
        type=Path,
        default=None,
        help=(
            "Directory of individual manipulation event NPZs (from create_sample_manipulation_events.py). "
            "When set, evaluates one prediction per physical event instead of NPZ holdout rows."
        ),
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-plots", action="store_true", help="Skip lineshape figure generation")
    p.add_argument(
        "--plot-examples",
        type=int,
        default=6,
        help="Number of holdout manipulated spectra to plot (default: 6)",
    )
    p.add_argument(
        "--plot-selection",
        choices=("spread", "worst_q", "random"),
        default="spread",
        help="How to pick plotted holdout rows",
    )
    p.add_argument("--plot-seed", type=int, default=0, help="RNG seed for --plot-selection random")
    p.add_argument(
        "--plot-all-sources",
        action="store_true",
        help="Include unmanipulated holdout rows in plots (default: ssRF + AFP only)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {DEVICE}", flush=True)
    model = load_trained_model(args.model, DEVICE)

    holdout_preds: HoldoutPredictions | None = None
    eval_mode = "npz_holdout"
    if args.events_dir is not None:
        eval_mode = "manipulation_events"
        holdout_preds = predict_manipulation_events(
            model, args.events_dir, batch_size=int(args.batch_size)
        )
    elif is_sharded_source(args.data):
        discover_spectrum_shards, read_light_columns, row_mask_by_shard = load_sharded_helpers()
        shard_paths = discover_spectrum_shards(args.data)
        light = read_light_columns(shard_paths)
        shard_row_counts = np.bincount(light["shard_id"], minlength=len(shard_paths)).tolist()
        _, holdout_sel = polarization_train_holdout_split(light["p0"])
        holdout_masks = row_mask_by_shard(light["shard_id"], light["local_idx"], shard_row_counts, holdout_sel)
        holdout_preds = predict_holdout_sharded(
            model, shard_paths, holdout_masks, batch_size=int(args.batch_size)
        )
    else:
        arrays = load_spectrum_npz(args.data)
        _, holdout_idx = polarization_train_holdout_split(arrays["p0"])
        holdout_preds = predict_holdout(model, arrays, holdout_idx, batch_size=int(args.batch_size))

    holdout_result = (
        metrics_from_predictions(holdout_preds)
        if holdout_preds is not None
        else {"n_samples": 0}
    )

    results: dict[str, Any] = {
        "model": str(args.model),
        "eval_mode": eval_mode,
        "holdout": holdout_result,
    }
    if args.events_dir is not None:
        results["events_dir"] = str(args.events_dir)
    else:
        results["data"] = str(args.data)

    if not args.no_plots and holdout_preds is not None:
        plot_dir = Path(args.output_dir) / "plots"
        plot_paths = plot_manipulated_lineshape_examples(
            holdout_preds,
            plot_dir,
            n_examples=int(args.plot_examples),
            selection=str(args.plot_selection),
            seed=int(args.plot_seed),
            manipulated_only=not bool(args.plot_all_sources),
        )
        if plot_paths:
            results["plot_paths"] = [str(path) for path in plot_paths]
            for path in plot_paths:
                print(f"Wrote {path}", flush=True)

    if args.test_pickle is not None:
        ext = evaluate_pickle_events(model, args.test_pickle)
        if ext is not None:
            results["external_pickle"] = ext

    out_json = Path(args.output_dir) / "test_statistics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    hold = results["holdout"]
    if hold.get("n_samples", 0) > 0:
        ov = hold["overall"]
        label = "Events" if eval_mode == "manipulation_events" else "Holdout"
        print(
            f"{label} n={hold['n_samples']}  L1_I+={ov['L1_Iplus']:.4f}  "
            f"L1_I-={ov['L1_Iminus']:.4f}  L1_alpha={ov['L1_alpha']:.4f}  "
            f"mean_RPE_P={ov['mean_RPE_P']:.2f}%  mean_RPE_Q={ov['mean_RPE_Q']:.2f}%  "
            f"mean_|Q_err|={ov['mean_abs_residual_Q']:.4g}  median_|Q_true|={ov['median_abs_Q_true']:.4g}",
            flush=True,
        )
        for src, metrics in hold.get("by_source", {}).items():
            print(
                f"  [{src}] L1_I+={metrics['L1_Iplus']:.4f}  L1_I-={metrics['L1_Iminus']:.4f}",
                flush=True,
            )
    print(f"Wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
