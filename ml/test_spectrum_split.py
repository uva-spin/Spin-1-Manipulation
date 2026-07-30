"""
Evaluate the global SpectrumSplitNet against holdout spectra and optional 500-bin baseline.

Run:
  python ml/test_spectrum_split.py \\
      --model ml/models/spectrum_split_model.pth \\
      --data Data_Creation/dulya_fit_v2/data/spectrum_train/spectrum_train.npz

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
from pathlib import Path
from typing import Any

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
    discover_spectrum_shards,
    is_sharded_source,
    load_spectrum_npz,
    load_trained_model,
    polarization_train_holdout_split,
    read_light_columns,
    row_mask_by_shard,
)

DEFAULT_OUT = ML_DIR / "results" / "test_spectrum_split"


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
    p_true = np.nansum(ip_true + im_true, axis=1)
    p_pred = np.nansum(ip_pred + im_pred, axis=1)
    q_true = np.nansum(ip_true - im_true, axis=1)
    q_pred = np.nansum(ip_pred - im_pred, axis=1)
    p_mask = np.abs(p_true) > 1e-10
    q_mask = np.abs(q_true) > 1e-10
    p_rpe = np.abs((p_pred - p_true) / np.where(p_mask, p_true, 1.0)) * 100.0
    q_rpe = np.abs((q_pred - q_true) / np.where(q_mask, q_true, 1.0)) * 100.0
    return {
        "L1_Iplus": l1_ip,
        "L1_Iminus": l1_im,
        "L1_alpha": l1_alpha,
        "max_conservation_residual": cons,
        "mean_RPE_P": float(np.nanmean(p_rpe[p_mask])) if p_mask.any() else float("nan"),
        "mean_RPE_Q": float(np.nanmean(q_rpe[q_mask])) if q_mask.any() else float("nan"),
    }


def evaluate_holdout(
    model: SpectrumSplitModel,
    arrays: dict[str, np.ndarray],
    holdout_idx: np.ndarray,
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    ps = np.asarray(arrays["ps"], dtype=np.float32)
    ip_true = np.asarray(arrays["iplus"], dtype=np.float32)
    im_true = np.asarray(arrays["iminus"], dtype=np.float32)
    p0 = np.asarray(arrays["p0"], dtype=np.float32)
    source = np.asarray(arrays.get("source", np.zeros(ps.shape[0], dtype=np.uint8)))

    idx = np.asarray(holdout_idx, dtype=np.int64)
    if idx.size == 0:
        return {"n_samples": 0}

    ip_pred = np.zeros_like(ip_true)
    im_pred = np.zeros_like(im_true)
    for start in range(0, int(idx.size), batch_size):
        sl = idx[start : start + batch_size]
        ip_b, im_b = model.predict_batch(ps[sl], p0_batch=p0[sl], device=DEVICE)
        ip_pred[sl] = ip_b
        im_pred[sl] = im_b

    overall = compute_metrics(ps[idx], ip_pred[idx], im_pred[idx], ip_true[idx], im_true[idx])
    by_source: dict[str, dict[str, float]] = {}
    for code, name in ((0, "ssrf"), (1, "afp"), (2, "unmanip")):
        m = source[idx] == code
        if not np.any(m):
            continue
        sel = idx[m]
        by_source[name] = compute_metrics(
            ps[sel], ip_pred[sel], im_pred[sel], ip_true[sel], im_true[sel]
        )

    return {
        "n_samples": int(idx.size),
        "overall": overall,
        "by_source": by_source,
    }


def evaluate_holdout_sharded(
    model: SpectrumSplitModel,
    shard_paths: list[Path],
    holdout_masks: dict[int, np.ndarray],
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    ps_parts: list[np.ndarray] = []
    ip_true_parts: list[np.ndarray] = []
    im_true_parts: list[np.ndarray] = []
    ip_pred_parts: list[np.ndarray] = []
    im_pred_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []

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

        ip_pred = np.zeros_like(ip_true)
        im_pred = np.zeros_like(im_true)
        for start in range(0, ps.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            ip_b, im_b = model.predict_batch(ps[sl], p0_batch=p0[sl], device=DEVICE)
            ip_pred[sl] = ip_b
            im_pred[sl] = im_b

        ps_parts.append(ps)
        ip_true_parts.append(ip_true)
        im_true_parts.append(im_true)
        ip_pred_parts.append(ip_pred)
        im_pred_parts.append(im_pred)
        source_parts.append(source)

    if not ps_parts:
        return {"n_samples": 0}

    ps_cat = np.concatenate(ps_parts)
    ip_true_cat = np.concatenate(ip_true_parts)
    im_true_cat = np.concatenate(im_true_parts)
    ip_pred_cat = np.concatenate(ip_pred_parts)
    im_pred_cat = np.concatenate(im_pred_parts)
    source_cat = np.concatenate(source_parts)

    overall = compute_metrics(ps_cat, ip_pred_cat, im_pred_cat, ip_true_cat, im_true_cat)
    by_source: dict[str, dict[str, float]] = {}
    for code, name in ((0, "ssrf"), (1, "afp"), (2, "unmanip")):
        m = source_cat == code
        if not np.any(m):
            continue
        by_source[name] = compute_metrics(
            ps_cat[m], ip_pred_cat[m], im_pred_cat[m], ip_true_cat[m], im_true_cat[m]
        )

    return {"n_samples": int(ps_cat.shape[0]), "overall": overall, "by_source": by_source}


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
    p.add_argument("--batch-size", type=int, default=32)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {DEVICE}", flush=True)
    model = load_trained_model(args.model, DEVICE)

    if is_sharded_source(args.data):
        shard_paths = discover_spectrum_shards(args.data)
        light = read_light_columns(shard_paths)
        shard_row_counts = np.bincount(light["shard_id"], minlength=len(shard_paths)).tolist()
        _, holdout_sel = polarization_train_holdout_split(light["p0"])
        holdout_masks = row_mask_by_shard(light["shard_id"], light["local_idx"], shard_row_counts, holdout_sel)
        holdout_result = evaluate_holdout_sharded(
            model, shard_paths, holdout_masks, batch_size=int(args.batch_size)
        )
    else:
        arrays = load_spectrum_npz(args.data)
        _, holdout_idx = polarization_train_holdout_split(arrays["p0"])
        holdout_result = evaluate_holdout(model, arrays, holdout_idx, batch_size=int(args.batch_size))

    results: dict[str, Any] = {
        "model": str(args.model),
        "data": str(args.data),
        "holdout": holdout_result,
    }

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
        print(
            f"Holdout n={hold['n_samples']}  L1_I+={ov['L1_Iplus']:.4f}  "
            f"L1_I-={ov['L1_Iminus']:.4f}  L1_alpha={ov['L1_alpha']:.4f}  "
            f"mean_RPE_P={ov['mean_RPE_P']:.2f}%",
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
