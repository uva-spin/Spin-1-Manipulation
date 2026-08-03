"""
Build per-bin train NPZs from pre-flattened ssRF spectrum rows (no unmanip).

For each observation bin ``b``, concatenates rows from every burn-center
``ssrf_spectrum_rows_XXXX.npz`` file and extracts ``ps/iplus/iminus/q/P/Q`` at
column ``b``.  ``P`` and ``Q`` are CC-calibrated at combine time (same as
``combine_all_train.py``).

Usage (from this directory):
  python flatten_spectrum_rows.py --source ssrf --flatten-all
  python combine_spectrum_train_bins.py \\
      --ssrf-rows-dir data/ssrf_spectrum_rows \\
      --output-dir data/combined_train_all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from bin_paths import ssrf_spectrum_rows_path
from combine_all_train import combined_bin_path
from common import (
    BURN_BIN_CHOICES,
    COMBINED_TRAIN_ALL_DIR,
    NUM_BINS,
    PHYSICS_MODEL,
    SOURCE_SSRF,
    SSRF_SPECTRUM_ROWS_DIR,
    intensity_pq,
)
from pq_calibration import calibrated_pq_fields, load_pq_calibration, validate_stored_per_bin_pq
from spectrum_rows import load_spectrum_rows_npz, validate_ps_iplus_iminus


def _slice_rows_at_bin(
    rows: dict[str, np.ndarray],
    obs_bin: int,
    *,
    num_bins: int,
    pq_calibration: dict,
) -> dict[str, np.ndarray]:
    ps_full = np.asarray(rows["ps"])
    n = int(ps_full.shape[0])
    if n <= 0:
        return {}
    obs = int(obs_bin)
    ps_b = np.asarray(ps_full[:, obs], dtype=np.float32)
    ip_b = np.asarray(rows["iplus"][:, obs], dtype=np.float32)
    im_b = np.asarray(rows["iminus"][:, obs], dtype=np.float32)
    _, q_b = intensity_pq(ip_b, im_b)
    p0 = np.asarray(rows["p0"], dtype=np.float32)
    slice_arrays = {
        "p0": p0,
        "ps": ps_b,
        "iplus": ip_b,
        "iminus": im_b,
        "q": np.asarray(q_b, dtype=np.float32),
    }
    p_b, q_cal_b = calibrated_pq_fields(
        slice_arrays,
        num_bins=int(num_bins),
        calibration=pq_calibration,
    )
    validate_stored_per_bin_pq(
        ps_b,
        q_b,
        p0,
        p_b,
        q_cal_b,
        calibration=pq_calibration,
        post_correct=True,
    )
    return {
        "p0": p0,
        "step": np.asarray(rows["step"], dtype=np.int32),
        "center_bin": np.asarray(rows["center_bin"], dtype=np.int32),
        "is_mirror": np.zeros(n, dtype=bool),
        "is_neighbor": np.zeros(n, dtype=bool),
        "source": np.full(n, np.uint8(SOURCE_SSRF), dtype=np.uint8),
        "ps": ps_b,
        "iplus": ip_b,
        "iminus": im_b,
        "q": np.asarray(q_b, dtype=np.float32),
        "P": np.asarray(p_b, dtype=np.float32),
        "Q": np.asarray(q_cal_b, dtype=np.float32),
        "amp": np.abs(ps_b).astype(np.float32, copy=False),
    }


def _concat_train_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    parts = [p for p in parts if int(np.asarray(p.get("ps", [])).size) > 0]
    if not parts:
        raise ValueError("no rows to concatenate")
    if len(parts) == 1:
        return parts[0]
    keys = parts[0].keys()
    out: dict[str, np.ndarray] = {}
    for key in keys:
        arrs = [p[key] for p in parts]
        if arrs[0].ndim == 1:
            out[key] = np.concatenate(arrs)
        else:
            out[key] = np.concatenate(arrs, axis=0)
    return out


def combine_ssrf_spectrum_rows_to_train_bins(
    ssrf_rows_dir: Path,
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    burn_centers: np.ndarray | None = None,
    strict: bool = False,
) -> dict:
    ssrf_rows_dir = Path(ssrf_rows_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nb = int(num_bins)
    pq_calibration = load_pq_calibration(num_bins=nb)
    centers = (
        np.asarray(burn_centers, dtype=int)
        if burn_centers is not None
        else np.asarray(BURN_BIN_CHOICES, dtype=int)
    )
    centers = centers[(centers >= 0) & (centers < nb)]

    missing_centers: list[int] = []
    for center in centers.tolist():
        if not ssrf_spectrum_rows_path(ssrf_rows_dir, int(center)).is_file():
            missing_centers.append(int(center))
    if strict and missing_centers:
        raise FileNotFoundError(
            f"Missing {len(missing_centers)} ssRF spectrum row files under {ssrf_rows_dir}; "
            f"first missing burn center={missing_centers[0]}"
        )

    samples_per_bin = np.zeros(nb, dtype=np.int64)
    n_centers_used = 0

    for obs_bin in range(nb):
        parts: list[dict[str, np.ndarray]] = []
        for center in centers.tolist():
            path = ssrf_spectrum_rows_path(ssrf_rows_dir, int(center))
            if not path.is_file():
                continue
            rows = load_spectrum_rows_npz(path)
            validate_ps_iplus_iminus(rows, label=str(path))
            sliced = _slice_rows_at_bin(
                rows,
                obs_bin,
                num_bins=nb,
                pq_calibration=pq_calibration,
            )
            if sliced:
                parts.append(sliced)
        if not parts:
            continue
        n_centers_used = max(n_centers_used, len(parts))
        merged = _concat_train_parts(parts)
        n = int(merged["ps"].size)
        samples_per_bin[obs_bin] = n

        meta = {
            "bin_idx": int(obs_bin),
            "n_samples": n,
            "n_ssrf": n,
            "n_afp": 0,
            "n_unmanip": 0,
            "source_codes": {"ssrf": SOURCE_SSRF},
            "physics_model": PHYSICS_MODEL,
            "dataset": "ssrf_spectrum_train_bin_v1",
            "combine_mode": "ssrf_spectrum_rows",
            "ssrf_rows_dir": str(ssrf_rows_dir),
            "include_unmanip": False,
            "fields": (
                "ps,q=raw I± sums; P,Q=CC-calibrated true polarizations at this bin"
            ),
            "pq_calibrated": True,
            "pq_target_scope": "per_bin",
            "pq_post_correct": True,
        }
        out_path = combined_bin_path(output_dir, obs_bin)
        tmp_path = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(
                tmp_path,
                meta_json=np.asarray(json.dumps(meta)),
                bin_idx=np.asarray(int(obs_bin), dtype=np.int32),
                **merged,
            )
            tmp_path.replace(out_path)
        except Exception:
            if tmp_path.is_file():
                tmp_path.unlink(missing_ok=True)
            raise

        if (obs_bin + 1) % 50 == 0 or obs_bin + 1 == nb:
            print(
                f"  wrote train bins through {obs_bin + 1}/{nb} "
                f"(samples so far={int(samples_per_bin[: obs_bin + 1].sum())})",
                flush=True,
            )

    return {
        "output_dir": str(output_dir),
        "n_samples": int(samples_per_bin.sum()),
        "samples_per_bin": samples_per_bin,
        "n_missing_row_centers": len(missing_centers),
        "n_burn_centers": int(centers.size),
        "dataset": "ssrf_spectrum_train_bin_v1",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Combine ssRF spectrum rows into per-bin train_bin_XXXX.npz (no unmanip)"
    )
    p.add_argument(
        "--ssrf-rows-dir",
        type=Path,
        default=SSRF_SPECTRUM_ROWS_DIR,
        help="Dir with ssrf_spectrum_rows_XXXX.npz from flatten_spectrum_rows.py",
    )
    p.add_argument("--output-dir", type=Path, default=COMBINED_TRAIN_ALL_DIR)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        f"Combining ssRF spectrum rows {args.ssrf_rows_dir} -> {args.output_dir}",
        flush=True,
    )
    result = combine_ssrf_spectrum_rows_to_train_bins(
        args.ssrf_rows_dir,
        args.output_dir,
        num_bins=int(args.num_bins),
        strict=bool(args.strict),
    )
    print(
        f"Wrote {result['n_samples']} samples across {int(args.num_bins)} bins -> "
        f"{result['output_dir']}  missing_row_centers={result['n_missing_row_centers']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
