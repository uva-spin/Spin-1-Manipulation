"""
Validate per-bin single_bin checkpoints against training NPZs.

Compares saved X_mean / P_mean in each binning_model_bin_XXXX.pth to the
statistics recomputed from the corresponding train_bin_XXXX.npz using the
same split logic as ml/single_bin.py.

Use before combine to catch corrupted normalization stats. Mismatched bins
must be retrained (do not patch stats in place — weights are tied to the
normalization used during training).

Examples:
  python ml/validate_single_bin_checkpoints.py \\
      --model-dir single_bin_models \\
      --data-dir combined_train_all

  # Print SLURM array list for bad bins only:
  python ml/validate_single_bin_checkpoints.py \\
      --model-dir single_bin_models_v3 \\
      --data-dir combined_train_all \\
      --print-array
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_bin import build_bin_datasets, load_bin_arrays

MODEL_PATTERN = re.compile(r"^binning_model_bin_(\d+)\.pth$")
DEFAULT_P0_MEAN_TOL = 0.05
DEFAULT_P_MEAN_TOL = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate per-bin checkpoint normalization against train NPZs"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory with binning_model_bin_XXXX.pth files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with train_bin_XXXX.npz files",
    )
    parser.add_argument(
        "--train-polarization-fraction",
        type=float,
        default=0.8,
        help="Must match the value used when training (default: 0.8)",
    )
    parser.add_argument(
        "--feature-clip-z",
        type=float,
        default=0.0,
        help="Must match the value used when training (default: 0.0)",
    )
    parser.add_argument(
        "--p0-mean-tol",
        type=float,
        default=DEFAULT_P0_MEAN_TOL,
        help="Max |checkpoint X_mean[p0] - expected| before flagging",
    )
    parser.add_argument(
        "--p-mean-tol",
        type=float,
        default=DEFAULT_P_MEAN_TOL,
        help="Max |checkpoint P_mean - expected| before flagging",
    )
    parser.add_argument(
        "--print-array",
        action="store_true",
        help="Print comma-separated bin indices suitable for sbatch --array",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any checkpoint fails validation",
    )
    return parser.parse_args()


def discover_model_bins(model_dir: Path) -> List[int]:
    bins: List[int] = []
    for path in sorted(model_dir.glob("binning_model_bin_*.pth")):
        match = MODEL_PATTERN.match(path.name)
        if match is not None:
            bins.append(int(match.group(1)))
    return bins


def _feature_vector(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(-1)
    return arr.reshape(-1)


def expected_stats(
    data_path: Path,
    *,
    train_polarization_fraction: float,
    feature_clip_z: float,
) -> Dict[str, np.ndarray | float]:
    arrays = load_bin_arrays(
        data_path=data_path,
        train_polarization_fraction=train_polarization_fraction,
        feature_clip_z=feature_clip_z,
    )
    _, _, _, stats = build_bin_datasets(arrays, validation_fraction=0.5)
    return {
        "x_mean": stats["x_mean"].numpy().reshape(-1),
        "x_std": stats["x_std"].numpy().reshape(-1),
        "P_mean": float(stats["P_mean"].item()),
        "P_std": float(stats["P_std"].item()),
        "Q_mean": float(stats["Q_mean"].item()),
        "Q_std": float(stats["Q_std"].item()),
    }


def validate_checkpoint(
    model_path: Path,
    data_path: Path,
    *,
    train_polarization_fraction: float,
    feature_clip_z: float,
    p0_mean_tol: float,
    p_mean_tol: float,
) -> Tuple[bool, Dict[str, Any]]:
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    ck_x_mean = _feature_vector(payload.get("X_mean"))
    ck_p_mean = float(payload["P_mean"])

    expected = expected_stats(
        data_path,
        train_polarization_fraction=train_polarization_fraction,
        feature_clip_z=feature_clip_z,
    )

    dp0 = abs(float(ck_x_mean[0]) - float(expected["x_mean"][0]))
    dp = abs(ck_p_mean - float(expected["P_mean"]))
    ok = dp0 <= p0_mean_tol and dp <= p_mean_tol
    return ok, {
        "bin_idx": int(payload.get("bin_idx", -1)),
        "ck_x_mean": ck_x_mean.tolist(),
        "exp_x_mean": expected["x_mean"].tolist(),
        "ck_P_mean": ck_p_mean,
        "exp_P_mean": float(expected["P_mean"]),
        "dp0_mean": dp0,
        "dP_mean": dp,
    }


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    bad: List[int] = []
    for bin_idx in discover_model_bins(model_dir):
        model_path = model_dir / f"binning_model_bin_{bin_idx:04d}.pth"
        data_path = data_dir / f"train_bin_{bin_idx:04d}.npz"
        if not data_path.is_file():
            print(f"SKIP bin {bin_idx}: missing {data_path}", flush=True)
            continue

        ok, info = validate_checkpoint(
            model_path,
            data_path,
            train_polarization_fraction=float(args.train_polarization_fraction),
            feature_clip_z=float(args.feature_clip_z),
            p0_mean_tol=float(args.p0_mean_tol),
            p_mean_tol=float(args.p_mean_tol),
        )
        if ok:
            continue

        bad.append(bin_idx)
        print(
            f"FAIL bin {bin_idx}: "
            f"X_mean[p0] checkpoint={info['ck_x_mean'][0]:.4f} "
            f"expected={info['exp_x_mean'][0]:.4f} (Δ={info['dp0_mean']:.4f})  "
            f"P_mean checkpoint={info['ck_P_mean']:.4f} "
            f"expected={info['exp_P_mean']:.4f} (Δ={info['dP_mean']:.4f})",
            flush=True,
        )

    print(f"Validated {model_dir}; failed bins: {len(bad)}", flush=True)
    if bad:
        if args.print_array:
            print(",".join(str(b) for b in bad), flush=True)
        else:
            print(f"Retrain with: sbatch --array={','.join(str(b) for b in bad)} ml/train_single_bin_array.slurm")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
