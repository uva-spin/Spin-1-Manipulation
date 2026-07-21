"""
Merge unmanipulated per-bin NPZs into combined ssRF/AFP train_bin_XXXX.npz files.

Reads:
  combined_train/train_bin_XXXX.npz   (source 0=ssrf, 1=afp)
  unmanip_train/unmanip_bin_XXXX.npz  (source 2=unmanipulated)

Writes one merged NPZ per bin (same schema, concatenated samples).

Usage:
  python merge_unmanip_into_combined.py \\
      --combined-dir combined_train --unmanip-dir unmanip_train \\
      --output-dir combined_train_all --strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

NUM_BINS = 500
SOURCE_SSRF = 0
SOURCE_AFP = 1
SOURCE_UNMANIP = 2

DEFAULT_COMBINED_DIR = Path(__file__).resolve().parent / "combined_train"
DEFAULT_UNMANIP_DIR = Path(__file__).resolve().parent / "unmanip_train"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "combined_train_all"

ARRAY_KEYS = (
    "p0",
    "step",
    "center_bin",
    "is_mirror",
    "source",
    "ps",
    "iplus",
    "iminus",
    "amp",
)


def combined_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"train_bin_{int(bin_idx):04d}.npz"


def unmanip_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"unmanip_bin_{int(bin_idx):04d}.npz"


def load_train_bin(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        out: dict[str, np.ndarray] = {}
        for key in ARRAY_KEYS:
            if key not in data.files:
                raise KeyError(f"{path}: missing required field {key!r}")
            out[key] = np.asarray(data[key])
        meta = {}
        if "meta_json" in data.files:
            meta = json.loads(str(data["meta_json"]))
        out["_meta"] = meta  # type: ignore[assignment]
        return out


def _as_dtype(key: str, arr: np.ndarray) -> np.ndarray:
    if key in ("step", "center_bin"):
        return np.asarray(arr, dtype=np.int32)
    if key == "is_mirror":
        return np.asarray(arr, dtype=bool)
    if key == "source":
        return np.asarray(arr, dtype=np.uint8)
    return np.asarray(arr, dtype=float)


def save_merged_bin(
    bin_idx: int,
    arrays: dict[str, np.ndarray],
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = np.asarray(arrays["source"], dtype=np.uint8)
    n_samples = int(np.asarray(arrays["ps"]).size)
    meta = {
        "bin_idx": int(bin_idx),
        "n_samples": n_samples,
        "n_ssrf": int(np.count_nonzero(source == SOURCE_SSRF)),
        "n_afp": int(np.count_nonzero(source == SOURCE_AFP)),
        "n_unmanip": int(np.count_nonzero(source == SOURCE_UNMANIP)),
        "source_codes": {
            "ssrf": SOURCE_SSRF,
            "afp": SOURCE_AFP,
            "unmanipulated": SOURCE_UNMANIP,
        },
        "dataset": "ssrf_afp_unmanip_train_bin",
        "fields": (
            "ps,iplus,iminus,amp at this bin; "
            "center_bin; is_mirror; source (0=ssrf,1=afp,2=unmanip)"
        ),
    }
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            bin_idx=np.asarray(int(bin_idx), dtype=np.int32),
            p0=_as_dtype("p0", arrays["p0"]),
            step=_as_dtype("step", arrays["step"]),
            center_bin=_as_dtype("center_bin", arrays["center_bin"]),
            is_mirror=_as_dtype("is_mirror", arrays["is_mirror"]),
            source=_as_dtype("source", arrays["source"]),
            ps=_as_dtype("ps", arrays["ps"]),
            iplus=_as_dtype("iplus", arrays["iplus"]),
            iminus=_as_dtype("iminus", arrays["iminus"]),
            amp=_as_dtype("amp", arrays["amp"]),
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def merge_bin(combined: dict, unmanip: dict) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for key in ARRAY_KEYS:
        merged[key] = np.concatenate(
            [_as_dtype(key, combined[key]), _as_dtype(key, unmanip[key])]
        )
    return merged


def merge_all(
    combined_dir: Path,
    unmanip_dir: Path,
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    strict: bool = True,
) -> dict:
    combined_dir = Path(combined_dir)
    unmanip_dir = Path(unmanip_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_combined: list[int] = []
    missing_unmanip: list[int] = []
    samples_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    counts = {
        "ssrf": np.zeros(int(num_bins), dtype=np.int64),
        "afp": np.zeros(int(num_bins), dtype=np.int64),
        "unmanip": np.zeros(int(num_bins), dtype=np.int64),
    }

    for bin_idx in range(int(num_bins)):
        c_path = combined_bin_path(combined_dir, bin_idx)
        u_path = unmanip_bin_path(unmanip_dir, bin_idx)
        if not c_path.is_file():
            missing_combined.append(bin_idx)
            continue
        if not u_path.is_file():
            missing_unmanip.append(bin_idx)
            continue

        merged = merge_bin(load_train_bin(c_path), load_train_bin(u_path))
        source = merged["source"]
        samples_per_bin[bin_idx] = int(merged["ps"].size)
        counts["ssrf"][bin_idx] = int(np.count_nonzero(source == SOURCE_SSRF))
        counts["afp"][bin_idx] = int(np.count_nonzero(source == SOURCE_AFP))
        counts["unmanip"][bin_idx] = int(np.count_nonzero(source == SOURCE_UNMANIP))
        save_merged_bin(bin_idx, merged, combined_bin_path(output_dir, bin_idx))

        if (bin_idx + 1) % 50 == 0 or bin_idx == int(num_bins) - 1:
            print(f"  merged through bin {bin_idx}", flush=True)

    if strict and (missing_combined or missing_unmanip):
        raise FileNotFoundError(
            f"Missing combined={len(missing_combined)} "
            f"unmanip={len(missing_unmanip)}; "
            f"first combined={missing_combined[:1]} "
            f"first unmanip={missing_unmanip[:1]}"
        )
    if missing_combined:
        print(
            f"WARNING: missing {len(missing_combined)} combined train bins",
            flush=True,
        )
    if missing_unmanip:
        print(
            f"WARNING: missing {len(missing_unmanip)} unmanip bins",
            flush=True,
        )

    return {
        "output_dir": str(output_dir),
        "n_samples": int(samples_per_bin.sum()),
        "samples_per_bin": samples_per_bin,
        "ssrf_per_bin": counts["ssrf"],
        "afp_per_bin": counts["afp"],
        "unmanip_per_bin": counts["unmanip"],
        "n_missing_combined": len(missing_combined),
        "n_missing_unmanip": len(missing_unmanip),
        "dataset": "ssrf_afp_unmanip_train_bin",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge unmanipulated NPZs into combined ssRF/AFP per-bin train files"
    )
    p.add_argument("--combined-dir", type=Path, default=DEFAULT_COMBINED_DIR)
    p.add_argument("--unmanip-dir", type=Path, default=DEFAULT_UNMANIP_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        f"Merging unmanip ({args.unmanip_dir}) into combined ({args.combined_dir}) "
        f"-> {args.output_dir}",
        flush=True,
    )
    result = merge_all(
        args.combined_dir,
        args.unmanip_dir,
        args.output_dir,
        num_bins=args.num_bins,
        strict=bool(args.strict),
    )
    print(
        f"Merged {result['n_samples']} samples -> {args.output_dir} "
        f"(missing_combined={result['n_missing_combined']} "
        f"missing_unmanip={result['n_missing_unmanip']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
