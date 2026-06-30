"""
Combine per-bin burn lookup shards into a single pickle file.

Example:
  python Data_Creation/rivanna/combine_burn_lookup_table.py \
    --shard-dir Data_Creation/burn_lookup_shards \
    --output Data_Creation/burn_lookup_table.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CREATION_DIR = SCRIPT_DIR.parent

if str(DATA_CREATION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_CREATION_DIR))

from burn_lookup_table import NUM_BINS, per_bin_output_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine burn_lookup_bin_{idx}.pkl shards into one table."
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=DATA_CREATION_DIR / "burn_lookup_shards",
        help="Directory containing per-bin shard pickles.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_CREATION_DIR / "burn_lookup_table.pkl",
        help="Path to combined output pickle.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=NUM_BINS,
        help=f"Number of burn bins (default: {NUM_BINS}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected burn-bin shard is missing.",
    )
    return parser.parse_args()


def expected_burn_bins(num_bins: int) -> List[int]:
    return list(range(num_bins))


def load_shards(shard_dir: Path, burn_bins: List[int], strict: bool) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    missing: List[int] = []

    for burn_idx in burn_bins:
        shard_path = per_bin_output_path(shard_dir, burn_idx)
        if not shard_path.exists():
            if strict:
                missing.append(burn_idx)
            else:
                print(f"Skipping missing shard: {shard_path}", flush=True)
            continue

        print(f"Loading {shard_path}", flush=True)
        frames.append(pd.read_pickle(shard_path))

    if strict and missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} shard(s): {preview}{suffix}"
        )

    if not frames:
        raise RuntimeError(f"No shards found in {shard_dir}")

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(
        ["burn_bin_idx", "P", "burn_step"],
        kind="mergesort",
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    burn_bins = expected_burn_bins(args.num_bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    combined = load_shards(args.shard_dir, burn_bins, strict=args.strict)
    combined.to_pickle(args.output)

    print(f"Saved {len(combined)} rows to {args.output}", flush=True)
    print(
        f"Loaded {combined['burn_bin_idx'].nunique()} burn bin(s) "
        f"from {args.shard_dir}",
        flush=True,
    )
    if combined["burn_bin_idx"].nunique() != len(burn_bins):
        print(
            f"Warning: expected {len(burn_bins)} bins, "
            f"found {combined['burn_bin_idx'].nunique()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
