"""Combine per-bin gamma-burn sweep shard CSVs into one table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine burning.slurm shard CSVs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "burning_shards",
        help="Directory containing sweep_bin_*.csv shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "sweep_gamma_burn_diffs.csv",
        help="Combined output CSV path.",
    )
    parser.add_argument("--pattern", type=str, default="sweep_bin_*.csv", help="Shard filename glob.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shards = sorted(args.input_dir.glob(args.pattern))
    if not shards:
        raise FileNotFoundError(f"No shards matching {args.pattern!r} in {args.input_dir}")

    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for shard in shards:
        with shard.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"Header mismatch in {shard}")
            rows.extend(reader)

    if fieldnames is None:
        raise ValueError("No rows found in shards")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"combined {len(shards)} shards ({len(rows)} rows) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
