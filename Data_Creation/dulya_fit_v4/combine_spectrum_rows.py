"""
Concatenate pre-flattened spectrum row NPZs into spectrum_train output.

Expects row files from ``flatten_spectrum_rows.py``:
  data/ssrf_spectrum_rows/ssrf_spectrum_rows_XXXX.npz
  data/afp_spectrum_rows/afp_spectrum_rows_XXXX.npz
  data/unmanip_spectrum_rows/unmanip_spectrum_rows.npz

Usage (from this directory):
  python combine_spectrum_rows.py --strict
  python combine_spectrum_rows.py \\
      --ssrf-rows-dir data/ssrf_spectrum_rows \\
      --afp-rows-dir data/afp_spectrum_rows \\
      --unmanip-rows-dir data/unmanip_spectrum_rows \\
      --output data/spectrum_train/spectrum_train.npz \\
      --shard-size 200000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from bin_io import (
    SPECTRUM_ROW_KEYS,
    SPECTRUM_TRAIN_MANIFEST_NAME,
    SpectrumShardWriter,
    _concat_spectrum_rows,
    _nonempty_spectrum_mask,
    afp_spectrum_rows_path,
    bin_index_range,
    format_missing_bins_error,
    load_spectrum_rows_npz,
    ssrf_spectrum_rows_path,
    unmanip_spectrum_rows_path,
)
from combine_spectrum_train import _validate_conservation
from common import (
    AFP_SPECTRUM_ROWS_DIR,
    NUM_BINS,
    PHYSICS_MODEL,
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    SPECTRUM_TRAIN_NPZ,
    SSRF_SPECTRUM_ROWS_DIR,
    UNMANIP_SPECTRUM_ROWS_DIR,
)


def _filter_rows(rows: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], int]:
    n_total = int(rows["ps"].shape[0])
    if n_total == 0:
        return rows, 0
    keep = _nonempty_spectrum_mask(rows["ps"], rows["iplus"], rows["iminus"])
    n_dropped = int(np.count_nonzero(~keep))
    if n_dropped == 0:
        return rows, 0
    return {key: values[keep] for key, values in rows.items()}, n_dropped


def _ingest_row_file(
    path: Path,
    *,
    source_key: str,
    stats: dict,
    writer: SpectrumShardWriter | None,
    parts: list[dict[str, np.ndarray]],
) -> None:
    loaded = load_spectrum_rows_npz(path)
    rows = {key: loaded[key] for key in SPECTRUM_ROW_KEYS}
    filtered, n_dropped = _filter_rows(rows)
    stats["n_filtered_empty"] += n_dropped
    n = int(filtered["ps"].shape[0])
    if n <= 0:
        return
    stats[source_key] += n
    res = _validate_conservation(filtered["ps"], filtered["iplus"], filtered["iminus"])
    stats["max_conservation_residual"] = max(stats["max_conservation_residual"], res)
    if writer is not None:
        writer.add(filtered)
    else:
        parts.append(filtered)


def combine_spectrum_row_dirs(
    ssrf_rows_dir: Path,
    afp_rows_dir: Path,
    output_path: Path,
    *,
    unmanip_rows_dir: Path = UNMANIP_SPECTRUM_ROWS_DIR,
    num_bins: int = NUM_BINS,
    strict: bool = True,
    shard_size: int = 0,
) -> dict:
    ssrf_rows_dir = Path(ssrf_rows_dir)
    afp_rows_dir = Path(afp_rows_dir)
    unmanip_rows_dir = Path(unmanip_rows_dir)
    output_path = Path(output_path)

    streaming = int(shard_size) > 0
    stats = {
        "n_ssrf": 0,
        "n_afp": 0,
        "n_unmanip": 0,
        "n_filtered_empty": 0,
        "max_conservation_residual": 0.0,
    }
    missing_ssrf: list[int] = []
    missing_afp: list[int] = []
    parts: list[dict[str, np.ndarray]] = []

    base_meta = {
        "num_bins": int(num_bins),
        "source_codes": {"ssrf": SOURCE_SSRF, "afp": SOURCE_AFP, "unmanipulated": SOURCE_UNMANIP},
        "physics_model": PHYSICS_MODEL,
        "dataset": "spectrum_train_v2",
        "fields": "ps,iplus,iminus shape (n_samples, num_bins)",
        "store_dtype": "float32",
        "combine_mode": "pre_flattened_rows",
        "ssrf_rows_dir": str(ssrf_rows_dir),
        "afp_rows_dir": str(afp_rows_dir),
        "unmanip_rows_dir": str(unmanip_rows_dir),
    }
    writer = (
        SpectrumShardWriter(
            output_path if output_path.is_dir() else output_path.parent,
            int(shard_size),
            base_meta,
        )
        if streaming
        else None
    )

    for bin_idx in bin_index_range(int(num_bins)):
        ssrf_path = ssrf_spectrum_rows_path(ssrf_rows_dir, bin_idx)
        if ssrf_path.is_file():
            _ingest_row_file(
                ssrf_path,
                source_key="n_ssrf",
                stats=stats,
                writer=writer,
                parts=parts,
            )
        else:
            missing_ssrf.append(bin_idx)

        afp_path = afp_spectrum_rows_path(afp_rows_dir, bin_idx)
        if afp_path.is_file():
            _ingest_row_file(
                afp_path,
                source_key="n_afp",
                stats=stats,
                writer=writer,
                parts=parts,
            )
        else:
            missing_afp.append(bin_idx)

        if (bin_idx + 1) % 100 == 0:
            print(f"  combined {bin_idx + 1}/{num_bins} bin row files", flush=True)

    if strict and (missing_ssrf or missing_afp):
        parts_err: list[str] = []
        if missing_ssrf:
            parts_err.append(
                format_missing_bins_error(
                    "ssRF spectrum row",
                    ssrf_rows_dir,
                    missing_ssrf,
                    num_bins=int(num_bins),
                    path_fn=ssrf_spectrum_rows_path,
                )
            )
        if missing_afp:
            parts_err.append(
                format_missing_bins_error(
                    "AFP spectrum row",
                    afp_rows_dir,
                    missing_afp,
                    num_bins=int(num_bins),
                    path_fn=afp_spectrum_rows_path,
                )
            )
        raise FileNotFoundError("\n".join(parts_err))

    unmanip_path = unmanip_spectrum_rows_path(unmanip_rows_dir)
    missing_unmanip = 0
    if unmanip_path.is_file():
        _ingest_row_file(
            unmanip_path,
            source_key="n_unmanip",
            stats=stats,
            writer=writer,
            parts=parts,
        )
    elif strict:
        raise FileNotFoundError(
            f"Missing unmanipulated spectrum rows under {unmanip_rows_dir}; "
            f"expected {unmanip_path.name}. Run: "
            "python flatten_spectrum_rows.py --source unmanip --strict"
        )
    else:
        missing_unmanip = 1
        print(
            f"WARNING: no {unmanip_path.name} under {unmanip_rows_dir}; skipping unmanip rows",
            flush=True,
        )

    if writer is not None:
        writer.close()
        n_samples = writer.n_written
        if n_samples <= 0:
            raise ValueError("No spectrum rows found in pre-flattened inputs")
        manifest = {
            "n_samples": n_samples,
            "n_shards": writer.shard_index,
            "shard_files": writer.shard_files,
            "shard_row_counts": writer.shard_row_counts,
            **stats,
            **base_meta,
        }
        (writer.output_dir / SPECTRUM_TRAIN_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2)
        )
        out_desc = str(writer.output_dir)
    else:
        if not parts:
            raise ValueError("No spectrum rows found in pre-flattened inputs")
        merged = _concat_spectrum_rows(parts)
        n_samples = int(merged["ps"].shape[0])
        if n_samples <= 0:
            raise ValueError("All spectrum rows were empty or invalid after filtering")
        meta = {"n_samples": n_samples, **stats, **base_meta}

        out_file = output_path
        if output_path.is_dir():
            out_file = output_path / "spectrum_train.npz"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_file.with_name(f".{out_file.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp,
            meta_json=np.asarray(json.dumps(meta)),
            **{k: merged[k] for k in SPECTRUM_ROW_KEYS},
        )
        tmp.replace(out_file)
        out_desc = str(out_file)

    return {
        "output": out_desc,
        "n_samples": n_samples,
        "max_conservation_residual": stats["max_conservation_residual"],
        "n_missing_ssrf": len(missing_ssrf),
        "n_missing_afp": len(missing_afp),
        "n_missing_unmanip": missing_unmanip,
        "n_ssrf": stats["n_ssrf"],
        "n_afp": stats["n_afp"],
        "n_unmanip": stats["n_unmanip"],
        "n_filtered_empty": stats["n_filtered_empty"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Concatenate pre-flattened spectrum row NPZs into spectrum_train"
    )
    p.add_argument(
        "--ssrf-rows-dir",
        type=Path,
        default=SSRF_SPECTRUM_ROWS_DIR,
        help="Dir with ssrf_spectrum_rows_XXXX.npz from flatten_spectrum_rows.py",
    )
    p.add_argument(
        "--afp-rows-dir",
        type=Path,
        default=AFP_SPECTRUM_ROWS_DIR,
        help="Dir with afp_spectrum_rows_XXXX.npz from flatten_spectrum_rows.py",
    )
    p.add_argument(
        "--unmanip-rows-dir",
        type=Path,
        default=UNMANIP_SPECTRUM_ROWS_DIR,
        help="Dir with unmanip_spectrum_rows.npz from flatten_spectrum_rows.py --source unmanip",
    )
    p.add_argument("--output", type=Path, default=SPECTRUM_TRAIN_NPZ)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help="If >0, stream output to sharded spectrum_train_XXXX.npz files",
    )
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        f"Combining pre-flattened rows ssrf={args.ssrf_rows_dir} afp={args.afp_rows_dir} "
        f"unmanip={args.unmanip_rows_dir} -> {args.output}",
        flush=True,
    )
    result = combine_spectrum_row_dirs(
        args.ssrf_rows_dir,
        args.afp_rows_dir,
        args.output,
        unmanip_rows_dir=args.unmanip_rows_dir,
        num_bins=int(args.num_bins),
        strict=bool(args.strict),
        shard_size=int(args.shard_size),
    )
    print(
        f"Wrote {result['n_samples']} spectrum rows -> {result['output']}  "
        f"ssrf={result['n_ssrf']} afp={result['n_afp']} unmanip={result['n_unmanip']}  "
        f"max|I++I--Ps|={result['max_conservation_residual']:.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
