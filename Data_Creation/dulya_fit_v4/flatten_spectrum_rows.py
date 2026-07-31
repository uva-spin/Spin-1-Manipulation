"""
Flatten per-bin spectrum shards (or unmanipulated bin NPZs) into pre-flattened
row NPZs for fast combining.

ssRF / AFP: reads ``*_spectrum_bin_XXXX.npz`` (or ``*_bin_XXXX.npz`` with
``ps_full``) and writes one row NPZ per bin:
  ps, iplus, iminus shape (n_rows, num_bins)

Unmanipulated: reads ``unmanip_bin_XXXX.npz`` equilibrium cubes and writes a
single ``unmanip_spectrum_rows.npz`` with one full-spectrum row per polarization.

Usage (from this directory):
  python flatten_spectrum_rows.py --source ssrf --bin-idx 0
  python flatten_spectrum_rows.py --source afp --flatten-all --strict
  python flatten_spectrum_rows.py --source unmanip --strict
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

import _bootstrap  # noqa: F401
from bin_io import (
    afp_spectrum_rows_path,
    afp_spectrum_shard_path,
    afp_shard_path,
    flatten_spectrum_shard_file_to_rows,
    flatten_spectrum_shard_files_to_rows,
    format_missing_bins_error,
    list_ssrf_spectrum_shard_paths,
    resolve_afp_spectrum_shard_path,
    resolve_ssrf_spectrum_shard_path,
    save_spectrum_rows_npz,
    ssrf_spectrum_rows_path,
    ssrf_spectrum_shard_path,
    ssrf_shard_path,
    unmanip_spectrum_rows_path,
)
from bin_setup import resolve_bin_idx
from combine_spectrum_train import _load_equilibrium_cube, _unmanip_rows_from_eq
from common import (
    AFP_SHARD_DIR,
    AFP_SPECTRUM_ROWS_DIR,
    AFP_STEP_SUBSAMPLE,
    BURN_BIN_CHOICES,
    NUM_BINS,
    SOURCE_AFP,
    SOURCE_SSRF,
    SSRF_SHARD_DIR,
    SSRF_SPECTRUM_ROWS_DIR,
    STORE_DTYPE,
    UNMANIP_SPECTRUM_ROWS_DIR,
    UNMANIP_TRAIN_DIR,
    is_burn_bin,
)
from unmanipulated_bin_lineshape import unmanip_bin_path

SourceName = Literal["ssrf", "afp", "unmanip"]


def _source_config(source: SourceName) -> dict:
    if source == "ssrf":
        return {
            "source_code": SOURCE_SSRF,
            "shard_dir_default": SSRF_SHARD_DIR,
            "rows_dir_default": SSRF_SPECTRUM_ROWS_DIR,
            "resolve_shard": resolve_ssrf_spectrum_shard_path,
            "rows_path_fn": ssrf_spectrum_rows_path,
            "spectrum_path_fn": ssrf_spectrum_shard_path,
            "traj_path_fn": ssrf_shard_path,
            "prefer_file_step_subsample": False,
        }
    if source == "afp":
        return {
            "source_code": SOURCE_AFP,
            "shard_dir_default": AFP_SHARD_DIR,
            "rows_dir_default": AFP_SPECTRUM_ROWS_DIR,
            "resolve_shard": resolve_afp_spectrum_shard_path,
            "rows_path_fn": afp_spectrum_rows_path,
            "spectrum_path_fn": afp_spectrum_shard_path,
            "traj_path_fn": afp_shard_path,
            "prefer_file_step_subsample": True,
        }
    return {
        "shard_dir_default": UNMANIP_TRAIN_DIR,
        "rows_dir_default": UNMANIP_SPECTRUM_ROWS_DIR,
    }


def flatten_one_bin(
    source: Literal["ssrf", "afp"],
    bin_idx: int,
    *,
    shard_dir: Path,
    output_dir: Path,
    afp_step_subsample: int = AFP_STEP_SUBSAMPLE,
) -> dict:
    cfg = _source_config(source)
    if source == "ssrf":
        shard_paths = list_ssrf_spectrum_shard_paths(shard_dir, int(bin_idx))
        if not shard_paths:
            raise FileNotFoundError(
                f"No spectrum shard for {source} bin_idx={bin_idx} under {shard_dir}"
            )
    else:
        shard_path = cfg["resolve_shard"](shard_dir, int(bin_idx))
        if shard_path is None:
            raise FileNotFoundError(
                f"No spectrum shard for {source} bin_idx={bin_idx} under {shard_dir} "
                f"(expected {cfg['spectrum_path_fn'](shard_dir, bin_idx).name} or "
                f"{cfg['traj_path_fn'](shard_dir, bin_idx).name} with ps_full)"
            )
        shard_paths = [shard_path]

    sub = 1 if source == "ssrf" else max(1, int(afp_step_subsample))
    if source == "ssrf" and len(shard_paths) > 1:
        rows = flatten_spectrum_shard_files_to_rows(
            shard_paths,
            source=int(cfg["source_code"]),
            center_bin=int(bin_idx),
            step_subsample=sub,
            prefer_file_step_subsample=bool(cfg["prefer_file_step_subsample"]),
        )
        input_desc = f"{len(shard_paths)} parts"
    else:
        rows = flatten_spectrum_shard_file_to_rows(
            shard_paths[0],
            source=int(cfg["source_code"]),
            center_bin=int(bin_idx),
            step_subsample=sub,
            prefer_file_step_subsample=bool(cfg["prefer_file_step_subsample"]),
        )
        input_desc = shard_paths[0].name
    n_rows = int(rows["ps"].shape[0])
    out_path = cfg["rows_path_fn"](output_dir, int(bin_idx))
    save_spectrum_rows_npz(
        out_path,
        rows,
        meta={
            "source": source,
            "bin_idx": int(bin_idx),
            "n_rows": n_rows,
            "input_shard": input_desc,
            "step_subsample": sub,
        },
    )
    return {"bin_idx": int(bin_idx), "n_rows": n_rows, "output": str(out_path)}


def flatten_all_bins(
    source: Literal["ssrf", "afp"],
    *,
    shard_dir: Path,
    output_dir: Path,
    num_bins: int = NUM_BINS,
    afp_step_subsample: int = AFP_STEP_SUBSAMPLE,
    strict: bool = True,
) -> dict:
    cfg = _source_config(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing: list[int] = []
    n_rows_total = 0
    n_bins_written = 0
    # Spectrum manip shards are generated only for burn-window bins.
    required_bins = [int(b) for b in np.asarray(BURN_BIN_CHOICES, dtype=int).tolist()]

    for bin_idx in required_bins:
        if source == "ssrf":
            if not list_ssrf_spectrum_shard_paths(shard_dir, bin_idx):
                missing.append(bin_idx)
                continue
        else:
            shard_path = cfg["resolve_shard"](shard_dir, bin_idx)
            if shard_path is None:
                missing.append(bin_idx)
                continue
        result = flatten_one_bin(
            source,
            bin_idx,
            shard_dir=shard_dir,
            output_dir=output_dir,
            afp_step_subsample=int(afp_step_subsample),
        )
        n_rows_total += int(result["n_rows"])
        n_bins_written += 1
        if (bin_idx + 1) % 100 == 0:
            print(f"  flattened {bin_idx + 1}/{num_bins} bins", flush=True)

    if strict and missing:
        raise FileNotFoundError(
            format_missing_bins_error(
                f"{source} spectrum shard",
                shard_dir,
                missing,
                num_bins=int(num_bins),
                path_fn=cfg["spectrum_path_fn"],
                traj_path_fn=cfg["traj_path_fn"],
            )
        )
    if missing and not strict:
        print(f"WARNING: skipped {len(missing)} bins with no spectrum shard", flush=True)

    return {
        "source": source,
        "output_dir": str(output_dir),
        "n_bins_written": n_bins_written,
        "n_rows": n_rows_total,
        "n_missing": len(missing),
    }


def flatten_unmanipulated_rows(
    *,
    unmanip_dir: Path,
    output_dir: Path,
    num_bins: int = NUM_BINS,
    strict: bool = True,
) -> dict:
    """Build full-spectrum unmanipulated rows from per-bin equilibrium NPZs."""
    unmanip_dir = Path(unmanip_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eq_p, eq_ps, eq_ip, eq_im, missing = _load_equilibrium_cube(
        unmanip_dir,
        num_bins=int(num_bins),
    )
    if strict and missing:
        raise FileNotFoundError(
            format_missing_bins_error(
                "unmanipulated bin",
                unmanip_dir,
                missing,
                num_bins=int(num_bins),
                path_fn=unmanip_bin_path,
            )
        )
    if int(eq_p.size) == 0:
        raise ValueError(f"No unmanip equilibrium data under {unmanip_dir}")

    rows = _unmanip_rows_from_eq(
        eq_p,
        np.asarray(eq_ps, dtype=STORE_DTYPE),
        np.asarray(eq_ip, dtype=STORE_DTYPE),
        np.asarray(eq_im, dtype=STORE_DTYPE),
        num_bins=int(num_bins),
    )
    n_rows = int(rows["ps"].shape[0])
    out_path = unmanip_spectrum_rows_path(output_dir)
    save_spectrum_rows_npz(
        out_path,
        rows,
        meta={
            "source": "unmanip",
            "n_rows": n_rows,
            "num_bins": int(num_bins),
            "unmanip_dir": str(unmanip_dir),
            "n_missing_bins": len(missing),
        },
    )
    return {
        "source": "unmanip",
        "output": str(out_path),
        "n_rows": n_rows,
        "n_missing": len(missing),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Flatten spectrum shards or unmanip bins into row-oriented NPZs"
    )
    p.add_argument("--source", choices=("ssrf", "afp", "unmanip"), required=True)
    p.add_argument("--bin-idx", type=int, default=None)
    p.add_argument(
        "--flatten-all",
        action="store_true",
        help="Flatten every bin (ssrf/afp only; omit --bin-idx)",
    )
    p.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Input shard dir (ssrf/afp) or unmanip bin dir (unmanip)",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--afp-step-subsample", type=int, default=AFP_STEP_SUBSAMPLE)
    p.add_argument("--skip-if-exists", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    source: SourceName = args.source
    cfg = _source_config(source)
    shard_dir = Path(args.shard_dir or cfg["shard_dir_default"])
    output_dir = Path(args.output_dir or cfg["rows_dir_default"])

    if source == "unmanip":
        out_path = unmanip_spectrum_rows_path(output_dir)
        if args.skip_if_exists and out_path.is_file():
            print(f"Skipping existing row file {out_path}", flush=True)
            return
        result = flatten_unmanipulated_rows(
            unmanip_dir=shard_dir,
            output_dir=output_dir,
            num_bins=int(args.num_bins),
            strict=bool(args.strict),
        )
        print(
            f"Wrote {result['n_rows']} unmanip rows -> {result['output']}  "
            f"missing_bins={result['n_missing']}",
            flush=True,
        )
        return

    if args.flatten_all:
        result = flatten_all_bins(
            source,
            shard_dir=shard_dir,
            output_dir=output_dir,
            num_bins=int(args.num_bins),
            afp_step_subsample=int(args.afp_step_subsample),
            strict=bool(args.strict),
        )
        print(
            f"Flattened {result['n_bins_written']} {source} bins -> {result['output_dir']}  "
            f"n_rows={result['n_rows']}  missing={result['n_missing']}",
            flush=True,
        )
        return

    bin_idx = resolve_bin_idx(args.bin_idx, num_bins=int(args.num_bins))
    if bin_idx is None:
        raise SystemExit(
            "Provide --bin-idx <int>, set SLURM_ARRAY_TASK_ID, or pass --flatten-all"
        )
    if not is_burn_bin(bin_idx):
        print(
            f"Skipping bin_idx={bin_idx}: outside burn window (no manip spectrum expected)",
            flush=True,
        )
        return

    out_path = cfg["rows_path_fn"](output_dir, bin_idx)
    if args.skip_if_exists and out_path.is_file():
        print(f"Skipping existing row file {out_path}", flush=True)
        return

    result = flatten_one_bin(
        source,
        bin_idx,
        shard_dir=shard_dir,
        output_dir=output_dir,
        afp_step_subsample=int(args.afp_step_subsample),
    )
    print(
        f"Wrote {result['n_rows']} {source} rows -> {result['output']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
