"""
Reconstruct dense num_bins-wide spectrum rows from independent per-bin
*trajectory* shards (single-bin mode, i.e. NOT --spectrum-mode).

Each bin_idx's shard was simulated on its own (burn centered at that bin),
but every bin in an array job shares the same (P, gamma_rf, burn_steps) combo
grid. So for a given combo index j and timestep t, stacking
shard[0]["ps"][j, t], shard[1]["ps"][j, t], ..., shard[499]["ps"][j, t] side
by side reconstructs the burn signal across the full spectrum for that
combo/timestep -- concatenation on bin_idx, nothing else. There is no
equilibrium fill and no cross-bin matching beyond the shared combo index; a
bin whose own combo was skipped (or missing) is simply left at 0 in that
column.

Every (row, bin) triple keeps its own ps/iplus/iminus together throughout
(they're always read from the same shard at the same (combo, step) index),
and by default the result is validated cell-by-cell against the physical
identity ps == iplus + iminus -- this is the cheapest possible check that
nothing got scrambled while stacking bins and concatenating sources.
Disable with --no-validate.

Reads (trajectory / single-bin mode shards, e.g. from ssrf_traj_array.slurm
and afp_traj_array.slurm -- do NOT use --spectrum-mode shards here):
  data/ssrf_shards/ssrf_bin_XXXX.npz
  data/afp_shards/afp_bin_XXXX.npz
  data/unmanip_train/unmanip_bin_XXXX.npz

Usage (from this directory):
  python combine_bin_events_simple.py --strict
  python combine_bin_events_simple.py \\
      --ssrf-shard-dir data/ssrf_shards \\
      --afp-shard-dir data/afp_shards \\
      --unmanip-dir data/unmanip_train \\
      --output data/spectrum_train/spectrum_train.npz --strict
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
    scan_traj_input_shards,
    stack_afp_traj_shards_from_dir,
    stack_ssrf_traj_shards_from_dir,
    stack_unmanip_bins_into_spectrum_rows,
    validate_ps_iplus_iminus,
)
from common import (
    AFP_SHARD_DIR,
    NUM_BINS,
    PHYSICS_MODEL,
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    SPECTRUM_TRAIN_NPZ,
    SSRF_SHARD_DIR,
    STORE_DTYPE,
    UNMANIP_TRAIN_DIR,
)
from unmanipulated_bin_lineshape import unmanip_bin_path


def _load_unmanip_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "p0": np.asarray(data["p0"], dtype=float),
            "ps": np.asarray(data["ps"], dtype=float),
            "iplus": np.asarray(data["iplus"], dtype=float),
            "iminus": np.asarray(data["iminus"], dtype=float),
        }


def combine_bin_events(
    ssrf_shard_dir: Path,
    afp_shard_dir: Path,
    output_path: Path,
    *,
    unmanip_dir: Path = UNMANIP_TRAIN_DIR,
    num_bins: int = NUM_BINS,
    strict: bool = True,
    validate: bool = True,
) -> dict:
    """Stack trajectory shards into full-spectrum rows without holding all bins in RAM."""
    ssrf_shard_dir = Path(ssrf_shard_dir)
    afp_shard_dir = Path(afp_shard_dir)
    unmanip_dir = Path(unmanip_dir)
    output_path = Path(output_path)
    nb = int(num_bins)

    availability = scan_traj_input_shards(
        ssrf_shard_dir, afp_shard_dir, unmanip_dir, num_bins=nb
    )
    missing_ssrf = availability["missing_ssrf"]
    missing_afp = availability["missing_afp"]
    missing_unmanip = availability["missing_unmanip"]

    print(
        f"Input shards present: ssrf={nb - len(missing_ssrf)}/{nb} "
        f"afp={nb - len(missing_afp)}/{nb} "
        f"unmanip={nb - len(missing_unmanip)}/{nb}",
        flush=True,
    )

    if strict and (missing_ssrf or missing_afp or missing_unmanip):
        raise FileNotFoundError(
            f"Missing ssrf={len(missing_ssrf)} afp={len(missing_afp)} "
            f"unmanip={len(missing_unmanip)}; "
            f"first ssrf={missing_ssrf[:1]} afp={missing_afp[:1]} "
            f"unmanip={missing_unmanip[:1]}"
        )

    print("Stacking ssRF shards into spectrum rows...", flush=True)
    ssrf_rows = stack_ssrf_traj_shards_from_dir(
        ssrf_shard_dir, num_bins=nb, strict=strict
    )
    print("Stacking AFP shards into spectrum rows...", flush=True)
    afp_rows = stack_afp_traj_shards_from_dir(
        afp_shard_dir, num_bins=nb, strict=strict
    )

    unmanip_arrays: list[dict | None] = [None] * nb
    for bin_idx in range(nb):
        upath = unmanip_bin_path(unmanip_dir, bin_idx)
        if upath.is_file():
            unmanip_arrays[bin_idx] = _load_unmanip_arrays(upath)
        if (bin_idx + 1) % 100 == 0 or bin_idx + 1 == nb:
            print(f"  loaded unmanip {bin_idx + 1}/{nb} bins", flush=True)

    print("Stacking unmanipulated bins into spectrum rows...", flush=True)
    unmanip_rows = stack_unmanip_bins_into_spectrum_rows(unmanip_arrays, num_bins=nb, strict=strict)
    print(
        f"  unmanip rows: {int(unmanip_rows['ps'].shape[0])}  "
        f"(ssRF={int(ssrf_rows['ps'].shape[0])}  AFP={int(afp_rows['ps'].shape[0])})",
        flush=True,
    )

    if validate:
        # Confirms ps/iplus/iminus stayed aligned to the same (row, bin) cell
        # through stacking + concatenation: ps must equal iplus + iminus
        # everywhere (including 0 == 0 + 0 at zero-filled cells).
        print("Validating ssRF rows...", flush=True)
        validate_ps_iplus_iminus(ssrf_rows, label="ssRF rows")
        print("Validating AFP rows...", flush=True)
        validate_ps_iplus_iminus(afp_rows, label="AFP rows")
        print("Validating unmanip rows...", flush=True)
        validate_ps_iplus_iminus(unmanip_rows, label="unmanip rows")
        print("Validated ps == iplus + iminus for ssRF, AFP, and unmanip rows.", flush=True)

    stats = {
        "n_ssrf": int(ssrf_rows["ps"].shape[0]),
        "n_afp": int(afp_rows["ps"].shape[0]),
        "n_unmanip": int(unmanip_rows["ps"].shape[0]),
    }

    print(f"Merging {stats['n_ssrf'] + stats['n_afp'] + stats['n_unmanip']} total rows...", flush=True)
    merged = {
        key: np.concatenate([ssrf_rows[key], afp_rows[key], unmanip_rows[key]], axis=0)
        for key in SPECTRUM_ROW_KEYS
    }
    n_samples = int(merged["ps"].shape[0])
    if n_samples == 0:
        raise ValueError("No events found in any ssRF/AFP/unmanip shard")

    if validate:
        print("Validating merged rows...", flush=True)
        validate_ps_iplus_iminus(merged, label="merged spectrum_train rows")
        print("Validated ps == iplus + iminus on the final concatenated array.", flush=True)

    print(f"Writing compressed NPZ ({n_samples} rows x {nb} bins)...", flush=True)

    meta = {
        "n_samples": n_samples,
        "num_bins": nb,
        "source_codes": {"ssrf": SOURCE_SSRF, "afp": SOURCE_AFP, "unmanipulated": SOURCE_UNMANIP},
        "physics_model": PHYSICS_MODEL,
        "dataset": "spectrum_train_bin_stack_v1",
        "fields": (
            "ps,iplus,iminus (n_samples, num_bins); column b is bin b's own "
            "single-bin trajectory value for the shared (p0, gamma_rf, "
            "burn_steps, step) combo -- 0 where that bin's shard is missing "
            "or its combo was skipped."
        ),
        "store_dtype": str(np.dtype(STORE_DTYPE)),
        "combine_mode": "stack_bins_shared_combo_grid",
        **stats,
        "n_missing_ssrf": len(missing_ssrf),
        "n_missing_afp": len(missing_afp),
        "n_missing_unmanip": len(missing_unmanip),
    }

    out_file = output_path
    if output_path.suffix != ".npz":
        out_file = output_path / "spectrum_train.npz"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_name(f".{out_file.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, meta_json=np.asarray(json.dumps(meta)), **merged)
        tmp.replace(out_file)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise

    return {
        "output": str(out_file),
        "n_samples": n_samples,
        **stats,
        "n_missing_ssrf": len(missing_ssrf),
        "n_missing_afp": len(missing_afp),
        "n_missing_unmanip": len(missing_unmanip),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stack ssRF+AFP+unmanip trajectory shards (shared combo grid, one column "
            "per bin_idx) into one num_bins-wide spectrum_train.npz."
        )
    )
    p.add_argument("--ssrf-shard-dir", type=Path, default=SSRF_SHARD_DIR)
    p.add_argument("--afp-shard-dir", type=Path, default=AFP_SHARD_DIR)
    p.add_argument("--unmanip-dir", type=Path, default=UNMANIP_TRAIN_DIR)
    p.add_argument("--output", type=Path, default=SPECTRUM_TRAIN_NPZ)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--strict", action="store_true")
    p.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip the ps == iplus + iminus alignment check (validated by default).",
    )
    p.set_defaults(validate=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        f"Combining ssRF ({args.ssrf_shard_dir}) + AFP ({args.afp_shard_dir}) + "
        f"unmanip ({args.unmanip_dir}) -> {args.output} (num_bins={args.num_bins})",
        flush=True,
    )
    result = combine_bin_events(
        args.ssrf_shard_dir,
        args.afp_shard_dir,
        args.output,
        unmanip_dir=args.unmanip_dir,
        num_bins=args.num_bins,
        strict=bool(args.strict),
        validate=bool(args.validate),
    )
    print(
        f"Wrote {result['n_samples']} events -> {result['output']} "
        f"(n_ssrf={result['n_ssrf']} n_afp={result['n_afp']} n_unmanip={result['n_unmanip']}; "
        f"missing_ssrf={result['n_missing_ssrf']} missing_afp={result['n_missing_afp']} "
        f"missing_unmanip={result['n_missing_unmanip']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
