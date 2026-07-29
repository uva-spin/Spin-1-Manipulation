"""
Merge ssRF + AFP spectrum shards into a unified full-spectrum training dataset.

Writes ``spectrum_train.npz`` (default) or sharded ``spectrum_train_XXXX.npz`` files
with rows:
  ps[500], iplus[500], iminus[500], p0, step, center_bin, source, gamma_rf, burn_steps

Usage (from this directory):
  python combine_spectrum_train.py --strict
  python combine_spectrum_train.py \\
      --ssrf-shard-dir data/spectrum_ssrf_shards \\
      --afp-shard-dir data/spectrum_afp_shards \\
      --output data/spectrum_train/spectrum_train.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from bin_io import (
    _concat_spectrum_rows,
    _flatten_spectrum_shard,
    afp_spectrum_shard_path,
    load_spectrum_shard,
    ssrf_spectrum_shard_path,
)
from bin_setup import generate_unmanipulated_cube, get_shape_params
from common import (
    AFP_STEP_SUBSAMPLE,
    NUM_BINS,
    PHYSICS_MODEL,
    SPECTRUM_AFP_SHARD_DIR,
    SPECTRUM_SSRF_SHARD_DIR,
    SPECTRUM_TRAIN_DIR,
    SPECTRUM_TRAIN_NPZ,
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    UNMANIP_TRAIN_FRACTION,
    P_MAX,
    P_MIN,
    P_STEP,
)

SPECTRUM_KEYS = (
    "p0",
    "step",
    "center_bin",
    "source",
    "gamma_rf",
    "burn_steps",
    "ps",
    "iplus",
    "iminus",
)


def _validate_conservation(ps: np.ndarray, ip: np.ndarray, im: np.ndarray) -> float:
    """Return max |I+ + I- - Ps| over rows."""
    residual = np.abs(ip + im - ps)
    return float(np.max(residual)) if residual.size else 0.0


def _add_unmanip_rows(
    rows: dict[str, np.ndarray],
    *,
    num_bins: int,
    target_fraction: float,
    p_min: float,
    p_max: float,
    p_step: float,
) -> dict[str, np.ndarray]:
    n_existing = int(rows["ps"].shape[0]) if rows else 0
    if target_fraction <= 0.0:
        return rows
    n_unmanip = max(1, int(round(target_fraction * n_existing / max(1e-12, 1.0 - target_fraction))))
    if n_unmanip <= 0:
        return rows

    cube = generate_unmanipulated_cube(
        num_bins=num_bins,
        p_min=p_min,
        p_max=p_max,
        p_step=p_step,
        shape_params=get_shape_params(),
    )
    p_values = np.asarray(cube["p_values"], dtype=float)
    if p_values.size == 0:
        return rows

    pick = np.linspace(0, p_values.size - 1, n_unmanip, dtype=int)
    unmanip = {
        "p0": p_values[pick],
        "step": np.zeros(n_unmanip, dtype=np.int32),
        "center_bin": np.full(n_unmanip, num_bins // 2, dtype=np.int32),
        "source": np.full(n_unmanip, SOURCE_UNMANIP, dtype=np.uint8),
        "gamma_rf": np.zeros(n_unmanip, dtype=float),
        "burn_steps": np.zeros(n_unmanip, dtype=np.int32),
        "ps": np.asarray(cube["ps"][pick], dtype=float),
        "iplus": np.asarray(cube["iplus"][pick], dtype=float),
        "iminus": np.asarray(cube["iminus"][pick], dtype=float),
    }
    if not rows:
        return unmanip
    return _concat_spectrum_rows([rows, unmanip])


def combine_spectrum_shards(
    ssrf_shard_dir: Path,
    afp_shard_dir: Path,
    output_path: Path,
    *,
    num_bins: int = NUM_BINS,
    afp_step_subsample: int = AFP_STEP_SUBSAMPLE,
    unmanip_fraction: float = UNMANIP_TRAIN_FRACTION,
    p_min: float = P_MIN,
    p_max: float = P_MAX,
    p_step: float = P_STEP,
    strict: bool = True,
    shard_size: int = 0,
) -> dict:
    ssrf_shard_dir = Path(ssrf_shard_dir)
    afp_shard_dir = Path(afp_shard_dir)
    output_path = Path(output_path)

    parts: list[dict[str, np.ndarray]] = []
    missing_ssrf: list[int] = []
    missing_afp: list[int] = []

    for bin_idx in range(int(num_bins)):
        ssrf_path = ssrf_spectrum_shard_path(ssrf_shard_dir, bin_idx)
        if ssrf_path.is_file():
            shard = load_spectrum_shard(ssrf_path)
            if "ps_full" in shard:
                parts.append(
                    _flatten_spectrum_shard(
                        shard,
                        source=SOURCE_SSRF,
                        center_bin=bin_idx,
                        step_subsample=1,
                    )
                )
        else:
            missing_ssrf.append(bin_idx)

        afp_path = afp_spectrum_shard_path(afp_shard_dir, bin_idx)
        if afp_path.is_file():
            shard = load_spectrum_shard(afp_path)
            if "ps_full" in shard:
                sub = int(shard.get("step_subsample", afp_step_subsample))
                parts.append(
                    _flatten_spectrum_shard(
                        shard,
                        source=SOURCE_AFP,
                        center_bin=bin_idx,
                        step_subsample=sub,
                    )
                )
        else:
            missing_afp.append(bin_idx)

        if (bin_idx + 1) % 100 == 0:
            print(f"  scanned {bin_idx + 1}/{num_bins} bins", flush=True)

    if strict and (missing_ssrf or missing_afp):
        raise FileNotFoundError(
            f"Missing spectrum shards: ssrf={len(missing_ssrf)} afp={len(missing_afp)}"
        )

    merged = _concat_spectrum_rows(parts) if parts else {}
    if not merged:
        raise ValueError("No spectrum rows found in input shards")

    merged = _add_unmanip_rows(
        merged,
        num_bins=int(num_bins),
        target_fraction=float(unmanip_fraction),
        p_min=float(p_min),
        p_max=float(p_max),
        p_step=float(p_step),
    )

    max_res = _validate_conservation(merged["ps"], merged["iplus"], merged["iminus"])
    n_samples = int(merged["ps"].shape[0])
    source = merged["source"]
    meta = {
        "n_samples": n_samples,
        "num_bins": int(num_bins),
        "max_conservation_residual": max_res,
        "n_ssrf": int(np.count_nonzero(source == SOURCE_SSRF)),
        "n_afp": int(np.count_nonzero(source == SOURCE_AFP)),
        "n_unmanip": int(np.count_nonzero(source == SOURCE_UNMANIP)),
        "source_codes": {"ssrf": SOURCE_SSRF, "afp": SOURCE_AFP, "unmanipulated": SOURCE_UNMANIP},
        "physics_model": PHYSICS_MODEL,
        "dataset": "spectrum_train_v2",
        "fields": "ps,iplus,iminus shape (n_samples, num_bins)",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if int(shard_size) > 0:
        n_shards = int(np.ceil(n_samples / int(shard_size)))
        shard_dir = output_path if output_path.is_dir() else output_path.parent
        shard_dir.mkdir(parents=True, exist_ok=True)
        for si in range(n_shards):
            sl = slice(si * int(shard_size), (si + 1) * int(shard_size))
            shard_path = shard_dir / f"spectrum_train_{si:04d}.npz"
            payload = {k: merged[k][sl] for k in SPECTRUM_KEYS}
            shard_meta = dict(meta)
            shard_meta["shard_index"] = si
            shard_meta["n_shards"] = n_shards
            tmp = shard_path.with_name(f".{shard_path.stem}.{os.getpid()}.tmp.npz")
            np.savez_compressed(tmp, meta_json=np.asarray(json.dumps(shard_meta)), **payload)
            tmp.replace(shard_path)
        out_desc = str(shard_dir)
    else:
        out_file = output_path
        if output_path.is_dir():
            out_file = output_path / "spectrum_train.npz"
        tmp = out_file.with_name(f".{out_file.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp,
            meta_json=np.asarray(json.dumps(meta)),
            **{k: merged[k] for k in SPECTRUM_KEYS},
        )
        tmp.replace(out_file)
        out_desc = str(out_file)

    return {
        "output": out_desc,
        "n_samples": n_samples,
        "max_conservation_residual": max_res,
        "n_missing_ssrf": len(missing_ssrf),
        "n_missing_afp": len(missing_afp),
        **{k: meta[k] for k in ("n_ssrf", "n_afp", "n_unmanip")},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Combine ssRF/AFP spectrum shards into spectrum_train.npz")
    p.add_argument("--ssrf-shard-dir", type=Path, default=SPECTRUM_SSRF_SHARD_DIR)
    p.add_argument("--afp-shard-dir", type=Path, default=SPECTRUM_AFP_SHARD_DIR)
    p.add_argument("--output", type=Path, default=SPECTRUM_TRAIN_NPZ)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--afp-step-subsample", type=int, default=AFP_STEP_SUBSAMPLE)
    p.add_argument("--unmanip-fraction", type=float, default=UNMANIP_TRAIN_FRACTION)
    p.add_argument("--p-min", type=float, default=P_MIN)
    p.add_argument("--p-max", type=float, default=P_MAX)
    p.add_argument("--p-step", type=float, default=P_STEP)
    p.add_argument("--shard-size", type=int, default=0, help="If >0, write sharded NPZs of this many rows")
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        f"Combining spectrum shards ssrf={args.ssrf_shard_dir} afp={args.afp_shard_dir} "
        f"-> {args.output}",
        flush=True,
    )
    result = combine_spectrum_shards(
        args.ssrf_shard_dir,
        args.afp_shard_dir,
        args.output,
        num_bins=args.num_bins,
        afp_step_subsample=int(args.afp_step_subsample),
        unmanip_fraction=float(args.unmanip_fraction),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        p_step=float(args.p_step),
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
