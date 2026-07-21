"""
Combine ssRF and AFP trajectory shards into one training NPZ per spectral bin.

Each modality's shards record (ps, iplus, iminus) at the center bin and its
mirror. Samples are routed to the bin they belong to (center → center file,
mirror → mirror file), then concatenated with a ``source`` tag:

  source == 0  → ssRF
  source == 1  → AFP

(Unmanipulated lineshapes are source == 2; see unmanipulated_bin_lineshape.py
and merge_unmanip_into_combined.py.)

Usage:
  python combine_ssrf_afp_train.py \\
      --ssrf-shard-dir ssrf_shards --afp-shard-dir afp_shards \\
      --output-dir combined_train --strict
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

from physics.lineshape.rate_eqs_test import afp_bin_traj as afp_mod
from physics.lineshape.rate_eqs_test import ssrf_bin_traj as ssrf_mod

NUM_BINS = 500
SOURCE_SSRF = 0
SOURCE_AFP = 1

DEFAULT_SSRF_SHARDS = Path(__file__).resolve().parent / "ssrf_shards"
DEFAULT_AFP_SHARDS = Path(__file__).resolve().parent / "afp_shards"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "combined_train"


def combined_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"train_bin_{int(bin_idx):04d}.npz"


def _empty_bags(num_bins: int) -> list[dict[str, list[np.ndarray]]]:
    keys = (
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
    return [{k: [] for k in keys} for _ in range(int(num_bins))]


def _append(
    bag: dict[str, list[np.ndarray]],
    *,
    p0: float,
    n: int,
    center_bin: int,
    is_mirror: bool,
    source: int,
    ps: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    amp: np.ndarray,
) -> None:
    bag["p0"].append(np.full(n, float(p0), dtype=float))
    bag["step"].append(np.arange(n, dtype=np.int32))
    bag["center_bin"].append(np.full(n, int(center_bin), dtype=np.int32))
    bag["is_mirror"].append(np.full(n, bool(is_mirror), dtype=bool))
    bag["source"].append(np.full(n, int(source), dtype=np.uint8))
    bag["ps"].append(np.asarray(ps, dtype=float))
    bag["iplus"].append(np.asarray(iplus, dtype=float))
    bag["iminus"].append(np.asarray(iminus, dtype=float))
    bag["amp"].append(np.asarray(amp, dtype=float))


def _finalize(bag: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    if not bag["ps"]:
        return {
            "p0": np.zeros(0, dtype=float),
            "step": np.zeros(0, dtype=np.int32),
            "center_bin": np.zeros(0, dtype=np.int32),
            "is_mirror": np.zeros(0, dtype=bool),
            "source": np.zeros(0, dtype=np.uint8),
            "ps": np.zeros(0, dtype=float),
            "iplus": np.zeros(0, dtype=float),
            "iminus": np.zeros(0, dtype=float),
            "amp": np.zeros(0, dtype=float),
        }
    return {k: np.concatenate(v) for k, v in bag.items()}


def _amp_pair(shard: dict) -> tuple[np.ndarray, np.ndarray]:
    ps = np.asarray(shard["ps"], dtype=float)
    ps_m = np.asarray(shard["ps_m"], dtype=float)
    amp = np.asarray(shard["amp"], dtype=float) if "amp" in shard else np.abs(ps)
    amp_m = np.asarray(shard["amp_m"], dtype=float) if "amp_m" in shard else np.abs(ps_m)
    return amp, amp_m


def _ingest_shard(bags: list[dict[str, list[np.ndarray]]], shard: dict, source: int) -> None:
    b = int(shard["bin_idx"])
    m = int(shard["mirror_idx"])
    p_values = np.asarray(shard["p_values"], dtype=float)
    n_steps = np.asarray(shard["n_steps"], dtype=np.int32)
    amp, amp_m = _amp_pair(shard)
    for j, p0 in enumerate(p_values):
        n = int(n_steps[j])
        if n <= 0:
            continue
        _append(
            bags[b],
            p0=float(p0),
            n=n,
            center_bin=b,
            is_mirror=False,
            source=source,
            ps=shard["ps"][j, :n],
            iplus=shard["iplus"][j, :n],
            iminus=shard["iminus"][j, :n],
            amp=amp[j, :n],
        )
        if m != b:
            _append(
                bags[m],
                p0=float(p0),
                n=n,
                center_bin=b,
                is_mirror=True,
                source=source,
                ps=shard["ps_m"][j, :n],
                iplus=shard["iplus_m"][j, :n],
                iminus=shard["iminus_m"][j, :n],
                amp=amp_m[j, :n],
            )


def _load_modality(
    bags: list[dict[str, list[np.ndarray]]],
    shard_dir: Path,
    *,
    num_bins: int,
    source: int,
    shard_path_fn,
    load_fn,
    label: str,
    strict: bool,
) -> list[int]:
    missing: list[int] = []
    shard_dir = Path(shard_dir)
    for bin_idx in range(int(num_bins)):
        path = shard_path_fn(shard_dir, bin_idx)
        if not path.is_file():
            missing.append(bin_idx)
            continue
        print(f"  [{label}] bin {bin_idx}/{num_bins - 1}", flush=True)
        _ingest_shard(bags, load_fn(path), source)
    if missing and strict:
        raise FileNotFoundError(
            f"Missing {len(missing)} {label} shard(s) under {shard_dir}; "
            f"first missing bin_idx={missing[0]}"
        )
    if missing:
        print(
            f"WARNING: missing {len(missing)} {label} shards; continuing",
            flush=True,
        )
    return missing


def save_combined_bin(
    bin_idx: int,
    arrays: dict[str, np.ndarray],
    path: Path,
    *,
    n_missing_ssrf: int,
    n_missing_afp: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = np.asarray(arrays["source"], dtype=np.uint8)
    n_samples = int(np.asarray(arrays["ps"]).size)
    n_ssrf = int(np.count_nonzero(source == SOURCE_SSRF))
    n_afp = int(np.count_nonzero(source == SOURCE_AFP))
    meta = {
        "bin_idx": int(bin_idx),
        "n_samples": n_samples,
        "n_ssrf": n_ssrf,
        "n_afp": n_afp,
        "n_missing_ssrf_shards": int(n_missing_ssrf),
        "n_missing_afp_shards": int(n_missing_afp),
        "source_codes": {"ssrf": SOURCE_SSRF, "afp": SOURCE_AFP},
        "dataset": "ssrf_afp_train_bin",
        "fields": (
            "ps,iplus,iminus,amp at this bin; "
            "center_bin=RF/AFP center; is_mirror; source"
        ),
    }
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            bin_idx=np.asarray(int(bin_idx), dtype=np.int32),
            p0=np.asarray(arrays["p0"], dtype=float),
            step=np.asarray(arrays["step"], dtype=np.int32),
            center_bin=np.asarray(arrays["center_bin"], dtype=np.int32),
            is_mirror=np.asarray(arrays["is_mirror"], dtype=bool),
            source=source,
            ps=np.asarray(arrays["ps"], dtype=float),
            iplus=np.asarray(arrays["iplus"], dtype=float),
            iminus=np.asarray(arrays["iminus"], dtype=float),
            amp=np.asarray(arrays["amp"], dtype=float),
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def combine_ssrf_afp(
    ssrf_shard_dir: Path,
    afp_shard_dir: Path,
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    strict: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bags = _empty_bags(num_bins)

    print(f"Ingesting ssRF shards from {ssrf_shard_dir}", flush=True)
    missing_ssrf = _load_modality(
        bags,
        ssrf_shard_dir,
        num_bins=num_bins,
        source=SOURCE_SSRF,
        shard_path_fn=ssrf_mod.shard_path,
        load_fn=ssrf_mod.load_shard,
        label="ssrf",
        strict=strict,
    )

    print(f"Ingesting AFP shards from {afp_shard_dir}", flush=True)
    missing_afp = _load_modality(
        bags,
        afp_shard_dir,
        num_bins=num_bins,
        source=SOURCE_AFP,
        shard_path_fn=afp_mod.shard_path,
        load_fn=afp_mod.load_shard,
        label="afp",
        strict=strict,
    )

    samples_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    ssrf_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    afp_per_bin = np.zeros(int(num_bins), dtype=np.int64)

    print(f"Writing {num_bins} combined per-bin files to {output_dir}", flush=True)
    for bin_idx in range(int(num_bins)):
        arrays = _finalize(bags[bin_idx])
        samples_per_bin[bin_idx] = int(arrays["ps"].size)
        src = arrays["source"]
        ssrf_per_bin[bin_idx] = int(np.count_nonzero(src == SOURCE_SSRF))
        afp_per_bin[bin_idx] = int(np.count_nonzero(src == SOURCE_AFP))
        save_combined_bin(
            bin_idx,
            arrays,
            combined_bin_path(output_dir, bin_idx),
            n_missing_ssrf=len(missing_ssrf),
            n_missing_afp=len(missing_afp),
        )
        if (bin_idx + 1) % 50 == 0 or bin_idx == int(num_bins) - 1:
            print(f"  wrote through bin {bin_idx}", flush=True)

    return {
        "output_dir": str(output_dir),
        "n_samples": int(samples_per_bin.sum()),
        "samples_per_bin": samples_per_bin,
        "ssrf_per_bin": ssrf_per_bin,
        "afp_per_bin": afp_per_bin,
        "n_missing_ssrf": len(missing_ssrf),
        "n_missing_afp": len(missing_afp),
        "dataset": "ssrf_afp_train_bin",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge ssRF + AFP shards into per-bin combined training NPZs"
    )
    p.add_argument("--ssrf-shard-dir", type=Path, default=DEFAULT_SSRF_SHARDS)
    p.add_argument("--afp-shard-dir", type=Path, default=DEFAULT_AFP_SHARDS)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = combine_ssrf_afp(
        args.ssrf_shard_dir,
        args.afp_shard_dir,
        args.output_dir,
        num_bins=args.num_bins,
        strict=bool(args.strict),
    )
    print(
        f"Combined {result['n_samples']} samples -> {args.output_dir} "
        f"({args.num_bins} bin files; "
        f"missing_ssrf={result['n_missing_ssrf']} "
        f"missing_afp={result['n_missing_afp']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
