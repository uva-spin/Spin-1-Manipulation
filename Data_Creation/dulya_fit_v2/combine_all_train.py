"""
Merge ssRF shards + AFP shards + unmanipulated bins into one NPZ per spectral bin.

Writes ``combined_train_all/train_bin_XXXX.npz`` with:
  p0, step, center_bin, is_mirror, source, ps, iplus, iminus, amp
  source: 0=ssrf, 1=afp, 2=unmanipulated

Streams one output bin at a time (own + mirror-partner shards only) so peak RAM
stays small.

Usage (from this directory):
  python combine_all_train.py --strict
  python combine_all_train.py \\
      --ssrf-shard-dir data/ssrf_shards \\
      --afp-shard-dir data/afp_shards \\
      --unmanip-dir data/unmanip_train \\
      --output-dir data/combined_train_all --strict
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from bin_io import (
    _organize_one_bin,
    afp_shard_path,
    load_afp_shard,
    load_ssrf_shard,
    ssrf_shard_path,
)
from common import (
    AFP_SHARD_DIR,
    COMBINED_TRAIN_ALL_DIR,
    NUM_BINS,
    PHYSICS_MODEL,
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    SSRF_SHARD_DIR,
    UNMANIP_TRAIN_DIR,
)

COMBINED_KEYS = (
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


def load_unmanip_bin(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        out: dict[str, np.ndarray] = {}
        for key in COMBINED_KEYS:
            if key not in data.files:
                raise KeyError(f"{path}: missing required field {key!r}")
            out[key] = np.asarray(data[key])
        return out


def _as_dtype(key: str, arr: np.ndarray) -> np.ndarray:
    if key in ("step", "center_bin"):
        return np.asarray(arr, dtype=np.int32)
    if key == "is_mirror":
        return np.asarray(arr, dtype=bool)
    if key == "source":
        return np.asarray(arr, dtype=np.uint8)
    return np.asarray(arr, dtype=float)


def _modality_to_combined(
    arrays: dict[str, np.ndarray],
    *,
    source: int,
    ref_key: str,
) -> dict[str, np.ndarray]:
    """Map a modality bag onto the combined schema (adds ``source``)."""
    n = int(np.asarray(arrays["ps"]).size)
    center = np.asarray(arrays[ref_key], dtype=np.int32)
    return {
        "p0": _as_dtype("p0", arrays["p0"]),
        "step": _as_dtype("step", arrays["step"]),
        "center_bin": center,
        "is_mirror": _as_dtype("is_mirror", arrays["is_mirror"]),
        "source": np.full(n, int(source), dtype=np.uint8),
        "ps": _as_dtype("ps", arrays["ps"]),
        "iplus": _as_dtype("iplus", arrays["iplus"]),
        "iminus": _as_dtype("iminus", arrays["iminus"]),
        "amp": _as_dtype("amp", arrays["amp"]),
    }


def _concat_combined(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    parts = [p for p in parts if int(np.asarray(p["ps"]).size) > 0]
    if not parts:
        return {k: _as_dtype(k, np.zeros(0)) for k in COMBINED_KEYS}
    if len(parts) == 1:
        return {k: _as_dtype(k, parts[0][k]) for k in COMBINED_KEYS}
    return {k: np.concatenate([_as_dtype(k, p[k]) for p in parts]) for k in COMBINED_KEYS}


def save_combined_bin(bin_idx: int, arrays: dict[str, np.ndarray], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = _as_dtype("source", arrays["source"])
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
        "physics_model": PHYSICS_MODEL,
        "dataset": "ssrf_afp_unmanip_train_bin_v2",
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
            **{k: _as_dtype(k, arrays[k]) for k in COMBINED_KEYS},
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def combine_all(
    ssrf_shard_dir: Path,
    afp_shard_dir: Path,
    unmanip_dir: Path,
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    strict: bool = True,
) -> dict:
    ssrf_shard_dir = Path(ssrf_shard_dir)
    afp_shard_dir = Path(afp_shard_dir)
    unmanip_dir = Path(unmanip_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_ssrf: list[int] = []
    missing_afp: list[int] = []
    missing_unmanip: list[int] = []
    samples_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    counts = {
        "ssrf": np.zeros(int(num_bins), dtype=np.int64),
        "afp": np.zeros(int(num_bins), dtype=np.int64),
        "unmanip": np.zeros(int(num_bins), dtype=np.int64),
    }

    for bin_idx in range(int(num_bins)):
        parts: list[dict[str, np.ndarray]] = []
        partner = int(num_bins) - 1 - int(bin_idx)

        has_ssrf = ssrf_shard_path(ssrf_shard_dir, bin_idx).is_file() or ssrf_shard_path(
            ssrf_shard_dir, partner
        ).is_file()
        if has_ssrf:
            ssrf = _organize_one_bin(
                bin_idx,
                num_bins=int(num_bins),
                shard_dir=ssrf_shard_dir,
                shard_path_fn=ssrf_shard_path,
                load_shard_fn=load_ssrf_shard,
                ref_key="center_bin",
                with_ssrf_params=True,
            )
            if int(ssrf["ps"].size) > 0:
                parts.append(
                    _modality_to_combined(ssrf, source=SOURCE_SSRF, ref_key="center_bin")
                )
        else:
            missing_ssrf.append(bin_idx)

        has_afp = afp_shard_path(afp_shard_dir, bin_idx).is_file() or afp_shard_path(
            afp_shard_dir, partner
        ).is_file()
        if has_afp:
            afp = _organize_one_bin(
                bin_idx,
                num_bins=int(num_bins),
                shard_dir=afp_shard_dir,
                shard_path_fn=afp_shard_path,
                load_shard_fn=load_afp_shard,
                ref_key="center_bin",
                with_ssrf_params=False,
            )
            if int(afp["ps"].size) > 0:
                parts.append(
                    _modality_to_combined(afp, source=SOURCE_AFP, ref_key="center_bin")
                )
        else:
            missing_afp.append(bin_idx)

        u_path = unmanip_bin_path(unmanip_dir, bin_idx)
        if u_path.is_file():
            parts.append(load_unmanip_bin(u_path))
        else:
            missing_unmanip.append(bin_idx)

        if not parts:
            continue

        merged = _concat_combined(parts)
        source = merged["source"]
        samples_per_bin[bin_idx] = int(merged["ps"].size)
        counts["ssrf"][bin_idx] = int(np.count_nonzero(source == SOURCE_SSRF))
        counts["afp"][bin_idx] = int(np.count_nonzero(source == SOURCE_AFP))
        counts["unmanip"][bin_idx] = int(np.count_nonzero(source == SOURCE_UNMANIP))
        save_combined_bin(bin_idx, merged, combined_bin_path(output_dir, bin_idx))

        if (bin_idx + 1) % 50 == 0 or bin_idx + 1 == int(num_bins):
            print(
                f"  combined {bin_idx + 1}/{int(num_bins)} bins "
                f"(running samples={int(samples_per_bin[: bin_idx + 1].sum())})",
                flush=True,
            )

    if strict and (missing_ssrf or missing_afp or missing_unmanip):
        raise FileNotFoundError(
            f"Missing ssrf={len(missing_ssrf)} afp={len(missing_afp)} "
            f"unmanip={len(missing_unmanip)}; "
            f"first ssrf={missing_ssrf[:1]} afp={missing_afp[:1]} "
            f"unmanip={missing_unmanip[:1]}"
        )
    if missing_ssrf:
        print(f"WARNING: missing ssRF data for {len(missing_ssrf)} bins", flush=True)
    if missing_afp:
        print(f"WARNING: missing AFP data for {len(missing_afp)} bins", flush=True)
    if missing_unmanip:
        print(
            f"WARNING: missing unmanip bins for {len(missing_unmanip)} bins",
            flush=True,
        )

    return {
        "output_dir": str(output_dir),
        "n_samples": int(samples_per_bin.sum()),
        "samples_per_bin": samples_per_bin,
        "ssrf_per_bin": counts["ssrf"],
        "afp_per_bin": counts["afp"],
        "unmanip_per_bin": counts["unmanip"],
        "n_missing_ssrf": len(missing_ssrf),
        "n_missing_afp": len(missing_afp),
        "n_missing_unmanip": len(missing_unmanip),
        "dataset": "ssrf_afp_unmanip_train_bin_v2",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge ssRF + AFP + unmanipulated into per-bin train_bin_XXXX.npz"
    )
    p.add_argument("--ssrf-shard-dir", type=Path, default=SSRF_SHARD_DIR)
    p.add_argument("--afp-shard-dir", type=Path, default=AFP_SHARD_DIR)
    p.add_argument("--unmanip-dir", type=Path, default=UNMANIP_TRAIN_DIR)
    p.add_argument("--output-dir", type=Path, default=COMBINED_TRAIN_ALL_DIR)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        f"Combining ssRF ({args.ssrf_shard_dir}) + AFP ({args.afp_shard_dir}) + "
        f"unmanip ({args.unmanip_dir}) -> {args.output_dir}",
        flush=True,
    )
    result = combine_all(
        args.ssrf_shard_dir,
        args.afp_shard_dir,
        args.unmanip_dir,
        args.output_dir,
        num_bins=args.num_bins,
        strict=bool(args.strict),
    )
    print(
        f"Combined {result['n_samples']} samples -> {args.output_dir} "
        f"(missing_ssrf={result['n_missing_ssrf']} "
        f"missing_afp={result['n_missing_afp']} "
        f"missing_unmanip={result['n_missing_unmanip']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
