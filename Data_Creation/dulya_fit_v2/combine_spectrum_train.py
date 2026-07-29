"""
Merge ssRF + AFP spectrum shards + unmanipulated bin NPZs into a unified full-spectrum
training dataset.

Writes ``spectrum_train.npz`` (default) or sharded ``spectrum_train_XXXX.npz`` files
with rows:
  ps[500], iplus[500], iminus[500], p0, step, center_bin, source, gamma_rf, burn_steps

Unmanipulated rows come from ``unmanip_bin_XXXX.npz`` files (``unmanipulated_bin_array``),
stacked into full 500-bin equilibrium spectra (one row per polarization). All available
unmanip rows are included (no subsampling).

Usage (from this directory):
  python combine_spectrum_train.py --strict
  python combine_spectrum_train.py \\
      --ssrf-shard-dir data/ssrf_shards \\
      --afp-shard-dir data/afp_shards \\
      --unmanip-dir data/unmanip_train \\
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
    afp_shard_path,
    bin_index_range,
    format_missing_bins_error,
    load_afp_shard,
    load_spectrum_shard,
    load_ssrf_shard,
    ssrf_shard_path,
)
from common import (
    AFP_SHARD_DIR,
    AFP_STEP_SUBSAMPLE,
    NUM_BINS,
    PHYSICS_MODEL,
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    SPECTRUM_TRAIN_NPZ,
    SSRF_SHARD_DIR,
    UNMANIP_TRAIN_DIR,
)
from unmanipulated_bin_lineshape import unmanip_bin_path

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


def _empty_spectrum_rows(num_bins: int) -> dict[str, np.ndarray]:
    nb = int(num_bins)
    return {
        "p0": np.zeros(0, dtype=float),
        "step": np.zeros(0, dtype=np.int32),
        "center_bin": np.zeros(0, dtype=np.int32),
        "source": np.zeros(0, dtype=np.uint8),
        "gamma_rf": np.zeros(0, dtype=float),
        "burn_steps": np.zeros(0, dtype=np.int32),
        "ps": np.zeros((0, nb), dtype=float),
        "iplus": np.zeros((0, nb), dtype=float),
        "iminus": np.zeros((0, nb), dtype=float),
    }


def _p_row_index(p_values: np.ndarray, p0: float) -> int:
    grid = np.asarray(p_values, dtype=float)
    idx = int(np.argmin(np.abs(grid - float(p0))))
    if abs(float(grid[idx]) - float(p0)) > 1e-5:
        raise ValueError(f"p0={p0} not on equilibrium grid (closest {grid[idx]})")
    return idx


def _load_equilibrium_cube(
    unmanip_dir: Path,
    *,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Load (p_values, ps, iplus, iminus) cubes from unmanip_bin_XXXX.npz files."""
    unmanip_dir = Path(unmanip_dir)
    missing: list[int] = []
    p_values: np.ndarray | None = None
    ps_cube: np.ndarray | None = None
    ip_cube: np.ndarray | None = None
    im_cube: np.ndarray | None = None

    for bin_idx in bin_index_range(int(num_bins)):
        path = unmanip_bin_path(unmanip_dir, bin_idx)
        if not path.is_file():
            missing.append(bin_idx)
            continue
        with np.load(path, allow_pickle=False) as data:
            p0 = np.asarray(data["p0"], dtype=float)
            if p_values is None:
                p_values = p0
                n_p = int(p0.size)
                ps_cube = np.full((n_p, int(num_bins)), np.nan, dtype=float)
                ip_cube = np.full((n_p, int(num_bins)), np.nan, dtype=float)
                im_cube = np.full((n_p, int(num_bins)), np.nan, dtype=float)
            elif p0.shape != p_values.shape or not np.allclose(p0, p_values):
                raise ValueError(f"{path}: p0 grid mismatch vs other unmanip bins")
            ps_cube[:, bin_idx] = np.asarray(data["ps"], dtype=float)
            ip_cube[:, bin_idx] = np.asarray(data["iplus"], dtype=float)
            im_cube[:, bin_idx] = np.asarray(data["iminus"], dtype=float)

    if p_values is None or ps_cube is None or ip_cube is None or im_cube is None:
        empty = np.zeros((0, int(num_bins)), dtype=float)
        return np.zeros(0, dtype=float), empty, empty, empty, missing
    return p_values, ps_cube, ip_cube, im_cube, missing


def _flatten_traj_shard_to_spectrum(
    shard: dict,
    *,
    source: int,
    center_bin: int,
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
    step_subsample: int = 1,
    with_burn_params: bool = True,
) -> dict[str, np.ndarray]:
    """Embed ssrf_bin/afp_bin trajectory shards into full-spectrum rows."""
    p_values = np.asarray(shard["p_values"], dtype=float)
    n_steps = np.asarray(shard["n_steps"], dtype=np.int32)
    skipped = np.asarray(shard.get("skipped", np.zeros_like(p_values, dtype=bool)), dtype=bool)
    ps = np.asarray(shard["ps"], dtype=float)
    ip = np.asarray(shard["iplus"], dtype=float)
    im = np.asarray(shard["iminus"], dtype=float)
    num_bins = int(eq_ps.shape[1])
    sub = max(1, int(step_subsample))
    c = int(center_bin)

    gamma_rf = np.full(int(p_values.size), np.nan, dtype=float)
    if with_burn_params and "gamma_rf" in shard:
        gamma_rf = np.asarray(shard["gamma_rf"], dtype=float)
    burn_steps = np.full(int(p_values.size), -1, dtype=np.int32)
    if with_burn_params and "burn_steps" in shard:
        burn_steps = np.asarray(shard["burn_steps"], dtype=np.int32)

    rows: list[tuple[int, int]] = []
    for j in range(int(p_values.size)):
        if bool(skipped[j]):
            continue
        n = int(n_steps[j])
        if n <= 0:
            continue
        for step in range(0, n, sub):
            rows.append((j, step))

    total = len(rows)
    if total <= 0:
        return _empty_spectrum_rows(num_bins)

    out = {
        "p0": np.empty(total, dtype=float),
        "step": np.empty(total, dtype=np.int32),
        "center_bin": np.empty(total, dtype=np.int32),
        "source": np.empty(total, dtype=np.uint8),
        "gamma_rf": np.empty(total, dtype=float),
        "burn_steps": np.empty(total, dtype=np.int32),
        "ps": np.empty((total, num_bins), dtype=float),
        "iplus": np.empty((total, num_bins), dtype=float),
        "iminus": np.empty((total, num_bins), dtype=float),
    }
    for idx, (j, step) in enumerate(rows):
        pi = _p_row_index(eq_p, float(p_values[j]))
        out["p0"][idx] = float(p_values[j])
        out["step"][idx] = int(step)
        out["center_bin"][idx] = c
        out["source"][idx] = np.uint8(int(source))
        out["gamma_rf"][idx] = float(gamma_rf[j])
        out["burn_steps"][idx] = int(burn_steps[j])
        out["ps"][idx] = eq_ps[pi]
        out["iplus"][idx] = eq_ip[pi]
        out["iminus"][idx] = eq_im[pi]
        out["ps"][idx, c] = float(ps[j, step])
        out["iplus"][idx, c] = float(ip[j, step])
        out["iminus"][idx, c] = float(im[j, step])
    return out


def _shard_has_ps_full(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as data:
        return "ps_full" in data.files


def _flatten_ssrf_shard(
    path: Path,
    *,
    center_bin: int,
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
) -> dict[str, np.ndarray]:
    if _shard_has_ps_full(path):
        shard = load_spectrum_shard(path)
        return _flatten_spectrum_shard(
            shard,
            source=SOURCE_SSRF,
            center_bin=center_bin,
            step_subsample=1,
            exclude_trailing_samples=int(shard.get("n_unmanip_samples", 0)),
        )
    shard = load_ssrf_shard(path)
    return _flatten_traj_shard_to_spectrum(
        shard,
        source=SOURCE_SSRF,
        center_bin=center_bin,
        eq_p=eq_p,
        eq_ps=eq_ps,
        eq_ip=eq_ip,
        eq_im=eq_im,
        step_subsample=1,
        with_burn_params=True,
    )


def _flatten_afp_shard(
    path: Path,
    *,
    center_bin: int,
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
    afp_step_subsample: int,
) -> dict[str, np.ndarray]:
    if _shard_has_ps_full(path):
        shard = load_spectrum_shard(path)
        sub = int(shard.get("step_subsample", afp_step_subsample))
        return _flatten_spectrum_shard(
            shard,
            source=SOURCE_AFP,
            center_bin=center_bin,
            step_subsample=sub,
            exclude_trailing_samples=int(shard.get("n_unmanip_samples", 0)),
        )
    shard = load_afp_shard(path)
    return _flatten_traj_shard_to_spectrum(
        shard,
        source=SOURCE_AFP,
        center_bin=center_bin,
        eq_p=eq_p,
        eq_ps=eq_ps,
        eq_ip=eq_ip,
        eq_im=eq_im,
        step_subsample=int(afp_step_subsample),
        with_burn_params=False,
    )


def _load_unmanip_spectrum_rows(
    unmanip_dir: Path,
    *,
    num_bins: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Stack per-bin ``unmanip_bin_XXXX.npz`` files into full-spectrum rows."""
    p_values, ps_cube, ip_cube, im_cube, missing = _load_equilibrium_cube(
        unmanip_dir,
        num_bins=int(num_bins),
    )
    if int(p_values.size) == 0:
        return {}, missing

    n_p = int(p_values.size)
    unmanip_center = int(num_bins // 2)
    rows = {
        "p0": p_values,
        "step": np.zeros(n_p, dtype=np.int32),
        "center_bin": np.full(n_p, unmanip_center, dtype=np.int32),
        "source": np.full(n_p, SOURCE_UNMANIP, dtype=np.uint8),
        "gamma_rf": np.zeros(n_p, dtype=float),
        "burn_steps": np.zeros(n_p, dtype=np.int32),
        "ps": ps_cube,
        "iplus": ip_cube,
        "iminus": im_cube,
    }
    return rows, missing


def combine_spectrum_shards(
    ssrf_shard_dir: Path,
    afp_shard_dir: Path,
    output_path: Path,
    *,
    unmanip_dir: Path = UNMANIP_TRAIN_DIR,
    num_bins: int = NUM_BINS,
    afp_step_subsample: int = AFP_STEP_SUBSAMPLE,
    strict: bool = True,
    shard_size: int = 0,
) -> dict:
    ssrf_shard_dir = Path(ssrf_shard_dir)
    afp_shard_dir = Path(afp_shard_dir)
    unmanip_dir = Path(unmanip_dir)
    output_path = Path(output_path)

    eq_p, eq_ps, eq_ip, eq_im, missing_unmanip = _load_equilibrium_cube(
        unmanip_dir,
        num_bins=int(num_bins),
    )
    if strict and missing_unmanip:
        raise FileNotFoundError(
            format_missing_bins_error(
                "unmanipulated bin",
                unmanip_dir,
                missing_unmanip,
                num_bins=int(num_bins),
                path_fn=unmanip_bin_path,
            )
        )
    if int(eq_p.size) == 0:
        raise ValueError(
            f"No unmanip equilibrium data under {unmanip_dir}; "
            "trajectory shards need unmanip_bin_XXXX.npz to fill full spectra"
        )

    parts: list[dict[str, np.ndarray]] = []
    missing_ssrf: list[int] = []
    missing_afp: list[int] = []

    for bin_idx in bin_index_range(int(num_bins)):
        ssrf_path = ssrf_shard_path(ssrf_shard_dir, bin_idx)
        if ssrf_path.is_file():
            parts.append(
                _flatten_ssrf_shard(
                    ssrf_path,
                    center_bin=bin_idx,
                    eq_p=eq_p,
                    eq_ps=eq_ps,
                    eq_ip=eq_ip,
                    eq_im=eq_im,
                )
            )
        else:
            missing_ssrf.append(bin_idx)

        afp_path = afp_shard_path(afp_shard_dir, bin_idx)
        if afp_path.is_file():
            parts.append(
                _flatten_afp_shard(
                    afp_path,
                    center_bin=bin_idx,
                    eq_p=eq_p,
                    eq_ps=eq_ps,
                    eq_ip=eq_ip,
                    eq_im=eq_im,
                    afp_step_subsample=int(afp_step_subsample),
                )
            )
        else:
            missing_afp.append(bin_idx)

        if (bin_idx + 1) % 100 == 0:
            print(f"  scanned {bin_idx + 1}/{num_bins} bins", flush=True)

    if strict and (missing_ssrf or missing_afp):
        parts_err: list[str] = []
        if missing_ssrf:
            parts_err.append(
                format_missing_bins_error(
                    "ssRF shard",
                    ssrf_shard_dir,
                    missing_ssrf,
                    num_bins=int(num_bins),
                    path_fn=ssrf_shard_path,
                )
            )
        if missing_afp:
            parts_err.append(
                format_missing_bins_error(
                    "AFP shard",
                    afp_shard_dir,
                    missing_afp,
                    num_bins=int(num_bins),
                    path_fn=afp_shard_path,
                )
            )
        raise FileNotFoundError("\n".join(parts_err))

    merged = _concat_spectrum_rows(parts) if parts else {}

    unmanip_rows, _ = _load_unmanip_spectrum_rows(
        unmanip_dir,
        num_bins=int(num_bins),
    )
    if missing_unmanip and not strict:
        print(
            f"WARNING: missing unmanip bins for {len(missing_unmanip)} bins "
            f"under {unmanip_dir}",
            flush=True,
        )
    if unmanip_rows:
        merged = _concat_spectrum_rows([merged, unmanip_rows]) if merged else unmanip_rows

    if not merged:
        raise ValueError("No spectrum rows found in input shards or unmanip bins")

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
        "unmanip_dir": str(unmanip_dir),
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
        "n_missing_unmanip": len(missing_unmanip),
        **{k: meta[k] for k in ("n_ssrf", "n_afp", "n_unmanip")},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Combine ssRF/AFP spectrum shards + unmanipulated bin NPZs into spectrum_train.npz"
        )
    )
    p.add_argument(
        "--ssrf-shard-dir",
        type=Path,
        default=SSRF_SHARD_DIR,
        help="Dir with ssrf_bin_XXXX.npz from ssrf_traj_array.slurm",
    )
    p.add_argument(
        "--afp-shard-dir",
        type=Path,
        default=AFP_SHARD_DIR,
        help="Dir with afp_bin_XXXX.npz from afp_traj_array.slurm",
    )
    p.add_argument(
        "--unmanip-dir",
        type=Path,
        default=UNMANIP_TRAIN_DIR,
        help="Directory of unmanip_bin_XXXX.npz from unmanipulated_bin_array",
    )
    p.add_argument("--output", type=Path, default=SPECTRUM_TRAIN_NPZ)
    p.add_argument(
        "--num-bins",
        type=int,
        default=NUM_BINS,
        help="Spectral bin count N; files use zero-indexed bin_idx 0..N-1 (default: 500 -> bins 0..499)",
    )
    p.add_argument("--afp-step-subsample", type=int, default=AFP_STEP_SUBSAMPLE)
    p.add_argument("--shard-size", type=int, default=0, help="If >0, write sharded NPZs of this many rows")
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(
        f"Combining spectrum shards ssrf={args.ssrf_shard_dir} afp={args.afp_shard_dir} "
        f"unmanip={args.unmanip_dir} -> {args.output}",
        flush=True,
    )
    result = combine_spectrum_shards(
        args.ssrf_shard_dir,
        args.afp_shard_dir,
        args.output,
        unmanip_dir=args.unmanip_dir,
        num_bins=args.num_bins,
        afp_step_subsample=int(args.afp_step_subsample),
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
