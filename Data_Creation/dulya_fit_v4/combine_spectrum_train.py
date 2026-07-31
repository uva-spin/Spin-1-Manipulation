"""
Merge ssRF + AFP spectrum shards + unmanipulated bin NPZs into a unified full-spectrum
training dataset.

Writes ``spectrum_train.npz`` (default, ``--shard-size 0``) or, when ``--shard-size``
is set, a shard directory of ``spectrum_train_XXXX.npz`` files plus a
``spectrum_train_manifest.json`` index. Each row has:
  ps[500], iplus[500], iminus[500], p0, step, center_bin, source, gamma_rf, burn_steps

Unmanipulated rows come from ``unmanip_bin_XXXX.npz`` files (``unmanipulated_bin_array``),
stacked into full 500-bin equilibrium spectra (one row per polarization). All available
unmanip rows are included (no subsampling).

Memory: prefer ``--shard-size N`` (N > 0). Rows are flattened and written in
bounded chunks (default chunk size matches ``N``, or 100k when ``N=0``), using
float32 spectra and memmap reads of ``ps_full`` cubes so peak RAM stays roughly
O(chunk) rather than O(full dataset). With ``--shard-size 0`` the merged output
is still held in memory before writing a single NPZ — fine for smoke tests, but
full-scale combines will OOM; use sharded output for production.

Usage (from this directory):
  python combine_spectrum_train.py --strict
  python combine_spectrum_train.py \\
      --ssrf-shard-dir data/ssrf_shards \\
      --afp-shard-dir data/afp_shards \\
      --unmanip-dir data/unmanip_train \\
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
    DEFAULT_SPECTRUM_ROW_CHUNK,
    SPECTRUM_ROW_KEYS,
    SPECTRUM_TRAIN_MANIFEST_NAME,
    SpectrumShardWriter,
    _build_flatten_row_indices,
    _concat_spectrum_rows,
    _filter_spectrum_rows,
    _iter_flatten_spectrum_shard_file,
    _iter_row_index_chunks,
    _nonempty_spectrum_mask,
    afp_shard_path,
    afp_spectrum_shard_path,
    bin_index_range,
    format_missing_bins_error,
    load_afp_shard,
    load_ssrf_shard,
    list_ssrf_spectrum_shard_paths,
    resolve_afp_spectrum_shard_path,
    resolve_ssrf_spectrum_shard_path,
    shard_has_ps_full,
    ssrf_spectrum_shard_complete,
    ssrf_shard_path,
    ssrf_spectrum_shard_path,
)
from common import (
    AFP_SHARD_DIR,
    AFP_STEP_SUBSAMPLE,
    BURN_BIN_CHOICES,
    FREQUENCY,
    NUM_BINS,
    PHYSICS_MODEL,
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    SPECTRUM_MAX_TRAIN_ROWS,
    SPECTRUM_MIN_BURN_BIN_COVERAGE,
    SPECTRUM_TRAIN_NPZ,
    SSRF_SHARD_DIR,
    STORE_DTYPE,
    UNMANIP_TRAIN_DIR,
    effective_afp_step_subsample,
)
from unmanipulated_bin_lineshape import unmanip_bin_path, verify_unmanip_train_dir

SPECTRUM_KEYS = SPECTRUM_ROW_KEYS


def _approx_spectrum_gib(n_rows: int, num_bins: int) -> float:
    """Rough float32 size of ps+iplus+iminus only."""
    return float(n_rows) * float(num_bins) * 3.0 * 4.0 / (1024.0**3)


def _histogram_finite(values: np.ndarray, *, decimals: int = 4) -> dict[str, int]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {}
    rounded = np.round(finite, decimals=int(decimals))
    unique, counts = np.unique(rounded, return_counts=True)
    return {str(float(k)): int(v) for k, v in zip(unique, counts)}


def _validate_conservation(
    ps: np.ndarray,
    ip: np.ndarray,
    im: np.ndarray,
    *,
    chunk_rows: int = 65536,
) -> float:
    """Return max |I+ + I- - Ps| over rows (chunked)."""
    if ps.size == 0:
        return 0.0
    n = int(ps.shape[0])
    step = max(1, int(chunk_rows))
    max_res = 0.0
    for start in range(0, n, step):
        end = min(start + step, n)
        chunk = np.add(ip[start:end], im[start:end])
        np.subtract(chunk, ps[start:end], out=chunk)
        np.abs(chunk, out=chunk)
        max_res = max(max_res, float(np.max(chunk)))
    return max_res


def _p_row_indices(eq_p: np.ndarray, p_values: np.ndarray) -> np.ndarray:
    grid = np.asarray(eq_p, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    idx = np.argmin(np.abs(grid[:, None] - p_values[None, :]), axis=0).astype(np.intp)
    if np.any(np.abs(grid[idx] - p_values) > 1e-5):
        bad = int(np.where(np.abs(grid[idx] - p_values) > 1e-5)[0][0])
        raise ValueError(
            f"p0={p_values[bad]} not on equilibrium grid (closest {grid[idx[bad]]})"
        )
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


def _iter_flatten_traj_shard_to_spectrum(
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
    chunk_rows: int = DEFAULT_SPECTRUM_ROW_CHUNK,
):
    """Yield full-spectrum row batches from a per-bin trajectory shard."""
    p_values = np.asarray(shard["p_values"], dtype=float)
    n_steps = np.asarray(shard["n_steps"], dtype=np.int32)
    skipped = np.asarray(shard.get("skipped", np.zeros_like(p_values, dtype=bool)), dtype=bool)
    ps = np.asarray(shard["ps"])
    ip = np.asarray(shard["iplus"])
    im = np.asarray(shard["iminus"])
    sub = max(1, int(step_subsample))
    c = int(center_bin)

    gamma_rf = np.full(int(p_values.size), np.nan, dtype=float)
    if with_burn_params and "gamma_rf" in shard:
        gamma_rf = np.asarray(shard["gamma_rf"], dtype=float)
    burn_steps = np.full(int(p_values.size), -1, dtype=np.int32)
    if with_burn_params and "burn_steps" in shard:
        burn_steps = np.asarray(shard["burn_steps"], dtype=np.int32)

    j_rep, step_rep = _build_flatten_row_indices(
        n_steps,
        skipped,
        step_subsample=sub,
    )
    if j_rep.size <= 0:
        return

    p_row = _p_row_indices(eq_p, p_values)
    for js, ss in _iter_row_index_chunks(j_rep, step_rep, chunk_rows=chunk_rows):
        n = int(js.size)
        pr = p_row[js]
        ps_out = np.asarray(eq_ps[pr], dtype=STORE_DTYPE)
        ip_out = np.asarray(eq_ip[pr], dtype=STORE_DTYPE)
        im_out = np.asarray(eq_im[pr], dtype=STORE_DTYPE)
        row_idx = np.arange(n, dtype=np.intp)
        ps_out[row_idx, c] = np.asarray(ps[js, ss], dtype=STORE_DTYPE)
        ip_out[row_idx, c] = np.asarray(ip[js, ss], dtype=STORE_DTYPE)
        im_out[row_idx, c] = np.asarray(im[js, ss], dtype=STORE_DTYPE)
        yield {
            "p0": p_values[js],
            "step": ss,
            "center_bin": np.full(n, c, dtype=np.int32),
            "source": np.full(n, np.uint8(source), dtype=np.uint8),
            "gamma_rf": gamma_rf[js],
            "burn_steps": burn_steps[js],
            "ps": ps_out,
            "iplus": ip_out,
            "iminus": im_out,
        }


def _iter_flatten_ssrf_shard(
    path: Path,
    *,
    center_bin: int,
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
    chunk_rows: int,
):
    yield from _iter_flatten_spectrum_shard_file(
        path,
        source=SOURCE_SSRF,
        center_bin=center_bin,
        step_subsample=1,
        prefer_file_step_subsample=False,
        chunk_rows=chunk_rows,
    )


def _iter_flatten_afp_shard(
    path: Path,
    *,
    center_bin: int,
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
    afp_step_subsample: int,
    chunk_rows: int,
):
    # Prefer shard meta; n_relax<=0 inside bin_io forces subsample=1.
    yield from _iter_flatten_spectrum_shard_file(
        path,
        source=SOURCE_AFP,
        center_bin=center_bin,
        step_subsample=max(1, int(afp_step_subsample)),
        prefer_file_step_subsample=True,
        chunk_rows=chunk_rows,
    )


def _iter_flatten_ssrf_shard_legacy(
    path: Path,
    *,
    center_bin: int,
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
    chunk_rows: int,
):
    shard = load_ssrf_shard(path)
    yield from _iter_flatten_traj_shard_to_spectrum(
        shard,
        source=SOURCE_SSRF,
        center_bin=center_bin,
        eq_p=eq_p,
        eq_ps=eq_ps,
        eq_ip=eq_ip,
        eq_im=eq_im,
        step_subsample=1,
        with_burn_params=True,
        chunk_rows=chunk_rows,
    )


def _iter_flatten_afp_shard_legacy(
    path: Path,
    *,
    center_bin: int,
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
    afp_step_subsample: int,
    chunk_rows: int,
):
    shard = load_afp_shard(path)
    yield from _iter_flatten_traj_shard_to_spectrum(
        shard,
        source=SOURCE_AFP,
        center_bin=center_bin,
        eq_p=eq_p,
        eq_ps=eq_ps,
        eq_ip=eq_ip,
        eq_im=eq_im,
        step_subsample=int(afp_step_subsample),
        with_burn_params=False,
        chunk_rows=chunk_rows,
    )


def _iter_ssrf_batches(
    path: Path,
    *,
    center_bin: int,
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
    chunk_rows: int,
):
    if shard_has_ps_full(path):
        yield from _iter_flatten_ssrf_shard(
            path,
            center_bin=center_bin,
            eq_p=eq_p,
            eq_ps=eq_ps,
            eq_ip=eq_ip,
            eq_im=eq_im,
            chunk_rows=chunk_rows,
        )
        return
    yield from _iter_flatten_ssrf_shard_legacy(
        path,
        center_bin=center_bin,
        eq_p=eq_p,
        eq_ps=eq_ps,
        eq_ip=eq_ip,
        eq_im=eq_im,
        chunk_rows=chunk_rows,
    )


def _iter_afp_batches(
    path: Path,
    *,
    center_bin: int,
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
    afp_step_subsample: int,
    chunk_rows: int,
):
    if shard_has_ps_full(path):
        yield from _iter_flatten_afp_shard(
            path,
            center_bin=center_bin,
            eq_p=eq_p,
            eq_ps=eq_ps,
            eq_ip=eq_ip,
            eq_im=eq_im,
            afp_step_subsample=int(afp_step_subsample),
            chunk_rows=chunk_rows,
        )
        return
    yield from _iter_flatten_afp_shard_legacy(
        path,
        center_bin=center_bin,
        eq_p=eq_p,
        eq_ps=eq_ps,
        eq_ip=eq_ip,
        eq_im=eq_im,
        afp_step_subsample=int(afp_step_subsample),
        chunk_rows=chunk_rows,
    )


def _unmanip_rows_from_eq(
    eq_p: np.ndarray,
    eq_ps: np.ndarray,
    eq_ip: np.ndarray,
    eq_im: np.ndarray,
    *,
    num_bins: int,
) -> dict[str, np.ndarray]:
    """Build unmanip spectrum rows from the already-loaded equilibrium cube."""
    n_p = int(eq_p.size)
    unmanip_center = int(num_bins // 2)
    return {
        "p0": np.asarray(eq_p, dtype=float),
        "step": np.zeros(n_p, dtype=np.int32),
        "center_bin": np.full(n_p, unmanip_center, dtype=np.int32),
        "source": np.full(n_p, SOURCE_UNMANIP, dtype=np.uint8),
        "gamma_rf": np.zeros(n_p, dtype=float),
        "burn_steps": np.zeros(n_p, dtype=np.int32),
        "ps": np.asarray(eq_ps, dtype=STORE_DTYPE),
        "iplus": np.asarray(eq_ip, dtype=STORE_DTYPE),
        "iminus": np.asarray(eq_im, dtype=STORE_DTYPE),
    }

def _filtered_rows_and_drop_count(
    rows: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
    """Drop empty/invalid rows from one flattened batch; return (rows, n_dropped)."""
    n_total = int(rows["ps"].shape[0])
    if n_total == 0:
        return rows, 0
    keep = _nonempty_spectrum_mask(rows["ps"], rows["iplus"], rows["iminus"])
    n_dropped = int(np.count_nonzero(~keep))
    if n_dropped == 0:
        return rows, 0
    return _filter_spectrum_rows(rows, keep), n_dropped


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
    max_train_rows: int = SPECTRUM_MAX_TRAIN_ROWS,
    min_burn_bin_coverage: float = SPECTRUM_MIN_BURN_BIN_COVERAGE,
) -> dict:
    ssrf_shard_dir = Path(ssrf_shard_dir)
    afp_shard_dir = Path(afp_shard_dir)
    unmanip_dir = Path(unmanip_dir)
    output_path = Path(output_path)

    unmanip_report = verify_unmanip_train_dir(unmanip_dir, num_bins=int(num_bins))
    if strict and not unmanip_report["ok"]:
        raise FileNotFoundError(
            f"unmanip_train incomplete under {unmanip_dir}: "
            f"missing={unmanip_report['n_missing']} "
            f"p_mismatch={len(unmanip_report['p_mismatch_bins'])}"
        )

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
    # Keep equilibrium spectra in store dtype for the rest of the combine.
    eq_ps = np.asarray(eq_ps, dtype=STORE_DTYPE)
    eq_ip = np.asarray(eq_ip, dtype=STORE_DTYPE)
    eq_im = np.asarray(eq_im, dtype=STORE_DTYPE)

    burn_bins = np.asarray(
        [int(b) for b in np.asarray(BURN_BIN_CHOICES, dtype=int).tolist() if 0 <= int(b) < int(num_bins)],
        dtype=int,
    )
    if burn_bins.size == 0:
        # Smoke / custom grids that do not overlap the global burn window.
        burn_bins = np.arange(int(num_bins), dtype=int)
    burn_set = {int(b) for b in burn_bins.tolist()}
    required_manip_bins = burn_bins

    streaming = int(shard_size) > 0
    chunk_rows = int(shard_size) if streaming else DEFAULT_SPECTRUM_ROW_CHUNK
    if not streaming:
        print(
            "WARNING: --shard-size 0 keeps the full merged dataset in RAM. "
            f"Flattening still uses {chunk_rows}-row chunks, but full-scale "
            "combines should pass --shard-size N (e.g. 200000).",
            flush=True,
        )
    stats = {
        "n_ssrf": 0,
        "n_afp": 0,
        "n_unmanip": 0,
        "n_filtered_empty": 0,
        "max_conservation_residual": 0.0,
    }
    missing_ssrf: list[int] = []
    missing_afp: list[int] = []
    present_ssrf_burn: set[int] = set()
    present_afp_burn: set[int] = set()
    gamma_hist_parts: list[np.ndarray] = []
    steps_hist_parts: list[np.ndarray] = []
    center_bins_seen: set[int] = set()
    parts: list[dict[str, np.ndarray]] = []

    base_meta = {
        "num_bins": int(num_bins),
        "source_codes": {"ssrf": SOURCE_SSRF, "afp": SOURCE_AFP, "unmanipulated": SOURCE_UNMANIP},
        "physics_model": PHYSICS_MODEL,
        "dataset": "spectrum_train_v2",
        "fields": "ps,iplus,iminus shape (n_samples, num_bins)",
        "store_dtype": str(np.dtype(STORE_DTYPE)),
        "unmanip_dir": str(unmanip_dir),
        "required_manip_bins": "burn_window",
        "n_burn_bins": int(burn_bins.size),
        "afp_step_subsample_requested": int(afp_step_subsample),
        "afp_step_subsample_effective_if_n_relax_0": int(
            effective_afp_step_subsample(0, int(afp_step_subsample))
        ),
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

    def _ingest(rows: dict[str, np.ndarray], source_key: str) -> None:
        filtered, n_dropped = _filtered_rows_and_drop_count(rows)
        stats["n_filtered_empty"] += n_dropped
        n = int(filtered["ps"].shape[0])
        if n <= 0:
            return
        stats[source_key] += n
        total_so_far = stats["n_ssrf"] + stats["n_afp"] + stats["n_unmanip"]
        if int(max_train_rows) > 0 and total_so_far > int(max_train_rows):
            raise RuntimeError(
                f"Train row cap exceeded: {total_so_far} > max_train_rows={int(max_train_rows)}"
            )
        res = _validate_conservation(filtered["ps"], filtered["iplus"], filtered["iminus"])
        stats["max_conservation_residual"] = max(stats["max_conservation_residual"], res)
        if "center_bin" in filtered:
            center_bins_seen.update(int(x) for x in np.unique(filtered["center_bin"]))
        if source_key == "n_ssrf":
            if "gamma_rf" in filtered:
                gamma_hist_parts.append(np.asarray(filtered["gamma_rf"], dtype=float))
            if "burn_steps" in filtered:
                steps_hist_parts.append(np.asarray(filtered["burn_steps"], dtype=float))
        if writer is not None:
            writer.add(filtered)
        else:
            parts.append(filtered)

    for bin_idx in bin_index_range(int(num_bins)):
        ssrf_paths = list_ssrf_spectrum_shard_paths(ssrf_shard_dir, bin_idx)
        if ssrf_paths:
            if int(bin_idx) in burn_set:
                present_ssrf_burn.add(int(bin_idx))
            for spath in ssrf_paths:
                for batch in _iter_ssrf_batches(
                    spath,
                    center_bin=bin_idx,
                    eq_p=eq_p,
                    eq_ps=eq_ps,
                    eq_ip=eq_ip,
                    eq_im=eq_im,
                    chunk_rows=chunk_rows,
                ):
                    _ingest(batch, "n_ssrf")
        else:
            legacy = ssrf_shard_path(ssrf_shard_dir, bin_idx)
            if legacy.is_file():
                if int(bin_idx) in burn_set:
                    present_ssrf_burn.add(int(bin_idx))
                for batch in _iter_ssrf_batches(
                    legacy,
                    center_bin=bin_idx,
                    eq_p=eq_p,
                    eq_ps=eq_ps,
                    eq_ip=eq_ip,
                    eq_im=eq_im,
                    chunk_rows=chunk_rows,
                ):
                    _ingest(batch, "n_ssrf")
            elif int(bin_idx) in burn_set:
                missing_ssrf.append(bin_idx)

        afp_path = resolve_afp_spectrum_shard_path(afp_shard_dir, bin_idx)
        if afp_path is None:
            legacy = afp_shard_path(afp_shard_dir, bin_idx)
            afp_path = legacy if legacy.is_file() else None
        if afp_path is not None:
            if int(bin_idx) in burn_set:
                present_afp_burn.add(int(bin_idx))
            for batch in _iter_afp_batches(
                afp_path,
                center_bin=bin_idx,
                eq_p=eq_p,
                eq_ps=eq_ps,
                eq_ip=eq_ip,
                eq_im=eq_im,
                afp_step_subsample=int(afp_step_subsample),
                chunk_rows=chunk_rows,
            ):
                _ingest(batch, "n_afp")
        elif int(bin_idx) in burn_set:
            missing_afp.append(bin_idx)

        if (bin_idx + 1) % 100 == 0:
            print(f"  scanned {bin_idx + 1}/{num_bins} bins", flush=True)

    n_required = max(1, int(required_manip_bins.size))
    ssrf_cov = float(len(present_ssrf_burn)) / float(n_required)
    afp_cov = float(len(present_afp_burn)) / float(n_required)
    if strict and (missing_ssrf or missing_afp):
        parts_err: list[str] = []
        if missing_ssrf:
            parts_err.append(
                format_missing_bins_error(
                    "ssRF shard (burn window)",
                    ssrf_shard_dir,
                    missing_ssrf,
                    num_bins=int(num_bins),
                    path_fn=ssrf_spectrum_shard_path,
                    traj_path_fn=ssrf_shard_path,
                )
            )
        if missing_afp:
            parts_err.append(
                format_missing_bins_error(
                    "AFP shard (burn window)",
                    afp_shard_dir,
                    missing_afp,
                    num_bins=int(num_bins),
                    path_fn=afp_spectrum_shard_path,
                    traj_path_fn=afp_shard_path,
                )
            )
        raise FileNotFoundError("\n".join(parts_err))
    if min(ssrf_cov, afp_cov) < float(min_burn_bin_coverage):
        msg = (
            f"Burn-bin coverage below {float(min_burn_bin_coverage):.0%}: "
            f"ssrf={ssrf_cov:.1%} ({len(present_ssrf_burn)}/{n_required}) "
            f"afp={afp_cov:.1%} ({len(present_afp_burn)}/{n_required})"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"WARNING: {msg}", flush=True)

    if missing_unmanip and not strict:
        print(
            f"WARNING: missing unmanip bins for {len(missing_unmanip)} bins "
            f"under {unmanip_dir}",
            flush=True,
        )
    unmanip_rows = _unmanip_rows_from_eq(
        eq_p, eq_ps, eq_ip, eq_im, num_bins=int(num_bins)
    )
    if int(unmanip_rows["ps"].shape[0]) > 0:
        _ingest(unmanip_rows, "n_unmanip")

    inventory = {
        "n_ssrf_burn_bins_present": len(present_ssrf_burn),
        "n_afp_burn_bins_present": len(present_afp_burn),
        "n_burn_bins_required": n_required,
        "ssrf_burn_bin_coverage": ssrf_cov,
        "afp_burn_bin_coverage": afp_cov,
        "n_unique_center_bins": len(center_bins_seen),
        "gamma_rf_histogram": _histogram_finite(
            np.concatenate(gamma_hist_parts) if gamma_hist_parts else np.zeros(0)
        ),
        "burn_steps_histogram": _histogram_finite(
            np.concatenate(steps_hist_parts) if steps_hist_parts else np.zeros(0),
            decimals=0,
        ),
        "approx_spectra_gib": None,
        "burn_r_min": float(FREQUENCY[burn_bins[0]]) if burn_bins.size else None,
        "burn_r_max": float(FREQUENCY[burn_bins[-1]]) if burn_bins.size else None,
        "unmanip_verify_ok": bool(unmanip_report["ok"]),
        "n_unmanip_p": int(unmanip_report["n_p"]),
    }

    if writer is not None:
        writer.close()
        n_samples = writer.n_written
        if n_samples <= 0:
            raise ValueError("No spectrum rows found in input shards or unmanip bins")
        inventory["approx_spectra_gib"] = _approx_spectrum_gib(n_samples, int(num_bins))
        manifest = {
            "n_samples": n_samples,
            "n_shards": writer.shard_index,
            "shard_files": writer.shard_files,
            "shard_row_counts": writer.shard_row_counts,
            **stats,
            **base_meta,
            "inventory": inventory,
        }
        (writer.output_dir / SPECTRUM_TRAIN_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        out_desc = str(writer.output_dir)
    else:
        if not parts:
            raise ValueError("No spectrum rows found in input shards or unmanip bins")
        merged = _concat_spectrum_rows(parts)
        n_samples = int(merged["ps"].shape[0])
        if n_samples <= 0:
            raise ValueError("All spectrum rows were empty or invalid after filtering")
        inventory["approx_spectra_gib"] = _approx_spectrum_gib(n_samples, int(num_bins))
        meta = {"n_samples": n_samples, **stats, **base_meta, "inventory": inventory}

        out_file = output_path
        if output_path.is_dir():
            out_file = output_path / "spectrum_train.npz"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_file.with_name(f".{out_file.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp,
            meta_json=np.asarray(json.dumps(meta)),
            **{k: merged[k] for k in SPECTRUM_KEYS},
        )
        tmp.replace(out_file)
        out_desc = str(out_file)

    print(
        f"Inventory: ssrf={stats['n_ssrf']} afp={stats['n_afp']} unmanip={stats['n_unmanip']}  "
        f"burn_cov ssrf={ssrf_cov:.1%} afp={afp_cov:.1%}  "
        f"~{inventory['approx_spectra_gib']:.2f} GiB spectra",
        flush=True,
    )

    return {
        "output": out_desc,
        "n_samples": n_samples,
        "max_conservation_residual": stats["max_conservation_residual"],
        "n_missing_ssrf": len(missing_ssrf),
        "n_missing_afp": len(missing_afp),
        "n_missing_unmanip": len(missing_unmanip),
        "n_ssrf": stats["n_ssrf"],
        "n_afp": stats["n_afp"],
        "n_unmanip": stats["n_unmanip"],
        "n_filtered_empty": stats["n_filtered_empty"],
        "inventory": inventory,
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
    p.add_argument(
        "--shard-size",
        type=int,
        default=0,
        help=(
            "If >0, stream rows to sharded NPZs of this many rows each "
            "(bounded memory; writes spectrum_train_manifest.json). "
            "Required for full-scale combines. If 0, build the full dataset "
            "in memory and write a single NPZ (smoke tests / small data only)."
        ),
    )
    p.add_argument(
        "--max-train-rows",
        type=int,
        default=SPECTRUM_MAX_TRAIN_ROWS,
        help="Hard cap on merged training rows (0 disables)",
    )
    p.add_argument(
        "--min-burn-bin-coverage",
        type=float,
        default=SPECTRUM_MIN_BURN_BIN_COVERAGE,
        help="Minimum fraction of burn-window bins that must have ssRF/AFP shards",
    )
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
        max_train_rows=int(args.max_train_rows),
        min_burn_bin_coverage=float(args.min_burn_bin_coverage),
    )
    print(
        f"Wrote {result['n_samples']} spectrum rows -> {result['output']}  "
        f"ssrf={result['n_ssrf']} afp={result['n_afp']} unmanip={result['n_unmanip']}  "
        f"max|I++I--Ps|={result['max_conservation_residual']:.2e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
