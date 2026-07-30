"""Shared NPZ shard / train I/O for dulya_fit_v2 workers."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from common import (
    NUM_BINS,
    PHYSICS_MODEL,
    PS_ABS_MIN,
    RF_MODE,
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    STORE_DTYPE,
)

# Default row batch when streaming flatten/ingest so peak RAM stays bounded.
DEFAULT_SPECTRUM_ROW_CHUNK = 100_000


def bin_index_range(num_bins: int) -> range:
    """Zero-indexed bin indices 0 .. num_bins-1."""
    nb = int(num_bins)
    if nb < 1:
        raise ValueError(f"num_bins must be >= 1, got {nb}")
    return range(nb)


def _traj_shard_mismatch_hint(
    shard_dir: Path,
    *,
    spectrum_path_fn,
    traj_path_fn,
) -> str:
    """Explain when trajectory shards exist but spectrum shards were requested."""
    spec0 = spectrum_path_fn(shard_dir, 0)
    traj0 = traj_path_fn(shard_dir, 0)
    if spec0.is_file() or not traj0.is_file():
        return ""
    return (
        f" Found {traj0.name} (per-bin trajectory shard) but not {spec0.name}. "
        "combine_spectrum_train needs full-spectrum shards from "
        f"{traj_path_fn(shard_dir, 0).parent}/ with --spectrum-mode, e.g. "
        f"{spec0.name}. For trajectory shards use combine_all_train.py instead."
    )


def format_missing_bins_error(
    label: str,
    shard_dir: Path,
    missing: list[int],
    *,
    num_bins: int,
    path_fn,
    traj_path_fn=None,
) -> str:
    """Human-readable strict-mode error for missing per-bin NPZ shards."""
    if not missing:
        return f"No missing {label} bins"
    nb = int(num_bins)
    last = nb - 1
    first = int(missing[0])
    example_lo = path_fn(shard_dir, 0).name
    example_hi = path_fn(shard_dir, last).name
    msg = (
        f"Missing {len(missing)} {label} file(s) under {shard_dir}; "
        f"expected {nb} zero-indexed bin_idx values 0..{last} "
        f"(e.g. {example_lo} .. {example_hi}); first missing bin_idx={first}"
    )
    if traj_path_fn is not None:
        msg += _traj_shard_mismatch_hint(
            shard_dir,
            spectrum_path_fn=path_fn,
            traj_path_fn=traj_path_fn,
        )
    if first == nb:
        msg += (
            f". bin_idx={nb} is invalid for num_bins={nb}; "
            f"use --num-bins {nb} for bins 0..{last} "
            f"(check SLURM --array=0-{last}, not 0-{nb})."
        )
    elif (
        len(missing) == 1
        and first == last
        and first > 0
        and path_fn(shard_dir, first - 1).is_file()
        and not path_fn(shard_dir, first).is_file()
    ):
        msg += (
            f". Found bins 0..{first - 1} only ({first} files); "
            f"use --num-bins {first} (zero-indexed 0..{first - 1}), not {nb}."
        )
    elif first == 0 and path_fn(shard_dir, nb).is_file():
        msg += (
            f". Found {path_fn(shard_dir, nb).name} but not {example_lo}; "
            "filenames look 1-based — regenerate with zero-indexed bin_idx 0.."
            f"{last} (SLURM --array=0-{last})."
        )
    elif first == 0 and path_fn(shard_dir, 1).is_file() and not path_fn(shard_dir, 0).is_file():
        msg += (
            f". Found {path_fn(shard_dir, 1).name} but not {example_lo}; "
            "expected zero-indexed filenames starting at 0000."
        )
    return msg


def ssrf_spectrum_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_spectrum_bin_{int(bin_idx):04d}.npz"


def ssrf_spectrum_shard_part_path(output_dir: Path, bin_idx: int, part_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_spectrum_bin_{int(bin_idx):04d}_part{int(part_idx):04d}.npz"


def ssrf_spectrum_shard_parts_manifest_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_spectrum_bin_{int(bin_idx):04d}_parts.json"


def list_ssrf_spectrum_shard_paths(shard_dir: Path, bin_idx: int) -> list[Path]:
    """Return monolithic shard and/or sorted part shards for one bin."""
    shard_dir = Path(shard_dir)
    main = ssrf_spectrum_shard_path(shard_dir, bin_idx)
    if main.is_file():
        return [main]
    manifest = ssrf_spectrum_shard_parts_manifest_path(shard_dir, bin_idx)
    if manifest.is_file():
        meta = json.loads(manifest.read_text())
        parts = [shard_dir / name for name in meta.get("part_files", [])]
        if parts and all(p.is_file() for p in parts):
            return parts
    parts = sorted(shard_dir.glob(f"ssrf_spectrum_bin_{int(bin_idx):04d}_part*.npz"))
    return parts


def ssrf_spectrum_shard_complete(shard_dir: Path, bin_idx: int) -> bool:
    shard_dir = Path(shard_dir)
    if ssrf_spectrum_shard_path(shard_dir, bin_idx).is_file():
        return True
    manifest = ssrf_spectrum_shard_parts_manifest_path(shard_dir, bin_idx)
    if not manifest.is_file():
        return False
    meta = json.loads(manifest.read_text())
    part_files = meta.get("part_files", [])
    return bool(part_files) and all((shard_dir / name).is_file() for name in part_files)


def resolve_ssrf_spectrum_shard_path(shard_dir: Path, bin_idx: int) -> Path | None:
    """Return ssRF spectrum shard path if present (monolithic, parts, or traj+ps_full)."""
    paths = list_ssrf_spectrum_shard_paths(shard_dir, bin_idx)
    if paths:
        return paths[0]
    shard_dir = Path(shard_dir)
    traj = ssrf_shard_path(shard_dir, bin_idx)
    if traj.is_file() and shard_has_ps_full(traj):
        return traj
    return None


def afp_spectrum_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"afp_spectrum_bin_{int(bin_idx):04d}.npz"


SPECTRUM_ROW_KEYS = (
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


def ssrf_spectrum_rows_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_spectrum_rows_{int(bin_idx):04d}.npz"


def afp_spectrum_rows_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"afp_spectrum_rows_{int(bin_idx):04d}.npz"


def unmanip_spectrum_rows_path(output_dir: Path) -> Path:
    return Path(output_dir) / "unmanip_spectrum_rows.npz"


def shard_has_ps_full(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as data:
        return "ps_full" in data.files


def resolve_afp_spectrum_shard_path(shard_dir: Path, bin_idx: int) -> Path | None:
    """Return AFP spectrum shard path if present (spectrum or traj+ps_full)."""
    shard_dir = Path(shard_dir)
    spec = afp_spectrum_shard_path(shard_dir, bin_idx)
    if spec.is_file():
        return spec
    traj = afp_shard_path(shard_dir, bin_idx)
    if traj.is_file() and shard_has_ps_full(traj):
        return traj
    return None


def ssrf_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_bin_{int(bin_idx):04d}.npz"


def afp_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"afp_bin_{int(bin_idx):04d}.npz"


def ssrf_shard_part_path(output_dir: Path, bin_idx: int, part_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_bin_{int(bin_idx):04d}_part{int(part_idx):04d}.npz"


def ssrf_shard_parts_manifest_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_bin_{int(bin_idx):04d}_parts.json"


def list_ssrf_shard_paths(shard_dir: Path, bin_idx: int) -> list[Path]:
    """Return the monolithic trajectory shard and/or sorted batched part shards for one bin."""
    shard_dir = Path(shard_dir)
    main = ssrf_shard_path(shard_dir, bin_idx)
    if main.is_file():
        return [main]
    manifest = ssrf_shard_parts_manifest_path(shard_dir, bin_idx)
    if manifest.is_file():
        meta = json.loads(manifest.read_text())
        parts = [shard_dir / name for name in meta.get("part_files", [])]
        if parts and all(p.is_file() for p in parts):
            return parts
    return sorted(shard_dir.glob(f"ssrf_bin_{int(bin_idx):04d}_part*.npz"))


def ssrf_shard_complete(shard_dir: Path, bin_idx: int) -> bool:
    """True if bin_idx's trajectory shard exists, monolithic or as complete batched parts."""
    shard_dir = Path(shard_dir)
    if ssrf_shard_path(shard_dir, bin_idx).is_file():
        return True
    manifest = ssrf_shard_parts_manifest_path(shard_dir, bin_idx)
    if not manifest.is_file():
        return False
    meta = json.loads(manifest.read_text())
    part_files = meta.get("part_files", [])
    return bool(part_files) and all((shard_dir / name).is_file() for name in part_files)


_SSRF_SHARD_ARRAY_KEYS = (
    "p_values",
    "gamma_rf",
    "burn_steps",
    "n_steps",
    "skipped",
    "ps",
    "iplus",
    "iminus",
    "amp",
    "ps_m",
    "iplus_m",
    "iminus_m",
    "amp_m",
)

_TRAJ_STACK_META_KEYS = (
    "p_values",
    "gamma_rf",
    "burn_steps",
    "n_steps",
    "skipped",
)


def load_ssrf_shard_meta(path: Path) -> dict:
    """Load only combo-grid / skip metadata from one ssRF trajectory shard."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        n_samp = int(np.asarray(data["p_values"]).shape[0])
        if "gamma_rf" in data.files:
            gamma_rf = np.asarray(data["gamma_rf"], dtype=float)
        else:
            gamma_rf = np.full(n_samp, float(meta.get("gamma_rf", np.nan)), dtype=float)
        if "burn_steps" in data.files:
            burn_steps = np.asarray(data["burn_steps"], dtype=np.int32)
        else:
            n_steps = np.asarray(data["n_steps"], dtype=np.int32)
            burn_steps = np.maximum(n_steps - 1, 0).astype(np.int32)
        return {
            "p_values": np.asarray(data["p_values"], dtype=float),
            "gamma_rf": gamma_rf,
            "burn_steps": burn_steps,
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
        }


def load_afp_shard_meta(path: Path) -> dict:
    """Load only combo-grid / skip metadata from one AFP trajectory shard."""
    with np.load(path, allow_pickle=False) as data:
        return {
            "p_values": np.asarray(data["p_values"], dtype=float),
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
        }


def _concat_ssrf_meta(parts: list[dict]) -> dict:
    """Merge batched trajectory-shard metadata (same combo grid, disjoint ranges)."""
    if not parts:
        raise ValueError("parts must be non-empty")
    if len(parts) == 1:
        return parts[0]
    merged = dict(parts[0])
    for key in _TRAJ_STACK_META_KEYS:
        if key in parts[0]:
            merged[key] = np.concatenate([p[key] for p in parts], axis=0)
    return merged


def load_ssrf_shard_meta_any(shard_dir: Path, bin_idx: int) -> dict | None:
    """Lightweight metadata load for one ssRF bin (merges batched parts). None if missing."""
    paths = list_ssrf_shard_paths(shard_dir, bin_idx)
    if not paths:
        return None
    return _concat_ssrf_meta([load_ssrf_shard_meta(p) for p in paths])


def ssrf_traj_shard_exists(shard_dir: Path, bin_idx: int) -> bool:
    return bool(list_ssrf_shard_paths(shard_dir, bin_idx))


def afp_traj_shard_exists(shard_dir: Path, bin_idx: int) -> bool:
    return afp_shard_path(shard_dir, bin_idx).is_file()


def scan_traj_input_shards(
    ssrf_shard_dir: Path,
    afp_shard_dir: Path,
    unmanip_dir: Path,
    *,
    num_bins: int,
) -> dict[str, list[int]]:
    """Fast existence scan before loading full trajectory arrays."""
    nb = int(num_bins)
    return {
        "missing_ssrf": [
            b for b in bin_index_range(nb) if not ssrf_traj_shard_exists(ssrf_shard_dir, b)
        ],
        "missing_afp": [
            b for b in bin_index_range(nb) if not afp_traj_shard_exists(afp_shard_dir, b)
        ],
        "missing_unmanip": [
            b
            for b in bin_index_range(nb)
            if not (Path(unmanip_dir) / f"unmanip_bin_{int(b):04d}.npz").is_file()
        ],
    }


def _concat_ssrf_shards(parts: list[dict]) -> dict:
    """Merge batched trajectory-shard parts (same combo grid, disjoint sample ranges)."""
    if not parts:
        raise ValueError("parts must be non-empty")
    if len(parts) == 1:
        return parts[0]
    merged = dict(parts[0])
    for key in _SSRF_SHARD_ARRAY_KEYS:
        if key in parts[0]:
            merged[key] = np.concatenate([p[key] for p in parts], axis=0)
    return merged


def load_ssrf_shard_any(shard_dir: Path, bin_idx: int) -> dict | None:
    """Load bin_idx's trajectory shard, transparently merging batched parts. None if missing."""
    paths = list_ssrf_shard_paths(shard_dir, bin_idx)
    if not paths:
        return None
    return _concat_ssrf_shards([load_ssrf_shard(p) for p in paths])


def ssrf_train_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_train_bin_{int(bin_idx):04d}.npz"


def afp_train_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"afp_train_bin_{int(bin_idx):04d}.npz"


def save_ssrf_shard(result: dict, path: Path, *, extra_meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gamma_values = np.asarray(result.get("gamma_values", []), dtype=float)
    steps_values = np.asarray(result.get("steps_values", []), dtype=np.int32)
    meta = {
        "bin_idx": int(result["bin_idx"]),
        "mirror_idx": int(result["mirror_idx"]),
        "R": float(result["R"]),
        "num_bins": int(result["num_bins"]),
        "dt": float(result["dt"]),
        "max_burn_steps": int(
            result.get("max_burn_steps", result.get("max_steps", 0))
        ),
        "gamma_values": [float(g) for g in gamma_values.tolist()],
        "steps_values": [int(s) for s in steps_values.tolist()],
        "physics_model": PHYSICS_MODEL,
        "rf_mode": str(result.get("rf_mode", RF_MODE)),
        "gaussian_fwhm_R": float(result.get("gaussian_fwhm_R", 0.0)),
        "lorentzian_fwhm_R": float(result.get("lorentzian_fwhm_R", 0.0)),
        "diffusion_scale": float(result.get("diffusion_scale", 0.0)),
        "sampling": "p_x_gamma_x_n_steps",
        "dataset": "ssrf_bin_traj_v2",
    }
    if extra_meta:
        meta.update(extra_meta)
    # STORE_DTYPE (float32): halves on-disk size for the (n_samples, t_max)
    # trajectory arrays, which dominate large-grid shards. load_ssrf_shard()
    # upcasts back to float64 on read, so this is transparent to consumers.
    ps = np.asarray(result["ps"], dtype=STORE_DTYPE)
    ps_m = np.asarray(result["ps_m"], dtype=STORE_DTYPE)
    gamma_rf = np.asarray(result["gamma_rf"], dtype=float)
    burn_steps = np.asarray(result["burn_steps"], dtype=np.int32)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            p_values=np.asarray(result["p_values"], dtype=float),
            gamma_rf=gamma_rf,
            burn_steps=burn_steps,
            n_steps=np.asarray(result["n_steps"], dtype=np.int32),
            skipped=np.asarray(result["skipped"], dtype=bool),
            ps=ps,
            iplus=np.asarray(result["iplus"], dtype=STORE_DTYPE),
            iminus=np.asarray(result["iminus"], dtype=STORE_DTYPE),
            amp=np.abs(ps),
            ps_m=ps_m,
            iplus_m=np.asarray(result["iplus_m"], dtype=STORE_DTYPE),
            iminus_m=np.asarray(result["iminus_m"], dtype=STORE_DTYPE),
            amp_m=np.abs(ps_m),
            bin_idx=np.asarray(int(result["bin_idx"]), dtype=np.int32),
            mirror_idx=np.asarray(int(result["mirror_idx"]), dtype=np.int32),
            dt=np.asarray(float(result["dt"]), dtype=float),
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def _save_spectrum_shard(
    result: dict,
    path: Path,
    *,
    dataset: str,
    extra_meta: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "bin_idx": int(result["bin_idx"]),
        "mirror_idx": int(result["mirror_idx"]),
        "R": float(result["R"]),
        "num_bins": int(result["num_bins"]),
        "dt": float(result["dt"]),
        "physics_model": PHYSICS_MODEL,
        "dataset": dataset,
        "fields": "ps_full,iplus_full,iminus_full (n_samples, n_steps, num_bins)",
    }
    if extra_meta:
        meta.update(extra_meta)
    for opt_key in (
        "rf_mode",
        "gaussian_fwhm_R",
        "lorentzian_fwhm_R",
        "diffusion_scale",
        "gamma_values",
        "steps_values",
        "max_burn_steps",
        "n_relax",
        "afp_window",
        "afp_efficiency",
        "afp_subset",
        "step_subsample",
        "n_random_samples",
        "n_unmanip_samples",
        "multi_burn",
    ):
        if opt_key in result:
            val = result[opt_key]
            if isinstance(val, np.ndarray):
                meta[opt_key] = val.tolist()
            else:
                meta[opt_key] = val

    payload = {
        "meta_json": np.asarray(json.dumps(meta)),
        "p_values": np.asarray(result["p_values"], dtype=float),
        "n_steps": np.asarray(result["n_steps"], dtype=np.int32),
        "skipped": np.asarray(result["skipped"], dtype=bool),
        "bin_idx": np.asarray(int(result["bin_idx"]), dtype=np.int32),
        "mirror_idx": np.asarray(int(result["mirror_idx"]), dtype=np.int32),
        "dt": np.asarray(float(result["dt"]), dtype=float),
    }
    if "gamma_rf" in result:
        payload["gamma_rf"] = np.asarray(result["gamma_rf"], dtype=float)
    if "burn_steps" in result:
        payload["burn_steps"] = np.asarray(result["burn_steps"], dtype=np.int32)
    for key in ("ps_full", "iplus_full", "iminus_full"):
        if key in result and result[key] is not None:
            payload[key] = np.asarray(result[key], dtype=STORE_DTYPE)

    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def save_ssrf_spectrum_shard(
    result: dict, path: Path, *, extra_meta: dict | None = None
) -> None:
    _save_spectrum_shard(
        result,
        path,
        dataset=str(result.get("dataset", "ssrf_spectrum_bin_v2")),
        extra_meta=extra_meta,
    )


def save_afp_spectrum_shard(
    result: dict, path: Path, *, extra_meta: dict | None = None
) -> None:
    _save_spectrum_shard(
        result,
        path,
        dataset=str(result.get("dataset", "afp_spectrum_bin_v2")),
        extra_meta=extra_meta,
    )


def load_spectrum_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        out = {
            **meta,
            "p_values": np.asarray(data["p_values"], dtype=float),
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
        }
        for key in ("gamma_rf", "burn_steps", "ps_full", "iplus_full", "iminus_full"):
            if key in data.files:
                out[key] = np.asarray(data[key])
        return out


def _nonempty_spectrum_mask(
    ps: np.ndarray,
    ip: np.ndarray,
    im: np.ndarray,
    *,
    min_ps_sum: float = PS_ABS_MIN,
    chunk_rows: int = 65536,
) -> np.ndarray:
    """Row mask for finite spectra with non-negligible |Ps| mass.

    Chunked to avoid full-size temporaries (isfinite / abs) on large batches.
    """
    ps = np.asarray(ps)
    ip = np.asarray(ip)
    im = np.asarray(im)
    n = int(ps.shape[0])
    keep = np.empty(n, dtype=bool)
    step = max(1, int(chunk_rows))
    for start in range(0, n, step):
        end = min(start + step, n)
        ps_c = ps[start:end]
        ip_c = ip[start:end]
        im_c = im[start:end]
        finite = (
            np.all(np.isfinite(ps_c), axis=1)
            & np.all(np.isfinite(ip_c), axis=1)
            & np.all(np.isfinite(im_c), axis=1)
        )
        keep[start:end] = finite & (np.sum(np.abs(ps_c), axis=1) > float(min_ps_sum))
    return keep


def _filter_spectrum_rows(
    rows: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    return {key: values[mask] for key, values in rows.items()}


def _empty_spectrum_row_arrays(num_bins: int) -> dict[str, np.ndarray]:
    nb = int(num_bins)
    return {
        "p0": np.zeros(0, dtype=float),
        "step": np.zeros(0, dtype=np.int32),
        "center_bin": np.zeros(0, dtype=np.int32),
        "source": np.zeros(0, dtype=np.uint8),
        "gamma_rf": np.zeros(0, dtype=float),
        "burn_steps": np.zeros(0, dtype=np.int32),
        "ps": np.zeros((0, nb), dtype=STORE_DTYPE),
        "iplus": np.zeros((0, nb), dtype=STORE_DTYPE),
        "iminus": np.zeros((0, nb), dtype=STORE_DTYPE),
    }


def _iter_row_index_chunks(
    j_rep: np.ndarray,
    step_rep: np.ndarray,
    *,
    chunk_rows: int,
):
    """Yield (j_rep_chunk, step_rep_chunk) slices of at most ``chunk_rows``."""
    n = int(j_rep.size)
    step = max(1, int(chunk_rows))
    for start in range(0, n, step):
        end = min(start + step, n)
        yield j_rep[start:end], step_rep[start:end]


def _build_flatten_row_indices(
    n_steps: np.ndarray,
    skipped: np.ndarray,
    *,
    n_keep: int | None = None,
    step_subsample: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = np.asarray(n_steps, dtype=np.int32)
    skipped = np.asarray(skipped, dtype=bool)
    n_samp = int(n_steps.size) if n_keep is None else int(n_keep)
    sub = max(1, int(step_subsample))

    valid = (~skipped[:n_samp]) & (n_steps[:n_samp] > 0)
    row_counts = np.zeros(n_samp, dtype=np.int64)
    row_counts[valid] = (n_steps[:n_samp][valid] + sub - 1) // sub
    total = int(row_counts.sum())
    if total <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)

    j_rep = np.repeat(np.arange(n_samp, dtype=np.int32), row_counts)
    offsets = np.zeros(n_samp + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(row_counts)
    within = np.arange(total, dtype=np.int32) - offsets[j_rep]
    step_rep = (within * sub).astype(np.int32)
    return j_rep, step_rep


def validate_ps_iplus_iminus(
    rows: dict[str, np.ndarray],
    *,
    label: str = "rows",
    atol: float = 1e-3,
    rtol: float = 1e-3,
    max_report: int = 5,
) -> None:
    """Raise if ps != iplus + iminus anywhere in the row batch."""
    ps = np.asarray(rows["ps"], dtype=np.float64)
    iplus = np.asarray(rows["iplus"], dtype=np.float64)
    iminus = np.asarray(rows["iminus"], dtype=np.float64)
    if ps.shape != iplus.shape or ps.shape != iminus.shape:
        raise ValueError(
            f"{label}: ps/iplus/iminus shape mismatch: "
            f"ps={ps.shape} iplus={iplus.shape} iminus={iminus.shape}"
        )
    if ps.size == 0:
        return
    expected = iplus + iminus
    bad = ~np.isclose(ps, expected, atol=atol, rtol=rtol, equal_nan=True)
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        return
    bad_idx = np.argwhere(bad)[:max_report]
    examples = [
        f"(row={int(r)}, bin={int(c)}): ps={ps[r, c]!r} iplus={iplus[r, c]!r} "
        f"iminus={iminus[r, c]!r} iplus+iminus={expected[r, c]!r}"
        for r, c in bad_idx
    ]
    raise ValueError(
        f"{label}: ps != iplus + iminus at {n_bad}/{ps.size} cells "
        f"(atol={atol}, rtol={rtol}); first mismatches: " + "; ".join(examples)
    )


class _ComboGridMismatch(ValueError):
    """Raised when a bin's shard was not run with the reference combo grid."""


def _check_combo_grid(
    ref_fields: dict[str, np.ndarray],
    bin_idx: int,
    fields: dict[str, np.ndarray],
    *,
    tol: float = 1e-6,
) -> None:
    """Verify a bin's per-combo fields (p_values, and gamma_rf/burn_steps when
    present) exactly match the reference bin's, so combo index j is guaranteed
    to mean the same (P, gamma_rf, burn_steps) across bins."""
    for key, ref_arr in ref_fields.items():
        arr = np.asarray(fields.get(key), dtype=float)
        if arr.shape != ref_arr.shape or not np.allclose(arr, ref_arr, atol=tol, rtol=0, equal_nan=True):
            raise _ComboGridMismatch(
                f"bin_idx={bin_idx}: combo grid ({key}) does not match the reference bin's grid; "
                "every bin must be generated with the same --p-min/--p-max/--p-step "
                "(and, for ssRF, --gamma-*/--steps-*) so combo index j means the same "
                "(P, gamma_rf, burn_steps) across all 500 bins."
            )


def _fill_traj_stack_column(
    ps_out: np.ndarray,
    ip_out: np.ndarray,
    im_out: np.ndarray,
    shard: dict,
    bin_idx: int,
    j_rep: np.ndarray,
    step_rep: np.ndarray,
) -> None:
    ps_b = np.asarray(shard["ps"])
    ip_b = np.asarray(shard["iplus"])
    im_b = np.asarray(shard["iminus"])
    t_max_b = ps_b.shape[1]
    in_range = step_rep < t_max_b
    if not np.any(in_range):
        return
    js = j_rep[in_range]
    ts = step_rep[in_range]
    ps_out[in_range, bin_idx] = np.nan_to_num(ps_b[js, ts], nan=0.0).astype(STORE_DTYPE)
    ip_out[in_range, bin_idx] = np.nan_to_num(ip_b[js, ts], nan=0.0).astype(STORE_DTYPE)
    im_out[in_range, bin_idx] = np.nan_to_num(im_b[js, ts], nan=0.0).astype(STORE_DTYPE)


def _stack_traj_shards_core(
    load_meta: Callable[[int], dict | None],
    load_full: Callable[[int], dict | None],
    *,
    num_bins: int,
    source: int,
    with_burn_params: bool = True,
    step_subsample: int = 1,
    strict: bool = True,
    progress_label: str | None = None,
) -> dict[str, np.ndarray]:
    """Plan row layout from lightweight metadata, then fill columns one bin at a time."""
    nb = int(num_bins)
    ref: dict | None = None
    ref_fields: dict[str, np.ndarray] | None = None
    ref_p: np.ndarray | None = None
    ref_gamma_rf: np.ndarray | None = None
    ref_burn_steps: np.ndarray | None = None
    max_steps: np.ndarray | None = None
    present_bins: list[int] = []

    for bin_idx in bin_index_range(nb):
        meta = load_meta(bin_idx)
        if meta is None:
            continue
        if ref is None:
            ref = meta
            ref_p = np.asarray(ref["p_values"], dtype=float)
            n_samples = int(ref_p.size)
            ref_gamma_rf = (
                np.asarray(ref["gamma_rf"], dtype=float)
                if with_burn_params and "gamma_rf" in ref
                else np.full(n_samples, np.nan)
            )
            ref_burn_steps = (
                np.asarray(ref["burn_steps"], dtype=np.int32)
                if with_burn_params and "burn_steps" in ref
                else np.full(n_samples, -1, dtype=np.int32)
            )
            ref_fields = {"p_values": ref_p}
            if with_burn_params and "gamma_rf" in ref:
                ref_fields["gamma_rf"] = ref_gamma_rf
            if with_burn_params and "burn_steps" in ref:
                ref_fields["burn_steps"] = ref_burn_steps.astype(float)
            max_steps = np.zeros(n_samples, dtype=np.int32)

        assert ref_fields is not None
        assert max_steps is not None
        try:
            check_fields: dict[str, np.ndarray] = {"p_values": meta["p_values"]}
            if "gamma_rf" in ref_fields:
                check_fields["gamma_rf"] = meta.get("gamma_rf")
            if "burn_steps" in ref_fields:
                check_fields["burn_steps"] = meta.get("burn_steps")
            _check_combo_grid(ref_fields, bin_idx, check_fields)
        except _ComboGridMismatch:
            if strict:
                raise
            continue
        present_bins.append(bin_idx)
        np.maximum(max_steps, np.asarray(meta["n_steps"], dtype=np.int32), out=max_steps)

        if progress_label and ((bin_idx + 1) % 100 == 0 or bin_idx + 1 == nb):
            print(f"  {progress_label}: planned {bin_idx + 1}/{nb} bins", flush=True)

    if ref is None or ref_p is None or ref_gamma_rf is None or ref_burn_steps is None:
        return _empty_spectrum_row_arrays(nb)
    assert max_steps is not None

    skipped_all = max_steps <= 0
    j_rep, step_rep = _build_flatten_row_indices(max_steps, skipped_all, step_subsample=step_subsample)
    n = int(j_rep.size)
    if n == 0:
        return _empty_spectrum_row_arrays(nb)

    ps_out = np.zeros((n, nb), dtype=STORE_DTYPE)
    ip_out = np.zeros((n, nb), dtype=STORE_DTYPE)
    im_out = np.zeros((n, nb), dtype=STORE_DTYPE)

    for fill_idx, bin_idx in enumerate(present_bins, start=1):
        shard = load_full(bin_idx)
        if shard is None:
            continue
        _fill_traj_stack_column(ps_out, ip_out, im_out, shard, bin_idx, j_rep, step_rep)
        if progress_label and (fill_idx % 100 == 0 or fill_idx == len(present_bins)):
            print(f"  {progress_label}: stacked {fill_idx}/{len(present_bins)} bins", flush=True)

    return {
        "p0": ref_p[j_rep],
        "step": step_rep,
        "center_bin": np.full(n, -1, dtype=np.int32),
        "source": np.full(n, np.uint8(source), dtype=np.uint8),
        "gamma_rf": ref_gamma_rf[j_rep],
        "burn_steps": ref_burn_steps[j_rep],
        "ps": ps_out,
        "iplus": ip_out,
        "iminus": im_out,
    }


def stack_traj_shards_into_spectrum_rows(
    shards_by_bin: list[dict | None],
    *,
    num_bins: int,
    source: int,
    with_burn_params: bool = True,
    step_subsample: int = 1,
    strict: bool = True,
) -> dict[str, np.ndarray]:
    """Reconstruct dense ``num_bins``-wide spectrum rows by stacking each bin's
    own trajectory shard as one column: ``row[..., bin_idx] = shards_by_bin[bin_idx]["ps"][j, t]``.

    Every column comes from an independent single-bin simulation run with the
    *same* (P, gamma_rf, burn_steps) combo grid (shared combo index ``j``
    across bins), so stacking bin 0 .. bin ``num_bins - 1`` side by side for a
    given combo/timestep reconstructs the burn signal across the full
    spectrum -- with no equilibrium fill and no cross-bin matching beyond
    that shared combo index. Bins with a missing shard, or whose own combo
    was skipped/out of range for a given (j, t), are left at 0 in that column.
    """
    nb = int(num_bins)
    if len(shards_by_bin) != nb:
        raise ValueError(f"expected {nb} entries in shards_by_bin, got {len(shards_by_bin)}")

    def load_meta(bin_idx: int) -> dict | None:
        if bin_idx >= len(shards_by_bin):
            return None
        shard = shards_by_bin[bin_idx]
        if shard is None:
            return None
        return {key: shard[key] for key in _TRAJ_STACK_META_KEYS if key in shard}

    def load_full(bin_idx: int) -> dict | None:
        if bin_idx >= len(shards_by_bin):
            return None
        return shards_by_bin[bin_idx]

    return _stack_traj_shards_core(
        load_meta,
        load_full,
        num_bins=nb,
        source=source,
        with_burn_params=with_burn_params,
        step_subsample=step_subsample,
        strict=strict,
    )


def stack_ssrf_traj_shards_from_dir(
    shard_dir: Path,
    *,
    num_bins: int,
    strict: bool = True,
    step_subsample: int = 1,
) -> dict[str, np.ndarray]:
    """Stream ssRF trajectory shards from disk, holding at most one full bin in RAM."""
    shard_dir = Path(shard_dir)
    nb = int(num_bins)

    def load_meta(bin_idx: int) -> dict | None:
        return load_ssrf_shard_meta_any(shard_dir, bin_idx)

    def load_full(bin_idx: int) -> dict | None:
        return load_ssrf_shard_any(shard_dir, bin_idx)

    return _stack_traj_shards_core(
        load_meta,
        load_full,
        num_bins=nb,
        source=SOURCE_SSRF,
        with_burn_params=True,
        step_subsample=step_subsample,
        strict=strict,
        progress_label="ssRF",
    )


def stack_afp_traj_shards_from_dir(
    shard_dir: Path,
    *,
    num_bins: int,
    strict: bool = True,
    step_subsample: int = 1,
) -> dict[str, np.ndarray]:
    """Stream AFP trajectory shards from disk, holding at most one full bin in RAM."""
    shard_dir = Path(shard_dir)
    nb = int(num_bins)

    def load_meta(bin_idx: int) -> dict | None:
        path = afp_shard_path(shard_dir, bin_idx)
        if not path.is_file():
            return None
        return load_afp_shard_meta(path)

    def load_full(bin_idx: int) -> dict | None:
        path = afp_shard_path(shard_dir, bin_idx)
        if not path.is_file():
            return None
        return load_afp_shard(path)

    return _stack_traj_shards_core(
        load_meta,
        load_full,
        num_bins=nb,
        source=SOURCE_AFP,
        with_burn_params=False,
        step_subsample=step_subsample,
        strict=strict,
        progress_label="AFP",
    )


def stack_unmanip_bins_into_spectrum_rows(
    unmanip_by_bin: list[dict | None],
    *,
    num_bins: int,
    strict: bool = True,
) -> dict[str, np.ndarray]:
    """Reconstruct dense ``num_bins``-wide equilibrium rows by stacking each
    bin's own ``unmanip_bin_XXXX.npz`` P-column as one column. Unlike ssRF/AFP
    there is no burn trajectory or skip mask -- every P value is valid, so
    this is a plain column stack (bins with a missing file are left at 0).
    """
    nb = int(num_bins)
    if len(unmanip_by_bin) != nb:
        raise ValueError(f"expected {nb} entries in unmanip_by_bin, got {len(unmanip_by_bin)}")

    ref = next((u for u in unmanip_by_bin if u is not None), None)
    if ref is None:
        return _empty_spectrum_row_arrays(nb)

    ref_p0 = np.asarray(ref["p0"], dtype=float)
    n = int(ref_p0.size)
    if n == 0:
        return _empty_spectrum_row_arrays(nb)

    ps_out = np.zeros((n, nb), dtype=STORE_DTYPE)
    ip_out = np.zeros((n, nb), dtype=STORE_DTYPE)
    im_out = np.zeros((n, nb), dtype=STORE_DTYPE)
    for bin_idx, u in enumerate(unmanip_by_bin):
        if u is None:
            continue
        try:
            _check_combo_grid({"p_values": ref_p0}, bin_idx, {"p_values": u["p0"]})
        except _ComboGridMismatch:
            if strict:
                raise
            continue
        ps_out[:, bin_idx] = np.asarray(u["ps"], dtype=STORE_DTYPE)
        ip_out[:, bin_idx] = np.asarray(u["iplus"], dtype=STORE_DTYPE)
        im_out[:, bin_idx] = np.asarray(u["iminus"], dtype=STORE_DTYPE)

    return {
        "p0": ref_p0,
        "step": np.zeros(n, dtype=np.int32),
        "center_bin": np.full(n, -1, dtype=np.int32),
        "source": np.full(n, np.uint8(SOURCE_UNMANIP), dtype=np.uint8),
        "gamma_rf": np.full(n, np.nan, dtype=float),
        "burn_steps": np.full(n, -1, dtype=np.int32),
        "ps": ps_out,
        "iplus": ip_out,
        "iminus": im_out,
    }


def _iter_flatten_spectrum_shard(
    shard: dict,
    *,
    source: int,
    center_bin: int,
    step_subsample: int = 1,
    exclude_trailing_samples: int = 0,
    chunk_rows: int = DEFAULT_SPECTRUM_ROW_CHUNK,
):
    """Yield flattened spectrum-row batches from an in-memory spectrum shard."""
    p_values = np.asarray(shard["p_values"], dtype=float)
    n_steps = np.asarray(shard["n_steps"], dtype=np.int32)
    skipped = np.asarray(shard.get("skipped", np.zeros_like(p_values, dtype=bool)), dtype=bool)
    # Keep native on-disk dtype (typically float32); do not upcast to float64.
    ps_full = np.asarray(shard["ps_full"])
    ip_full = np.asarray(shard["iplus_full"])
    im_full = np.asarray(shard["iminus_full"])
    n_samp = int(p_values.size)
    num_bins = int(ps_full.shape[-1])
    n_exclude = max(0, min(int(exclude_trailing_samples), n_samp))
    n_keep = n_samp - n_exclude

    gamma_rf = np.full(n_samp, np.nan, dtype=float)
    if "gamma_rf" in shard:
        gamma_rf = np.asarray(shard["gamma_rf"], dtype=float)
    burn_steps = np.full(n_samp, -1, dtype=np.int32)
    if "burn_steps" in shard:
        burn_steps = np.asarray(shard["burn_steps"], dtype=np.int32)

    j_rep, step_rep = _build_flatten_row_indices(
        n_steps,
        skipped,
        n_keep=n_keep,
        step_subsample=step_subsample,
    )
    if j_rep.size <= 0:
        return

    for js, ss in _iter_row_index_chunks(j_rep, step_rep, chunk_rows=chunk_rows):
        yield {
            "p0": p_values[js],
            "step": ss,
            "center_bin": np.full(js.size, center_bin, dtype=np.int32),
            "source": np.full(js.size, np.uint8(source), dtype=np.uint8),
            "gamma_rf": gamma_rf[js],
            "burn_steps": burn_steps[js],
            "ps": np.asarray(ps_full[js, ss], dtype=STORE_DTYPE),
            "iplus": np.asarray(ip_full[js, ss], dtype=STORE_DTYPE),
            "iminus": np.asarray(im_full[js, ss], dtype=STORE_DTYPE),
        }


def _iter_flatten_spectrum_shard_file(
    path: Path,
    *,
    source: int,
    center_bin: int,
    step_subsample: int = 1,
    exclude_trailing_samples: int | None = None,
    prefer_file_step_subsample: bool = False,
    chunk_rows: int = DEFAULT_SPECTRUM_ROW_CHUNK,
):
    """Yield flattened rows from a spectrum NPZ using memmap (no full-cube RAM load)."""
    with np.load(path, allow_pickle=False, mmap_mode="r") as data:
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
        p_values = np.asarray(data["p_values"], dtype=float)
        n_steps = np.asarray(data["n_steps"], dtype=np.int32)
        skipped = np.asarray(
            data["skipped"] if "skipped" in data.files else np.zeros_like(p_values, dtype=bool),
            dtype=bool,
        )
        ps_full = data["ps_full"]
        ip_full = data["iplus_full"]
        im_full = data["iminus_full"]
        n_samp = int(p_values.size)

        if exclude_trailing_samples is None:
            n_exclude = max(0, min(int(meta.get("n_unmanip_samples", 0)), n_samp))
        else:
            n_exclude = max(0, min(int(exclude_trailing_samples), n_samp))
        n_keep = n_samp - n_exclude

        gamma_rf = np.full(n_samp, np.nan, dtype=float)
        if "gamma_rf" in data.files:
            gamma_rf = np.asarray(data["gamma_rf"], dtype=float)
        burn_steps = np.full(n_samp, -1, dtype=np.int32)
        if "burn_steps" in data.files:
            burn_steps = np.asarray(data["burn_steps"], dtype=np.int32)

        if prefer_file_step_subsample and "step_subsample" in meta:
            sub = max(1, int(meta["step_subsample"]))
        else:
            sub = max(1, int(step_subsample))

        j_rep, step_rep = _build_flatten_row_indices(
            n_steps,
            skipped,
            n_keep=n_keep,
            step_subsample=sub,
        )
        if j_rep.size <= 0:
            return

        for js, ss in _iter_row_index_chunks(j_rep, step_rep, chunk_rows=chunk_rows):
            yield {
                "p0": p_values[js],
                "step": ss,
                "center_bin": np.full(js.size, center_bin, dtype=np.int32),
                "source": np.full(js.size, np.uint8(source), dtype=np.uint8),
                "gamma_rf": gamma_rf[js],
                "burn_steps": burn_steps[js],
                "ps": np.asarray(ps_full[js, ss], dtype=STORE_DTYPE),
                "iplus": np.asarray(ip_full[js, ss], dtype=STORE_DTYPE),
                "iminus": np.asarray(im_full[js, ss], dtype=STORE_DTYPE),
            }


def _flatten_spectrum_shard(
    shard: dict,
    *,
    source: int,
    center_bin: int,
    step_subsample: int = 1,
    exclude_trailing_samples: int = 0,
) -> dict[str, np.ndarray]:
    """Flatten (sample, timestep) trajectories into spectrum-level rows."""
    parts = list(
        _iter_flatten_spectrum_shard(
            shard,
            source=source,
            center_bin=center_bin,
            step_subsample=step_subsample,
            exclude_trailing_samples=exclude_trailing_samples,
            chunk_rows=DEFAULT_SPECTRUM_ROW_CHUNK,
        )
    )
    if not parts:
        ps_full = np.asarray(shard["ps_full"])
        return _empty_spectrum_row_arrays(int(ps_full.shape[-1]))
    if len(parts) == 1:
        return parts[0]
    return _concat_spectrum_rows(parts)


def _concat_spectrum_rows(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    parts = [p for p in parts if int(np.asarray(p["ps"]).shape[0]) > 0]
    if not parts:
        return parts[0] if parts else {}
    if len(parts) == 1:
        return parts[0]
    keys = parts[0].keys()
    out: dict[str, np.ndarray] = {}
    for key in keys:
        arrs = [p[key] for p in parts]
        if arrs[0].ndim == 1:
            out[key] = np.concatenate(arrs)
        else:
            out[key] = np.concatenate(arrs, axis=0)
    return out


def flatten_spectrum_shard_file_to_rows(
    path: Path,
    *,
    source: int,
    center_bin: int,
    step_subsample: int = 1,
    prefer_file_step_subsample: bool = False,
    chunk_rows: int = DEFAULT_SPECTRUM_ROW_CHUNK,
) -> dict[str, np.ndarray]:
    """Flatten one spectrum shard NPZ into training rows (n_rows, num_bins)."""
    parts = list(
        _iter_flatten_spectrum_shard_file(
            path,
            source=source,
            center_bin=center_bin,
            step_subsample=step_subsample,
            prefer_file_step_subsample=prefer_file_step_subsample,
            chunk_rows=chunk_rows,
        )
    )
    if not parts:
        with np.load(path, allow_pickle=False) as data:
            num_bins = int(data["ps_full"].shape[-1])
        return _empty_spectrum_row_arrays(num_bins)
    rows = _concat_spectrum_rows(parts)
    keep = _nonempty_spectrum_mask(rows["ps"], rows["iplus"], rows["iminus"])
    return _filter_spectrum_rows(rows, keep)


def flatten_spectrum_shard_files_to_rows(
    paths: list[Path],
    *,
    source: int,
    center_bin: int,
    step_subsample: int = 1,
    prefer_file_step_subsample: bool = False,
    chunk_rows: int = DEFAULT_SPECTRUM_ROW_CHUNK,
) -> dict[str, np.ndarray]:
    """Flatten one or more spectrum shard NPZs (e.g. batched parts) into rows."""
    if not paths:
        raise ValueError("paths must be non-empty")
    row_parts: list[dict[str, np.ndarray]] = []
    for path in paths:
        row_parts.append(
            flatten_spectrum_shard_file_to_rows(
                path,
                source=source,
                center_bin=center_bin,
                step_subsample=step_subsample,
                prefer_file_step_subsample=prefer_file_step_subsample,
                chunk_rows=chunk_rows,
            )
        )
    if len(row_parts) == 1:
        return row_parts[0]
    rows = _concat_spectrum_rows(row_parts)
    keep = _nonempty_spectrum_mask(rows["ps"], rows["iplus"], rows["iminus"])
    return _filter_spectrum_rows(rows, keep)


def save_spectrum_rows_npz(
    path: Path,
    rows: dict[str, np.ndarray],
    *,
    meta: dict | None = None,
) -> None:
    """Write pre-flattened full-spectrum training rows to NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_meta = {"dataset": "spectrum_rows_v2", "fields": "ps,iplus,iminus (n_rows, num_bins)"}
    if meta:
        payload_meta.update(meta)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(payload_meta)),
            **{key: rows[key] for key in SPECTRUM_ROW_KEYS},
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def load_spectrum_rows_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
        rows = {key: np.asarray(data[key]) for key in SPECTRUM_ROW_KEYS}
        rows["meta"] = meta
        return rows


SPECTRUM_TRAIN_MANIFEST_NAME = "spectrum_train_manifest.json"


class SpectrumShardWriter:
    """Buffers full-spectrum rows and flushes fixed-size shard NPZs."""

    def __init__(self, output_dir: Path, shard_size: int, base_meta: dict):
        self.output_dir = Path(output_dir)
        self.shard_size = int(shard_size)
        self.base_meta = base_meta
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: dict[str, list[np.ndarray]] = {k: [] for k in SPECTRUM_ROW_KEYS}
        self._buffered = 0
        self.shard_index = 0
        self.shard_files: list[str] = []
        self.shard_row_counts: list[int] = []
        self.n_written = 0

    def add(self, rows: dict[str, np.ndarray]) -> None:
        n = int(rows["ps"].shape[0])
        if n <= 0:
            return
        start = 0
        while start < n:
            room = self.shard_size - self._buffered
            if room <= 0:
                self._flush(self.shard_size)
                room = self.shard_size
            take = min(room, n - start)
            end = start + take
            for key in SPECTRUM_ROW_KEYS:
                self._buffer[key].append(rows[key][start:end])
            self._buffered += take
            start = end
            if self._buffered >= self.shard_size:
                self._flush(self.shard_size)

    def _flush(self, take: int | None) -> None:
        if self._buffered <= 0:
            return
        merged = {
            key: (parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0))
            for key, parts in self._buffer.items()
        }
        n_take = self._buffered if take is None else min(int(take), self._buffered)
        payload = {key: merged[key][:n_take] for key in SPECTRUM_ROW_KEYS}
        remainder_n = int(merged["ps"].shape[0]) - n_take

        shard_path = self.output_dir / f"spectrum_train_{self.shard_index:04d}.npz"
        shard_meta = dict(self.base_meta)
        shard_meta["shard_index"] = self.shard_index
        tmp = shard_path.with_name(f".{shard_path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, meta_json=np.asarray(json.dumps(shard_meta)), **payload)
        tmp.replace(shard_path)

        self.shard_files.append(shard_path.name)
        self.shard_row_counts.append(n_take)
        self.n_written += n_take
        self.shard_index += 1
        if remainder_n > 0:
            self._buffer = {key: [merged[key][n_take:]] for key in SPECTRUM_ROW_KEYS}
            self._buffered = remainder_n
        else:
            self._buffer = {k: [] for k in SPECTRUM_ROW_KEYS}
            self._buffered = 0

    def close(self) -> None:
        self._flush(None)


def load_ssrf_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        ps = np.asarray(data["ps"], dtype=float)
        ps_m = np.asarray(data["ps_m"], dtype=float)
        amp = np.asarray(data["amp"], dtype=float) if "amp" in data.files else np.abs(ps)
        amp_m = np.asarray(data["amp_m"], dtype=float) if "amp_m" in data.files else np.abs(ps_m)
        n_samp = int(np.asarray(data["p_values"]).shape[0])
        if "gamma_rf" in data.files:
            gamma_rf = np.asarray(data["gamma_rf"], dtype=float)
        else:
            gamma_rf = np.full(n_samp, float(meta.get("gamma_rf", np.nan)), dtype=float)
        if "burn_steps" in data.files:
            burn_steps = np.asarray(data["burn_steps"], dtype=np.int32)
        else:
            # Legacy continuous-burn shards: burn length = traj length - 1.
            n_steps = np.asarray(data["n_steps"], dtype=np.int32)
            burn_steps = np.maximum(n_steps - 1, 0).astype(np.int32)
        return {
            **meta,
            "p_values": np.asarray(data["p_values"], dtype=float),
            "gamma_rf": gamma_rf,
            "burn_steps": burn_steps,
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
            "ps": ps,
            "iplus": np.asarray(data["iplus"], dtype=float),
            "iminus": np.asarray(data["iminus"], dtype=float),
            "amp": amp,
            "ps_m": ps_m,
            "iplus_m": np.asarray(data["iplus_m"], dtype=float),
            "iminus_m": np.asarray(data["iminus_m"], dtype=float),
            "amp_m": amp_m,
        }


def save_afp_shard(result: dict, path: Path, *, extra_meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "bin_idx": int(result["bin_idx"]),
        "mirror_idx": int(result["mirror_idx"]),
        "R": float(result["R"]),
        "num_bins": int(result["num_bins"]),
        "dt": float(result["dt"]),
        "n_relax": int(result["n_relax"]),
        "afp_window": int(result["afp_window"]),
        "afp_efficiency": float(result["afp_efficiency"]),
        "afp_subset": [int(i) for i in np.asarray(result["afp_subset"]).tolist()],
        "physics_model": PHYSICS_MODEL,
        "dataset": "afp_bin_traj_v2",
    }
    if extra_meta:
        meta.update(extra_meta)
    ps = np.asarray(result["ps"], dtype=STORE_DTYPE)
    ps_m = np.asarray(result["ps_m"], dtype=STORE_DTYPE)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            p_values=np.asarray(result["p_values"], dtype=float),
            n_steps=np.asarray(result["n_steps"], dtype=np.int32),
            skipped=np.asarray(result["skipped"], dtype=bool),
            ps=ps,
            iplus=np.asarray(result["iplus"], dtype=STORE_DTYPE),
            iminus=np.asarray(result["iminus"], dtype=STORE_DTYPE),
            amp=np.abs(ps),
            ps_m=ps_m,
            iplus_m=np.asarray(result["iplus_m"], dtype=STORE_DTYPE),
            iminus_m=np.asarray(result["iminus_m"], dtype=STORE_DTYPE),
            amp_m=np.abs(ps_m),
            afp_subset=np.asarray(result["afp_subset"], dtype=np.int32),
            bin_idx=np.asarray(int(result["bin_idx"]), dtype=np.int32),
            mirror_idx=np.asarray(int(result["mirror_idx"]), dtype=np.int32),
            dt=np.asarray(float(result["dt"]), dtype=float),
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def load_afp_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        ps = np.asarray(data["ps"], dtype=float)
        ps_m = np.asarray(data["ps_m"], dtype=float)
        amp = np.asarray(data["amp"], dtype=float) if "amp" in data.files else np.abs(ps)
        amp_m = np.asarray(data["amp_m"], dtype=float) if "amp_m" in data.files else np.abs(ps_m)
        return {
            **meta,
            "p_values": np.asarray(data["p_values"], dtype=float),
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
            "ps": ps,
            "iplus": np.asarray(data["iplus"], dtype=float),
            "iminus": np.asarray(data["iminus"], dtype=float),
            "amp": amp,
            "ps_m": ps_m,
            "iplus_m": np.asarray(data["iplus_m"], dtype=float),
            "iminus_m": np.asarray(data["iminus_m"], dtype=float),
            "amp_m": amp_m,
            "afp_subset": np.asarray(data["afp_subset"], dtype=np.int32),
        }


def _mirror_bin_idx(num_bins: int, bin_idx: int) -> int:
    return int(num_bins) - 1 - int(bin_idx)


def _empty_arrays(ref_key: str) -> dict[str, np.ndarray]:
    return {
        "p0": np.zeros(0, dtype=float),
        "step": np.zeros(0, dtype=np.int32),
        "gamma_rf": np.zeros(0, dtype=float),
        "burn_steps": np.zeros(0, dtype=np.int32),
        ref_key: np.zeros(0, dtype=np.int32),
        "is_mirror": np.zeros(0, dtype=bool),
        "ps": np.zeros(0, dtype=float),
        "iplus": np.zeros(0, dtype=float),
        "iminus": np.zeros(0, dtype=float),
        "amp": np.zeros(0, dtype=float),
    }


def _arrays_from_shard_side(
    shard: dict,
    *,
    ref_key: str,
    is_mirror: bool,
    gamma_rf: np.ndarray | None = None,
    burn_steps: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Flatten one shard side (burn or mirror) into per-timestep training rows."""
    p_values = np.asarray(shard["p_values"], dtype=float)
    n_steps = np.asarray(shard["n_steps"], dtype=np.int32)
    n_samp = int(p_values.shape[0])
    if gamma_rf is None:
        gamma_rf = np.full(n_samp, np.nan, dtype=float)
    else:
        gamma_rf = np.asarray(gamma_rf, dtype=float)
    if burn_steps is None:
        burn_steps = np.full(n_samp, -1, dtype=np.int32)
    else:
        burn_steps = np.asarray(burn_steps, dtype=np.int32)

    lengths = np.maximum(n_steps, 0).astype(np.int64)
    total = int(lengths.sum())
    if total <= 0:
        return _empty_arrays(ref_key)

    ref_bin = int(shard["bin_idx"])
    if is_mirror:
        ps_src = np.asarray(shard["ps_m"], dtype=float)
        ip_src = np.asarray(shard["iplus_m"], dtype=float)
        im_src = np.asarray(shard["iminus_m"], dtype=float)
        amp_src = np.asarray(shard["amp_m"], dtype=float)
    else:
        ps_src = np.asarray(shard["ps"], dtype=float)
        ip_src = np.asarray(shard["iplus"], dtype=float)
        im_src = np.asarray(shard["iminus"], dtype=float)
        amp_src = np.asarray(shard["amp"], dtype=float)

    out = {
        "p0": np.empty(total, dtype=float),
        "step": np.empty(total, dtype=np.int32),
        "gamma_rf": np.empty(total, dtype=float),
        "burn_steps": np.empty(total, dtype=np.int32),
        ref_key: np.empty(total, dtype=np.int32),
        "is_mirror": np.empty(total, dtype=bool),
        "ps": np.empty(total, dtype=float),
        "iplus": np.empty(total, dtype=float),
        "iminus": np.empty(total, dtype=float),
        "amp": np.empty(total, dtype=float),
    }
    offset = 0
    for j in range(n_samp):
        n = int(lengths[j])
        if n <= 0:
            continue
        sl = slice(offset, offset + n)
        out["p0"][sl] = float(p_values[j])
        out["step"][sl] = np.arange(n, dtype=np.int32)
        out["gamma_rf"][sl] = float(gamma_rf[j])
        out["burn_steps"][sl] = int(burn_steps[j])
        out[ref_key][sl] = ref_bin
        out["is_mirror"][sl] = bool(is_mirror)
        out["ps"][sl] = ps_src[j, :n]
        out["iplus"][sl] = ip_src[j, :n]
        out["iminus"][sl] = im_src[j, :n]
        out["amp"][sl] = amp_src[j, :n]
        offset += n
    return out


def _concat_arrays(
    parts: list[dict[str, np.ndarray]],
    ref_key: str,
) -> dict[str, np.ndarray]:
    parts = [p for p in parts if int(np.asarray(p["ps"]).size) > 0]
    if not parts:
        return _empty_arrays(ref_key)
    if len(parts) == 1:
        return parts[0]
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def _save_train_bin(
    bin_idx: int,
    arrays: dict[str, np.ndarray],
    path: Path,
    *,
    dataset: str,
    ref_key: str,
    fields: str,
    n_missing: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(np.asarray(arrays["ps"]).size)
    meta = {
        "bin_idx": int(bin_idx),
        "n_samples": n_samples,
        "n_missing_shards": int(n_missing),
        "physics_model": PHYSICS_MODEL,
        "dataset": dataset,
        "fields": fields,
    }
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            bin_idx=np.asarray(int(bin_idx), dtype=np.int32),
            p0=np.asarray(arrays["p0"], dtype=float),
            step=np.asarray(arrays["step"], dtype=np.int32),
            gamma_rf=np.asarray(arrays["gamma_rf"], dtype=float),
            burn_steps=np.asarray(arrays["burn_steps"], dtype=np.int32),
            **{ref_key: np.asarray(arrays[ref_key], dtype=np.int32)},
            is_mirror=np.asarray(arrays["is_mirror"], dtype=bool),
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


def _missing_shards(
    shard_dir: Path,
    num_bins: int,
    shard_path_fn,
) -> list[int]:
    missing: list[int] = []
    for bin_idx in bin_index_range(num_bins):
        if not shard_path_fn(shard_dir, bin_idx).is_file():
            missing.append(bin_idx)
    return missing


def _organize_one_bin(
    out_bin: int,
    *,
    num_bins: int,
    shard_dir: Path,
    shard_path_fn,
    load_shard_fn,
    ref_key: str,
    with_ssrf_params: bool,
) -> dict[str, np.ndarray]:
    """Build training rows for one spectral bin from at most two shards.

    Own shard contributes burn/center samples; the mirror-partner shard
    contributes the mirrored observations filed under ``out_bin``.
    """
    parts: list[dict[str, np.ndarray]] = []
    own_path = shard_path_fn(shard_dir, out_bin)
    if own_path.is_file():
        shard = load_shard_fn(own_path)
        kwargs = {}
        if with_ssrf_params:
            kwargs = {
                "gamma_rf": shard["gamma_rf"],
                "burn_steps": shard["burn_steps"],
            }
        parts.append(
            _arrays_from_shard_side(
                shard,
                ref_key=ref_key,
                is_mirror=False,
                **kwargs,
            )
        )
        del shard

    partner = _mirror_bin_idx(num_bins, out_bin)
    if partner != out_bin:
        partner_path = shard_path_fn(shard_dir, partner)
        if partner_path.is_file():
            shard = load_shard_fn(partner_path)
            if int(shard["mirror_idx"]) == int(out_bin):
                kwargs = {}
                if with_ssrf_params:
                    kwargs = {
                        "gamma_rf": shard["gamma_rf"],
                        "burn_steps": shard["burn_steps"],
                    }
                parts.append(
                    _arrays_from_shard_side(
                        shard,
                        ref_key=ref_key,
                        is_mirror=True,
                        **kwargs,
                    )
                )
            del shard

    return _concat_arrays(parts, ref_key)


def organize_ssrf_shards(
    shard_dir: Path,
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    strict: bool = True,
) -> dict:
    """Route ssRF shards into one training NPZ per spectral bin.

    Streams one output bin at a time (own shard + mirror partner only) so peak
    RAM stays O(one shard + one train bin), not O(all shards).
    """
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_key = "burn_bin"
    missing = _missing_shards(shard_dir, num_bins, ssrf_shard_path)

    if missing and strict:
        raise FileNotFoundError(
            format_missing_bins_error(
                "ssRF shard",
                shard_dir,
                missing,
                num_bins=int(num_bins),
                path_fn=ssrf_shard_path,
            )
        )
    if missing:
        print(f"WARNING: missing {len(missing)} shards; continuing", flush=True)

    samples_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    for bin_idx in bin_index_range(num_bins):
        arrays = _organize_one_bin(
            bin_idx,
            num_bins=int(num_bins),
            shard_dir=shard_dir,
            shard_path_fn=ssrf_shard_path,
            load_shard_fn=load_ssrf_shard,
            ref_key=ref_key,
            with_ssrf_params=True,
        )
        samples_per_bin[bin_idx] = int(arrays["ps"].size)
        _save_train_bin(
            bin_idx,
            arrays,
            ssrf_train_bin_path(output_dir, bin_idx),
            dataset="ssrf_train_bin_v2",
            ref_key=ref_key,
            fields=(
                "ps,iplus,iminus,amp at this bin; burn_bin=RF center; is_mirror; "
                "gamma_rf; burn_steps; step along that fixed burn"
            ),
            n_missing=len(missing),
        )
        if (bin_idx + 1) % 50 == 0 or bin_idx + 1 == int(num_bins):
            print(
                f"  organized {bin_idx + 1}/{int(num_bins)} bins "
                f"(running samples={int(samples_per_bin[: bin_idx + 1].sum())})",
                flush=True,
            )

    return {
        "output_dir": str(output_dir),
        "samples_per_bin": samples_per_bin,
        "n_samples": int(samples_per_bin.sum()),
        "n_missing": len(missing),
        "dataset": "ssrf_train_bin_v2",
    }


def organize_afp_shards(
    shard_dir: Path,
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    strict: bool = True,
) -> dict:
    """Route AFP shards into one training NPZ per spectral bin (streaming)."""
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_key = "center_bin"
    missing = _missing_shards(shard_dir, num_bins, afp_shard_path)

    if missing and strict:
        raise FileNotFoundError(
            format_missing_bins_error(
                "AFP shard",
                shard_dir,
                missing,
                num_bins=int(num_bins),
                path_fn=afp_shard_path,
            )
        )
    if missing:
        print(f"WARNING: missing {len(missing)} shards; continuing", flush=True)

    samples_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    for bin_idx in bin_index_range(num_bins):
        arrays = _organize_one_bin(
            bin_idx,
            num_bins=int(num_bins),
            shard_dir=shard_dir,
            shard_path_fn=afp_shard_path,
            load_shard_fn=load_afp_shard,
            ref_key=ref_key,
            with_ssrf_params=False,
        )
        samples_per_bin[bin_idx] = int(arrays["ps"].size)
        _save_train_bin(
            bin_idx,
            arrays,
            afp_train_bin_path(output_dir, bin_idx),
            dataset="afp_train_bin_v2",
            ref_key=ref_key,
            fields="ps,iplus,iminus,amp at this bin; center_bin=AFP center; is_mirror",
            n_missing=len(missing),
        )

    return {
        "output_dir": str(output_dir),
        "samples_per_bin": samples_per_bin,
        "n_samples": int(samples_per_bin.sum()),
        "n_missing": len(missing),
        "dataset": "afp_train_bin_v2",
    }
