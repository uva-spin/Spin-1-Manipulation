import json
from pathlib import Path

import numpy as np


def bin_index_range(num_bins: int) -> range:
    return range(int(num_bins))


def _shard_npz(output_dir: Path, stem: str) -> Path:
    return Path(output_dir) / stem


def traj_shard_path(output_dir: Path, prefix: str, bin_idx: int) -> Path:
    return _shard_npz(output_dir, f"{prefix}_bin_{int(bin_idx):04d}.npz")


def spectrum_shard_path(output_dir: Path, prefix: str, bin_idx: int) -> Path:
    return _shard_npz(output_dir, f"{prefix}_spectrum_bin_{int(bin_idx):04d}.npz")


def spectrum_rows_path(output_dir: Path, prefix: str, bin_idx: int) -> Path:
    return _shard_npz(output_dir, f"{prefix}_spectrum_rows_{int(bin_idx):04d}.npz")


def train_bin_path(output_dir: Path, prefix: str, bin_idx: int) -> Path:
    return _shard_npz(output_dir, f"{prefix}_train_bin_{int(bin_idx):04d}.npz")


def shard_part_path(output_dir: Path, prefix: str, bin_idx: int, part_idx: int) -> Path:
    return _shard_npz(output_dir, f"{prefix}_bin_{int(bin_idx):04d}_part{int(part_idx):04d}.npz")


def spectrum_shard_part_path(output_dir: Path, prefix: str, bin_idx: int, part_idx: int) -> Path:
    return _shard_npz(
        output_dir, f"{prefix}_spectrum_bin_{int(bin_idx):04d}_part{int(part_idx):04d}.npz"
    )


def shard_parts_manifest_path(output_dir: Path, prefix: str, bin_idx: int) -> Path:
    return Path(output_dir) / f"{prefix}_bin_{int(bin_idx):04d}_parts.json"


def spectrum_shard_parts_manifest_path(output_dir: Path, prefix: str, bin_idx: int) -> Path:
    return Path(output_dir) / f"{prefix}_spectrum_bin_{int(bin_idx):04d}_parts.json"


def unmanip_spectrum_rows_path(output_dir: Path) -> Path:
    return Path(output_dir) / "unmanip_spectrum_rows.npz"


# --- ssRF wrappers ---

def ssrf_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return traj_shard_path(output_dir, "ssrf", bin_idx)


def ssrf_spectrum_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return spectrum_shard_path(output_dir, "ssrf", bin_idx)


def ssrf_spectrum_rows_path(output_dir: Path, bin_idx: int) -> Path:
    return spectrum_rows_path(output_dir, "ssrf", bin_idx)


def ssrf_train_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return train_bin_path(output_dir, "ssrf", bin_idx)


def ssrf_shard_part_path(output_dir: Path, bin_idx: int, part_idx: int) -> Path:
    return shard_part_path(output_dir, "ssrf", bin_idx, part_idx)


def ssrf_spectrum_shard_part_path(output_dir: Path, bin_idx: int, part_idx: int) -> Path:
    return spectrum_shard_part_path(output_dir, "ssrf", bin_idx, part_idx)


def ssrf_shard_parts_manifest_path(output_dir: Path, bin_idx: int) -> Path:
    return shard_parts_manifest_path(output_dir, "ssrf", bin_idx)


def ssrf_spectrum_shard_parts_manifest_path(output_dir: Path, bin_idx: int) -> Path:
    return spectrum_shard_parts_manifest_path(output_dir, "ssrf", bin_idx)


# --- AFP wrappers ---

def afp_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return traj_shard_path(output_dir, "afp", bin_idx)


def afp_spectrum_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return spectrum_shard_path(output_dir, "afp", bin_idx)


def afp_spectrum_rows_path(output_dir: Path, bin_idx: int) -> Path:
    return spectrum_rows_path(output_dir, "afp", bin_idx)


def afp_train_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return train_bin_path(output_dir, "afp", bin_idx)


def list_batched_shard_paths(
    shard_dir: Path,
    *,
    main_path_fn,
    manifest_path_fn,
    glob_pattern: str,
) -> list[Path]:
    """Return monolithic shard and/or sorted part shards for one bin."""
    shard_dir = Path(shard_dir)
    main = main_path_fn(shard_dir)
    if main.is_file():
        return [main]
    manifest = manifest_path_fn(shard_dir)
    if manifest.is_file():
        meta = json.loads(manifest.read_text())
        parts = [shard_dir / name for name in meta.get("part_files", [])]
        if parts and all(p.is_file() for p in parts):
            return parts
    return sorted(shard_dir.glob(glob_pattern))


def batched_shard_complete(
    shard_dir: Path,
    *,
    main_path_fn,
    manifest_path_fn,
) -> bool:
    shard_dir = Path(shard_dir)
    if main_path_fn(shard_dir).is_file():
        return True
    manifest = manifest_path_fn(shard_dir)
    if not manifest.is_file():
        return False
    meta = json.loads(manifest.read_text())
    part_files = meta.get("part_files", [])
    return bool(part_files) and all((shard_dir / name).is_file() for name in part_files)


def list_ssrf_shard_paths(shard_dir: Path, bin_idx: int) -> list[Path]:
    return list_batched_shard_paths(
        shard_dir,
        main_path_fn=lambda d: ssrf_shard_path(d, bin_idx),
        manifest_path_fn=lambda d: ssrf_shard_parts_manifest_path(d, bin_idx),
        glob_pattern=f"ssrf_bin_{int(bin_idx):04d}_part*.npz",
    )


def list_ssrf_spectrum_shard_paths(shard_dir: Path, bin_idx: int) -> list[Path]:
    return list_batched_shard_paths(
        shard_dir,
        main_path_fn=lambda d: ssrf_spectrum_shard_path(d, bin_idx),
        manifest_path_fn=lambda d: ssrf_spectrum_shard_parts_manifest_path(d, bin_idx),
        glob_pattern=f"ssrf_spectrum_bin_{int(bin_idx):04d}_part*.npz",
    )


def ssrf_shard_complete(shard_dir: Path, bin_idx: int) -> bool:
    return batched_shard_complete(
        shard_dir,
        main_path_fn=lambda d: ssrf_shard_path(d, bin_idx),
        manifest_path_fn=lambda d: ssrf_shard_parts_manifest_path(d, bin_idx),
    )


def ssrf_spectrum_shard_complete(shard_dir: Path, bin_idx: int) -> bool:
    return batched_shard_complete(
        shard_dir,
        main_path_fn=lambda d: ssrf_spectrum_shard_path(d, bin_idx),
        manifest_path_fn=lambda d: ssrf_spectrum_shard_parts_manifest_path(d, bin_idx),
    )


def ssrf_traj_shard_exists(shard_dir: Path, bin_idx: int) -> bool:
    return bool(list_ssrf_shard_paths(shard_dir, bin_idx))


def afp_traj_shard_exists(shard_dir: Path, bin_idx: int) -> bool:
    return afp_shard_path(shard_dir, bin_idx).is_file()


def shard_has_ps_full(path: Path) -> bool:
    with np.load(path, allow_pickle=False) as data:
        return "ps_full" in data.files


def resolve_spectrum_shard_path(
    shard_dir: Path,
    bin_idx: int,
    *,
    prefix: str,
    list_parts_fn,
    traj_path_fn,
) -> Path | None:
    paths = list_parts_fn(shard_dir, bin_idx)
    if paths:
        return paths[0]
    shard_dir = Path(shard_dir)
    traj = traj_path_fn(shard_dir, bin_idx)
    if traj.is_file() and shard_has_ps_full(traj):
        return traj
    return None


def resolve_ssrf_spectrum_shard_path(shard_dir: Path, bin_idx: int) -> Path | None:
    return resolve_spectrum_shard_path(
        shard_dir,
        bin_idx,
        prefix="ssrf",
        list_parts_fn=list_ssrf_spectrum_shard_paths,
        traj_path_fn=ssrf_shard_path,
    )


def resolve_afp_spectrum_shard_path(shard_dir: Path, bin_idx: int) -> Path | None:
    spec = afp_spectrum_shard_path(shard_dir, bin_idx)
    if spec.is_file():
        return spec
    traj = afp_shard_path(shard_dir, bin_idx)
    if traj.is_file() and shard_has_ps_full(traj):
        return traj
    return None


def _traj_shard_mismatch_hint(
    shard_dir: Path,
    *,
    spectrum_path_fn,
    traj_path_fn,
) -> str:
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
            shard_dir, spectrum_path_fn=path_fn, traj_path_fn=traj_path_fn
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


def missing_shards(shard_dir: Path, num_bins: int, shard_path_fn) -> list[int]:
    return [
        bin_idx
        for bin_idx in range(int(num_bins))
        if not shard_path_fn(shard_dir, bin_idx).is_file()
    ]
