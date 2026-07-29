"""Shared NPZ shard / train I/O for dulya_fit_v2 workers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from common import NUM_BINS, PHYSICS_MODEL, RF_MODE, SOURCE_AFP, SOURCE_SSRF, SOURCE_UNMANIP


def ssrf_spectrum_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_spectrum_bin_{int(bin_idx):04d}.npz"


def afp_spectrum_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"afp_spectrum_bin_{int(bin_idx):04d}.npz"


def ssrf_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"ssrf_bin_{int(bin_idx):04d}.npz"


def afp_shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"afp_bin_{int(bin_idx):04d}.npz"


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
    ps = np.asarray(result["ps"], dtype=float)
    ps_m = np.asarray(result["ps_m"], dtype=float)
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
            iplus=np.asarray(result["iplus"], dtype=float),
            iminus=np.asarray(result["iminus"], dtype=float),
            amp=np.abs(ps),
            ps_m=ps_m,
            iplus_m=np.asarray(result["iplus_m"], dtype=float),
            iminus_m=np.asarray(result["iminus_m"], dtype=float),
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
            payload[key] = np.asarray(result[key], dtype=float)

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


def _flatten_spectrum_shard(
    shard: dict,
    *,
    source: int,
    center_bin: int,
    step_subsample: int = 1,
) -> dict[str, np.ndarray]:
    """Flatten (sample, timestep) trajectories into spectrum-level rows."""
    p_values = np.asarray(shard["p_values"], dtype=float)
    n_steps = np.asarray(shard["n_steps"], dtype=np.int32)
    skipped = np.asarray(shard.get("skipped", np.zeros_like(p_values, dtype=bool)), dtype=bool)
    ps_full = np.asarray(shard["ps_full"], dtype=float)
    ip_full = np.asarray(shard["iplus_full"], dtype=float)
    im_full = np.asarray(shard["iminus_full"], dtype=float)
    n_samp = int(p_values.size)
    num_bins = int(ps_full.shape[-1])
    sub = max(1, int(step_subsample))

    gamma_rf = np.full(n_samp, np.nan, dtype=float)
    if "gamma_rf" in shard:
        gamma_rf = np.asarray(shard["gamma_rf"], dtype=float)
    burn_steps = np.full(n_samp, -1, dtype=np.int32)
    if "burn_steps" in shard:
        burn_steps = np.asarray(shard["burn_steps"], dtype=np.int32)

    rows: list[int] = []
    for j in range(n_samp):
        if bool(skipped[j]):
            continue
        n = int(n_steps[j])
        if n <= 0:
            continue
        for step in range(0, n, sub):
            rows.append((j, step))

    total = len(rows)
    if total <= 0:
        return {
            "p0": np.zeros(0, dtype=float),
            "step": np.zeros(0, dtype=np.int32),
            "center_bin": np.zeros(0, dtype=np.int32),
            "source": np.zeros(0, dtype=np.uint8),
            "gamma_rf": np.zeros(0, dtype=float),
            "burn_steps": np.zeros(0, dtype=np.int32),
            "ps": np.zeros((0, num_bins), dtype=float),
            "iplus": np.zeros((0, num_bins), dtype=float),
            "iminus": np.zeros((0, num_bins), dtype=float),
        }

    out = {
        "p0": np.empty(total, dtype=float),
        "step": np.empty(total, dtype=np.int32),
        "center_bin": np.full(total, int(center_bin), dtype=np.int32),
        "source": np.full(total, int(source), dtype=np.uint8),
        "gamma_rf": np.empty(total, dtype=float),
        "burn_steps": np.empty(total, dtype=np.int32),
        "ps": np.empty((total, num_bins), dtype=float),
        "iplus": np.empty((total, num_bins), dtype=float),
        "iminus": np.empty((total, num_bins), dtype=float),
    }
    for idx, (j, step) in enumerate(rows):
        out["p0"][idx] = float(p_values[j])
        out["step"][idx] = int(step)
        out["gamma_rf"][idx] = float(gamma_rf[j])
        out["burn_steps"][idx] = int(burn_steps[j])
        out["ps"][idx] = ps_full[j, step]
        out["iplus"][idx] = ip_full[j, step]
        out["iminus"][idx] = im_full[j, step]
    return out


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
    ps = np.asarray(result["ps"], dtype=float)
    ps_m = np.asarray(result["ps_m"], dtype=float)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            p_values=np.asarray(result["p_values"], dtype=float),
            n_steps=np.asarray(result["n_steps"], dtype=np.int32),
            skipped=np.asarray(result["skipped"], dtype=bool),
            ps=ps,
            iplus=np.asarray(result["iplus"], dtype=float),
            iminus=np.asarray(result["iminus"], dtype=float),
            amp=np.abs(ps),
            ps_m=ps_m,
            iplus_m=np.asarray(result["iplus_m"], dtype=float),
            iminus_m=np.asarray(result["iminus_m"], dtype=float),
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
    for bin_idx in range(int(num_bins)):
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
            f"Missing {len(missing)} shard(s) under {shard_dir}; "
            f"first missing bin_idx={missing[0]}"
        )
    if missing:
        print(f"WARNING: missing {len(missing)} shards; continuing", flush=True)

    samples_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    for bin_idx in range(int(num_bins)):
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
            f"Missing {len(missing)} shard(s) under {shard_dir}; "
            f"first missing bin_idx={missing[0]}"
        )
    if missing:
        print(f"WARNING: missing {len(missing)} shards; continuing", flush=True)

    samples_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    for bin_idx in range(int(num_bins)):
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
