import json
import os
from pathlib import Path

import numpy as np

from bin_paths import list_ssrf_shard_paths
from common import PHYSICS_MODEL, RF_MODE, STORE_DTYPE, intensity_pq

_SSRF_SHARD_ARRAY_KEYS = (
    "p_values",
    "gamma_rf",
    "burn_steps",
    "n_steps",
    "skipped",
    "ps",
    "iplus",
    "iminus",
    "q",
    "amp",
    "ps_m",
    "iplus_m",
    "iminus_m",
    "q_m",
    "amp_m",
    "track_lo",
    "track_hi",
    "ps_lo",
    "iplus_lo",
    "iminus_lo",
    "q_lo",
    "ps_hi",
    "iplus_hi",
    "iminus_hi",
    "q_hi",
)

_TRAJ_STACK_META_KEYS = (
    "p_values",
    "gamma_rf",
    "burn_steps",
    "n_steps",
    "skipped",
)


def _save_npz_atomic(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def _gamma_burn_from_shard(data, meta: dict) -> tuple[np.ndarray, np.ndarray]:
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
    return gamma_rf, burn_steps


def load_ssrf_shard_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        gamma_rf, burn_steps = _gamma_burn_from_shard(data, {})
        return {
            "p_values": np.asarray(data["p_values"], dtype=float),
            "gamma_rf": gamma_rf,
            "burn_steps": burn_steps,
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
        }


def load_afp_shard_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "p_values": np.asarray(data["p_values"], dtype=float),
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
        }


def _concat_shard_parts(parts: list[dict], keys: tuple[str, ...]) -> dict:
    if not parts:
        raise ValueError("parts must be non-empty")
    if len(parts) == 1:
        return parts[0]
    merged = dict(parts[0])
    for key in keys:
        if key in parts[0]:
            merged[key] = np.concatenate([p[key] for p in parts], axis=0)
    return merged


def load_ssrf_shard_meta_any(shard_dir: Path, bin_idx: int) -> dict | None:
    paths = list_ssrf_shard_paths(shard_dir, bin_idx)
    if not paths:
        return None
    return _concat_shard_parts(
        [load_ssrf_shard_meta(p) for p in paths], _TRAJ_STACK_META_KEYS
    )


def load_ssrf_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        gamma_rf, burn_steps = _gamma_burn_from_shard(data, meta)
        ps = np.asarray(data["ps"], dtype=float)
        ps_m = np.asarray(data["ps_m"], dtype=float)
        amp = np.asarray(data["amp"], dtype=float) if "amp" in data.files else np.abs(ps)
        amp_m = np.asarray(data["amp_m"], dtype=float) if "amp_m" in data.files else np.abs(ps_m)
        iplus = np.asarray(data["iplus"], dtype=float)
        iminus = np.asarray(data["iminus"], dtype=float)
        iplus_m = np.asarray(data["iplus_m"], dtype=float)
        iminus_m = np.asarray(data["iminus_m"], dtype=float)
        q = np.asarray(data["q"], dtype=float) if "q" in data.files else iplus - iminus
        q_m = np.asarray(data["q_m"], dtype=float) if "q_m" in data.files else iplus_m - iminus_m
        out = {
            **meta,
            "p_values": np.asarray(data["p_values"], dtype=float),
            "gamma_rf": gamma_rf,
            "burn_steps": burn_steps,
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
            "ps": ps,
            "iplus": iplus,
            "iminus": iminus,
            "q": q,
            "amp": amp,
            "ps_m": ps_m,
            "iplus_m": iplus_m,
            "iminus_m": iminus_m,
            "q_m": q_m,
            "amp_m": amp_m,
        }
        for key in (
            "track_lo",
            "track_hi",
            "ps_lo",
            "iplus_lo",
            "iminus_lo",
            "q_lo",
            "ps_hi",
            "iplus_hi",
            "iminus_hi",
            "q_hi",
        ):
            if key in data.files:
                out[key] = np.asarray(data[key])
        return out


def load_ssrf_shard_any(shard_dir: Path, bin_idx: int) -> dict | None:
    paths = list_ssrf_shard_paths(shard_dir, bin_idx)
    if not paths:
        return None
    return _concat_shard_parts([load_ssrf_shard(p) for p in paths], _SSRF_SHARD_ARRAY_KEYS)


def load_afp_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        ps = np.asarray(data["ps"], dtype=float)
        ps_m = np.asarray(data["ps_m"], dtype=float)
        amp = np.asarray(data["amp"], dtype=float) if "amp" in data.files else np.abs(ps)
        amp_m = np.asarray(data["amp_m"], dtype=float) if "amp_m" in data.files else np.abs(ps_m)
        iplus = np.asarray(data["iplus"], dtype=float)
        iminus = np.asarray(data["iminus"], dtype=float)
        iplus_m = np.asarray(data["iplus_m"], dtype=float)
        iminus_m = np.asarray(data["iminus_m"], dtype=float)
        q = np.asarray(data["q"], dtype=float) if "q" in data.files else iplus - iminus
        q_m = np.asarray(data["q_m"], dtype=float) if "q_m" in data.files else iplus_m - iminus_m
        out = {
            **meta,
            "p_values": np.asarray(data["p_values"], dtype=float),
            "n_steps": np.asarray(data["n_steps"], dtype=np.int32),
            "skipped": np.asarray(data["skipped"], dtype=bool),
            "ps": ps,
            "iplus": iplus,
            "iminus": iminus,
            "q": q,
            "amp": amp,
            "ps_m": ps_m,
            "iplus_m": iplus_m,
            "iminus_m": iminus_m,
            "q_m": q_m,
            "amp_m": amp_m,
            "afp_subset": np.asarray(data["afp_subset"], dtype=np.int32),
        }
        if "track_lo" in data.files:
            out["track_lo"] = np.asarray(data["track_lo"], dtype=bool)
        if "track_hi" in data.files:
            out["track_hi"] = np.asarray(data["track_hi"], dtype=bool)
        for side in ("lo", "hi"):
            ip_key = f"iplus_{side}"
            if ip_key in data.files:
                ip_side = np.asarray(data[ip_key], dtype=float)
                im_side = np.asarray(data[f"iminus_{side}"], dtype=float)
                out[f"ps_{side}"] = np.asarray(data[f"ps_{side}"], dtype=float)
                out[ip_key] = ip_side
                out[f"iminus_{side}"] = im_side
                out[f"q_{side}"] = (
                    np.asarray(data[f"q_{side}"], dtype=float)
                    if f"q_{side}" in data.files
                    else ip_side - im_side
                )
        return out


def save_ssrf_shard(result: dict, path: Path, *, extra_meta: dict | None = None) -> None:
    gamma_values = np.asarray(result.get("gamma_values", []), dtype=float)
    steps_values = np.asarray(result.get("steps_values", []), dtype=np.int32)
    meta = {
        "bin_idx": int(result["bin_idx"]),
        "mirror_idx": int(result["mirror_idx"]),
        "R": float(result["R"]),
        "num_bins": int(result["num_bins"]),
        "dt": float(result["dt"]),
        "max_burn_steps": int(result.get("max_burn_steps", result.get("max_steps", 0))),
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
    ps = np.asarray(result["ps"], dtype=STORE_DTYPE)
    ps_m = np.asarray(result["ps_m"], dtype=STORE_DTYPE)
    iplus = np.asarray(result["iplus"], dtype=STORE_DTYPE)
    iminus = np.asarray(result["iminus"], dtype=STORE_DTYPE)
    iplus_m = np.asarray(result["iplus_m"], dtype=STORE_DTYPE)
    iminus_m = np.asarray(result["iminus_m"], dtype=STORE_DTYPE)
    _, q = intensity_pq(iplus, iminus)
    _, q_m = intensity_pq(iplus_m, iminus_m)
    payload = {
        "meta_json": np.asarray(json.dumps(meta)),
        "p_values": np.asarray(result["p_values"], dtype=float),
        "gamma_rf": np.asarray(result["gamma_rf"], dtype=float),
        "burn_steps": np.asarray(result["burn_steps"], dtype=np.int32),
        "n_steps": np.asarray(result["n_steps"], dtype=np.int32),
        "skipped": np.asarray(result["skipped"], dtype=bool),
        "ps": ps,
        "iplus": iplus,
        "iminus": iminus,
        "q": np.asarray(q, dtype=STORE_DTYPE),
        "amp": np.abs(ps),
        "ps_m": ps_m,
        "iplus_m": iplus_m,
        "iminus_m": iminus_m,
        "q_m": np.asarray(q_m, dtype=STORE_DTYPE),
        "amp_m": np.abs(ps_m),
        "bin_idx": np.asarray(int(result["bin_idx"]), dtype=np.int32),
        "mirror_idx": np.asarray(int(result["mirror_idx"]), dtype=np.int32),
        "dt": np.asarray(float(result["dt"]), dtype=float),
    }
    if "track_lo" in result:
        payload["track_lo"] = np.asarray(result["track_lo"], dtype=bool)
    if "track_hi" in result:
        payload["track_hi"] = np.asarray(result["track_hi"], dtype=bool)
    for side in ("lo", "hi"):
        ip_key = f"iplus_{side}"
        if ip_key in result and result[ip_key] is not None:
            ip_side = np.asarray(result[ip_key], dtype=STORE_DTYPE)
            im_side = np.asarray(result[f"iminus_{side}"], dtype=STORE_DTYPE)
            _, q_side = intensity_pq(ip_side, im_side)
            payload[f"ps_{side}"] = np.asarray(result[f"ps_{side}"], dtype=STORE_DTYPE)
            payload[ip_key] = ip_side
            payload[f"iminus_{side}"] = im_side
            payload[f"q_{side}"] = np.asarray(q_side, dtype=STORE_DTYPE)
    _save_npz_atomic(path, **payload)


def save_afp_shard(result: dict, path: Path, *, extra_meta: dict | None = None) -> None:
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
    iplus = np.asarray(result["iplus"], dtype=STORE_DTYPE)
    iminus = np.asarray(result["iminus"], dtype=STORE_DTYPE)
    iplus_m = np.asarray(result["iplus_m"], dtype=STORE_DTYPE)
    iminus_m = np.asarray(result["iminus_m"], dtype=STORE_DTYPE)
    _, q = intensity_pq(iplus, iminus)
    _, q_m = intensity_pq(iplus_m, iminus_m)
    payload = {
        "meta_json": np.asarray(json.dumps(meta)),
        "p_values": np.asarray(result["p_values"], dtype=float),
        "n_steps": np.asarray(result["n_steps"], dtype=np.int32),
        "skipped": np.asarray(result["skipped"], dtype=bool),
        "ps": ps,
        "iplus": iplus,
        "iminus": iminus,
        "q": np.asarray(q, dtype=STORE_DTYPE),
        "amp": np.abs(ps),
        "ps_m": ps_m,
        "iplus_m": iplus_m,
        "iminus_m": iminus_m,
        "q_m": np.asarray(q_m, dtype=STORE_DTYPE),
        "amp_m": np.abs(ps_m),
        "afp_subset": np.asarray(result["afp_subset"], dtype=np.int32),
        "bin_idx": np.asarray(int(result["bin_idx"]), dtype=np.int32),
        "mirror_idx": np.asarray(int(result["mirror_idx"]), dtype=np.int32),
        "dt": np.asarray(float(result["dt"]), dtype=float),
    }
    if "track_lo" in result:
        payload["track_lo"] = np.asarray(result["track_lo"], dtype=bool)
    if "track_hi" in result:
        payload["track_hi"] = np.asarray(result["track_hi"], dtype=bool)
    for side in ("lo", "hi"):
        ip_key = f"iplus_{side}"
        if ip_key in result and result[ip_key] is not None:
            ip_side = np.asarray(result[ip_key], dtype=STORE_DTYPE)
            im_side = np.asarray(result[f"iminus_{side}"], dtype=STORE_DTYPE)
            _, q_side = intensity_pq(ip_side, im_side)
            payload[f"ps_{side}"] = np.asarray(result[f"ps_{side}"], dtype=STORE_DTYPE)
            payload[ip_key] = ip_side
            payload[f"iminus_{side}"] = im_side
            payload[f"q_{side}"] = np.asarray(q_side, dtype=STORE_DTYPE)
    _save_npz_atomic(path, **payload)


_SPECTRUM_META_OPTIONAL = (
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
)


def _save_spectrum_shard(
    result: dict,
    path: Path,
    *,
    dataset: str,
    extra_meta: dict | None = None,
) -> None:
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
    for opt_key in _SPECTRUM_META_OPTIONAL:
        if opt_key in result:
            val = result[opt_key]
            meta[opt_key] = val.tolist() if isinstance(val, np.ndarray) else val

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
    _save_npz_atomic(path, **payload)


def save_ssrf_spectrum_shard(result: dict, path: Path, *, extra_meta: dict | None = None) -> None:
    _save_spectrum_shard(
        result,
        path,
        dataset=str(result.get("dataset", "ssrf_spectrum_bin_v2")),
        extra_meta=extra_meta,
    )


def save_afp_spectrum_shard(result: dict, path: Path, *, extra_meta: dict | None = None) -> None:
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
