"""Organize per-bin trajectory shards into per-spectral-bin training NPZs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from bin_paths import (
    afp_shard_path,
    afp_train_bin_path,
    format_missing_bins_error,
    missing_shards,
    ssrf_shard_path,
    ssrf_train_bin_path,
)
from common import NUM_BINS, PHYSICS_MODEL
from pq_calibration import calibrated_pq_fields, load_pq_calibration, validate_stored_per_bin_pq
from shard_store import load_afp_shard, load_ssrf_shard

_ORGANIZE_CONFIG = {
    "ssrf": {
        "ref_key": "burn_bin",
        "dataset": "ssrf_train_bin_v2",
        "fields": (
            "ps,q=raw I± sums; P,Q=CC-calibrated true polarizations at this bin; "
            "burn_bin=RF center; is_mirror; is_neighbor; gamma_rf; burn_steps; step"
        ),
        "with_ssrf_params": True,
    },
    "afp": {
        "ref_key": "center_bin",
        "dataset": "afp_train_bin_v2",
        "fields": (
            "ps,q=raw I± sums; P,Q=CC-calibrated true polarizations at this bin; "
            "center_bin=AFP center; is_mirror; is_neighbor"
        ),
        "with_ssrf_params": False,
    },
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
        "is_neighbor": np.zeros(0, dtype=bool),
        "ps": np.zeros(0, dtype=float),
        "iplus": np.zeros(0, dtype=float),
        "iminus": np.zeros(0, dtype=float),
        "q": np.zeros(0, dtype=float),
        "amp": np.zeros(0, dtype=float),
        "P": np.zeros(0, dtype=float),
        "Q": np.zeros(0, dtype=float),
    }


def _arrays_from_shard_side(
    shard: dict,
    *,
    ref_key: str,
    is_mirror: bool,
    is_neighbor: bool = False,
    side_suffix: str = "",
    track_mask: np.ndarray | None = None,
    gamma_rf: np.ndarray | None = None,
    burn_steps: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
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
    if track_mask is not None:
        active = np.asarray(track_mask, dtype=bool) & (lengths > 0)
        total = int(lengths[active].sum())
    else:
        total = int(lengths.sum())
    if total <= 0:
        return _empty_arrays(ref_key)

    ref_bin = int(shard["bin_idx"])
    suffix = side_suffix if side_suffix else ("_m" if is_mirror else "")
    ps_src = np.asarray(shard[f"ps{suffix}"], dtype=float)
    ip_src = np.asarray(shard[f"iplus{suffix}"], dtype=float)
    im_src = np.asarray(shard[f"iminus{suffix}"], dtype=float)
    q_key = f"q{suffix}"
    q_src = (
        np.asarray(shard[q_key], dtype=float)
        if q_key in shard
        else ip_src - im_src
    )
    if suffix in ("", "_m") and f"amp{suffix}" in shard:
        amp_src = np.asarray(shard[f"amp{suffix}"], dtype=float)
    else:
        amp_src = np.abs(ps_src)

    out = {
        "p0": np.empty(total, dtype=float),
        "step": np.empty(total, dtype=np.int32),
        "gamma_rf": np.empty(total, dtype=float),
        "burn_steps": np.empty(total, dtype=np.int32),
        ref_key: np.empty(total, dtype=np.int32),
        "is_mirror": np.empty(total, dtype=bool),
        "is_neighbor": np.empty(total, dtype=bool),
        "ps": np.empty(total, dtype=float),
        "iplus": np.empty(total, dtype=float),
        "iminus": np.empty(total, dtype=float),
        "q": np.empty(total, dtype=float),
        "amp": np.empty(total, dtype=float),
    }
    offset = 0
    for j in range(n_samp):
        n = int(lengths[j])
        if n <= 0:
            continue
        if track_mask is not None and not bool(track_mask[j]):
            continue
        sl = slice(offset, offset + n)
        out["p0"][sl] = float(p_values[j])
        out["step"][sl] = np.arange(n, dtype=np.int32)
        out["gamma_rf"][sl] = float(gamma_rf[j])
        out["burn_steps"][sl] = int(burn_steps[j])
        out[ref_key][sl] = ref_bin
        out["is_mirror"][sl] = bool(is_mirror)
        out["is_neighbor"][sl] = bool(is_neighbor)
        out["ps"][sl] = ps_src[j, :n]
        out["iplus"][sl] = ip_src[j, :n]
        out["iminus"][sl] = im_src[j, :n]
        out["q"][sl] = q_src[j, :n]
        out["amp"][sl] = amp_src[j, :n]
        offset += n
    if offset <= 0:
        return _empty_arrays(ref_key)
    if offset < total:
        for key in out:
            out[key] = out[key][:offset]
    return out


def _concat_arrays(parts: list[dict[str, np.ndarray]], ref_key: str) -> dict[str, np.ndarray]:
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
    num_bins: int = NUM_BINS,
    pq_calibration: Optional[dict] = None,
    pq_post_correct: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p_true, q_true = calibrated_pq_fields(
        arrays,
        num_bins=int(num_bins),
        calibration=pq_calibration,
        post_correct=pq_post_correct,
    )
    cal = pq_calibration or load_pq_calibration(num_bins=int(num_bins))
    validate_stored_per_bin_pq(
        arrays["ps"],
        arrays["q"],
        arrays["p0"],
        p_true,
        q_true,
        calibration=cal,
        post_correct=pq_post_correct,
    )
    meta = {
        "bin_idx": int(bin_idx),
        "n_samples": int(np.asarray(arrays["ps"]).size),
        "n_missing_shards": int(n_missing),
        "physics_model": PHYSICS_MODEL,
        "dataset": dataset,
        "fields": fields,
        "pq_calibrated": True,
        "pq_target_scope": "per_bin",
        "pq_cc_scale": "cc_bin",
        "pq_post_correct": bool(pq_post_correct),
        "pq_cc_bin": float(cal["cc_bin"]),
        "pq_amp": float(cal["amp"]),
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
            is_neighbor=np.asarray(arrays["is_neighbor"], dtype=bool),
            ps=np.asarray(arrays["ps"], dtype=float),
            iplus=np.asarray(arrays["iplus"], dtype=float),
            iminus=np.asarray(arrays["iminus"], dtype=float),
            q=np.asarray(arrays["q"], dtype=float),
            amp=np.asarray(arrays["amp"], dtype=float),
            P=np.asarray(p_true, dtype=float),
            Q=np.asarray(q_true, dtype=float),
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


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
    parts: list[dict[str, np.ndarray]] = []
    own_path = shard_path_fn(shard_dir, out_bin)
    if own_path.is_file():
        shard = load_shard_fn(own_path)
        kwargs = (
            {"gamma_rf": shard["gamma_rf"], "burn_steps": shard["burn_steps"]}
            if with_ssrf_params
            else {}
        )
        parts.append(_arrays_from_shard_side(shard, ref_key=ref_key, is_mirror=False, **kwargs))
        del shard

    partner = _mirror_bin_idx(num_bins, out_bin)
    if partner != out_bin:
        partner_path = shard_path_fn(shard_dir, partner)
        if partner_path.is_file():
            shard = load_shard_fn(partner_path)
            if int(shard["mirror_idx"]) == int(out_bin):
                kwargs = (
                    {"gamma_rf": shard["gamma_rf"], "burn_steps": shard["burn_steps"]}
                    if with_ssrf_params
                    else {}
                )
                parts.append(_arrays_from_shard_side(shard, ref_key=ref_key, is_mirror=True, **kwargs))
            del shard

    for burn_center, side_suffix, track_key in (
        (int(out_bin) - 1, "_hi", "track_hi"),
        (int(out_bin) + 1, "_lo", "track_lo"),
    ):
        if burn_center < 0 or burn_center >= int(num_bins):
            continue
        center_path = shard_path_fn(shard_dir, burn_center)
        if not center_path.is_file():
            continue
        shard = load_shard_fn(center_path)
        track = shard.get(track_key)
        if track is None or not np.any(np.asarray(track, dtype=bool)):
            del shard
            continue
        kwargs: dict = {}
        if with_ssrf_params:
            kwargs["gamma_rf"] = shard["gamma_rf"]
            kwargs["burn_steps"] = shard["burn_steps"]
        parts.append(
            _arrays_from_shard_side(
                shard,
                ref_key=ref_key,
                is_mirror=False,
                is_neighbor=True,
                side_suffix=side_suffix,
                track_mask=np.asarray(track, dtype=bool),
                **kwargs,
            )
        )
        del shard

    return _concat_arrays(parts, ref_key)


def _organize_shards(
    kind: str,
    shard_dir: Path,
    output_dir: Path,
    *,
    num_bins: int,
    strict: bool,
    shard_path_fn,
    train_path_fn,
    load_shard_fn,
) -> dict:
    cfg = _ORGANIZE_CONFIG[kind]
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_key = cfg["ref_key"]
    missing = missing_shards(shard_dir, num_bins, shard_path_fn)

    if missing and strict:
        raise FileNotFoundError(
            format_missing_bins_error(
                f"{kind.upper()} shard",
                shard_dir,
                missing,
                num_bins=int(num_bins),
                path_fn=shard_path_fn,
            )
        )
    if missing:
        print(f"WARNING: missing {len(missing)} shards; continuing", flush=True)

    pq_calibration = load_pq_calibration(num_bins=int(num_bins))
    samples_per_bin = np.zeros(int(num_bins), dtype=np.int64)
    for bin_idx in range(int(num_bins)):
        arrays = _organize_one_bin(
            bin_idx,
            num_bins=int(num_bins),
            shard_dir=shard_dir,
            shard_path_fn=shard_path_fn,
            load_shard_fn=load_shard_fn,
            ref_key=ref_key,
            with_ssrf_params=cfg["with_ssrf_params"],
        )
        samples_per_bin[bin_idx] = int(arrays["ps"].size)
        _save_train_bin(
            bin_idx,
            arrays,
            train_path_fn(output_dir, bin_idx),
            dataset=cfg["dataset"],
            ref_key=ref_key,
            fields=cfg["fields"],
            n_missing=len(missing),
            num_bins=int(num_bins),
            pq_calibration=pq_calibration,
        )
        if kind == "ssrf" and ((bin_idx + 1) % 50 == 0 or bin_idx + 1 == int(num_bins)):
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
        "dataset": cfg["dataset"],
    }


def organize_ssrf_shards(
    shard_dir: Path,
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    strict: bool = True,
) -> dict:
    return _organize_shards(
        "ssrf",
        shard_dir,
        output_dir,
        num_bins=num_bins,
        strict=strict,
        shard_path_fn=ssrf_shard_path,
        train_path_fn=ssrf_train_bin_path,
        load_shard_fn=load_ssrf_shard,
    )


def organize_afp_shards(
    shard_dir: Path,
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    strict: bool = True,
) -> dict:
    return _organize_shards(
        "afp",
        shard_dir,
        output_dir,
        num_bins=num_bins,
        strict=strict,
        shard_path_fn=afp_shard_path,
        train_path_fn=afp_train_bin_path,
        load_shard_fn=load_afp_shard,
    )
