"""
Single-manipulation event NPZ I/O.

Each file stores one physical manipulation (final spectrum only), not a stacked
training row or timestep grid. Intended for realistic evaluation of
SpectrumSplitNet and related models.

Event NPZ layout
----------------
Arrays (length ``num_bins``, typically 500):
  ps, iplus, iminus, frequency

Scalars (0-d arrays):
  p0, source, center_bin, step, burn_steps, gamma_rf, is_final_step

Metadata:
  meta_json  — JSON string with event_id, manipulation_mode, provenance, etc.

Example:
  python Data_Creation/create_sample_manipulation_events.py
  python -c "from manipulation_event_io import load_events_directory; ..."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_NAME = "manifest.json"
DATASET_NAME = "manipulation_event_v1"

EVENT_ARRAY_KEYS = ("ps", "iplus", "iminus", "frequency")
EVENT_SCALAR_KEYS = (
    "p0",
    "source",
    "center_bin",
    "step",
    "burn_steps",
    "gamma_rf",
    "is_final_step",
)

SOURCE_LABELS = {0: "ssrf", 1: "afp", 2: "unmanip"}


def _scalar_array(value: Any, dtype: np.dtype | type) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(())


def save_manipulation_event(
    path: Path,
    *,
    ps: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    frequency: np.ndarray,
    p0: float,
    source: int,
    manipulation_mode: str,
    center_bin: int,
    step: int = 0,
    burn_steps: int = -1,
    gamma_rf: float = float("nan"),
    is_final_step: bool = True,
    event_id: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Write one manipulation event to ``path`` (compressed NPZ)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ps_arr = np.asarray(ps, dtype=np.float32).reshape(-1)
    ip_arr = np.asarray(iplus, dtype=np.float32).reshape(-1)
    im_arr = np.asarray(iminus, dtype=np.float32).reshape(-1)
    freq_arr = np.asarray(frequency, dtype=np.float32).reshape(-1)
    if not (ps_arr.shape == ip_arr.shape == im_arr.shape == freq_arr.shape):
        raise ValueError("ps, iplus, iminus, and frequency must have the same length")

    src = int(source)
    if event_id is None:
        event_id = f"{SOURCE_LABELS.get(src, f'source{src}')}_p{float(p0):.3f}_bin{int(center_bin)}"

    meta: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "event_id": str(event_id),
        "manipulation_mode": str(manipulation_mode),
        "source_code": src,
        "source_label": SOURCE_LABELS.get(src, f"source={src}"),
        "num_bins": int(ps_arr.size),
        "is_final_step": bool(is_final_step),
    }
    if extra_meta:
        meta.update(extra_meta)

    tmp = path.with_name(f".{path.stem}.tmp.npz")
    try:
        np.savez_compressed(
            tmp,
            meta_json=np.asarray(json.dumps(meta, indent=2)),
            ps=ps_arr,
            iplus=ip_arr,
            iminus=im_arr,
            frequency=freq_arr,
            p0=_scalar_array(p0, np.float32),
            source=_scalar_array(src, np.uint8),
            center_bin=_scalar_array(center_bin, np.int32),
            step=_scalar_array(step, np.int32),
            burn_steps=_scalar_array(burn_steps, np.int32),
            gamma_rf=_scalar_array(gamma_rf, np.float32),
            is_final_step=_scalar_array(is_final_step, np.bool_),
        )
        tmp.replace(path)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    return path


def load_manipulation_event(path: Path) -> dict[str, Any]:
    """Load one event NPZ into a plain dict."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
        event = {
            "path": str(path),
            "meta": meta,
            "event_id": str(meta.get("event_id", path.stem)),
            **{key: np.asarray(data[key]) for key in EVENT_ARRAY_KEYS},
            **{key: data[key].item() if key in data.files else None for key in EVENT_SCALAR_KEYS},
        }
    if event["is_final_step"] is None:
        event["is_final_step"] = True
    return event


def discover_event_files(root: Path) -> list[Path]:
    """Return sorted event NPZ paths under ``root`` (excluding manifest/tmp files)."""
    root = Path(root)
    if root.is_file() and root.suffix == ".npz":
        return [root]

    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths: list[Path] = []
        for entry in manifest.get("events", []):
            rel = entry.get("path") or entry.get("file")
            if rel:
                paths.append((root / str(rel)).resolve())
        if paths:
            return sorted(paths)

    return sorted(
        p
        for p in root.rglob("*.npz")
        if p.is_file() and not p.name.startswith(".") and p.name != MANIFEST_NAME
    )


def load_events_directory(root: Path) -> list[dict[str, Any]]:
    """Load all manipulation events under ``root``."""
    paths = discover_event_files(root)
    if not paths:
        raise FileNotFoundError(f"No manipulation event NPZ files found under {root}")
    return [load_manipulation_event(path) for path in paths]


def events_to_batch(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Stack loaded events into batch arrays for model evaluation."""
    if not events:
        raise ValueError("events list is empty")
    return {
        "ps": np.stack([np.asarray(e["ps"], dtype=np.float32) for e in events], axis=0),
        "iplus": np.stack([np.asarray(e["iplus"], dtype=np.float32) for e in events], axis=0),
        "iminus": np.stack([np.asarray(e["iminus"], dtype=np.float32) for e in events], axis=0),
        "p0": np.asarray([float(e["p0"]) for e in events], dtype=np.float32),
        "source": np.asarray([int(e["source"]) for e in events], dtype=np.uint8),
        "event_id": np.asarray([str(e["event_id"]) for e in events], dtype=object),
    }


def write_manifest(root: Path, event_paths: list[Path]) -> Path:
    """Write ``manifest.json`` listing event files relative to ``root``."""
    root = Path(root)
    entries = []
    for path in sorted(event_paths):
        path = Path(path)
        event = load_manipulation_event(path)
        entries.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "event_id": event["event_id"],
                "source": SOURCE_LABELS.get(int(event["source"]), "unknown"),
                "p0": float(event["p0"]),
                "center_bin": int(event["center_bin"]),
            }
        )
    manifest = {
        "dataset": DATASET_NAME,
        "n_events": len(entries),
        "events": entries,
    }
    out = root / MANIFEST_NAME
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
