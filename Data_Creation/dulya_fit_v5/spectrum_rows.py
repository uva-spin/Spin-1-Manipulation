"""Spectrum row flattening, validation, and NPZ I/O for dulya_fit_v5."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from common import PS_ABS_MIN, STORE_DTYPE

DEFAULT_SPECTRUM_ROW_CHUNK = 100_000

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

SPECTRUM_TRAIN_MANIFEST_NAME = "spectrum_train_manifest.json"

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
        # Instant AFP flip (n_relax=0) must keep the single post-flip spectrum.
        if int(meta.get("n_relax", -1)) <= 0 and "n_relax" in meta:
            sub = 1

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


