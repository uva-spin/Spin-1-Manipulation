"""
Plot one full-spectrum row from spectrum_train.npz (or a sharded train directory).

Each row is a single (P, source, step, ...) event with ps/iplus/iminus vectors
over all spectral bins.

Examples (from this directory):
  python plot_spectrum_event.py --index 0
  python plot_spectrum_event.py --input data/spectrum_train/spectrum_train.npz --index 42
  python plot_spectrum_event.py --input data/spectrum_train --index 1000
  python plot_spectrum_event.py --source ssrf --pick first
  python plot_spectrum_event.py --source afp --pick random --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from bin_io import SPECTRUM_ROW_KEYS, SPECTRUM_TRAIN_MANIFEST_NAME
from common import (
    F_MAX,
    F_MIN,
    NUM_BINS,
    PLOT_DIR,
    SOURCE_AFP,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
    SPECTRUM_TRAIN_NPZ,
)

_SOURCE_BY_NAME = {
    "ssrf": SOURCE_SSRF,
    "afp": SOURCE_AFP,
    "unmanip": SOURCE_UNMANIP,
    "unmanipulated": SOURCE_UNMANIP,
}
_SOURCE_LABEL = {
    SOURCE_SSRF: "ssRF",
    SOURCE_AFP: "AFP",
    SOURCE_UNMANIP: "unmanipulated",
}


def _frequency_axis(num_bins: int) -> np.ndarray:
    nb = int(num_bins)
    return np.linspace(float(F_MIN), float(F_MAX), nb, dtype=float)


def _resolve_index(index: int, n_samples: int) -> int:
    idx = int(index)
    if idx < 0:
        idx = int(n_samples) + idx
    if idx < 0 or idx >= int(n_samples):
        raise IndexError(f"event index {index} out of range for n_samples={n_samples}")
    return idx


def _load_meta(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if "meta_json" not in data.files:
            return {}
        return json.loads(str(data["meta_json"]))


def _event_from_mmap(data: np.lib.npyio.NpzFile, index: int) -> dict[str, np.ndarray]:
    return {key: np.asarray(data[key][index]) for key in SPECTRUM_ROW_KEYS}


def load_spectrum_event(input_path: Path, index: int) -> tuple[dict[str, np.ndarray], dict, int]:
    """Load one event by global row index from a single NPZ or sharded directory."""
    input_path = Path(input_path)
    if input_path.is_file():
        with np.load(input_path, mmap_mode="r", allow_pickle=False) as data:
            n_samples = int(data["ps"].shape[0])
            idx = _resolve_index(index, n_samples)
            event = _event_from_mmap(data, idx)
            meta = _load_meta(input_path)
        return event, meta, n_samples

    if not input_path.is_dir():
        raise FileNotFoundError(f"input not found: {input_path}")

    single = input_path / "spectrum_train.npz"
    if single.is_file():
        return load_spectrum_event(single, index)

    manifest_path = input_path / SPECTRUM_TRAIN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"expected {single} or {manifest_path} under {input_path}"
        )

    manifest = json.loads(manifest_path.read_text())
    shard_files = [str(name) for name in manifest.get("shard_files", [])]
    shard_counts = [int(n) for n in manifest.get("shard_row_counts", [])]
    if not shard_files or len(shard_files) != len(shard_counts):
        raise ValueError(f"invalid manifest at {manifest_path}")

    n_samples = int(manifest.get("n_samples", sum(shard_counts)))
    idx = _resolve_index(index, n_samples)
    offset = 0
    for shard_name, shard_n in zip(shard_files, shard_counts, strict=True):
        if idx < offset + shard_n:
            local_idx = idx - offset
            shard_path = input_path / shard_name
            with np.load(shard_path, mmap_mode="r", allow_pickle=False) as data:
                event = _event_from_mmap(data, local_idx)
            return event, manifest, n_samples
        offset += shard_n

    raise IndexError(f"event index {index} out of range for n_samples={n_samples}")


def find_event_index(
    input_path: Path,
    *,
    source: int,
    pick: str,
    seed: int | None,
) -> int:
    """Return a global row index matching source and pick mode (first/random)."""
    input_path = Path(input_path)
    rng = np.random.default_rng(seed)

    def _scan_file(path: Path) -> int:
        with np.load(path, mmap_mode="r", allow_pickle=False) as data:
            sources = np.asarray(data["source"], dtype=np.int32)
            mask = sources == int(source)
            hits = np.flatnonzero(mask)
            if hits.size == 0:
                raise ValueError(f"no events with source={_SOURCE_LABEL.get(source, source)!r}")
            if pick == "first":
                return int(hits[0])
            if pick == "random":
                return int(rng.choice(hits))
            raise ValueError(f"unsupported pick={pick!r}")

    if input_path.is_file():
        return _scan_file(input_path)

    single = input_path / "spectrum_train.npz"
    if single.is_file():
        return _scan_file(single)

    manifest_path = input_path / SPECTRUM_TRAIN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(input_path)

    manifest = json.loads(manifest_path.read_text())
    offset = 0
    collected_first: int | None = None
    random_choices: list[int] = []

    for shard_name, shard_n in zip(
        manifest.get("shard_files", []),
        manifest.get("shard_row_counts", []),
        strict=True,
    ):
        shard_path = input_path / str(shard_name)
        with np.load(shard_path, mmap_mode="r", allow_pickle=False) as data:
            sources = np.asarray(data["source"], dtype=np.int32)
            hits = np.flatnonzero(sources == int(source))
            if hits.size:
                global_hits = hits + offset
                if collected_first is None:
                    collected_first = int(global_hits[0])
                random_choices.extend(int(i) for i in global_hits)
        offset += int(shard_n)

    if collected_first is None:
        raise ValueError(f"no events with source={_SOURCE_LABEL.get(source, source)!r}")
    if pick == "first":
        return collected_first
    if pick == "random":
        return int(rng.choice(np.asarray(random_choices, dtype=np.int64)))
    raise ValueError(f"unsupported pick={pick!r}")


def source_name(code: int) -> str:
    return _SOURCE_LABEL.get(int(code), f"source={int(code)}")


def format_event_title(event: dict[str, np.ndarray], index: int) -> str:
    src = source_name(int(event["source"]))
    p0 = float(event["p0"])
    step = int(event["step"])
    parts = [f"event {index}", f"P={p0:.3f}", f"source={src}", f"step={step}"]
    gamma = float(event["gamma_rf"])
    burn = int(event["burn_steps"])
    if np.isfinite(gamma):
        parts.append(f"γ_rf={gamma:.3g}")
    if burn >= 0:
        parts.append(f"burn_steps={burn}")
    center = int(event["center_bin"])
    if center >= 0:
        parts.append(f"center_bin={center}")
    return "  |  ".join(parts)


def plot_spectrum_event(
    event: dict[str, np.ndarray],
    *,
    index: int,
    frequency: np.ndarray | None = None,
    output: Path | None = None,
    show: bool = False,
) -> Path:
    ps = np.asarray(event["ps"], dtype=float)
    iplus = np.asarray(event["iplus"], dtype=float)
    iminus = np.asarray(event["iminus"], dtype=float)
    nb = int(ps.size)
    f = _frequency_axis(nb) if frequency is None else np.asarray(frequency, dtype=float)
    if f.size != nb:
        f = _frequency_axis(nb)

    q = iplus - iminus
    nonzero_bins = int(np.count_nonzero(ps))
    peak_bin = int(np.argmax(np.abs(ps)))

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True, layout="constrained")
    ax0, ax1 = axes

    ax0.plot(f, iplus, color="tab:red", lw=1.3, label=r"$I_+$")
    ax0.plot(f, iminus, color="tab:blue", lw=1.3, label=r"$I_-$")
    ax0.plot(f, ps, color="black", lw=1.0, ls="--", alpha=0.85, label=r"$P_s$")
    ax0.axhline(0.0, color="black", ls=":", alpha=0.35, lw=0.8)
    ax0.set_ylabel("intensity")
    ax0.set_title(format_event_title(event, index))
    ax0.legend(fontsize=9, loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1.plot(f, q, color="tab:purple", lw=1.3, label=r"$Q = I_+ - I_-$")
    ax1.axhline(0.0, color="black", ls=":", alpha=0.35, lw=0.8)
    ax1.axvline(float(f[peak_bin]), color="green", ls=":", alpha=0.6, label=f"peak |Ps| bin {peak_bin}")
    ax1.set_xlabel("R")
    ax1.set_ylabel(r"$Q$")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        f"spectrum_train row  |  {nb} bins  |  {nonzero_bins} nonzero Ps bins",
        fontsize=11,
    )

    out = Path(output) if output is not None else PLOT_DIR / f"spectrum_event_{index:06d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved {out}", flush=True)
    print(
        f"  peak |Ps|={abs(ps[peak_bin]):.4g} at R={f[peak_bin]:.3f} (bin {peak_bin}); "
        f"Ps sum={float(np.sum(ps)):.4g}",
        flush=True,
    )
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot one spectrum_train event (full 500-bin row)")
    p.add_argument(
        "--input",
        type=Path,
        default=SPECTRUM_TRAIN_NPZ,
        help="spectrum_train.npz or directory with NPZ / manifest shards",
    )
    p.add_argument("--index", type=int, default=None, help="Global event row index (supports negative indices)")
    p.add_argument(
        "--source",
        choices=sorted(_SOURCE_BY_NAME),
        default=None,
        help="With --pick, choose an event from this source subset",
    )
    p.add_argument(
        "--pick",
        choices=("first", "random"),
        default=None,
        help="Pick by source instead of --index (use with --source)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for --pick random")
    p.add_argument("--output", type=Path, default=None, help="Output PNG path")
    p.add_argument("--show", action="store_true", help="Show the plot interactively")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.index is None and args.pick is None:
        args.index = 0
    if args.pick is not None and args.source is None:
        raise SystemExit("--pick requires --source")
    if args.index is not None and args.pick is not None:
        raise SystemExit("Use either --index or --source/--pick, not both")

    if args.pick is not None:
        index = find_event_index(
            args.input,
            source=_SOURCE_BY_NAME[args.source],
            pick=str(args.pick),
            seed=args.seed,
        )
    else:
        index = int(args.index)

    event, meta, n_samples = load_spectrum_event(args.input, index)
    num_bins = int(meta.get("num_bins", event["ps"].size or NUM_BINS))
    frequency = _frequency_axis(num_bins)

    print(f"Loaded event {index}/{n_samples - 1} from {args.input}", flush=True)
    plot_spectrum_event(
        event,
        index=index,
        frequency=frequency,
        output=args.output,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
