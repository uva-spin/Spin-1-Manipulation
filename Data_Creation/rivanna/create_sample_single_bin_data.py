"""
Create sample ssRF + unmanipulated data for ml/single_bin.py P/Q training and testing.

Outputs under ``<output>/``:

  combined_train_all/train_bin_XXXX.npz
      Per-bin training rows (p0, ps, P, Q, …) from ssRF shards + unmanipulated bins.

  test_events/*.npz
      Full-spectrum evaluation events with ground-truth ``P_bins`` / ``Q_bins`` at
      every spectral bin (for holdout polarizations not used in shard generation).

Examples (from this directory):
  python create_sample_single_bin_data.py --quick
  python create_sample_single_bin_data.py --output sample_single_bin
  python create_sample_single_bin_data.py --tiny --quick
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from bin_paths import ssrf_shard_path
from bin_setup import generate_unmanipulated_cube, get_shape_params
from burn_selection import is_manipulation_shard_bin
from combine_all_train import combine_all, combined_bin_path
from common import (
    BURN_BIN_CHOICES,
    F_MAX,
    F_MIN,
    NUM_BINS,
    RF_MODE_PHYSICAL_VOIGT,
    SOURCE_SSRF,
    SOURCE_UNMANIP,
)
from pq_calibration import calibrated_pq_spectrum, load_pq_calibration
from shard_store import save_ssrf_shard
from ssrf_bin_traj import run_one_bin as run_ssrf_bin
from ssrf_bin_traj import run_one_polarization as run_ssrf_event
from ssrf_bin_traj import run_unmanipulated_polarization
from unmanipulated_bin_lineshape import save_unmanip_bin, unmanip_bin_path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sample_single_bin"

TRAIN_P_VALUES = (0.30, 0.40, 0.50, 0.60)
TEST_P_VALUES = (0.45, 0.55)
QUICK_TRAIN_P = (0.35, 0.45, 0.55)
QUICK_TEST_P = (0.40, 0.50)
# Spread across burn window: negative-Q wing, near R≈0 (not bin 250), positive-Q wing.
QUICK_SSRF_BINS = (172, 193, 195)
DEFAULT_SSRF_BINS = (206, 249, 294, 326, 340)
TINY_SSRF_BINS = (13, 19, 21)


def filter_ssrf_bins_for_train_p(
    ssrf_bins: tuple[int, ...],
    train_p_values: np.ndarray | tuple[float, ...],
    *,
    num_bins: int = NUM_BINS,
    require_all_p: bool = True,
) -> tuple[int, ...]:
    """Keep burn-window manipulation centers valid for shard generation."""
    del train_p_values, require_all_p
    kept: list[int] = []
    dropped: list[int] = []
    for bin_idx in ssrf_bins:
        bi = int(bin_idx)
        if is_manipulation_shard_bin(bi):
            kept.append(bi)
        else:
            dropped.append(bi)
    if dropped:
        print(
            f"  dropping ssRF bins outside allowed manipulation centers: {dropped}",
            flush=True,
        )
    return tuple(kept)


def _scale_ssrf_bins(ssrf_bins: tuple[int, ...], num_bins: int) -> tuple[int, ...]:
    """Map full-grid burn centers onto a smaller spectral grid."""
    nb = int(num_bins)
    bins = tuple(int(b) for b in ssrf_bins)
    if all(0 <= b < nb for b in bins):
        return bins
    if nb == int(NUM_BINS):
        return bins
    scaled = {
        int(round(float(b) / float(NUM_BINS - 1) * float(nb - 1)))
        for b in bins
    }
    scaled = {min(max(b, 1), nb - 2) for b in scaled}
    return tuple(sorted(scaled))


def _frequency_axis(num_bins: int) -> np.ndarray:
    return np.linspace(float(F_MIN), float(F_MAX), int(num_bins), dtype=np.float32)


def _save_test_event(
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
    step: int,
    extra_meta: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cal = load_pq_calibration(num_bins=int(ps.size))
    p_bins, q_bins = calibrated_pq_spectrum(
        ps, iplus, iminus, float(p0), calibration=cal, num_bins=int(ps.size)
    )
    p_total = float(np.mean(p_bins))
    q_total = float(np.mean(q_bins))
    meta = {
        "dataset": "single_bin_test_event_v1",
        "manipulation_mode": str(manipulation_mode),
        "source_code": int(source),
        "p0": float(p0),
        "center_bin": int(center_bin),
        "step": int(step),
        "num_bins": int(ps.size),
        "fields": (
            "P_bins,Q_bins = CC-calibrated per-bin targets; "
            "P_total,Q_total = mean(P_bins/Q_bins) integrated lineshape P/Q"
        ),
        "pq_calibrated": True,
        "P_total": p_total,
        "Q_total": q_total,
    }
    if extra_meta:
        meta.update(extra_meta)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp,
            meta_json=np.asarray(json.dumps(meta)),
            ps=np.asarray(ps, dtype=np.float32),
            iplus=np.asarray(iplus, dtype=np.float32),
            iminus=np.asarray(iminus, dtype=np.float32),
            frequency=np.asarray(frequency, dtype=np.float32),
            P_bins=np.asarray(p_bins, dtype=np.float32),
            Q_bins=np.asarray(q_bins, dtype=np.float32),
            P_total=np.asarray(p_total, dtype=np.float32),
            Q_total=np.asarray(q_total, dtype=np.float32),
            p0=np.asarray(float(p0), dtype=np.float32),
            source=np.asarray(int(source), dtype=np.uint8),
            center_bin=np.asarray(int(center_bin), dtype=np.int32),
            step=np.asarray(int(step), dtype=np.int32),
        )
        tmp.replace(path)
    except Exception:
        if tmp.is_file():
            tmp.unlink(missing_ok=True)
        raise
    return path


def _write_ssrf_shard(
    ssrf_dir: Path,
    *,
    bin_idx: int,
    p_values: np.ndarray,
    gamma_rf: float,
    burn_steps: int,
    num_bins: int,
) -> None:
    result = run_ssrf_bin(
        int(bin_idx),
        p_values=np.asarray(p_values, dtype=float),
        gamma_values=np.array([float(gamma_rf)], dtype=float),
        steps_values=np.array([int(burn_steps)], dtype=np.int32),
        num_bins=int(num_bins),
        rf_mode=RF_MODE_PHYSICAL_VOIGT,
    )
    if not np.any(~np.asarray(result["skipped"], dtype=bool)):
        raise RuntimeError(
            f"ssRF shard bin {bin_idx}: all polarizations skipped "
            "(check ps0 / burn parameters)"
        )
    save_ssrf_shard(result, ssrf_shard_path(ssrf_dir, int(bin_idx)))


def _write_unmanip_dir(
    unmanip_dir: Path,
    *,
    p_values: np.ndarray,
    num_bins: int,
) -> None:
    unmanip_dir = Path(unmanip_dir)
    unmanip_dir.mkdir(parents=True, exist_ok=True)
    shape = get_shape_params()
    p_arr = np.asarray(p_values, dtype=float)
    p_min = float(p_arr.min())
    p_max = float(p_arr.max())
    p_step = float(p_arr[1] - p_arr[0]) if p_arr.size > 1 else 0.1
    cube = generate_unmanipulated_cube(
        num_bins=int(num_bins),
        p_min=p_min,
        p_max=p_max,
        p_step=p_step,
        shape_params=shape,
    )
    print(f"  writing {int(num_bins)} unmanip_bin_*.npz files...", flush=True)
    pq_calibration = load_pq_calibration(num_bins=int(num_bins))
    for bin_idx in range(int(num_bins)):
        save_unmanip_bin(
            bin_idx,
            p_values=cube["p_values"],
            ps=cube["ps"][:, bin_idx],
            iplus=cube["iplus"][:, bin_idx],
            iminus=cube["iminus"][:, bin_idx],
            amp=cube["amp"][:, bin_idx],
            R=float(cube["R"][bin_idx]),
            path=unmanip_bin_path(unmanip_dir, bin_idx),
            p_min=p_min,
            p_max=p_max,
            p_step=p_step,
            num_bins=int(num_bins),
            shape_params=shape,
            pq_calibration=pq_calibration,
        )
        if (bin_idx + 1) % 50 == 0 or bin_idx + 1 == int(num_bins):
            print(f"  unmanip through bin {bin_idx + 1}/{int(num_bins)}", flush=True)


def _write_test_events(
    test_dir: Path,
    *,
    test_p_values: tuple[float, ...],
    ssrf_bins: tuple[int, ...],
    gamma_rf: float,
    burn_steps: int,
    frequency: np.ndarray,
) -> list[Path]:
    test_dir = Path(test_dir)
    saved: list[Path] = []
    for p0 in test_p_values:
        traj = run_unmanipulated_polarization(float(p0), num_bins=int(frequency.size))
        ps = np.asarray(traj["ps_full"][0], dtype=np.float32)
        iplus = np.asarray(traj["iplus_full"][0], dtype=np.float32)
        iminus = np.asarray(traj["iminus_full"][0], dtype=np.float32)
        event_id = f"unmanip_p{p0:.3f}"
        saved.append(
            _save_test_event(
                test_dir / f"{event_id}.npz",
                ps=ps,
                iplus=iplus,
                iminus=iminus,
                frequency=frequency,
                p0=float(p0),
                source=SOURCE_UNMANIP,
                manipulation_mode="unmanip",
                center_bin=int(traj.get("center_bin", frequency.size // 2)),
                step=0,
            )
        )
        print(f"  test event {event_id}", flush=True)

        for bin_idx in ssrf_bins:
            traj = run_ssrf_event(
                int(bin_idx),
                float(p0),
                gamma_rf=float(gamma_rf),
                n_steps=int(burn_steps),
                capture_spectrum=True,
                num_bins=int(frequency.size),
            )
            if bool(traj.get("skipped", False)):
                print(f"  skip ssRF test p0={p0:.3f} bin={bin_idx}", flush=True)
                continue
            iplus = np.asarray(traj["ip_spectrum"], dtype=np.float32)
            iminus = np.asarray(traj["im_spectrum"], dtype=np.float32)
            ps = iplus + iminus
            event_id = f"ssrf_p{p0:.3f}_g{gamma_rf:.2g}_s{burn_steps}_bin{bin_idx}"
            saved.append(
                _save_test_event(
                    test_dir / f"{event_id}.npz",
                    ps=ps,
                    iplus=iplus,
                    iminus=iminus,
                    frequency=frequency,
                    p0=float(p0),
                    source=SOURCE_SSRF,
                    manipulation_mode="ssrf",
                    center_bin=int(bin_idx),
                    step=int(burn_steps),
                    extra_meta={
                        "gamma_rf": float(gamma_rf),
                        "burn_steps": int(burn_steps),
                    },
                )
            )
            print(f"  test event {event_id}", flush=True)
    return saved


def create_sample_single_bin_data(
    output_dir: Path,
    *,
    train_p_values: tuple[float, ...],
    test_p_values: tuple[float, ...],
    ssrf_bins: tuple[int, ...],
    gamma_rf: float = 50.0,
    burn_steps: int = 50,
    num_bins: int = NUM_BINS,
) -> dict:
    output_dir = Path(output_dir)
    work = output_dir / "_work"
    ssrf_dir = work / "ssrf_shards"
    afp_dir = work / "afp_shards"
    unmanip_dir = work / "unmanip_train"
    combined_dir = output_dir / "combined_train_all"
    test_dir = output_dir / "test_events"
    for d in (ssrf_dir, afp_dir, unmanip_dir, combined_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_p = np.asarray(train_p_values, dtype=float)
    frequency = _frequency_axis(num_bins)
    if int(num_bins) != int(NUM_BINS):
        ssrf_bins = _scale_ssrf_bins(ssrf_bins, int(num_bins))
    ssrf_bins = filter_ssrf_bins_for_train_p(
        ssrf_bins, train_p, num_bins=int(num_bins), require_all_p=True
    )
    if not ssrf_bins:
        raise ValueError(
            "No ssRF burn bins remain after burn-window filtering; "
            "adjust --ssrf-bins"
        )

    print("Writing ssRF training shards...", flush=True)
    for bin_idx in ssrf_bins:
        print(f"  ssrf shard bin {bin_idx}", flush=True)
        _write_ssrf_shard(
            ssrf_dir,
            bin_idx=int(bin_idx),
            p_values=train_p,
            gamma_rf=float(gamma_rf),
            burn_steps=int(burn_steps),
            num_bins=int(num_bins),
        )

    print("Writing unmanipulated training bins...", flush=True)
    _write_unmanip_dir(unmanip_dir, p_values=train_p, num_bins=int(num_bins))

    print("Combining -> combined_train_all...", flush=True)
    combine_result = combine_all(
        ssrf_dir,
        afp_dir,
        unmanip_dir,
        combined_dir,
        num_bins=int(num_bins),
        p_min=float(train_p.min()),
        p_max=float(train_p.max()),
        p_step=float(train_p[1] - train_p[0]) if train_p.size > 1 else 0.1,
        strict=False,
        include_unmanip=True,
    )

    n_train_files = sum(
        1 for b in range(int(num_bins)) if combined_bin_path(combined_dir, b).is_file()
    )
    sample_train = combined_bin_path(combined_dir, int(ssrf_bins[len(ssrf_bins) // 2]))
    with np.load(sample_train, allow_pickle=False) as data:
        assert "P" in data.files and "Q" in data.files

    print("Writing holdout test events (full spectra + P_bins/Q_bins)...", flush=True)
    test_paths = _write_test_events(
        test_dir,
        test_p_values=test_p_values,
        ssrf_bins=ssrf_bins[: min(2, len(ssrf_bins))],
        gamma_rf=float(gamma_rf),
        burn_steps=int(burn_steps),
        frequency=frequency,
    )

    manifest = {
        "dataset": "sample_single_bin_v1",
        "num_bins": int(num_bins),
        "train_p_values": [float(x) for x in train_p_values],
        "test_p_values": [float(x) for x in test_p_values],
        "ssrf_bins": [int(x) for x in ssrf_bins],
        "gamma_rf": float(gamma_rf),
        "burn_steps": int(burn_steps),
        "combined_train_dir": str(combined_dir),
        "test_events_dir": str(test_dir),
        "n_train_bin_files": int(n_train_files),
        "n_test_events": len(test_paths),
        "combine_stats": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in combine_result.items()
            if k != "samples_per_bin"
        },
        "usage": {
            "train": (
                "python ml/single_bin.py --bin-idx B --data-dir "
                f"{combined_dir} --output-dir single_bin_models"
            ),
            "combine": (
                "python ml/combine_single_bin_models.py --model-dir single_bin_models "
                f"--output single_bin_models/combined_bin_model.pth --num-bins {int(num_bins)}"
            ),
            "eval": (
                "python ml/test_single_bin_pq.py --sample-dir "
                + str(output_dir)
                + " --combined-model single_bin_models/combined_bin_model.pth"
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Done -> {output_dir}", flush=True)
    print(f"  train NPZs: {combined_dir} ({n_train_files} bins with data)", flush=True)
    print(f"  test events: {test_dir} ({len(test_paths)} files)", flush=True)
    print(f"  manifest: {manifest_path}", flush=True)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample ssRF + unmanipulated data for single_bin P/Q training/testing"
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller p grid, 3 ssRF bins, single gamma/burn combo",
    )
    p.add_argument(
        "--tiny",
        action="store_true",
        help="Use 32 spectral bins (fast local smoke only)",
    )
    p.add_argument("--gamma-rf", type=float, default=10.0)
    p.add_argument("--burn-steps", type=int, default=500)
    p.add_argument("--train-p", type=float, nargs="+", default=None)
    p.add_argument("--test-p", type=float, nargs="+", default=None)
    p.add_argument("--ssrf-bins", type=int, nargs="+", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    num_bins = 32 if args.tiny else NUM_BINS

    if args.quick:
        train_p = tuple(args.train_p or QUICK_TRAIN_P)
        test_p = tuple(args.test_p or QUICK_TEST_P)
        ssrf_bins = tuple(args.ssrf_bins or (TINY_SSRF_BINS if args.tiny else QUICK_SSRF_BINS))
    else:
        train_p = tuple(args.train_p or TRAIN_P_VALUES)
        test_p = tuple(args.test_p or TEST_P_VALUES)
        ssrf_bins = tuple(args.ssrf_bins or DEFAULT_SSRF_BINS)
        if args.tiny and args.ssrf_bins is None:
            ssrf_bins = TINY_SSRF_BINS

    overlap = set(train_p) & set(test_p)
    if overlap:
        raise ValueError(
            f"train and test p0 must not overlap for holdout eval; shared: {sorted(overlap)}"
        )

    create_sample_single_bin_data(
        args.output,
        train_p_values=train_p,
        test_p_values=test_p,
        ssrf_bins=ssrf_bins,
        gamma_rf=float(args.gamma_rf),
        burn_steps=int(args.burn_steps),
        num_bins=int(num_bins),
    )


if __name__ == "__main__":
    main()
