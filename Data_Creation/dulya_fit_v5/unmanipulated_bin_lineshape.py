"""
Unmanipulated per-bin lineshape NPZs using the frozen Dulya fit (v2 pipeline tags).

Equilibrium spectra always come from ``GenerateDulyaLineshape``.

Examples (from this directory):
  python unmanipulated_bin_lineshape.py
  python unmanipulated_bin_lineshape.py --bin-idx 172
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from burn_selection import positive_polarization_grid
from bin_setup import (
    LINESHape_MODEL,
    equilibrium_lineshape,
    generate_unmanipulated_cube,
    get_shape_params,
    polarization_grid,
    print_shape_banner,
    resolve_bin_idx,
    shape_meta,
)
from common import (
    F_MAX,
    F_MIN,
    NUM_BINS,
    P_MAX,
    P_MIN,
    P_STEP,
    UNMANIP_TRAIN_DIR,
    intensity_pq,
)
from pq_calibration import calibrated_pq_fields, load_pq_calibration, validate_stored_per_bin_pq

SOURCE_UNMANIP = 2


def unmanip_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"unmanip_bin_{int(bin_idx):04d}.npz"


def verify_unmanip_train_dir(
    unmanip_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    p_min: float = P_MIN,
    p_max: float = P_MAX,
    p_step: float = P_STEP,
) -> dict:
    """Check that per-bin unmanip NPZs cover ``0..num_bins-1`` with a shared P grid."""
    unmanip_dir = Path(unmanip_dir)
    expected_p = positive_polarization_grid(float(p_min), float(p_max), float(p_step))
    missing: list[int] = []
    p_mismatch: list[int] = []
    p_values: np.ndarray | None = None

    for bin_idx in range(int(num_bins)):
        path = unmanip_bin_path(unmanip_dir, bin_idx)
        if not path.is_file():
            missing.append(int(bin_idx))
            continue
        with np.load(path, allow_pickle=False) as data:
            p0 = np.asarray(data["p0"], dtype=float)
        if p_values is None:
            p_values = p0
            if p0.shape != expected_p.shape or not np.allclose(p0, expected_p, atol=1e-5):
                p_mismatch.append(int(bin_idx))
        elif p0.shape != p_values.shape or not np.allclose(p0, p_values, atol=1e-5):
            p_mismatch.append(int(bin_idx))

    ok = not missing and not p_mismatch and p_values is not None
    return {
        "ok": bool(ok),
        "unmanip_dir": str(unmanip_dir),
        "num_bins": int(num_bins),
        "n_present": int(num_bins) - len(missing),
        "n_missing": len(missing),
        "missing_bins": missing,
        "p_mismatch_bins": p_mismatch,
        "n_p": int(p_values.size) if p_values is not None else 0,
        "p_values": (
            np.asarray(p_values, dtype=float)
            if p_values is not None
            else np.zeros(0, dtype=float)
        ),
    }


def save_unmanip_bin(
    bin_idx: int,
    *,
    p_values: np.ndarray,
    ps: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    amp: np.ndarray,
    R: float,
    path: Path,
    p_min: float,
    p_max: float,
    p_step: float,
    num_bins: int,
    shape_params: dict[str, float],
    pq_calibration: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(np.asarray(p_values).size)
    ps_arr = np.asarray(ps, dtype=float)
    ip_arr = np.asarray(iplus, dtype=float)
    im_arr = np.asarray(iminus, dtype=float)
    _, q_arr = intensity_pq(ip_arr, im_arr)
    row_arrays = {
        "p0": np.asarray(p_values, dtype=float),
        "ps": ps_arr,
        "iplus": ip_arr,
        "iminus": im_arr,
        "q": q_arr,
    }
    pq_calibration = pq_calibration or load_pq_calibration(num_bins=int(num_bins))
    p_true, q_true = calibrated_pq_fields(
        row_arrays,
        num_bins=int(num_bins),
        calibration=pq_calibration,
    )
    validate_stored_per_bin_pq(
        row_arrays["ps"],
        row_arrays["q"],
        row_arrays["p0"],
        p_true,
        q_true,
        calibration=pq_calibration,
    )
    meta = {
        "bin_idx": int(bin_idx),
        "n_samples": n,
        "num_bins": int(num_bins),
        "R": float(R),
        "p_min": float(p_min),
        "p_max": float(p_max),
        "p_step": float(p_step),
        "source": int(SOURCE_UNMANIP),
        "source_codes": {"ssrf": 0, "afp": 1, "unmanipulated": SOURCE_UNMANIP},
        "dataset": "dulya_unmanip_bin_v2",
        "fields": (
            "ps,q=raw I± sums; P,Q=CC-calibrated true polarizations at this bin "
            "(unmanipulated equilibrium)"
        ),
        "pq_calibrated": True,
        "pq_target_scope": "per_bin",
        "pq_cc_scale": "cc_bin",
        "pq_post_correct": True,
        "pq_cc_bin": float(pq_calibration["cc_bin"]),
        "pq_amp": float(pq_calibration["amp"]),
        **shape_meta(shape_params),
    }
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            bin_idx=np.asarray(int(bin_idx), dtype=np.int32),
            p0=np.asarray(p_values, dtype=float),
            step=np.zeros(n, dtype=np.int32),
            center_bin=np.full(n, int(bin_idx), dtype=np.int32),
            is_mirror=np.zeros(n, dtype=bool),
            is_neighbor=np.zeros(n, dtype=bool),
            source=np.full(n, int(SOURCE_UNMANIP), dtype=np.uint8),
            ps=ps_arr,
            iplus=ip_arr,
            iminus=im_arr,
            q=np.asarray(q_arr, dtype=float),
            amp=np.asarray(amp, dtype=float),
            P=np.asarray(p_true, dtype=float),
            Q=np.asarray(q_true, dtype=float),
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def run_one_bin(
    bin_idx: int,
    *,
    output_dir: Path,
    num_bins: int,
    p_min: float,
    p_max: float,
    p_step: float,
    skip_if_exists: bool,
    shape_params: dict[str, float],
) -> Path:
    out = unmanip_bin_path(output_dir, bin_idx)
    if skip_if_exists and out.is_file():
        print(f"Skipping existing {out}", flush=True)
        return out

    p_values = positive_polarization_grid(p_min, p_max, p_step)
    f = np.linspace(float(F_MIN), float(F_MAX), int(num_bins))
    n_p = int(p_values.size)

    ps_col = np.zeros(n_p, dtype=float)
    ip_col = np.zeros(n_p, dtype=float)
    im_col = np.zeros(n_p, dtype=float)

    for j, p0 in enumerate(p_values):
        if (j + 1) % 10 == 0 or j == 0 or j == n_p - 1:
            print(f"  bin {bin_idx}  P={p0:+.3f} ({j + 1}/{n_p})", flush=True)
        signal, ip, im = equilibrium_lineshape(float(p0), f, shape_params)
        ps_col[j] = float(np.asarray(signal, dtype=float)[bin_idx])
        ip_col[j] = float(np.asarray(ip, dtype=float)[bin_idx])
        im_col[j] = float(np.asarray(im, dtype=float)[bin_idx])

    save_unmanip_bin(
        bin_idx,
        p_values=p_values,
        ps=ps_col,
        iplus=ip_col,
        iminus=im_col,
        amp=np.abs(ps_col),
        R=float(f[bin_idx]),
        path=out,
        p_min=p_min,
        p_max=p_max,
        p_step=p_step,
        num_bins=num_bins,
        shape_params=shape_params,
    )
    print(f"Wrote {out}", flush=True)
    return out


def run_all_bins(
    *,
    output_dir: Path,
    num_bins: int,
    p_min: float,
    p_max: float,
    p_step: float,
    shape_params: dict[str, float],
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating Dulya unmanipulated lineshapes (v2): "
        f"P=[{p_min},{p_max}] step={p_step}  bins={num_bins}",
        flush=True,
    )
    cube = generate_unmanipulated_cube(
        num_bins=num_bins,
        p_min=p_min,
        p_max=p_max,
        p_step=p_step,
        shape_params=shape_params,
    )
    p_values = cube["p_values"]
    n_p = int(p_values.size)

    print(f"Writing {num_bins} unmanipulated per-bin NPZs to {output_dir}", flush=True)
    pq_calibration = load_pq_calibration(num_bins=int(num_bins))
    for bin_idx in range(int(num_bins)):
        save_unmanip_bin(
            bin_idx,
            p_values=p_values,
            ps=cube["ps"][:, bin_idx],
            iplus=cube["iplus"][:, bin_idx],
            iminus=cube["iminus"][:, bin_idx],
            amp=cube["amp"][:, bin_idx],
            R=float(cube["R"][bin_idx]),
            path=unmanip_bin_path(output_dir, bin_idx),
            p_min=p_min,
            p_max=p_max,
            p_step=p_step,
            num_bins=num_bins,
            shape_params=shape_params,
            pq_calibration=pq_calibration,
        )
        if (bin_idx + 1) % 50 == 0 or bin_idx == int(num_bins) - 1:
            print(f"  wrote through bin {bin_idx}", flush=True)

    return {
        "output_dir": str(output_dir),
        "n_bins": int(num_bins),
        "n_p": n_p,
        "n_samples_total": int(num_bins) * n_p,
        "p_min": float(p_min),
        "p_max": float(p_max),
        "p_step": float(p_step),
        "dataset": "dulya_unmanip_bin_v2",
        "lineshape_model": LINESHape_MODEL,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dulya-fit unmanipulated per-bin lineshape generator (v2)"
    )
    p.add_argument(
        "--bin-idx",
        type=int,
        default=None,
        help="Single bin (or SLURM_ARRAY_TASK_ID); omit to write all bins",
    )
    p.add_argument("--output-dir", type=Path, default=UNMANIP_TRAIN_DIR)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--p-min", type=float, default=P_MIN)
    p.add_argument("--p-max", type=float, default=P_MAX)
    p.add_argument("--p-step", type=float, default=P_STEP)
    p.add_argument("--skip-if-exists", action="store_true")
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing unmanip_bin_XXXX.npz coverage/P-grid and exit",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.verify_only:
        report = verify_unmanip_train_dir(
            args.output_dir,
            num_bins=int(args.num_bins),
            p_min=float(args.p_min),
            p_max=float(args.p_max),
            p_step=float(args.p_step),
        )
        print(
            f"unmanip verify: ok={report['ok']}  present={report['n_present']}/"
            f"{report['num_bins']}  n_p={report['n_p']}  "
            f"missing={report['n_missing']}  p_mismatch={len(report['p_mismatch_bins'])}",
            flush=True,
        )
        if not report["ok"]:
            raise SystemExit(1)
        return

    shape = get_shape_params()
    print_shape_banner(shape, num_bins=int(args.num_bins))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bin_idx = resolve_bin_idx(args.bin_idx, num_bins=int(args.num_bins))

    if bin_idx is not None:
        run_one_bin(
            bin_idx,
            output_dir=args.output_dir,
            num_bins=int(args.num_bins),
            p_min=float(args.p_min),
            p_max=float(args.p_max),
            p_step=float(args.p_step),
            skip_if_exists=bool(args.skip_if_exists),
            shape_params=shape,
        )
        return

    result = run_all_bins(
        output_dir=args.output_dir,
        num_bins=int(args.num_bins),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        p_step=float(args.p_step),
        shape_params=shape,
    )
    print(
        f"Wrote {result['n_bins']} files ({result['n_p']} P values each) -> "
        f"{result['output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
