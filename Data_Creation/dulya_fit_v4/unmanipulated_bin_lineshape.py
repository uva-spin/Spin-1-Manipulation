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

import _bootstrap  # noqa: F401
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
)

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
    expected_p = polarization_grid(float(p_min), float(p_max), float(p_step))
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
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(np.asarray(p_values).size)
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
        "fields": "unmanipulated GenerateDulyaLineshape samples at this bin",
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
            source=np.full(n, int(SOURCE_UNMANIP), dtype=np.uint8),
            ps=np.asarray(ps, dtype=float),
            iplus=np.asarray(iplus, dtype=float),
            iminus=np.asarray(iminus, dtype=float),
            amp=np.asarray(amp, dtype=float),
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

    p_values = polarization_grid(p_min, p_max, p_step)
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
