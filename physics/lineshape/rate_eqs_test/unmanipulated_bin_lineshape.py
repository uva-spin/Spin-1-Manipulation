"""
Generate unmanipulated (unburned) per-bin lineshape training NPZs.

For each initial vector polarization P in [P_MIN, P_MAX] with step P_STEP,
calls ``GenerateVectorLineshape`` once and records (ps, iplus, iminus, amp) at
every spectral bin. Writes one NPZ per bin (same field layout as combined
train files, with ``source=2``).

Usage:
  python unmanipulated_bin_lineshape.py
  python unmanipulated_bin_lineshape.py --output-dir unmanip_train --p-step 0.005
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape

NUM_BINS = 500
P_MIN = -0.70
P_MAX = 0.70
P_STEP = 0.005
# Match combined_train source codes: 0=ssrf, 1=afp, 2=unmanipulated.
SOURCE_UNMANIP = 2

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "unmanip_train"


def polarization_grid(p_min: float, p_max: float, p_step: float) -> np.ndarray:
    return np.arange(float(p_min), float(p_max) + 1e-12, float(p_step), dtype=float)


def unmanip_bin_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"unmanip_bin_{int(bin_idx):04d}.npz"


def generate_unmanipulated_cube(
    *,
    num_bins: int = NUM_BINS,
    p_min: float = P_MIN,
    p_max: float = P_MAX,
    p_step: float = P_STEP,
) -> dict[str, np.ndarray]:
    """
    Build (n_p, n_bins) cubes of unmanipulated intensities.

    Returns p_values plus ps/iplus/iminus/amp with shape (n_p, n_bins).
    """
    p_values = polarization_grid(p_min, p_max, p_step)
    f = np.linspace(-3.0, 3.0, int(num_bins))
    n_p = int(p_values.size)
    n_bins = int(num_bins)

    ps = np.zeros((n_p, n_bins), dtype=float)
    iplus = np.zeros((n_p, n_bins), dtype=float)
    iminus = np.zeros((n_p, n_bins), dtype=float)

    for j, p0 in enumerate(p_values):
        if (j + 1) % 50 == 0 or j == 0 or j == n_p - 1:
            print(f"  GenerateVectorLineshape P={p0:+.3f} ({j + 1}/{n_p})", flush=True)
        signal, ip, im = GenerateVectorLineshape(float(p0), f)
        iplus[j] = np.asarray(ip, dtype=float)
        iminus[j] = np.asarray(im, dtype=float)
        ps[j] = np.asarray(signal, dtype=float)

    return {
        "p_values": p_values,
        "ps": ps,
        "iplus": iplus,
        "iminus": iminus,
        "amp": np.abs(ps),
        "R": f,
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
        "dataset": "unmanip_bin",
        "fields": "unmanipulated GenerateVectorLineshape samples at this bin",
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


def write_all_bins(
    output_dir: Path,
    *,
    num_bins: int = NUM_BINS,
    p_min: float = P_MIN,
    p_max: float = P_MAX,
    p_step: float = P_STEP,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating unmanipulated lineshapes: "
        f"P=[{p_min},{p_max}] step={p_step}  bins={num_bins}",
        flush=True,
    )
    cube = generate_unmanipulated_cube(
        num_bins=num_bins,
        p_min=p_min,
        p_max=p_max,
        p_step=p_step,
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
        "dataset": "unmanip_bin",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate unmanipulated per-bin lineshape NPZs via GenerateVectorLineshape"
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--p-min", type=float, default=P_MIN)
    p.add_argument("--p-max", type=float, default=P_MAX)
    p.add_argument("--p-step", type=float, default=P_STEP)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = write_all_bins(
        args.output_dir,
        num_bins=args.num_bins,
        p_min=args.p_min,
        p_max=args.p_max,
        p_step=args.p_step,
    )
    print(
        f"Wrote {result['n_bins']} files ({result['n_p']} P values each) -> "
        f"{result['output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
