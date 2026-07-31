"""
Per-bin Dulya-fit v2 MC data (unmanipulated / ssRF / AFP).

Thin dispatcher — prefer the standalone workers for parallel runs:
  ssrf_bin_traj.py, afp_bin_traj.py, unmanipulated_bin_lineshape.py

Examples (from this directory):
  python generate_bins.py --mode unmanipulated
  python generate_bins.py --mode ssrf --bin-idx 172
  python generate_bins.py --mode afp --bin-idx 172
  python generate_bins.py --mode ssrf --organize
  python generate_bins.py --mode all --smoke --bin-idx 208

SLURM (submit from this directory):
  sbatch ssrf_traj_array.slurm
  sbatch afp_traj_array.slurm
  sbatch unmanipulated_bin_array.slurm
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import _bootstrap  # noqa: F401
from afp_bin_traj import main as afp_main
from common import (
    AFP_N_RELAX,
    AFP_SHARD_DIR,
    AFP_TRAIN_DIR,
    DATA_DIR,
    NUM_BINS,
    P_MAX,
    P_MIN,
    P_STEP,
    RF_MODE,
    RF_MODE_PHYSICAL_VOIGT,
    RF_MODE_SINGLE_BIN,
    SSRF_SHARD_DIR,
    SSRF_TRAIN_DIR,
    UNMANIP_TRAIN_DIR,
)
from ssrf_bin_traj import main as ssrf_main
from unmanipulated_bin_lineshape import main as unmanip_main

ModeName = Literal["unmanipulated", "ssrf", "afp", "all"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-bin Dulya-fit v2 MC data dispatcher (see mode-specific scripts)"
    )
    p.add_argument(
        "--mode",
        choices=("unmanipulated", "ssrf", "afp", "all"),
        default="all",
    )
    p.add_argument("--bin-idx", type=int, default=None)
    p.add_argument("--organize", action="store_true")
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--p-min", type=float, default=P_MIN)
    p.add_argument("--p-max", type=float, default=P_MAX)
    p.add_argument("--p-step", type=float, default=P_STEP)
    p.add_argument("--ssrf-shard-dir", type=Path, default=SSRF_SHARD_DIR)
    p.add_argument("--ssrf-train-dir", type=Path, default=SSRF_TRAIN_DIR)
    p.add_argument("--afp-shard-dir", type=Path, default=AFP_SHARD_DIR)
    p.add_argument("--afp-train-dir", type=Path, default=AFP_TRAIN_DIR)
    p.add_argument("--unmanip-dir", type=Path, default=UNMANIP_TRAIN_DIR)
    p.add_argument("--skip-if-exists", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Coarse P grid + short AFP/ssRF for a quick per-bin smoke test",
    )
    p.add_argument(
        "--rf-mode",
        choices=(RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN),
        default=RF_MODE,
        help="ssRF profile mode for --mode ssrf/all",
    )
    return p.parse_args()


def _argv_for_mode(args: argparse.Namespace, mode: str) -> list[str]:
    argv = [
        "--num-bins",
        str(args.num_bins),
        "--p-min",
        str(args.p_min),
        "--p-max",
        str(args.p_max),
        "--p-step",
        str(args.p_step),
    ]
    p_step = float(args.p_step)
    if args.smoke:
        p_step = max(p_step, 0.3)
        argv.extend(["--p-step", str(p_step)])

    if mode == "unmanipulated":
        argv.extend(["--output-dir", str(args.unmanip_dir)])
        if args.bin_idx is not None:
            argv.extend(["--bin-idx", str(args.bin_idx)])
        if args.skip_if_exists:
            argv.append("--skip-if-exists")
        return argv

    if mode == "ssrf":
        if args.organize:
            argv.extend(
                [
                    "--organize",
                    "--shard-dir",
                    str(args.ssrf_shard_dir),
                    "--output-dir",
                    str(args.ssrf_train_dir),
                ]
            )
            if args.strict:
                argv.append("--strict")
            return argv
        if args.bin_idx is None and args.smoke:
            argv.extend(["--bin-idx", "208"])
        elif args.bin_idx is not None:
            argv.extend(["--bin-idx", str(args.bin_idx)])
        argv.extend(
            [
                "--shard-dir",
                str(args.ssrf_shard_dir),
                "--rf-mode",
                str(args.rf_mode),
            ]
        )
        if args.smoke:
            # Tiny discrete grids for a quick smoke shard.
            argv.extend(
                [
                    "--gamma-min",
                    "1.0",
                    "--gamma-max",
                    "2.0",
                    "--gamma-step",
                    "1.0",
                    "--steps-min",
                    "20",
                    "--steps-max",
                    "40",
                    "--steps-step",
                    "20",
                ]
            )
        if args.skip_if_exists:
            argv.append("--skip-if-exists")
        return argv

    if mode == "afp":
        if args.organize:
            argv.extend(
                [
                    "--organize",
                    "--shard-dir",
                    str(args.afp_shard_dir),
                    "--output-dir",
                    str(args.afp_train_dir),
                ]
            )
            if args.strict:
                argv.append("--strict")
            return argv
        if args.bin_idx is None and args.smoke:
            argv.extend(["--bin-idx", "208"])
        elif args.bin_idx is not None:
            argv.extend(["--bin-idx", str(args.bin_idx)])
        argv.extend(
            [
                "--shard-dir",
                str(args.afp_shard_dir),
                "--n-relax",
                str(min(int(AFP_N_RELAX), 50) if args.smoke else int(AFP_N_RELAX)),
            ]
        )
        if args.skip_if_exists:
            argv.append("--skip-if-exists")
        return argv

    raise ValueError(f"unknown mode {mode!r}")


def main() -> None:
    args = _parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "all":
        if args.organize:
            ssrf_main(_argv_for_mode(args, "ssrf"))
            afp_main(_argv_for_mode(args, "afp"))
            return
        modes = ("unmanipulated", "ssrf", "afp")
        if args.bin_idx is None and not args.smoke:
            modes = ("unmanipulated",)
            print(
                "Note: --mode all without --bin-idx runs unmanipulated only; "
                "pass --bin-idx for ssRF/AFP shards.",
                flush=True,
            )
    else:
        modes = (args.mode,)

    for mode in modes:
        argv = _argv_for_mode(args, mode)
        if mode == "unmanipulated":
            unmanip_main(argv)
        elif mode == "ssrf":
            if args.bin_idx is None and not args.smoke and not args.organize:
                raise SystemExit(
                    "Provide --bin-idx for ssRF, or pass --organize / --smoke"
                )
            ssrf_main(argv)
        elif mode == "afp":
            if args.bin_idx is None and not args.smoke and not args.organize:
                raise SystemExit(
                    "Provide --bin-idx for AFP, or pass --organize / --smoke"
                )
            afp_main(argv)


if __name__ == "__main__":
    main()
