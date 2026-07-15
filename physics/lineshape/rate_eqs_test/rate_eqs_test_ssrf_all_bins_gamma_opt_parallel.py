"""
Parallel (independent) gamma-opt burns for SLURM array jobs.

Each task burns one R-bin starting from the same unburned full spectrum
(GenerateVectorLineshape), using the same burn-down physics as
``rate_eqs_test_ssrf_all_bins_gamma_opt`` (fixed ``DT``, min ``gamma_rf`` to
null local Q; RF + sameθ + neighbors in the simulation; commit only
burn/mirror bins).

Unlike the sequential optimizer, bins do not see each other's burns.

Usage (single bin, local or array task):
  python rate_eqs_test_ssrf_all_bins_gamma_opt_parallel.py --bin-idx 172
  # Or rely on SLURM_ARRAY_TASK_ID when submitted as an array job.

Usage (combine shards after array completes):
  python rate_eqs_test_ssrf_all_bins_gamma_opt_parallel.py --combine \\
      --shard-dir gamma_opt_shards --output-dir gamma_opt_combined

SLURM (submit from repo root):
  ARRAY_JOB_ID=$(sbatch --parsable \\
      physics/lineshape/rate_eqs_test/gamma_opt_array.slurm)
  sbatch --dependency=afterok:${ARRAY_JOB_ID} \\
      physics/lineshape/rate_eqs_test/gamma_opt_combine.slurm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Headless cluster nodes: set before importing matplotlib via gopt.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
import physics.lineshape.rate_eqs_test.rate_eqs_test_ssrf_all_bins_gamma_opt as gopt

DEFAULT_SHARD_DIR = Path(__file__).resolve().parent / "gamma_opt_shards"
DEFAULT_COMBINED_DIR = Path(__file__).resolve().parent / "gamma_opt_combined"


def shard_path(output_dir: Path, bin_idx: int) -> Path:
    return Path(output_dir) / f"gamma_opt_bin_{int(bin_idx):04d}.npz"


def run_one_bin(
    bin_idx: int,
    *,
    polarization: float = gopt.P,
    num_bins: int = gopt.NUM_BINS,
    dt: float = gopt.DT,
    n_steps: int = gopt.N_STEPS,
    gamma_hi: float = gopt.GAMMA_MAX,
    n_bisect: int = gopt.N_BISECT,
) -> dict:
    """
    Find the minimum gamma_rf that nulls |Q(R)| at ``bin_idx`` for fixed ``dt``.

    Starts from the unburned full lineshape. Returns a JSON-serializable
    summary plus arrays for the committed spectrum.
    """
    bin_idx = int(bin_idx)
    if bin_idx < 0 or bin_idx >= int(num_bins):
        raise ValueError(f"bin_idx={bin_idx} out of range for num_bins={num_bins}")

    f = np.linspace(-3.0, 3.0, int(num_bins))
    _, iplus0, iminus0 = GenerateVectorLineshape(float(polarization), f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    mirror_idx = gopt.mirror_bin_idx(int(num_bins), bin_idx)
    q_before = gopt.q_at_r_bin(iplus, iminus, bin_idx)
    q0_total = gopt.q_total(iplus, iminus)
    p0_total = gopt.p_total(iplus, iminus)
    area0 = gopt.lineshape_area(iplus, iminus, f)
    gamma_max = float(min(gamma_hi, gopt.GAMMA_MAX))
    dt_f = float(dt)

    base = {
        "bin_idx": bin_idx,
        "mirror_idx": mirror_idx,
        "R": float(f[bin_idx]),
        "polarization": float(polarization),
        "num_bins": int(num_bins),
        "dt": dt_f,
        "gamma_max": gamma_max,
        "n_steps_max": int(n_steps),
        "q_before": float(q_before),
        "q_total_unburned": float(q0_total),
        "p_total_unburned": float(p0_total),
        "area_unburned": float(area0),
        "f": f,
        "iplus_unburned": iplus,
        "iminus_unburned": iminus,
    }

    if q_before >= 0.0:
        return {
            **base,
            "gamma_rf": 0.0,
            "n_steps": 0,
            "t_burn": 0.0,
            "q_after": float(q_before),
            "q_gain": 0.0,
            "q_total_after": float(q0_total),
            "q_total_gain": 0.0,
            "p_total_after": float(p0_total),
            "area_after": float(area0),
            "area_loss": 0.0,
            "skipped": True,
            "iplus": iplus,
            "iminus": iminus,
        }

    trial = gopt.find_gamma_to_null_q_r(
        iplus,
        iminus,
        bin_idx,
        f=f,
        polarization=float(polarization),
        dt=dt_f,
        n_steps=int(n_steps),
        gamma_hi=float(gamma_hi),
        n_bisect=int(n_bisect),
        gamma_guess=None,
    )
    if trial is None:
        return {
            **base,
            "gamma_rf": 0.0,
            "n_steps": 0,
            "t_burn": 0.0,
            "q_after": float(q_before),
            "q_gain": 0.0,
            "q_total_after": float(q0_total),
            "q_total_gain": 0.0,
            "p_total_after": float(p0_total),
            "area_after": float(area0),
            "area_loss": 0.0,
            "skipped": True,
            "iplus": iplus,
            "iminus": iminus,
        }

    iplus_c = np.asarray(trial["iplus"], dtype=float)
    iminus_c = np.asarray(trial["iminus"], dtype=float)
    return {
        **base,
        "gamma_rf": float(trial["gamma_rf"]),
        "n_steps": int(trial["n_steps"]),
        "t_burn": float(trial["t_burn"]),
        "q_after": float(trial["q_after"]),
        "q_gain": float(trial["q_gain"]),
        "q_total_after": float(trial["q_total_after"]),
        "q_total_gain": float(trial["q_total_gain"]),
        "p_total_after": gopt.p_total(iplus_c, iminus_c),
        "area_after": float(trial["area_after"]),
        "area_loss": float(trial["area_loss"]),
        "skipped": False,
        "iplus": iplus_c,
        "iminus": iminus_c,
    }


def save_shard(result: dict, path: Path) -> None:
    """Write shard atomically (temp file + replace) so kills leave no partial .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "bin_idx": int(result["bin_idx"]),
        "mirror_idx": int(result["mirror_idx"]),
        "R": float(result["R"]),
        "polarization": float(result["polarization"]),
        "num_bins": int(result["num_bins"]),
        "dt": float(result["dt"]),
        "gamma_max": float(result["gamma_max"]),
        "n_steps_max": int(result["n_steps_max"]),
        "gamma_rf": float(result["gamma_rf"]),
        "n_steps": int(result["n_steps"]),
        "t_burn": float(result["t_burn"]),
        "q_before": float(result["q_before"]),
        "q_after": float(result["q_after"]),
        "q_gain": float(result["q_gain"]),
        "q_total_unburned": float(result["q_total_unburned"]),
        "q_total_after": float(result["q_total_after"]),
        "q_total_gain": float(result["q_total_gain"]),
        "p_total_unburned": float(result["p_total_unburned"]),
        "p_total_after": float(result["p_total_after"]),
        "area_unburned": float(result["area_unburned"]),
        "area_after": float(result["area_after"]),
        "area_loss": float(result["area_loss"]),
        "skipped": bool(result["skipped"]),
    }
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            meta_json=np.asarray(json.dumps(meta)),
            f=np.asarray(result["f"], dtype=float),
            iplus_unburned=np.asarray(result["iplus_unburned"], dtype=float),
            iminus_unburned=np.asarray(result["iminus_unburned"], dtype=float),
            iplus=np.asarray(result["iplus"], dtype=float),
            iminus=np.asarray(result["iminus"], dtype=float),
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def load_shard(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta_json"]))
        return {
            **meta,
            "f": np.asarray(data["f"], dtype=float),
            "iplus_unburned": np.asarray(data["iplus_unburned"], dtype=float),
            "iminus_unburned": np.asarray(data["iminus_unburned"], dtype=float),
            "iplus": np.asarray(data["iplus"], dtype=float),
            "iminus": np.asarray(data["iminus"], dtype=float),
        }


def combine_shards(
    shard_dir: Path,
    *,
    num_bins: int,
    polarization: float,
    n_steps: int,
    dt: float = gopt.DT,
    gamma_max: float = gopt.GAMMA_MAX,
    strict: bool = True,
) -> dict:
    """
    Merge independent per-bin shards into a gamma profile + overlaid lineshape.

    Composite spectrum starts from unburned and overlays each applied burn's
    committed burn/mirror bins (independent burns; overlapping mirrors last-write).
    """
    shard_dir = Path(shard_dir)
    f = np.linspace(-3.0, 3.0, int(num_bins))
    _, iplus0, iminus0 = GenerateVectorLineshape(float(polarization), f)
    iplus_unburned = np.asarray(iplus0, dtype=float).copy()
    iminus_unburned = np.asarray(iminus0, dtype=float).copy()
    iplus = iplus_unburned.copy()
    iminus = iminus_unburned.copy()
    q0 = iplus_unburned - iminus_unburned

    gamma_profile = np.zeros(int(num_bins), dtype=float)
    steps_profile = np.zeros(int(num_bins), dtype=int)
    trace: list[dict] = []
    applied = 0
    skipped = 0
    missing: list[int] = []

    for bin_idx in range(int(num_bins)):
        path = shard_path(shard_dir, bin_idx)
        if not path.is_file():
            missing.append(bin_idx)
            continue
        shard = load_shard(path)
        if int(shard["num_bins"]) != int(num_bins):
            raise ValueError(
                f"{path}: num_bins={shard['num_bins']} != expected {num_bins}"
            )
        gamma_profile[bin_idx] = float(shard["gamma_rf"])
        steps_profile[bin_idx] = int(shard["n_steps"])
        row = {
            "bin_idx": bin_idx,
            "mirror_idx": int(shard["mirror_idx"]),
            "f": float(shard["R"]),
            "gamma_rf": float(shard["gamma_rf"]),
            "n_steps": int(shard["n_steps"]),
            "t_burn": float(shard["t_burn"]),
            "q_before": float(shard["q_before"]),
            "q_after": float(shard["q_after"]),
            "q_gain": float(shard["q_gain"]),
            "q_total_gain": float(shard["q_total_gain"]),
            "q_total": float(shard["q_total_after"]),
            "p_total": float(shard["p_total_after"]),
            "area_loss": float(shard["area_loss"]),
            "skipped": bool(shard["skipped"]),
        }
        trace.append(row)
        if shard["skipped"]:
            skipped += 1
            continue
        applied += 1
        for idx in (bin_idx, int(shard["mirror_idx"])):
            iplus[idx] = float(shard["iplus"][idx])
            iminus[idx] = float(shard["iminus"][idx])

    if missing and strict:
        raise FileNotFoundError(
            f"Missing {len(missing)} shard(s) under {shard_dir}; "
            f"first missing bin_idx={missing[0]}"
        )
    if missing:
        print(f"WARNING: missing {len(missing)} shards; continuing", flush=True)

    iplus_pre_afp, iminus_pre_afp = iplus.copy(), iminus.copy()
    afp_subset: list[int] = []
    afp_on = bool(getattr(gopt, "AFP_ENABLED", True))
    if afp_on:
        burned = [i for i in range(int(num_bins)) if float(gamma_profile[i]) > 0.0]
        subset = None
        if gopt.AFP_BIN_RANGE is None and burned:
            subset = sorted(
                {int(i) for i in burned}
                | {gopt.mirror_bin_idx(int(num_bins), int(i)) for i in burned}
            )
        iplus, iminus, afp_subset = gopt.apply_afp_sweep(
            iplus,
            iminus,
            bin_range=gopt.AFP_BIN_RANGE,
            subset_indices=subset,
            efficiency=gopt.AFP_EFFICIENCY,
            center_margin=gopt.AFP_CENTER_EXCLUSION_BINS,
        )

    initial_q = gopt.q_total(iplus_unburned, iminus_unburned)
    initial_p = gopt.p_total(iplus_unburned, iminus_unburned)
    area0 = gopt.lineshape_area(iplus_unburned, iminus_unburned, f)
    final_q = gopt.q_total(iplus, iminus)
    final_p = gopt.p_total(iplus, iminus)
    final_area = gopt.lineshape_area(iplus, iminus, f)
    n_candidates = sum(1 for i in range(int(num_bins)) if float(q0[i]) < 0.0)

    return {
        "polarization": float(polarization),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "f": f,
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus_pre_afp": iplus_pre_afp,
        "iminus_pre_afp": iminus_pre_afp,
        "iplus": iplus,
        "iminus": iminus,
        "gamma_profile": gamma_profile,
        "steps_profile": steps_profile,
        "q0": q0,
        "area0": area0,
        "area_final": final_area,
        "area_loss_total": area0 - final_area,
        "initial_q": initial_q,
        "final_q": final_q,
        "initial_p": initial_p,
        "final_p": final_p,
        "q_pre_afp": gopt.q_total(iplus_pre_afp, iminus_pre_afp),
        "p_pre_afp": gopt.p_total(iplus_pre_afp, iminus_pre_afp),
        "n_applied": applied,
        "n_skipped": skipped,
        "n_candidates": n_candidates,
        "n_missing": len(missing),
        "trace": trace,
        "parallel_independent": True,
        "afp_enabled": afp_on,
        "afp_efficiency": float(gopt.AFP_EFFICIENCY),
        "afp_subset": afp_subset,
        "afp_center_margin": int(gopt.AFP_CENTER_EXCLUSION_BINS),
    }


def _print_combined_summary(result: dict) -> None:
    q_final = result["iplus"] - result["iminus"]
    burned = result["gamma_profile"] > 0.0
    if np.any(burned):
        max_abs_q = float(np.max(np.abs(q_final[burned])))
        mean_abs_q = float(np.mean(np.abs(q_final[burned])))
        steps_used = result["steps_profile"][burned]
        mean_steps = float(np.mean(steps_used))
        max_steps = int(np.max(steps_used))
        gamma_min = float(np.min(result["gamma_profile"][burned]))
    else:
        max_abs_q = mean_abs_q = mean_steps = float("nan")
        max_steps = 0
        gamma_min = 0.0

    print(flush=True)
    print(
        f"P0={result['polarization']}  dt={result['dt']}  "
        f"n_steps≤{result['n_steps']}  (independent parallel burns)",
        flush=True,
    )
    print(
        f"RF bins applied: {result['n_applied']}/{result['n_candidates']}  "
        f"skipped={result['n_skipped']}  missing={result.get('n_missing', 0)}",
        flush=True,
    )
    print(
        f"P total (overlay): {result['initial_p']:.8f} -> {result['final_p']:.8f}",
        flush=True,
    )
    print(
        f"Q total (overlay): {result['initial_q']:.8f} -> {result['final_q']:.8f}",
        flush=True,
    )
    print(
        f"|Q(R)| on burned bins: mean={mean_abs_q:.3e}  max={max_abs_q:.3e}",
        flush=True,
    )
    print(
        f"steps used: mean={mean_steps:.1f}  max={max_steps}/{result['n_steps']}",
        flush=True,
    )
    print(
        f"gamma_rf:  min={gamma_min:.6g}  "
        f"max={float(np.max(result['gamma_profile'])):.6g}",
        flush=True,
    )


def _resolve_bin_idx(cli_bin_idx: int | None) -> int | None:
    """Prefer --bin-idx; fall back to SLURM_ARRAY_TASK_ID for array jobs."""
    if cli_bin_idx is not None:
        return int(cli_bin_idx)
    env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_idx is not None and str(env_idx).strip() != "":
        return int(env_idx)
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Independent per-bin gamma-opt worker / shard combiner for SLURM"
    )
    p.add_argument(
        "--bin-idx",
        type=int,
        default=None,
        help="Burn bin index (array task). Defaults to SLURM_ARRAY_TASK_ID if set.",
    )
    p.add_argument("--combine", action="store_true", help="Combine shards instead of running a bin")
    p.add_argument(
        "--shard-dir",
        type=Path,
        default=DEFAULT_SHARD_DIR,
        help="Directory for per-bin .npz shards",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_COMBINED_DIR,
        help="Directory for combined plots/CSV (combine mode)",
    )
    p.add_argument("--polarization", type=float, default=gopt.P)
    p.add_argument("--num-bins", type=int, default=gopt.NUM_BINS)
    p.add_argument("--dt", type=float, default=gopt.DT)
    p.add_argument("--n-steps", type=int, default=gopt.N_STEPS)
    p.add_argument("--gamma-hi", type=float, default=gopt.GAMMA_MAX)
    p.add_argument("--n-bisect", type=int, default=gopt.N_BISECT)
    p.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip bin if shard already exists",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Combine mode: fail if any shard is missing",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.combine:
        result = combine_shards(
            args.shard_dir,
            num_bins=args.num_bins,
            polarization=args.polarization,
            n_steps=args.n_steps,
            dt=args.dt,
            gamma_max=args.gamma_hi,
            strict=bool(args.strict),
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "rate_eqs_test_ssrf_all_bins_gamma_opt_parallel"
        lineshape_path = out_dir / f"{stem}_lineshape.png"
        gains_path = out_dir / f"{stem}_gains.png"
        profile_path = out_dir / f"{stem}_burn_profile.png"
        csv_path = out_dir / f"{stem}_gamma_profile.csv"

        gopt.plot_result(result, lineshape_path)
        gopt.plot_q_gains(result, gains_path)
        gopt.plot_gamma_profile(result, profile_path)
        gopt.save_gamma_profile(result, csv_path)
        _print_combined_summary(result)
        print(
            f"Saved {lineshape_path}, {gains_path}, {profile_path}, {csv_path}",
            flush=True,
        )
        return

    bin_idx = _resolve_bin_idx(args.bin_idx)
    if bin_idx is None:
        raise SystemExit(
            "Provide --bin-idx <int>, or set SLURM_ARRAY_TASK_ID, or pass --combine"
        )

    out = shard_path(args.shard_dir, bin_idx)
    if args.skip_if_exists and out.is_file():
        print(f"Skipping existing shard {out}", flush=True)
        return

    print(
        f"bin_idx={bin_idx}  P={args.polarization}  "
        f"num_bins={args.num_bins}  dt={args.dt}  "
        f"γ≤{min(args.gamma_hi, gopt.GAMMA_MAX)}  n_steps≤{args.n_steps}",
        flush=True,
    )
    result = run_one_bin(
        bin_idx,
        polarization=args.polarization,
        num_bins=args.num_bins,
        dt=args.dt,
        n_steps=args.n_steps,
        gamma_hi=args.gamma_hi,
        n_bisect=args.n_bisect,
    )
    save_shard(result, out)
    status = "skipped" if result["skipped"] else "applied"
    print(
        f"{status}: R={result['R']:+.4f}  gamma={result['gamma_rf']:.6g}  "
        f"steps={result['n_steps']}/{args.n_steps}  "
        f"Q={result['q_before']:.3e}->{result['q_after']:.3e}",
        flush=True,
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
