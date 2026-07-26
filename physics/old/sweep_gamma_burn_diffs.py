from __future__ import annotations

import argparse
import csv
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

V14_ROOT = Path(__file__).resolve().parent / "spin1_ssrf_realtime"
if str(V14_ROOT) not in sys.path:
    sys.path.insert(0, str(V14_ROOT))

from ssrf_realtime.model import Spin1Model, Spin1Params

PS_DTYPE = np.float64


@dataclass(frozen=True)
class SweepPoint:
    bin_idx: int
    dt: float
    p0: float
    rf_burn_R: float
    target_ps_fraction: float
    ps_initial: float
    target_ps: float
    gamma_a: float
    gamma_b: float
    steps_a: int
    steps_b: int
    time_a: float
    time_b: float
    ps_a: float
    ps_b: float
    iplus_a: float
    iplus_b: float
    iminus_a: float
    iminus_b: float
    d_iplus: float
    d_iminus: float
    d_ps: float
    status: str


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep dt, polarization, RF position, and bin height; "
            "compare two gamma_rf burns at matched Ps and record dI+/dI-."
        )
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs=2,
        default=None,
        metavar=("GAMMA_A", "GAMMA_B"),
        help="Single gamma_a/gamma_b pair when --gamma-as/--gamma-bs are not set.",
    )
    parser.add_argument(
        "--gamma-as",
        type=parse_float_list,
        default=None,
        help="RF burn rates for branch A (comma-separated).",
    )
    parser.add_argument(
        "--gamma-bs",
        type=parse_float_list,
        default=None,
        help="RF burn rates for branch B (comma-separated).",
    )
    parser.add_argument(
        "--bin-idx",
        type=int,
        default=None,
        help=(
            "Burn-bin index in [0, n_bins). Sets rf_burn_R from linspace(r_min, r_max, n_bins). "
            "Use with burning.slurm array tasks."
        ),
    )
    parser.add_argument("--r-min", type=float, default=Spin1Params.r_min, help="Physical R grid minimum.")
    parser.add_argument("--r-max", type=float, default=Spin1Params.r_max, help="Physical R grid maximum.")
    parser.add_argument(
        "--dts",
        type=parse_float_list,
        default=None,
        help="Integration timesteps to sweep (comma-separated).",
    )
    parser.add_argument(
        "--p0s",
        type=parse_float_list,
        default=None,
        help="Initial polarizations to sweep (comma-separated).",
    )
    parser.add_argument(
        "--rf-burn-Rs",
        type=parse_float_list,
        default=None,
        help="RF burn positions on physical R (comma-separated).",
    )
    parser.add_argument(
        "--bin-heights",
        type=parse_float_list,
        default=None,
        help=(
            "Remaining bin heights as fractions of initial Ps at burn bin "
            "(comma-separated, e.g. 0.3,0.5,0.7)."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small default grid for a fast smoke sweep.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Safety cap on RF integration steps per burn.",
    )
    parser.add_argument(
        "--ps-tolerance",
        type=float,
        default=1e-6,
        help="Acceptable |Ps - target| when stopping the burn search.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to physics/sweep_gamma_burn_diffs.csv.",
    )
    parser.add_argument("--n-bins", type=int, default=Spin1Params.n_bins, help="Number of R bins.")
    parser.add_argument("--line-gamma", type=float, default=Spin1Params.line_gamma, help="Pake branch broadening.")
    parser.add_argument("--line-asym", type=float, default=Spin1Params.line_asym, help="Pake branch asymmetry.")
    return parser.parse_args()


def rf_burn_R_from_bin_idx(bin_idx: int, *, n_bins: int, r_min: float, r_max: float) -> float:
    if bin_idx < 0 or bin_idx >= n_bins:
        raise ValueError(f"bin-idx must be in [0, {n_bins}), got {bin_idx}")
    return float(np.linspace(r_min, r_max, n_bins)[bin_idx])


def resolve_gamma_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.gamma_as is not None or args.gamma_bs is not None:
        gamma_as = args.gamma_as or []
        gamma_bs = args.gamma_bs or []
        if not gamma_as or not gamma_bs:
            raise ValueError("both --gamma-as and --gamma-bs are required when either is set")
        return list(itertools.product(gamma_as, gamma_bs))

    if args.gammas is not None:
        return [(float(args.gammas[0]), float(args.gammas[1]))]

    return [(0.1, 0.2)]


def resolve_grids(
    args: argparse.Namespace,
) -> tuple[int | None, list[float], list[float], list[float], list[float], list[tuple[float, float]]]:
    if args.quick:
        dts = args.dts or [Spin1Params.dt, Spin1Params.dt * 2.0]
        p0s = args.p0s or [0.35, 0.45, 0.55]
        rf_burn_rs = args.rf_burn_Rs or [-0.95, -0.75, -0.55]
        bin_heights = args.bin_heights or [0.4, 0.5, 0.6]
    else:
        dts = args.dts or [Spin1Params.dt]
        p0s = args.p0s or [Spin1Params.p0]
        bin_heights = args.bin_heights or [0.5]
        if args.bin_idx is not None:
            rf_burn_rs = [
                rf_burn_R_from_bin_idx(
                    args.bin_idx,
                    n_bins=args.n_bins,
                    r_min=args.r_min,
                    r_max=args.r_max,
                )
            ]
        else:
            rf_burn_rs = args.rf_burn_Rs or [Spin1Params.rf_burn_R]

    gamma_pairs = resolve_gamma_pairs(args)

    for name, values in (
        ("dts", dts),
        ("p0s", p0s),
        ("rf_burn_Rs", rf_burn_rs),
        ("bin_heights", bin_heights),
        ("gamma_pairs", gamma_pairs),
    ):
        if not values:
            raise ValueError(f"{name} must contain at least one value")
        if name == "bin_heights" and any(v <= 0.0 or v >= 1.0 for v in values):
            raise ValueError("bin-heights must be between 0 and 1 (exclusive)")

    return args.bin_idx, dts, p0s, rf_burn_rs, bin_heights, gamma_pairs


def build_params(
    *,
    dt: float,
    p0: float,
    rf_burn_R: float,
    n_bins: int,
    line_gamma: float,
    line_asym: float,
) -> Spin1Params:
    return Spin1Params(
        n_bins=n_bins,
        line_gamma=line_gamma,
        line_asym=line_asym,
        p0=p0,
        rf_burn_R=rf_burn_R,
        rf_enabled=True,
        dnp_enabled=False,
        dt=dt,
    )


def local_ps(model: Spin1Model) -> np.float64:
    return PS_DTYPE(model.local_intensities(model.params.rf_burn_R)["total"])


def local_branch_intensities(model: Spin1Model) -> tuple[float, float, float]:
    loc = model.local_intensities(model.params.rf_burn_R)
    return float(loc["Iplus"]), float(loc["Iminus"]), float(loc["total"])


def burn_to_target_ps_fast(
    base_params: Spin1Params,
    *,
    gamma_rf: float,
    target_ps: float,
    max_steps: int,
    ps_tolerance: float,
) -> tuple[int, float, float, float, float, str]:
    target_ps = PS_DTYPE(target_ps)
    ps_tolerance = PS_DTYPE(ps_tolerance)

    probe = Spin1Model(base_params)
    ps_initial = local_ps(probe)
    if target_ps >= ps_initial:
        return 0, 0.0, float("nan"), float("nan"), float("nan"), "target_above_initial"

    def simulate(n_steps: int) -> tuple[Spin1Model, float]:
        model = Spin1Model(base_params)
        model.params.gamma_rf = float(gamma_rf)
        model.params.rf_enabled = True
        model.params.dnp_enabled = False
        if n_steps > 0:
            model.step(n_steps, rf_on=True, dnp_on=False)
        return model, local_ps(model)

    block = 500
    march_model = Spin1Model(base_params)
    march_model.params.gamma_rf = float(gamma_rf)
    march_model.params.rf_enabled = True
    march_model.params.dnp_enabled = False
    marched_steps = 0
    ps = ps_initial
    while ps > target_ps and marched_steps < max_steps:
        n = min(block, max_steps - marched_steps)
        march_model.step(n, rf_on=True, dnp_on=False)
        marched_steps += n
        ps = local_ps(march_model)

    if ps > target_ps:
        return marched_steps, float(march_model.t), float("nan"), float("nan"), float("nan"), "max_steps"

    lo = max(0, marched_steps - block)
    hi = marched_steps
    best_steps = hi
    while lo <= hi:
        mid = (lo + hi) // 2
        _, ps_mid = simulate(mid)
        if ps_mid > target_ps:
            lo = mid + 1
        else:
            best_steps = mid
            hi = mid - 1

    model, ps = simulate(best_steps)
    if abs(ps - target_ps) > ps_tolerance:
        _, ps_next = simulate(best_steps + 1)
        if abs(ps_next - target_ps) < abs(ps - target_ps):
            best_steps += 1
            model, ps = simulate(best_steps)

    iplus, iminus, _ = local_branch_intensities(model)
    return best_steps, float(model.t), float(ps), iplus, iminus, "ok"


def evaluate_point(
    *,
    bin_idx: int | None,
    dt: float,
    p0: float,
    rf_burn_R: float,
    target_ps_fraction: float,
    gamma_a: float,
    gamma_b: float,
    n_bins: int,
    line_gamma: float,
    line_asym: float,
    max_steps: int,
    ps_tolerance: float,
) -> SweepPoint:
    base_params = build_params(
        dt=dt,
        p0=p0,
        rf_burn_R=rf_burn_R,
        n_bins=n_bins,
        line_gamma=line_gamma,
        line_asym=line_asym,
    )
    ps_initial = float(local_ps(Spin1Model(base_params)))
    target_ps = float(PS_DTYPE(ps_initial * target_ps_fraction))

    steps_a, time_a, ps_a, iplus_a, iminus_a, status_a = burn_to_target_ps_fast(
        base_params,
        gamma_rf=gamma_a,
        target_ps=target_ps,
        max_steps=max_steps,
        ps_tolerance=ps_tolerance,
    )
    steps_b, time_b, ps_b, iplus_b, iminus_b, status_b = burn_to_target_ps_fast(
        base_params,
        gamma_rf=gamma_b,
        target_ps=target_ps,
        max_steps=max_steps,
        ps_tolerance=ps_tolerance,
    )

    if status_a != "ok" or status_b != "ok":
        status = status_a if status_a != "ok" else status_b
        d_iplus = float("nan")
        d_iminus = float("nan")
        d_ps = float("nan")
    else:
        status = "ok"
        d_iplus = iplus_b - iplus_a
        d_iminus = iminus_b - iminus_a
        d_ps = ps_b - ps_a

    return SweepPoint(
        bin_idx=-1 if bin_idx is None else bin_idx,
        dt=dt,
        p0=p0,
        rf_burn_R=rf_burn_R,
        target_ps_fraction=target_ps_fraction,
        ps_initial=ps_initial,
        target_ps=target_ps,
        gamma_a=gamma_a,
        gamma_b=gamma_b,
        steps_a=steps_a,
        steps_b=steps_b,
        time_a=time_a,
        time_b=time_b,
        ps_a=ps_a,
        ps_b=ps_b,
        iplus_a=iplus_a,
        iplus_b=iplus_b,
        iminus_a=iminus_a,
        iminus_b=iminus_b,
        d_iplus=d_iplus,
        d_iminus=d_iminus,
        d_ps=d_ps,
        status=status,
    )


def default_output_path(bin_idx: int | None = None) -> Path:
    parent = Path(__file__).resolve().parent
    if bin_idx is None:
        return parent / "sweep_gamma_burn_diffs.csv"
    return parent / "burning_shards" / f"sweep_bin_{bin_idx}.csv"


def write_csv(path: Path, rows: list[SweepPoint]) -> None:
    fieldnames = [
        "bin_idx",
        "dt",
        "p0",
        "rf_burn_R",
        "target_ps_fraction",
        "ps_initial",
        "target_ps",
        "gamma_a",
        "gamma_b",
        "steps_a",
        "steps_b",
        "time_a",
        "time_b",
        "ps_a",
        "ps_b",
        "iplus_a",
        "iplus_b",
        "iminus_a",
        "iminus_b",
        "d_iplus",
        "d_iminus",
        "d_ps",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: getattr(row, key) for key in fieldnames})


def print_summary(rows: list[SweepPoint], *, output_path: Path) -> None:
    ok_rows = [r for r in rows if r.status == "ok"]
    print(f"wrote {len(rows)} rows to {output_path} ({len(ok_rows)} ok)")
    if not ok_rows:
        return

    d_iplus = np.array([r.d_iplus for r in ok_rows], dtype=float)
    d_iminus = np.array([r.d_iminus for r in ok_rows], dtype=float)
    print(
        f"d_iplus: min={d_iplus.min():+.6g} max={d_iplus.max():+.6g} "
        f"mean={d_iplus.mean():+.6g} std={d_iplus.std():.6g}"
    )
    print(
        f"d_iminus: min={d_iminus.min():+.6g} max={d_iminus.max():+.6g} "
        f"mean={d_iminus.mean():+.6g} std={d_iminus.std():.6g}"
    )

    worst = max(ok_rows, key=lambda r: abs(r.d_iplus) + abs(r.d_iminus))
    print(
        "largest combined split: "
        f"dt={worst.dt:g} p0={worst.p0:g} R={worst.rf_burn_R:g} "
        f"bin_height={worst.target_ps_fraction:g} "
        f"d_iplus={worst.d_iplus:+.6g} d_iminus={worst.d_iminus:+.6g}"
    )


def main() -> int:
    args = parse_args()
    bin_idx, dts, p0s, rf_burn_rs, bin_heights, gamma_pairs = resolve_grids(args)

    combos = list(itertools.product(dts, p0s, rf_burn_rs, bin_heights, gamma_pairs))
    rows: list[SweepPoint] = []
    for dt, p0, rf_burn_R, target_ps_fraction, (gamma_a, gamma_b) in tqdm(combos, desc="sweep", unit="pt"):
        rows.append(
            evaluate_point(
                bin_idx=bin_idx,
                dt=dt,
                p0=p0,
                rf_burn_R=rf_burn_R,
                target_ps_fraction=target_ps_fraction,
                gamma_a=gamma_a,
                gamma_b=gamma_b,
                n_bins=args.n_bins,
                line_gamma=args.line_gamma,
                line_asym=args.line_asym,
                max_steps=args.max_steps,
                ps_tolerance=args.ps_tolerance,
            )
        )

    output_path = args.output or default_output_path(bin_idx)
    write_csv(output_path, rows)
    print_summary(rows, output_path=output_path)
    if bin_idx is not None:
        print(f"bin_idx={bin_idx} rf_burn_R={rf_burn_rs[0]:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
