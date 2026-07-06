"""Compare v14 burn outcomes at two RF rates burned to the same local Ps."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import tqdm as tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V14_ROOT = Path(__file__).resolve().parent / "spin1_ssrf_realtime_sim_v14"
if str(V14_ROOT) not in sys.path:
    sys.path.insert(0, str(V14_ROOT))

from ssrf_realtime.model import Spin1Model, Spin1Params

PS_DTYPE = np.float64


@dataclass(frozen=True)
class BurnOutcome:
    gamma_rf: float
    steps: int
    time: float
    ps: float
    iplus: float
    iminus: float
    p_global: float
    q: float
    ps_trace: tuple[float, ...]
    iplus_trace: tuple[float, ...]
    iminus_trace: tuple[float, ...]
    time_trace: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Burn the v14 spin-1 model with two different gamma_rf values until each "
            "reaches the same local Ps (= I+ + I- at the burn bin), then compare I+ and I-."
        )
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs=2,
        default=(0.1, 0.2),
        metavar=("GAMMA_A", "GAMMA_B"),
        help="Two RF burn rates to compare (default: 0.1 0.2).",
    )
    parser.add_argument(
        "--target-ps",
        type=float,
        default=None,
        help="Target local Ps at the burn bin. If omitted, use --target-ps-fraction of the initial Ps.",
    )
    parser.add_argument(
        "--target-ps-fraction",
        type=float,
        default=0.5,
        help="Fraction of initial burn-bin Ps to burn down to when --target-ps is not set.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Safety cap on RF integration steps while searching for the target Ps.",
    )
    parser.add_argument(
        "--ps-tolerance",
        type=float,
        default=1e-8,
        help="Acceptable |Ps - target| when stopping the burn search.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to physics/compare_gamma_burn_<target>.png.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving it.")

    parser.add_argument("--rf-burn-R", type=float, default=Spin1Params.rf_burn_R, help="Physical R coordinate of the RF burn bin.")
    parser.add_argument("--dt", type=float, default=Spin1Params.dt, help="Integration timestep.")
    parser.add_argument("--p0", type=float, default=Spin1Params.p0, help="Initial vector polarization.")
    parser.add_argument("--n-bins", type=int, default=Spin1Params.n_bins, help="Number of R bins in the simulation grid.")
    parser.add_argument("--line-gamma", type=float, default=Spin1Params.line_gamma, help="Pake branch broadening parameter.")
    parser.add_argument("--line-asym", type=float, default=Spin1Params.line_asym, help="Pake branch asymmetry parameter.")
    return parser.parse_args()


def build_params(args: argparse.Namespace) -> Spin1Params:
    return Spin1Params(
        n_bins=args.n_bins,
        line_gamma=args.line_gamma,
        line_asym=args.line_asym,
        p0=args.p0,
        rf_burn_R=args.rf_burn_R,
        rf_enabled=True,
        dnp_enabled=False,
        dt=args.dt,
    )


def local_ps(model: Spin1Model) -> np.float64:
    return PS_DTYPE(model.local_intensities(model.params.rf_burn_R)["total"])


def burn_to_target_ps(
    base_params: Spin1Params,
    *,
    gamma_rf: float,
    target_ps: float,
    max_steps: int,
    ps_tolerance: float,
) -> tuple[Spin1Model, BurnOutcome]:
    target_ps = PS_DTYPE(target_ps)
    ps_tolerance = PS_DTYPE(ps_tolerance)

    probe = Spin1Model(base_params)
    ps_initial = local_ps(probe)
    if target_ps >= ps_initial:
        raise ValueError(
            f"target Ps ({target_ps:.6g}) must be below the initial burn-bin Ps ({ps_initial:.6g})"
        )

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
        raise RuntimeError(
            f"Did not reach target Ps={target_ps:.6g} with gamma_rf={gamma_rf:.6g} "
            f"within {max_steps} steps (Ps={ps:.6g})."
        )

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

    loc_initial = probe.local_intensities(probe.params.rf_burn_R)
    ps_trace: list[float] = [ps_initial]
    iplus_trace: list[float] = [float(loc_initial["Iplus"])]
    iminus_trace: list[float] = [float(loc_initial["Iminus"])]
    time_trace: list[float] = [0.0]
    trace_block = max(1, best_steps // 40)
    trace_model = Spin1Model(base_params)
    trace_model.params.gamma_rf = float(gamma_rf)
    trace_model.params.rf_enabled = True
    trace_model.params.dnp_enabled = False
    done = 0
    while done < best_steps:
        n = min(trace_block, best_steps - done)
        trace_model.step(n, rf_on=True, dnp_on=False)
        done += n
        loc = trace_model.local_intensities(trace_model.params.rf_burn_R)
        ps_trace.append(float(loc["total"]))
        iplus_trace.append(float(loc["Iplus"]))
        iminus_trace.append(float(loc["Iminus"]))
        time_trace.append(float(trace_model.t))

    loc = model.local_intensities(model.params.rf_burn_R)
    pol = model.polarizations()
    outcome = BurnOutcome(
        gamma_rf=float(gamma_rf),
        steps=best_steps,
        time=float(model.t),
        ps=float(loc["total"]),
        iplus=float(loc["Iplus"]),
        iminus=float(loc["Iminus"]),
        p_global=float(pol["P"]),
        q=float(pol["Q"]),
        ps_trace=tuple(ps_trace),
        iplus_trace=tuple(iplus_trace),
        iminus_trace=tuple(iminus_trace),
        time_trace=tuple(time_trace),
    )
    return model, outcome


def default_output_path(target_ps: float) -> Path:
    target_tag = f"{target_ps:.4g}".replace(".", "p")
    return Path(__file__).resolve().parent / f"compare_gamma_burn_Ps_{target_tag}.png"


def plot_comparison(
    models: dict[float, Spin1Model],
    outcomes: list[BurnOutcome],
    *,
    target_ps: float,
    ps_initial: float,
    output_path: Path,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(11.0, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.4, 1.0, 1.0])

    ax_spec = fig.add_subplot(gs[0, :])
    ax_bars = fig.add_subplot(gs[1, 0])
    ax_traj = fig.add_subplot(gs[1, 1])

    ref_model = next(iter(models.values()))
    R_ref, _Ip_ref, _Im_ref, total_ref = ref_model.reference_spectrum()
    ax_spec.plot(R_ref, total_ref, color="0.70", linewidth=1.2, label="initial total")

    colors = ["tab:blue", "tab:orange"]
    for color, outcome in zip(colors, outcomes):
        model = models[outcome.gamma_rf]
        R, Ip, Im, total = model.spectrum()
        label = f"Γ_RF={outcome.gamma_rf:g} ({outcome.steps} steps, t={outcome.time:.4g})"
        ax_spec.plot(R, Ip, color=color, linewidth=1.0, linestyle="-", alpha=0.95, label=f"I+ {label}")
        ax_spec.plot(R, Im, color=color, linewidth=1.0, linestyle="--", alpha=0.95, label=f"I- {label}")
        ax_spec.plot(R, total, color=color, linewidth=1.4, alpha=0.55, label=f"total {label}")

    Rb = ref_model.params.rf_burn_R
    ax_spec.axvline(Rb, color="tab:red", linestyle="--", linewidth=1.0, label="RF R")
    ax_spec.axvline(-Rb, color="tab:red", linestyle=":", linewidth=1.0, label="mirror -R")
    ax_spec.set_xlabel("physical R")
    ax_spec.set_ylabel("intensity density")
    ax_spec.set_title(
        f"v14 gamma burn comparison to Ps={target_ps:.5g} "
        f"(initial Ps={ps_initial:.5g}, fraction={target_ps / ps_initial:.3f})"
    )
    ax_spec.legend(fontsize=7, ncols=2, loc="upper right")

    gammas = [o.gamma_rf for o in outcomes]
    x = np.arange(len(gammas))
    width = 0.35
    iplus_vals = [o.iplus for o in outcomes]
    iminus_vals = [o.iminus for o in outcomes]
    ax_bars.bar(x - width / 2, iplus_vals, width, label="I+", color="tab:blue", alpha=0.85)
    ax_bars.bar(x + width / 2, iminus_vals, width, label="I-", color="tab:orange", alpha=0.85)
    ax_bars.axhline(target_ps, color="black", linestyle=":", linewidth=1.0, label=f"target Ps={target_ps:.4g}")
    for idx, outcome in enumerate(outcomes):
        ax_bars.text(idx - width / 2, outcome.iplus, f"{outcome.iplus:.4g}", ha="center", va="bottom", fontsize=8)
        ax_bars.text(idx + width / 2, outcome.iminus, f"{outcome.iminus:.4g}", ha="center", va="bottom", fontsize=8)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([f"Γ={g:g}" for g in gammas])
    ax_bars.set_ylabel("burn-bin intensity")
    ax_bars.set_title("I+ and I- at burn bin after matched Ps")
    ax_bars.legend(fontsize=8)

    for color, outcome in zip(colors, outcomes):
        ax_traj.plot(outcome.time_trace, outcome.ps_trace, color=color, linewidth=1.2, label=f"Γ={outcome.gamma_rf:g}")

    ax_traj.axhline(target_ps, color="black", linestyle=":", linewidth=1.0, label="target Ps")
    ax_traj.set_xlabel("time")
    ax_traj.set_ylabel("local Ps at burn bin")
    ax_traj.set_title("Ps burn trajectories")
    ax_traj.legend(fontsize=8)

    for idx, (color, outcome) in enumerate(zip(colors, outcomes)):
        ax_i = fig.add_subplot(gs[2, idx])
        ax_i.plot(outcome.time_trace, outcome.iplus_trace, color="tab:red", linewidth=1.2, label="I+")
        ax_i.plot(outcome.time_trace, outcome.iminus_trace, color="tab:blue", linewidth=1.2, label="I-")
        ax_i.plot(
            outcome.time_trace,
            outcome.ps_trace,
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.75,
            label="Ps",
        )
        ax_i.set_xlabel("time")
        ax_i.set_ylabel("burn-bin intensity")
        ax_i.set_title(f"I+ / I- trajectories (Γ_RF={outcome.gamma_rf:g})")
        ax_i.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def print_summary(outcomes: list[BurnOutcome], *, target_ps: float, ps_initial: float) -> None:
    print(f"initial burn-bin Ps={ps_initial:.8g}")
    print(f"target burn-bin Ps={target_ps:.8g}")
    print()
    header = (
        f"{'gamma_rf':>10} {'steps':>8} {'time':>12} {'Ps':>14} "
        f"{'Iplus':>14} {'Iminus':>14} {'P_global':>12} {'Q':>12}"
    )
    print(header)
    print("-" * len(header))
    for o in outcomes:
        print(
            f"{o.gamma_rf:10.4g} {o.steps:8d} {o.time:12.6g} {o.ps:14.8g} "
            f"{o.iplus:14.8g} {o.iminus:14.8g} {o.p_global:12.8g} {o.q:12.8g}"
        )

    if len(outcomes) == 2:
        d_iplus = outcomes[1].iplus - outcomes[0].iplus
        d_iminus = outcomes[1].iminus - outcomes[0].iminus
        d_ps = outcomes[1].ps - outcomes[0].ps
        print()
        print(
            f"delta (gamma {outcomes[1].gamma_rf:g} - {outcomes[0].gamma_rf:g}): "
            f"dPs={d_ps:+.6g}, dIplus={d_iplus:+.6g}, dIminus={d_iminus:+.6g}"
        )


def main() -> int:
    args = parse_args()
    if args.target_ps_fraction <= 0.0 or args.target_ps_fraction >= 1.0:
        raise ValueError("--target-ps-fraction must be between 0 and 1 (exclusive).")

    base_params = build_params(args)
    probe = Spin1Model(base_params)
    ps_initial = local_ps(probe)
    target_ps = (
        PS_DTYPE(args.target_ps)
        if args.target_ps is not None
        else PS_DTYPE(ps_initial * args.target_ps_fraction)
    )

    outcomes: list[BurnOutcome] = []
    models: dict[float, Spin1Model] = {}
    for gamma in args.gammas:
        model, outcome = burn_to_target_ps(
            base_params,
            gamma_rf=gamma,
            target_ps=target_ps,
            max_steps=args.max_steps,
            ps_tolerance=args.ps_tolerance,
        )
        models[gamma] = model
        outcomes.append(outcome)

    output_path = args.output or default_output_path(target_ps)
    plot_comparison(
        models,
        outcomes,
        target_ps=target_ps,
        ps_initial=ps_initial,
        output_path=output_path,
        show=args.show,
    )
    print_summary(outcomes, target_ps=target_ps, ps_initial=ps_initial)
    print(f"\nwrote comparison plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
