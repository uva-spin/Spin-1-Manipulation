"""Offline step-count runner for the spin-1 ss-RF v14 lineshape model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

V14_ROOT = Path(__file__).resolve().parent / "spin1_ssrf_realtime_sim_v14"
if str(V14_ROOT) not in sys.path:
    sys.path.insert(0, str(V14_ROOT))

from ssrf_realtime.model import Spin1Model, Spin1Params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the v14 spin-1 ss-RF model for a fixed number of steps and plot the resulting lineshape."
    )
    parser.add_argument("--steps", type=int, required=True, help="Number of integration steps to perform.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to physics/offline_lineshape_<steps>_steps.png.",
    )
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving it.")

    parser.add_argument("--rf-on", dest="rf_on", action="store_true", default=True, help="Enable RF during the offline steps.")
    parser.add_argument("--rf-off", dest="rf_on", action="store_false", help="Disable RF during the offline steps.")
    parser.add_argument("--dnp-on", dest="dnp_on", action="store_true", default=False, help="Enable DNP during the offline steps.")
    parser.add_argument("--dnp-off", dest="dnp_on", action="store_false", help="Disable DNP during the offline steps.")

    parser.add_argument("--rf-burn-R", type=float, default=Spin1Params.rf_burn_R, help="Physical R coordinate of the RF burn bin.")
    parser.add_argument("--gamma-rf", type=float, default=Spin1Params.gamma_rf, help="RF burn rate.")
    parser.add_argument("--dt", type=float, default=Spin1Params.dt, help="Integration timestep.")
    parser.add_argument("--p0", type=float, default=Spin1Params.p0, help="Initial vector polarization.")
    parser.add_argument("--p-dnp-sat", type=float, default=Spin1Params.p_dnp_sat, help="DNP saturation polarization.")
    parser.add_argument("--dnp-rate", type=float, default=Spin1Params.dnp_rate, help="DNP build rate.")
    parser.add_argument("--n-bins", type=int, default=Spin1Params.n_bins, help="Number of R bins in the simulation grid.")
    parser.add_argument("--line-gamma", type=float, default=Spin1Params.line_gamma, help="Pake branch broadening parameter.")
    parser.add_argument("--line-asym", type=float, default=Spin1Params.line_asym, help="Pake branch asymmetry parameter.")
    parser.add_argument("--noise-sigma", type=float, default=0.0, help="Optional Gaussian noise added to the final spectrum.")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> Spin1Model:
    params = Spin1Params(
        n_bins=args.n_bins,
        line_gamma=args.line_gamma,
        line_asym=args.line_asym,
        p0=args.p0,
        rf_burn_R=args.rf_burn_R,
        rf_enabled=args.rf_on,
        gamma_rf=args.gamma_rf,
        dnp_enabled=args.dnp_on,
        p_dnp_sat=args.p_dnp_sat,
        dnp_rate=args.dnp_rate,
        dt=args.dt,
        noise_sigma=args.noise_sigma,
    )
    return Spin1Model(params)


def default_output_path(steps: int) -> Path:
    return Path(__file__).resolve().parent / f"offline_lineshape_{steps}_steps.png"


def plot_lineshape(model: Spin1Model, steps: int, output_path: Path, show: bool = False) -> None:
    R_ref, _Ip_ref, _Im_ref, total_ref = model.reference_spectrum()
    R_final, Ip_final, Im_final, total_final = model.spectrum()
    pol = model.polarizations()
    areas = model.branch_areas()

    fig = plt.figure(figsize=(9.0, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(R_ref, total_ref, color="0.65", linewidth=1.2, label="initial total")
    ax.plot(R_final, Ip_final, color="tab:blue", linewidth=1.1, label="final I+")
    ax.plot(R_final, Im_final, color="tab:orange", linewidth=1.1, label="final I-")
    ax.plot(R_final, total_final, color="black", linewidth=1.5, label="final total")
    ax.axvline(model.params.rf_burn_R, color="tab:red", linestyle="--", linewidth=1.0, label="RF R")
    ax.axvline(-model.params.rf_burn_R, color="tab:red", linestyle=":", linewidth=1.0, label="mirror -R")
    ax.set_xlabel("physical R")
    ax.set_ylabel("intensity density")
    ax.set_title(
        f"v14 offline lineshape after {steps} steps "
        f"(t={model.t:.4g}, RF={'on' if model.params.rf_enabled else 'off'}, DNP={'on' if model.params.dnp_enabled else 'off'})"
    )
    ax.text(
        0.015,
        0.975,
        f"P={pol['P']:.5f}\nQ={pol['Q']:.5f}\nA_total={areas['A_total']:.5g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.steps < 0:
        raise ValueError("--steps must be non-negative")

    model = build_model(args)
    if args.steps > 0:
        model.step(args.steps, rf_on=args.rf_on, dnp_on=args.dnp_on)

    output_path = args.output or default_output_path(args.steps)
    plot_lineshape(model, args.steps, output_path, show=args.show)

    pol = model.polarizations()
    print(f"wrote offline lineshape to {output_path}")
    print(f"steps={args.steps} time={model.t:.6g} P={pol['P']:.6g} Q={pol['Q']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
