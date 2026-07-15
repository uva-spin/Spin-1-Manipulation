"""
Generate a sample ss-RF burn event and save a before/after lineshape plot.

Uses ``physics.burn_lookup_realtime`` and ``ssrf_realtime.rate_equations_realtime``
to start from an equilibrium lineshape, apply repeated single-bin RF burns, and
write a PNG comparing unburned vs burned Ps, I+, and I-.

Run from anywhere:
  python physics/ssrf_realtime/tests/plot_sample_burn_event.py

Or from ``physics/ssrf_realtime``:
  python tests/plot_sample_burn_event.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TESTS_DIR = Path(__file__).resolve().parent
SSRF_DIR = TESTS_DIR.parent
REPO_ROOT = SSRF_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.burn_lookup_realtime import BurnTrajectoryConfig, initial_lineshape  
from physics.ssrf_realtime.rate_equations_realtime import ( 
    burn_preserves_ps_sign,
    solve_rate_equations,
)

DEFAULT_OUTPUT = TESTS_DIR / "output" / "sample_burn_event_01_rf_4000_steps_dt_0015.png"
DEFAULT_TRAJECTORY_OUTPUT = TESTS_DIR / "output" / "sample_burn_event_01_rf_4000_steps_dt_0015_trajectory.png"
DEFAULT_EVENT = TESTS_DIR / "output" / "sample_burn_event_01_rf_4000_steps_dt_0015.npz"

DT_RF = 0.0015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a sample ss-RF burn event.")
    parser.add_argument("--polarization", type=float, default=0.45, help="Initial vector polarization P")
    parser.add_argument("--burn-r", type=float, default=-0.92, help="Burn frequency R (MHz)")
    parser.add_argument("--gamma-rf", type=float, default=0.2, help="RF equalization rate")
    parser.add_argument("--burn-steps", type=int, default=100, help="Number of RF integration steps")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument(
        "--event-output",
        type=Path,
        default=DEFAULT_EVENT,
        help="Optional NPZ path for the event arrays/metadata",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    return parser.parse_args()


def apply_burn_event(
    polarization: float,
    burn_r: float,
    *,
    gamma_rf: float,
    burn_steps: int,
    config: BurnTrajectoryConfig | None = None,
) -> dict:
    """Build equilibrium lineshape, apply RF burns, return event dict."""
    cfg = config or BurnTrajectoryConfig(gamma_rf=gamma_rf, dt=DT_RF)
    f, iplus, iminus, ps = initial_lineshape(polarization, config=cfg)

    burn_bin_idx = int(np.argmin(np.abs(f - float(burn_r))))
    burn_freq = float(f[burn_bin_idx])
    mirror_bin_idx = len(f) - 1 - burn_bin_idx

    params = cfg.spin1_params(polarization)
    params.gamma_rf = float(gamma_rf)

    iplus_cur = iplus.copy()
    iminus_cur = iminus.copy()
    steps_applied = 0
    iplus_trace = [float(iplus_cur[burn_bin_idx])]
    iminus_trace = [float(iminus_cur[burn_bin_idx])]

    for _ in range(max(1, int(burn_steps))):
        iplus_new, iminus_new, _, _, _ = solve_rate_equations(
            iplus_cur,
            iminus_cur,
            cfg.dt,
            gamma_rf,
            burn_bin_idx,
            params=params,
            initial_polarization=polarization,
            rf_only=True,
        )
        iplus_new = np.asarray(iplus_new, dtype=float)
        iminus_new = np.asarray(iminus_new, dtype=float)
        if not burn_preserves_ps_sign(iplus_cur, iminus_cur, iplus_new, iminus_new, burn_bin_idx):
            break
        ps_before = float(iplus_cur[burn_bin_idx] + iminus_cur[burn_bin_idx])
        ps_after = float(iplus_new[burn_bin_idx] + iminus_new[burn_bin_idx])
        if ps_after == ps_before:
            break
        iplus_cur = iplus_new
        iminus_cur = iminus_new
        steps_applied += 1
        iplus_trace.append(float(iplus_cur[burn_bin_idx]))
        iminus_trace.append(float(iminus_cur[burn_bin_idx]))

    ps_after_full = iplus_cur + iminus_cur

    return {
        "polarization": float(polarization),
        "burn_bin_idx": burn_bin_idx,
        "mirror_bin_idx": mirror_bin_idx,
        "burn_freq": burn_freq,
        "gamma_rf": float(gamma_rf),
        "dt": float(cfg.dt),
        "steps_applied": steps_applied,
        "f_mhz": f,
        "iplus_before": iplus,
        "iminus_before": iminus,
        "ps_before": ps,
        "iplus_after": iplus_cur,
        "iminus_after": iminus_cur,
        "ps_after": ps_after_full,
        "ps_at_burn_before": float(ps[burn_bin_idx]),
        "ps_at_burn_after": float(ps_after_full[burn_bin_idx]),
        "iplus_trace": np.asarray(iplus_trace, dtype=float),
        "iminus_trace": np.asarray(iminus_trace, dtype=float),
    }


def plot_burn_event(event: dict, output_path: Path, *, dpi: int = 200, show: bool = False) -> None:
    """Save a two-panel before/after burn plot."""
    f = event["f_mhz"]
    burn_freq = event["burn_freq"]
    burn_step = event["steps_applied"]

    fig, axes = plt.subplots(
        1,
        1,
        figsize=(7.4, 3.8),
        sharex=True,
        gridspec_kw={"wspace": 0.28},
        layout="constrained",
    )
    ax_sig = axes

    ax_sig.plot(f, event["iplus_before"], color="tab:red", alpha=0.45, linestyle="--", linewidth=1.0, label=r"$I_+$ unburned")
    ax_sig.plot(f, event["iplus_after"], color="tab:red", linestyle="-", linewidth=1.25, alpha=0.4, label=rf"$I_+$ after {burn_step} steps")
    ax_sig.plot(f, event["iminus_before"], color="tab:blue", alpha=0.45, linestyle="--", linewidth=1.0, label=r"$I_-$ unburned")
    ax_sig.plot(f, event["iminus_after"], color="tab:blue", linestyle="-", linewidth=1.25, alpha=0.4, label=rf"$I_-$ after {burn_step} steps")
    ax_sig.axvline(burn_freq, color="green", alpha=0.55, linestyle=":", linewidth=1.2, label="burn freq")
    ax_sig.set_xlabel("frequency (MHz)")
    ax_sig.grid(True, alpha=0.3)
    ax_sig.legend(loc="upper right", fontsize=8)
    ax_sig.set_ylabel(r"$I_\pm$")
    ax_sig.set_title("Iplus / Iminus lineshape", fontsize=10)

    fig.suptitle(
        f"Sample ss-RF burn event  |  "
        f"Ps(burn): {event['ps_at_burn_before']:.4g} -> {event['ps_at_burn_after']:.4g}  |  "
        f"gamma_rf = {event['gamma_rf']:.2f}",
        fontsize=12,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_burn_trajectory(event: dict, output_path: Path, *, dpi: int = 200, show: bool = False) -> None:
    """Plot I+ and I- at the burn bin as they decrease over RF steps."""
    steps = np.arange(len(event["iplus_trace"]))
    burn_freq = event["burn_freq"]

    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    ax.plot(steps, event["iplus_trace"], color="tab:red", linewidth=1.5, label=r"$I_+$ at burn bin")
    ax.plot(steps, event["iminus_trace"], color="tab:blue", linewidth=1.5, label=r"$I_-$ at burn bin")
    ax.plot(
        steps,
        event["iplus_trace"] + event["iminus_trace"],
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.7,
        label=r"$P_s = I_+ + I_-$",
    )

    ax.set_xlabel("RF integration step")
    ax.set_ylabel(r"intensity at burn bin")
    ax.set_title(
        f"$I_\\pm$ burn-down at f = {burn_freq:.3f} MHz  |  "
        f"{event['steps_applied']} steps applied",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(
        f"Sample ss-RF burn trajectory  |  P = {event['polarization']:.3f}  |  "
        f"gamma_rf = {event['gamma_rf']:.2f}",
        fontsize=12,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved trajectory plot: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def save_event(event: dict, output_path: Path) -> None:
    """Persist arrays and scalar metadata for the burn event."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        f_mhz=event["f_mhz"],
        iplus_before=event["iplus_before"],
        iminus_before=event["iminus_before"],
        ps_before=event["ps_before"],
        iplus_after=event["iplus_after"],
        iminus_after=event["iminus_after"],
        ps_after=event["ps_after"],
        polarization=event["polarization"],
        burn_bin_idx=event["burn_bin_idx"],
        mirror_bin_idx=event["mirror_bin_idx"],
        burn_freq=event["burn_freq"],
        gamma_rf=event["gamma_rf"],
        dt=event["dt"],
        steps_applied=event["steps_applied"],
        ps_at_burn_before=event["ps_at_burn_before"],
        ps_at_burn_after=event["ps_at_burn_after"],
        iplus_trace=event["iplus_trace"],
        iminus_trace=event["iminus_trace"],
    )
    print(f"Saved event: {output_path}")


def main() -> None:
    args = parse_args()
    event = apply_burn_event(
        args.polarization,
        args.burn_r,
        gamma_rf=args.gamma_rf,
        burn_steps=args.burn_steps,
    )
    print(
        f"Burn at f={event['burn_freq']:.4f} MHz (bin {event['burn_bin_idx']}): "
        f"Ps {event['ps_at_burn_before']:.6e} -> {event['ps_at_burn_after']:.6e} "
        f"in {event['steps_applied']} steps"
    )
    plot_burn_event(event, args.output, dpi=args.dpi, show=args.show)
    plot_burn_trajectory(event, DEFAULT_TRAJECTORY_OUTPUT, dpi=args.dpi, show=args.show)
    if args.event_output is not None:
        save_event(event, args.event_output)


if __name__ == "__main__":
    main()
