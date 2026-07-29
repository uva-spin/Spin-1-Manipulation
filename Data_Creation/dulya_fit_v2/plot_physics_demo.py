"""
Demo plots at a realistic Dulya-fit polarization (default P=0.48).

Shows equilibrium lineshape, physical-Voigt (and optional single-bin) ssRF burn
effects, burn/mirror trajectories, and AFP + relaxation — all driven by the
vendored ``ssrf_realtime_v2`` package with frozen ``fit_params.json``.

Examples (from this directory):
  python plot_physics_demo.py
  python plot_physics_demo.py --p 0.48 --burn-bin 228
  python plot_physics_demo.py --compare-rf-modes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import _bootstrap  # noqa: F401
from afp_bin_traj import run_one_polarization as run_afp
from bin_setup import get_shape_params, print_shape_banner
from common import (
    DEMO_BURN_BIN,
    DEMO_P,
    DIFFUSION_SCALE,
    DT,
    NUM_BINS,
    PLOT_DIR,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE_PHYSICAL_VOIGT,
    RF_MODE_SINGLE_BIN,
    SSRF_GAMMA_RF,
)
from model_bridge import mirror_bin_idx
from ssrf_bin_traj import run_one_polarization as run_ssrf


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}", flush=True)
    return path


def _plot_q_lineshape_panel(
    ax: plt.Axes,
    f: np.ndarray,
    ip0: np.ndarray,
    im0: np.ndarray,
    ip1: np.ndarray,
    im1: np.ndarray,
    *,
    q_initial: float | None = None,
    q_final: float | None = None,
    burn_R: float | None = None,
    mirror_R: float | None = None,
    afp_span: tuple[float, float] | None = None,
) -> None:
    q0 = np.asarray(ip0, dtype=float) - np.asarray(im0, dtype=float)
    q1 = np.asarray(ip1, dtype=float) - np.asarray(im1, dtype=float)
    ax.plot(f, q0, color="tab:purple", alpha=0.35, ls="--", lw=1.0, label=r"$Q$ before")
    ax.plot(f, q1, color="tab:purple", lw=1.4, label=r"$Q$ after")
    ax.axhline(0.0, color="black", ls=":", alpha=0.5, lw=0.8)
    if burn_R is not None:
        ax.axvline(float(burn_R), color="green", ls=":", alpha=0.7)
    if mirror_R is not None:
        ax.axvline(float(mirror_R), color="orange", ls=":", alpha=0.6)
    if afp_span is not None:
        ax.axvspan(float(afp_span[0]), float(afp_span[1]), color="green", alpha=0.12)
    ax.set_xlabel("R")
    ax.set_ylabel(r"$Q = I_+ - I_-$")
    title = "Tensor Q lineshape"
    if q_initial is not None and q_final is not None:
        title += f"  (Q: {q_initial:.4g} → {q_final:.4g})"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_ssrf_burn_and_trajectory(
    *,
    polarization: float,
    burn_bin: int,
    gamma_rf: float,
    max_steps: int,
    rf_mode: str,
    out_dir: Path,
) -> Path:
    traj = run_ssrf(
        int(burn_bin),
        float(polarization),
        gamma_rf=float(gamma_rf),
        n_steps=int(max_steps),
        rf_mode=str(rf_mode),
        capture_spectrum=True,
    )
    if traj["skipped"]:
        raise RuntimeError(f"ssRF trajectory skipped at bin={burn_bin} P={polarization}")

    f = np.asarray(traj["frequency"], dtype=float)
    ip0 = np.asarray(traj["ip_spectrum0"], dtype=float)
    im0 = np.asarray(traj["im_spectrum0"], dtype=float)
    ip1 = np.asarray(traj["ip_spectrum"], dtype=float)
    im1 = np.asarray(traj["im_spectrum"], dtype=float)
    mirror = mirror_bin_idx(len(f), int(burn_bin))
    n = int(traj["n_steps"])
    t = np.arange(n)
    burn_R = float(f[burn_bin])

    fig = plt.figure(figsize=(12.5, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.15, 1.0, 0.75), hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(f, ip0, color="tab:red", alpha=0.35, ls="--", lw=1.0, label=r"$I_+$ before")
    ax0.plot(f, im0, color="tab:blue", alpha=0.35, ls="--", lw=1.0, label=r"$I_-$ before")
    ax0.plot(f, ip1, color="tab:red", lw=1.4, label=r"$I_+$ after ssRF")
    ax0.plot(f, im1, color="tab:blue", lw=1.4, label=r"$I_-$ after ssRF")
    ax0.axvline(burn_R, color="green", ls=":", alpha=0.7, label=f"burn R={burn_R:.3f}")
    ax0.axvline(float(f[mirror]), color="orange", ls=":", alpha=0.6, label=f"mirror bin {mirror}")
    ax0.set_xlabel("R")
    ax0.set_ylabel("intensity (fit scale)")
    ax0.set_title(
        f"Dulya-fit ssRF burn  P={polarization:.2f}  mode={rf_mode}  "
        f"γ={gamma_rf:.2f}  steps={n} ({traj.get('stop_reason', '')})"
    )
    ax0.legend(fontsize=8, ncols=3, loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, traj["iplus"][:n], color="tab:red", lw=1.4, label=r"$I_+$ burn")
    ax1.plot(t, traj["iminus"][:n], color="tab:blue", lw=1.4, label=r"$I_-$ burn")
    ax1.plot(t, traj["ps"][:n], color="black", lw=1.1, ls="--", alpha=0.8, label=r"$P_s$ burn")
    ax1.plot(t, traj["iplus"][:n] - traj["iminus"][:n], color="tab:orange", lw=1.1, ls="--", alpha=0.8, label=r"$Q$")
    ax1.set_xlabel("step")
    ax1.set_ylabel("intensity (fit scale)")
    ax1.set_title(f"Burn bin {burn_bin}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(t, traj["iplus_m"][:n], color="tab:red", lw=1.4, label=r"$I_+$ mirror")
    ax2.plot(t, traj["iminus_m"][:n], color="tab:blue", lw=1.4, label=r"$I_-$ mirror")
    ax2.plot(t, traj["ps_m"][:n], color="black", lw=1.1, ls="--", alpha=0.8, label=r"$P_s$ mirror")
    ax2.plot(t, traj["iplus_m"][:n] - traj["iminus_m"][:n], color="tab:orange", lw=1.1, ls="--", alpha=0.8, label=r"$Q$ mirror")
    ax2.set_xlabel("step")
    ax2.set_ylabel("intensity (fit scale)")
    ax2.set_title(f"Mirror bin {mirror}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2, :])
    _plot_q_lineshape_panel(
        ax3,
        f,
        ip0,
        im0,
        ip1,
        im1,
        q_initial=float(traj["q_initial"]),
        q_final=float(traj["q_final"]),
        burn_R=burn_R,
        mirror_R=float(f[mirror]),
    )

    fig.suptitle(
        f"ssrf_realtime_v2 + Dulya fit_params  |  diffusion={DIFFUSION_SCALE}  "
        f"Gauss={RF_GAUSSIAN_FWHM_R:.3f} Lorentz={RF_LORENTZIAN_FWHM_R:.3f}",
        fontsize=11,
        y=0.995,
    )
    print(
        f"ssRF P/Q: initial P={traj['p_initial']:.6g}  Q={traj['q_initial']:.6g}  "
        f"final P={traj['p_final']:.6g}  Q={traj['q_final']:.6g}",
        flush=True,
    )
    print(
        f"ssRF burn bin final: I+={traj['iplus'][n-1]:.6g}  I-={traj['iminus'][n-1]:.6g}  "
        f"Ps={traj['ps'][n-1]:.6g}",
        flush=True,
    )
    return _save(fig, out_dir / f"dulya_v2_ssrf_{rf_mode}_P{polarization:.2f}_bin{burn_bin:04d}.png")


def plot_afp_and_relaxation(
    *,
    polarization: float,
    burn_bin: int,
    n_relax: int,
    out_dir: Path,
) -> Path:
    traj = run_afp(
        int(burn_bin),
        float(polarization),
        n_relax=int(n_relax),
        capture_spectrum=True,
    )
    f = np.asarray(traj["frequency"], dtype=float)
    ip0 = np.asarray(traj["ip_spectrum0"], dtype=float)
    im0 = np.asarray(traj["im_spectrum0"], dtype=float)
    ip1 = np.asarray(traj["ip_spectrum"], dtype=float)
    im1 = np.asarray(traj["im_spectrum"], dtype=float)
    mirror = mirror_bin_idx(len(f), int(burn_bin))
    n = int(traj["n_steps"])
    t = np.arange(n) * float(DT)
    subset = list(traj["afp_subset"])

    fig = plt.figure(figsize=(12.5, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.15, 1.0, 0.75), hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(f, ip0, color="tab:red", alpha=0.35, ls="--", lw=1.0, label=r"$I_+$ before AFP")
    ax0.plot(f, im0, color="tab:blue", alpha=0.35, ls="--", lw=1.0, label=r"$I_-$ before AFP")
    ax0.plot(f, ip1, color="tab:red", lw=1.4, label=r"$I_+$ after AFP+relax")
    ax0.plot(f, im1, color="tab:blue", lw=1.4, label=r"$I_-$ after AFP+relax")
    ax0.axvspan(float(f[subset[0]]), float(f[subset[-1]]), color="green", alpha=0.12, label="AFP window")
    ax0.axvline(float(f[burn_bin]), color="green", ls=":", alpha=0.7)
    ax0.axvline(float(f[mirror]), color="orange", ls=":", alpha=0.6)
    ax0.set_xlabel("R")
    ax0.set_ylabel("intensity (fit scale)")
    ax0.set_title(
        f"Dulya-fit AFP  P={polarization:.2f}  center={burn_bin}  "
        f"window={subset[0]}–{subset[-1]}  n_relax={n_relax}"
    )
    ax0.legend(fontsize=8, ncols=3, loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, traj["iplus"][:n], color="tab:red", lw=1.4, label=r"$I_+$ center")
    ax1.plot(t, traj["iminus"][:n], color="tab:blue", lw=1.4, label=r"$I_-$ center")
    ax1.plot(t, traj["ps"][:n], color="black", lw=1.1, ls="--", alpha=0.8, label=r"$P_s$ center")
    ax1.plot(t, traj["iplus"][:n] - traj["iminus"][:n], color="tab:orange", lw=1.1, ls="--", alpha=0.8, label=r"$Q$")
    ax1.set_xlabel("time [arb.]")
    ax1.set_ylabel("intensity (fit scale)")
    ax1.set_title(f"Center bin {burn_bin} during relaxation")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(t, traj["iplus_m"][:n], color="tab:red", lw=1.4, label=r"$I_+$ mirror")
    ax2.plot(t, traj["iminus_m"][:n], color="tab:blue", lw=1.4, label=r"$I_-$ mirror")
    ax2.plot(t, traj["ps_m"][:n], color="black", lw=1.1, ls="--", alpha=0.8, label=r"$P_s$ mirror")
    ax2.set_xlabel("time [arb.]")
    ax2.set_ylabel("intensity (fit scale)")
    ax2.set_title(f"Mirror bin {mirror} during relaxation")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2, :])
    _plot_q_lineshape_panel(
        ax3,
        f,
        ip0,
        im0,
        ip1,
        im1,
        q_initial=float(traj["q_initial"]),
        q_final=float(traj["q_final"]),
        burn_R=float(f[burn_bin]),
        mirror_R=float(f[mirror]),
        afp_span=(float(f[subset[0]]), float(f[subset[-1]])),
    )

    fig.suptitle(
        r"ssrf_realtime_v2 AFP + Boltzmann recovery at $P_{\mathrm{AFP}}$ ($Q\to Q_B(P)$)",
        fontsize=11,
        y=0.995,
    )
    print(
        f"AFP P/Q: initial P={traj['p_initial']:.6g}  Q={traj['q_initial']:.6g}  "
        f"final P={traj['p_final']:.6g}  Q={traj['q_final']:.6g}",
        flush=True,
    )
    print(
        f"AFP center final: I+={traj['iplus'][n-1]:.6g}  I-={traj['iminus'][n-1]:.6g}  "
        f"Ps={traj['ps'][n-1]:.6g}",
        flush=True,
    )
    return _save(fig, out_dir / f"dulya_v2_afp_P{polarization:.2f}_bin{burn_bin:04d}.png")


def plot_rf_mode_compare(
    *,
    polarization: float,
    burn_bin: int,
    gamma_rf: float,
    max_steps: int,
    out_dir: Path,
) -> Path:
    voigt = run_ssrf(
        int(burn_bin),
        float(polarization),
        gamma_rf=float(gamma_rf),
        max_steps=int(max_steps),
        rf_mode=RF_MODE_PHYSICAL_VOIGT,
        capture_spectrum=True,
    )
    single = run_ssrf(
        int(burn_bin),
        float(polarization),
        gamma_rf=float(gamma_rf),
        max_steps=int(max_steps),
        rf_mode=RF_MODE_SINGLE_BIN,
        capture_spectrum=True,
    )
    f = np.asarray(voigt["frequency"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), layout="constrained")
    for col, run, title in (
        (0, single, "single-bin RF"),
        (1, voigt, "physical-R Voigt RF"),
    ):
        ax_ip = axes[0, col]
        ax_ip.plot(f, run["ip_spectrum0"], color="tab:red", alpha=0.3, ls="--")
        ax_ip.plot(f, run["im_spectrum0"], color="tab:blue", alpha=0.3, ls="--")
        ax_ip.plot(f, run["ip_spectrum"], color="tab:red", lw=1.3, label=r"$I_+$ after")
        ax_ip.plot(f, run["im_spectrum"], color="tab:blue", lw=1.3, label=r"$I_-$ after")
        ax_ip.axvline(float(f[burn_bin]), color="green", ls=":", alpha=0.6)
        ax_ip.set_xlabel("R")
        ax_ip.set_ylabel("intensity (fit scale)")
        ax_ip.set_title(f"{title}  steps={run['n_steps']}")
        ax_ip.legend(fontsize=8)
        ax_ip.grid(True, alpha=0.3)

        ax_q = axes[1, col]
        _plot_q_lineshape_panel(
            ax_q,
            f,
            run["ip_spectrum0"],
            run["im_spectrum0"],
            run["ip_spectrum"],
            run["im_spectrum"],
            q_initial=float(run["q_initial"]),
            q_final=float(run["q_final"]),
            burn_R=float(f[burn_bin]),
        )
    fig.suptitle(
        f"Same Dulya equilibrium + recovery  |  P={polarization:.2f}  γ={gamma_rf}",
        fontsize=11,
    )
    return _save(fig, out_dir / f"dulya_v2_ssrf_mode_compare_P{polarization:.2f}_bin{burn_bin:04d}.png")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Dulya-fit v2 physics demo plots at one polarization")
    p.add_argument("--p", type=float, default=DEMO_P, dest="polarization")
    p.add_argument("--burn-bin", type=int, default=220)
    p.add_argument("--gamma-rf", type=float, default=SSRF_GAMMA_RF)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--n-relax", type=int, default=2)
    p.add_argument(
        "--rf-mode",
        choices=(RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN),
        default=RF_MODE_PHYSICAL_VOIGT,
    )
    p.add_argument(
        "--compare-rf-modes",
        action="store_true",
        help="Also plot single-bin vs physical Voigt side-by-side",
    )
    p.add_argument("--output-dir", type=Path, default=PLOT_DIR)
    args = p.parse_args(argv)

    shape = get_shape_params()
    print_shape_banner(shape, num_bins=NUM_BINS)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_ssrf_burn_and_trajectory(
        polarization=float(args.polarization),
        burn_bin=int(args.burn_bin),
        gamma_rf=float(args.gamma_rf),
        max_steps=int(args.max_steps),
        rf_mode=str(args.rf_mode),
        out_dir=out_dir,
    )
    plot_afp_and_relaxation(
        polarization=float(args.polarization),
        burn_bin=int(args.burn_bin),
        n_relax=int(args.n_relax),
        out_dir=out_dir,
    )
    if args.compare_rf_modes:
        plot_rf_mode_compare(
            polarization=float(args.polarization),
            burn_bin=int(args.burn_bin),
            gamma_rf=float(args.gamma_rf),
            max_steps=int(args.max_steps),
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
