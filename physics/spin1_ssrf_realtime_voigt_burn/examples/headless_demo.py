"""Headless validation examples for the Voigt-profile ss-RF simulator."""

from __future__ import annotations

from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from ssrf_realtime.model import Spin1Model, Spin1Params


OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
REFERENCE = ROOT / "tests" / "data" / "previous_working_dynamics.npz"


def zero_width_regression() -> None:
    """Verify that zero Voigt widths reproduce the prior one-bin RF dynamics."""
    ref = np.load(REFERENCE)
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=0.4,
        gamma_rf=2.0,
        rf_gaussian_fwhm_R=0.0,
        rf_lorentzian_fwhm_R=0.0,
        diffusion_scale=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
        dt=0.0015,
        capacity_rate_power=1.0,
    )
    m = Spin1Model(p)
    for _ in range(1000):
        m.step(rf_on=True, dnp_on=False)

    R, Ip_new, Im_new, total_new = m.spectrum()
    _, Ip_ref, Im_ref, total_ref = m.spectrum_from_state(ref["rf_after"])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.8, 5.3), constrained_layout=True)
    ax0.plot(R, Ip_ref, label="previous I+")
    ax0.plot(R, Ip_new, linestyle="--", label="Voigt package I+, widths=0")
    ax0.plot(R, Im_ref, label="previous I-")
    ax0.plot(R, Im_new, linestyle="--", label="Voigt package I-, widths=0")
    ax0.set_title("Zero-width regression: exact recovery of the one-bin RF model")
    ax0.set_xlabel("physical R")
    ax0.set_ylabel("intensity [arb.]")
    ax0.legend(fontsize=8, ncols=2)

    ax1.plot(R, total_new - total_ref, label="updated - previous")
    ax1.set_xlabel("physical R")
    ax1.set_ylabel("difference")
    ax1.set_title(f"maximum absolute difference = {np.max(np.abs(total_new-total_ref)):.3e}")
    ax1.legend()
    fig.savefig(OUT / "voigt_zero_width_regression.png", dpi=180)
    plt.close(fig)


def profile_examples() -> None:
    """Plot the finite-bin rate fields for Gaussian, Lorentzian, and Voigt cases."""
    cases = [
        (0.08, 0.00, "Gaussian only"),
        (0.00, 0.08, "Lorentzian only"),
        (0.08, 0.04, "Voigt"),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    for gf, lf, label in cases:
        m = Spin1Model(
            Spin1Params(
                rf_burn_R=0.35,
                rf_gaussian_fwhm_R=gf,
                rf_lorentzian_fwhm_R=lf,
                rf_profile_normalization="center_bin",
            )
        )
        R, profile = m.rf_profile_physical()
        ax.plot(R, profile, drawstyle="steps-mid", label=label)
    ax.axvline(0.35, linestyle="--", linewidth=1.0, label="RF center")
    ax.set_xlim(0.0, 0.7)
    ax.set_xlabel("physical R")
    ax.set_ylabel("normalized bin-averaged RF rate")
    ax.set_title("One adjustable RF rate profile, integrated over finite R bins")
    ax.legend()
    fig.savefig(OUT / "voigt_rf_rate_profiles.png", dpi=180)
    plt.close(fig)


def spectral_evolution() -> None:
    """Show that the imposed rate is Voigt while the finite-time hole evolves nonlinearly."""
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=0.40,
        gamma_rf=5.0,
        rf_gaussian_fwhm_R=0.10,
        rf_lorentzian_fwhm_R=0.035,
        rf_profile_normalization="center_bin",
        capacity_rate_power=0.0,
        diffusion_scale=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
        dt=2.5e-4,
    )
    m = Spin1Model(p)
    snapshots: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    targets = [0.0, 0.03, 0.12, 0.45]
    next_target = 0
    while m.t <= targets[-1] + 0.5 * p.dt:
        if next_target < len(targets) and m.t >= targets[next_target] - 0.5 * p.dt:
            R, Ip, Im, total = m.spectrum()
            snapshots.append((m.t, Ip.copy(), Im.copy(), total.copy()))
            next_target += 1
        if m.t < targets[-1]:
            m.step(rf_on=True, dnp_on=False)
        else:
            break

    Rref, Ipref, Imref, totalref = m.reference_spectrum()
    Rprof, profile = m.rf_profile_physical()
    scale = 0.34 * float(np.max(totalref))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.0, 6.0), constrained_layout=True)
    ax0.plot(Rref, totalref, linewidth=1.2, label="initial total")
    for t, _, _, total in snapshots[1:]:
        ax0.plot(Rref, total, label=f"t={t:.2f}")
    ax0.plot(Rprof, scale * profile, linestyle="--", label="RF rate profile (scaled)")
    ax0.axvline(p.rf_burn_R, linestyle=":", linewidth=1.0)
    ax0.set_xlim(0.05, 0.75)
    ax0.set_xlabel("physical R")
    ax0.set_ylabel("total intensity [arb.]")
    ax0.set_title("Finite-power burn: center saturates before the Voigt wings")
    ax0.legend(fontsize=8, ncols=3)

    for t, Ip, _, _ in snapshots[1:]:
        ax1.plot(Rref, Ip - Ipref, label=f"Delta I+, t={t:.2f}")
    ax1.axhline(0.0, linewidth=0.8)
    ax1.set_xlim(0.05, 0.75)
    ax1.set_xlabel("physical R")
    ax1.set_ylabel("change from initial I+")
    ax1.set_title("The resulting hole is calculated from populations, not imposed as a fixed Voigt")
    ax1.legend(fontsize=8)
    fig.savefig(OUT / "voigt_burn_spectral_evolution.png", dpi=180)
    plt.close(fig)


def center_wing_dynamics() -> Path:
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=0.35,
        gamma_rf=3.0,
        rf_gaussian_fwhm_R=0.12,
        rf_lorentzian_fwhm_R=0.04,
        rf_profile_normalization="center_bin",
        capacity_rate_power=0.0,
        diffusion_scale=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
        dt=5e-4,
    )
    m = Spin1Model(p)
    offsets = [0.0, 0.06, 0.14, 0.28]
    indices = [int(np.argmin(np.abs(m.Rplus - (p.rf_burn_R + d)))) for d in offsets]
    initial = m.packet_intensities(density=False)[0][indices].copy()

    tvals: list[float] = []
    curves = [[] for _ in offsets]
    Pvals: list[float] = []
    for step in range(1201):
        if step % 4 == 0:
            current = m.packet_intensities(density=False)[0][indices]
            tvals.append(m.t)
            for j, value in enumerate(current):
                curves[j].append(float(value / initial[j]))
            Pvals.append(float(m.polarizations()["P"]))
        if step < 1200:
            m.step(rf_on=True, dnp_on=False)

    csv_path = OUT / "voigt_center_wing_dynamics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time"] + [f"Iplus_norm_offset_{d:.3f}" for d in offsets] + ["P"])
        writer.writerows(zip(tvals, *curves, Pvals))

    fig, ax = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    for offset, values in zip(offsets, curves):
        ax.plot(tvals, values, label=f"R-Rb={offset:+.2f}")
    ax.set_xlabel("time [arb.]")
    ax.set_ylabel("I+(R,t) / I+(R,0)")
    ax.set_title("Voigt-rate burn dynamics: center and wings saturate at different rates")
    ax.legend()
    fig.savefig(OUT / "voigt_center_wing_dynamics.png", dpi=180)
    plt.close(fig)
    return csv_path


def full_ode_direct_mirror() -> None:
    """Demonstrate direct and mirror changes from the full ODE for a broad profile."""
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=0.18,
        gamma_rf=2.5,
        rf_gaussian_fwhm_R=0.75,
        rf_lorentzian_fwhm_R=0.30,
        rf_profile_normalization="center_bin",
        capacity_rate_power=0.0,
        diffusion_scale=10.0,
        zq_width_R=0.05,
        dnp_enabled=True,
        dnp_rate=0.04,
        p_dnp_sat=0.58,
        t1_rate=0.0,
        dt=2.5e-4,
    )
    m = Spin1Model(p)
    tvals: list[float] = []
    dIp: list[float] = []
    dIm: list[float] = []
    dIp_m: list[float] = []
    dIm_m: list[float] = []
    ratio: list[float] = []
    for step in range(1801):
        if step % 5 == 0:
            v = m.response_values()
            tvals.append(m.t)
            dIp.append(v["dIplus_R"])
            dIm.append(v["dIminus_R"])
            dIp_m.append(v["dIplus_minusR"])
            dIm_m.append(v["dIminus_minusR"])
            denom = -v["dIplus_R"]
            ratio.append(np.nan if abs(denom) < 1e-15 else v["dIminus_minusR"] / denom)
        if step < 1800:
            m.step(rf_on=True, dnp_on=True)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.8, 5.7), constrained_layout=True)
    ax0.plot(tvals, dIp, label="Delta I+(+R), direct")
    ax0.plot(tvals, dIm, label="Delta I-(+R), direct")
    ax0.plot(tvals, dIp_m, linestyle="--", label="Delta I+(-R), mirror")
    ax0.plot(tvals, dIm_m, linestyle="--", label="Delta I-(-R), mirror")
    ax0.axhline(0.0, linewidth=0.8)
    ax0.set_ylabel("change from initial intensity")
    ax0.set_title("Broad-profile direct and mirror response from the coupled population ODE")
    ax0.legend(fontsize=8, ncols=2)

    ax1.plot(tvals, ratio, label="Delta I-(-R) / [-Delta I+(+R)]")
    ax1.axhline(0.5, linestyle="--", label="isolated narrow-RF limit")
    ax1.set_xlabel("time [arb.]")
    ax1.set_ylabel("response ratio")
    ax1.set_title("No mirror ratio is imposed; RF wings, diffusion, and DNP determine it dynamically")
    ax1.legend(fontsize=8)
    fig.savefig(OUT / "voigt_direct_mirror_full_ode.png", dpi=180)
    plt.close(fig)


def no_dnp_burn_recovery() -> Path:
    p = Spin1Params(
        p0=0.45,
        rf_burn_R=0.4,
        gamma_rf=6.0,
        rf_gaussian_fwhm_R=0.055,
        rf_lorentzian_fwhm_R=0.020,
        diffusion_scale=50.0,
        zq_width_R=0.05,
        cross_branch_ratio=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
        dt=5e-4,
    )
    m = Spin1Model(p)
    burn_end = 0.125
    total_time = 1.75

    t_out: list[float] = []
    Ip_out: list[float] = []
    Im_out: list[float] = []
    P_out: list[float] = []
    rf_out: list[int] = []

    n_steps = int(total_time / p.dt)
    for step in range(n_steps + 1):
        if step % 5 == 0:
            loc = m.local_intensities()
            t_out.append(m.t)
            Ip_out.append(loc["Iplus"])
            Im_out.append(loc["Iminus"])
            P_out.append(m.polarizations()["P"])
            rf_out.append(1 if m.t < burn_end else 0)
        if step < n_steps:
            m.step(rf_on=m.t < burn_end, dnp_on=False)

    csv_path = OUT / "voigt_no_dnp_burn_recovery.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "Iplus_R", "Iminus_R", "P", "rf_on"])
        writer.writerows(zip(t_out, Ip_out, Im_out, P_out, rf_out))

    R, Ip, Im, total = m.spectrum()
    Rprof, profile = m.rf_profile_physical()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.0, 5.8), constrained_layout=True)
    ax0.plot(R, Ip, drawstyle="steps-mid", label="I+(R)")
    ax0.plot(R, Im, drawstyle="steps-mid", label="I-(R)")
    ax0.plot(R, total, label="total")
    ax0.plot(Rprof, 0.30 * np.max(total) * profile, linestyle="--", label="RF rate profile (scaled)")
    ax0.axvline(p.rf_burn_R, linestyle=":", linewidth=1.0)
    ax0.set_xlabel("physical R")
    ax0.set_ylabel("intensity [arb.]")
    ax0.set_title("Voigt burn followed by population-dependent recovery with DNP off")
    ax0.legend(fontsize=8, ncols=3)

    ax1.plot(t_out, Ip_out, label="I+(R_RF,t)")
    ax1.plot(t_out, Im_out, label="I-(R_RF,t)")
    ax1.axvspan(0.0, burn_end, alpha=0.12, label="RF on")
    ax1.set_xlabel("time [arb.]")
    ax1.set_ylabel("burn-center intensity")
    ax1.set_title(f"RF lowers P from {P_out[0]:.5f} to {min(P_out):.5f}; diffusion redistributes the reduced state")
    ax1.legend()
    fig.savefig(OUT / "voigt_no_dnp_burn_recovery.png", dpi=180)
    plt.close(fig)
    return csv_path


def grid_convergence() -> None:
    values = []
    for n_bins in (351, 701, 1401):
        m = Spin1Model(
            Spin1Params(
                n_bins=n_bins,
                rf_burn_R=0.31,
                rf_gaussian_fwhm_R=0.10,
                rf_lorentzian_fwhm_R=0.05,
                rf_profile_normalization="continuous_peak",
            )
        )
        R, profile = m.rf_profile_physical()
        values.append((n_bins, R, profile, float(np.sum(profile) * m.dR)))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.8, 5.4), constrained_layout=True)
    for n_bins, R, profile, area in values:
        ax0.plot(R, profile, drawstyle="steps-mid", label=f"{n_bins} bins")
        ax1.scatter([n_bins], [area], label=f"{n_bins}: {area:.6f}")
    ax0.set_xlim(0.0, 0.65)
    ax0.set_xlabel("physical R")
    ax0.set_ylabel("continuous-peak profile")
    ax0.set_title("Finite-bin Voigt integration converges as the R grid is refined")
    ax0.legend()
    ax1.set_xlabel("number of bins")
    ax1.set_ylabel("integrated profile width [R]")
    ax1.legend(fontsize=8)
    fig.savefig(OUT / "voigt_grid_convergence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    zero_width_regression()
    profile_examples()
    spectral_evolution()
    center_wing_dynamics()
    full_ode_direct_mirror()
    no_dnp_burn_recovery()
    grid_convergence()
    print(f"Wrote Voigt-profile diagnostics to {OUT}")
