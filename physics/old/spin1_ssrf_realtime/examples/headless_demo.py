"""Headless examples for the v11 ss-RF real-time simulation model."""

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
from ssrf_realtime.lineshape import plot_signal_reference


OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def set_bar(ax, x: float, y0: float, y1: float, **kwargs):
    if np.isfinite(y0) and np.isfinite(y1):
        ax.plot([x, x], [y0, y1], **kwargs)


def run_no_dnp_burn_recovery():
    params = Spin1Params(
        rf_burn_R=-0.90,
        p0=0.45,
        gamma_rf=3.0,
        line_gamma=0.05,
        line_asym=0.04,
        d_same_plus0=0.20,
        d_same_0minus=0.12,
        d_spec_plus0=1.8,
        d_spec_0minus=1.0,
        dnp_enabled=False,
        t2_width_R=0.055,
        dt=0.0015,
    )
    model = Spin1Model(params)
    Rb = params.rf_burn_R

    times: list[float] = []
    Ip_trace: list[float] = []
    Im_trace: list[float] = []
    P_trace: list[float] = []
    Q_trace: list[float] = []
    rf_state: list[int] = []
    dnp_state: list[int] = []

    def record():
        loc = model.local_intensities(Rb)
        pol = model.polarizations()
        times.append(model.t)
        Ip_trace.append(loc["Iplus"])
        Im_trace.append(loc["Iminus"])
        P_trace.append(pol["P"])
        Q_trace.append(pol["Q"])
        rf_state.append(1 if params.rf_enabled else 0)
        dnp_state.append(1 if params.dnp_enabled else 0)

    record()
    n_total = int(8.0 / params.dt)
    for i in range(n_total):
        t = model.t
        params.rf_enabled = 1.0 <= t < 3.0
        params.dnp_enabled = False
        model.step(1)
        if i % 8 == 0:
            record()

    csv_path = OUT / "v11_no_dnp_burn_recovery.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "Iplus_at_R", "Iminus_at_R", "P", "Q", "rf_on", "dnp_on"])
        writer.writerows(zip(times, Ip_trace, Im_trace, P_trace, Q_trace, rf_state, dnp_state))

    R, Ip, Im, total = model.spectrum()
    vals = model.response_values(Rb)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6.2), constrained_layout=True)
    ax0.plot(R, Ip, drawstyle="steps-mid", label="I+(R)")
    ax0.plot(R, Im, drawstyle="steps-mid", label="I-(R)")
    ax0.plot(R, total, linewidth=1.3, label="total")
    ax0.axvline(Rb, linestyle="--", linewidth=1.1, label="RF bin")
    ax0.axvline(-Rb, linestyle=":", linewidth=1.1, label="mirror")
    ax0.plot([Rb], [vals["Iplus_R"]], marker="o", linestyle="None", label="I+ direct")
    ax0.plot([Rb], [vals["Iminus_R"]], marker="s", linestyle="None", label="I- direct")
    ax0.plot([-Rb], [vals["Iplus_minusR"]], marker="^", linestyle="None", label="I+ mirror")
    ax0.plot([-Rb], [vals["Iminus_minusR"]], marker="v", linestyle="None", label="I- mirror")
    set_bar(ax0, Rb, vals["Iplus_R_ref"], vals["Iplus_R"], linewidth=5, solid_capstyle="round")
    set_bar(ax0, Rb, vals["Iminus_R_ref"], vals["Iminus_R"], linewidth=5, solid_capstyle="round")
    set_bar(ax0, -Rb, vals["Iplus_minusR_ref"], vals["Iplus_minusR"], linewidth=5, solid_capstyle="round")
    set_bar(ax0, -Rb, vals["Iminus_minusR_ref"], vals["Iminus_minusR"], linewidth=5, solid_capstyle="round")
    ax0.set_title("v11: DNP off, RF removes area and diffusion redistributes what remains")
    ax0.set_xlabel("physical R")
    ax0.set_ylabel("intensity [arb.]")
    ax0.set_xlim(-3.05, 3.05)
    ax0.legend(loc="upper left", ncols=4, fontsize=8)

    ax1.plot(times, Ip_trace, label="I+(R_RF,t)")
    ax1.plot(times, Im_trace, label="I-(R_RF,t)")
    ax1.axvspan(1.0, 3.0, alpha=0.12, label="RF on interval")
    ax1.set_title(f"Burn-location intensities from selected initial values [R={Rb:.2f}, DNP off]")
    ax1.set_xlabel("time since this R was selected [arb.]")
    ax1.set_ylabel("intensity at RF bin [arb.]")
    ax1.legend(loc="best")
    fig.savefig(OUT / "v11_two_dynamic_plots_no_dnp.png", dpi=180)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    ax.plot(times, P_trace, label="P(t)")
    ax.plot(times, Q_trace, label="Q(t)")
    ax.axvspan(1.0, 3.0, alpha=0.12, label="RF on interval")
    ax.set_title("DNP off: RF lowers total vector polarization; recovery conserves the reduced P")
    ax.set_xlabel("time [arb.]")
    ax.set_ylabel("dimensionless polarization")
    ax.legend(loc="best")
    fig2.savefig(OUT / "v11_no_dnp_total_area_loss.png", dpi=180)
    plt.close(fig2)

    return csv_path


def run_dnp_build_demo():
    params = Spin1Params(
        p0=0.10,
        p_dnp_sat=0.58,
        dnp_enabled=True,
        dnp_rate=0.65,
        rf_enabled=False,
        dt=0.0015,
    )
    model = Spin1Model(params)
    times: list[float] = []
    P_trace: list[float] = []
    Q_trace: list[float] = []
    for i in range(int(9.0 / params.dt)):
        if i % 10 == 0:
            pol = model.polarizations()
            times.append(model.t)
            P_trace.append(pol["P"])
            Q_trace.append(pol["Q"])
        model.step(1)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    ax.plot(times, P_trace, label="P(t)")
    ax.plot(times, Q_trace, label="Q(t)")
    ax.axhline(params.p_dnp_sat, linestyle="--", linewidth=1.0, label="P saturation setting")
    ax.set_title("DNP on: finite-rate build toward the selected saturation polarization")
    ax.set_xlabel("time [arb.]")
    ax.set_ylabel("dimensionless polarization")
    ax.legend(loc="best")
    fig.savefig(OUT / "v11_dnp_build_to_saturation.png", dpi=180)
    plt.close(fig)

    return OUT / "v11_dnp_build_to_saturation.png"


def run_negative_p_demo():
    params = Spin1Params(p0=-0.45, rf_burn_R=0.4, rf_enabled=False)
    model = Spin1Model(params)
    R, Ip, Im, total = model.spectrum()
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    ax.plot(R, Ip, drawstyle="steps-mid", label="I+(R)")
    ax.plot(R, Im, drawstyle="steps-mid", label="I-(R)")
    ax.plot(R, total, linewidth=1.3, label="total")
    ax.axhline(0, linewidth=1)
    ax.set_title("Signed negative initial vector polarization")
    ax.set_xlabel("physical R")
    ax.set_ylabel("intensity [arb.]")
    ax.legend(loc="best")
    fig.savefig(OUT / "v11_negative_initial_polarization.png", dpi=180)
    plt.close(fig)


def lineshape_validation():
    R = np.linspace(-3, 3, 701)
    Ip_ref, Im_ref, total_ref = plot_signal_reference(R, P=0.50, gamma=0.05, asym=0.04)
    model = Spin1Model(Spin1Params(p0=0.50, calibration_p=0.50, rf_enabled=False, line_gamma=0.05, line_asym=0.04))
    Rm, Ip_model, Im_model, total_model = model.reference_spectrum()

    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    ax.plot(R, total_ref, label="Plot_Signal.py total", linewidth=2)
    ax.plot(Rm, total_model, linestyle="--", label="v11 model reference", linewidth=1.5)
    ax.plot(R, Ip_ref, label="Plot_Signal.py I+")
    ax.plot(R, Im_ref, label="Plot_Signal.py I-")
    ax.set_xlim(-3.0, 3.0)
    ax.set_xlabel("R")
    ax.set_ylabel("intensity [Plot_Signal-like arb.]")
    ax.set_title("Static lineshape validation against the attached analytic example")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "v11_lineshape_validation_against_plot_signal.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    lineshape_validation()
    run_no_dnp_burn_recovery()
    run_dnp_build_demo()
    run_negative_p_demo()
    print(f"Wrote outputs to {OUT}")
