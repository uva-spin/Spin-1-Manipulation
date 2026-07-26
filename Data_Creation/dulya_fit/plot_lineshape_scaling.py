"""
Diagnostic plots for Dulya-fit lineshape scaling.

Verifies:
  - Stored I± are at fit.py signal scale (no Σ→P rescale)
  - P/Q from amp integration with equilibrium post-correction
  - Shape matches fit.component_curves

Examples:
  python Data_Creation/dulya_fit/plot_lineshape_scaling.py
  python Data_Creation/dulya_fit/plot_lineshape_scaling.py --p-highlight 0.48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import DATA_DIR, F_MAX, F_MIN, FREQUENCY, NUM_BINS, P_MAX, P_MIN  # noqa: E402
from lineshape import GenerateDulyaLineshape, shape_params_from_fit  # noqa: E402
from physics.lineshape.rate_eqs_test.fit import component_curves  # noqa: E402
from polarization import (  # noqa: E402
    apply_amp_post_correction,
    build_amp_integration_calibration,
    integrated_polarizations,
)

DEFAULT_P_HIGHLIGHT = 0.48
OVERLAY_P = (-0.9, -0.5, -0.25, 0.25, 0.5, 0.9)
OUT_PATH = DATA_DIR / "lineshape_scaling_test.png"


def build_scaling_report(
    f: np.ndarray,
    shape: dict[str, float],
    calibration: dict[str, np.ndarray | float | int],
    *,
    p_min: float,
    p_max: float,
    p_step: float,
) -> dict[str, np.ndarray]:
    amp = float(shape["amp"])
    n_bins = int(f.size)
    p_grid = np.arange(float(p_min), float(p_max) + 1e-12, float(p_step), dtype=float)
    p_raw = np.zeros_like(p_grid)
    q_raw = np.zeros_like(p_grid)
    p_amp_naive = np.zeros_like(p_grid)
    q_amp_naive = np.zeros_like(p_grid)
    p_vec = np.zeros_like(p_grid)
    q_tens = np.zeros_like(p_grid)
    for i, p0 in enumerate(p_grid):
        if abs(p0) < 1e-12:
            continue
        _, ip, im = GenerateDulyaLineshape(float(p0), f, shape)
        p_raw[i], q_raw[i] = integrated_polarizations(ip, im)
        p_amp_naive[i], q_amp_naive[i] = integrated_polarizations(
            ip, im, amp=amp, n_bins=n_bins
        )
        p_vec[i], q_tens[i] = apply_amp_post_correction(
            p_amp_naive[i], q_amp_naive[i], calibration
        )
    return {
        "p_grid": p_grid,
        "p_raw": p_raw,
        "q_raw": q_raw,
        "p_amp_naive": p_amp_naive,
        "q_amp_naive": q_amp_naive,
        "p_vec": p_vec,
        "q_tens": q_tens,
    }


def plot_lineshape_scaling(
    *,
    p_highlight: float = DEFAULT_P_HIGHLIGHT,
    output_path: Path = OUT_PATH,
) -> Path:
    shape = shape_params_from_fit()
    amp = float(shape["amp"])
    f = np.asarray(FREQUENCY, dtype=float)
    n_bins = int(f.size)
    calibration = build_amp_integration_calibration(
        f, shape, p_min=P_MIN, p_max=P_MAX, p_step=0.01
    )
    report = build_scaling_report(
        f, shape, calibration, p_min=P_MIN, p_max=P_MAX, p_step=0.05
    )

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.2, 1.0, 1.0), hspace=0.32, wspace=0.28)

    ps, ip, im = GenerateDulyaLineshape(p_highlight, f, shape)
    p_raw, q_raw = integrated_polarizations(ip, im)
    p_amp_naive, q_amp_naive = integrated_polarizations(ip, im, amp=amp, n_bins=n_bins)
    p_vec, q_tens = apply_amp_post_correction(p_amp_naive, q_amp_naive, calibration)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(f, ps, color="C0", lw=2.0, label=f"Ps  Σ_raw={p_raw:.4f}")
    ax0.axhline(0.0, color="0.5", lw=0.8)
    ax0.set_xlim(F_MIN, F_MAX)
    ax0.set_ylabel("Ps")
    ax0.set_title(
        f"P = {p_highlight:.2f}  fit-scale (no area rescale)\n"
        f"P_vec = post-corr amp int = {p_vec:.4f}  Q = {q_tens:.4f}  "
        f"peak Ps = {float(np.min(ps)):.5f}"
    )
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(f, ip, color="C0", lw=1.8, label="I+")
    ax1.plot(f, im, color="C3", lw=1.8, label="I−")
    ax1.axhline(0.0, color="0.5", lw=0.8)
    ax1.set_xlim(F_MIN, F_MAX)
    ax1.set_ylabel("I±")
    ax1.set_title(f"I± at P = {p_highlight:.2f} (fit units)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    for p0 in OVERLAY_P:
        ps_p, _, _ = GenerateDulyaLineshape(float(p0), f, shape)
        ax2.plot(
            f,
            ps_p,
            lw=1.4,
            label=f"P={p0:+.2f}  Σ_raw={np.sum(ps_p):+.3f}",
        )
    ax2.axhline(0.0, color="0.5", lw=0.8)
    ax2.set_xlim(F_MIN, F_MAX)
    ax2.set_ylabel("Ps = I+ + I−")
    ax2.set_title("Full lineshape vs P (fit-scale storage)")
    ax2.legend(fontsize=7, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    fit_params = {**shape, "P": float(p_highlight)}
    curves = component_curves(fit_params, f, exact_intensity=True, nphi=64)
    ax3.plot(f, curves["total"], color="0.65", lw=1.4, label="fit.component_curves")
    ax3.plot(f, ps, color="C2", lw=1.8, ls="--", label="GenerateDulyaLineshape Ps")
    max_diff = float(np.max(np.abs(curves["total"] - ps)))
    ax3.axhline(0.0, color="0.5", lw=0.8)
    ax3.set_xlim(F_MIN, F_MAX)
    ax3.set_ylabel("signal (fit units)")
    ax3.set_title(f"fit.py match at P={p_highlight:.2f}  max|Δ|={max_diff:.2e}")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[2, 0])
    mask = np.abs(report["p_grid"]) >= 1e-12
    p_in = report["p_grid"][mask]
    p_out = report["p_vec"][mask]
    p_naive = report["p_amp_naive"][mask]
    ax4.plot(
        p_in,
        p_naive,
        ".",
        ms=3,
        color="0.65",
        alpha=0.75,
        label="naive Σ/(amp·n)",
    )
    ax4.plot(
        p_in,
        p_out,
        "o",
        ms=4,
        color="C0",
        label="post-corrected P_vec",
    )
    lims = [min(P_MIN, float(p_in.min())), max(P_MAX, float(p_in.max()))]
    ax4.plot(lims, lims, "k--", lw=1.0, label="y = x")
    resid = p_out - p_in
    ax4.set_xlabel("P_input")
    ax4.set_ylabel("integrated P")
    ax4.set_title(
        f"Amp integration + post-correction  max|P_vec − P| = "
        f"{float(np.max(np.abs(resid))):.2e}"
    )
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect("equal", adjustable="box")

    ax5 = fig.add_subplot(gs[2, 1])
    corr = np.asarray(calibration["correction"], dtype=float)
    p_cal = np.asarray(calibration["p_input"], dtype=float)
    ax5.plot(p_cal, corr, "-", color="C4", lw=1.6, label="post-corr factor P / [Σ/(amp·n)]")
    ax5.plot(p_in, report["q_tens"][mask], "o", ms=4, color="C1", label="Q (post-corr)")
    ax5.set_xlabel("P_input")
    ax5.set_ylabel("correction / Q")
    ax5.set_title("Amp post-correction factor vs P")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    fig.suptitle(
        "Dulya-fit lineshape scaling test\n"
        f"amp={shape['amp']:.4g}  R=[{F_MIN},{F_MAX}]  n={NUM_BINS}",
        fontsize=12,
        y=0.98,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Dulya lineshape scaling diagnostics")
    p.add_argument("--p-highlight", type=float, default=DEFAULT_P_HIGHLIGHT)
    p.add_argument("--output", type=Path, default=OUT_PATH)
    args = p.parse_args()

    shape = shape_params_from_fit()
    f = np.asarray(FREQUENCY, dtype=float)
    n_bins = int(f.size)
    amp = float(shape["amp"])
    print("Shape params:", ", ".join(f"{k}={v:.6g}" for k, v in shape.items()))
    print(f"R grid: [{F_MIN}, {F_MAX}]  n={NUM_BINS}")

    calibration = build_amp_integration_calibration(
        f, shape, p_min=P_MIN, p_max=P_MAX, p_step=0.01
    )
    out = plot_lineshape_scaling(p_highlight=float(args.p_highlight), output_path=args.output)
    print(f"Saved {out}")

    P = float(args.p_highlight)
    ps, ip, im = GenerateDulyaLineshape(P, f, shape)
    p_raw, q_raw = integrated_polarizations(ip, im)
    p_amp_naive, q_amp_naive = integrated_polarizations(ip, im, amp=amp, n_bins=n_bins)
    p_vec, q_tens = apply_amp_post_correction(p_amp_naive, q_amp_naive, calibration)
    fit_params = {**shape, "P": P}
    curves = component_curves(fit_params, f, exact_intensity=True, nphi=64)
    print(f"\nP = {P:.3f}:")
    print(f"  sum(I+ + I-) raw = {p_raw:.6f}")
    print(f"  naive sum/(amp*n) = {p_amp_naive:.6f}")
    print(f"  P_vec post-corrected = {p_vec:.6f}  (target {P:.6f})")
    print(f"  Q_tensor post-corrected = {q_tens:.6f}")
    print(f"  peak Ps = {float(np.min(ps)):.6f}")
    print(f"  max|fit.total - Ps| = {float(np.max(np.abs(curves['total'] - ps))):.2e}")


if __name__ == "__main__":
    main()
