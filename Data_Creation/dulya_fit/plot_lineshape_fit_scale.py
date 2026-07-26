"""
Plot Dulya-fit lineshapes at fit.py signal scale (no P normalization).

Compares ``GenerateDulyaLineshapeFitScale`` to ``fit.signal_model`` and
``fit.component_curves`` so peak heights match the fitted diagnostic.

Examples:
  python Data_Creation/dulya_fit/plot_lineshape_fit_scale.py
  python Data_Creation/dulya_fit/plot_lineshape_fit_scale.py --p-highlight 0.48
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

from common import DATA_DIR, F_MAX, F_MIN, FREQUENCY, NUM_BINS  # noqa: E402
from lineshape import (  # noqa: E402
    GenerateDulyaLineshapeFitScale,
    shape_params_from_fit,
)
from physics.lineshape.rate_eqs_test.fit import (  # noqa: E402
    component_curves,
    signal_model,
)

DEFAULT_P = 0.48
OVERLAY_P = (-0.9, -0.5, -0.25, 0.25, 0.5, 0.9)
OUT_PATH = DATA_DIR / "lineshape_fit_scale_test.png"


def _fit_signal_model(P: float, f: np.ndarray, shape: dict[str, float]) -> np.ndarray:
    return np.asarray(
        signal_model(
            f,
            P,
            shape["amp"],
            shape["center"],
            shape["cc"],
            shape["split"],
            shape["sigma"],
            shape["eta"],
            shape["xi"],
            shape["b0"],
            shape["b1"],
            shape["b2"],
            shape["b3"],
            exact_intensity=True,
            nphi=64,
        ),
        dtype=float,
    )


def plot_fit_scale(*, p_highlight: float, output_path: Path) -> Path:
    shape = shape_params_from_fit()
    f = np.asarray(FREQUENCY, dtype=float)
    fit_params = {**shape, "P": float(p_highlight)}

    ps, ip, im = GenerateDulyaLineshapeFitScale(p_highlight, f, shape)
    curves = component_curves(fit_params, f, exact_intensity=True, nphi=64)
    sig = _fit_signal_model(p_highlight, f, shape)

    gain = curves["qmeter_gain"]
    bg = curves["background"]
    ip_fit = curves["plus"] * gain + 0.5 * bg
    im_fit = curves["minus"] * gain + 0.5 * bg

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col")

    # --- Ps total vs fit.py ---
    ax = axes[0, 0]
    ax.plot(f, sig, color="0.55", lw=2.0, label="fit.signal_model")
    ax.plot(f, curves["total"], color="C2", lw=1.4, ls="--", label="component_curves total")
    ax.plot(f, ps, color="C0", lw=1.8, label="GenerateDulyaFitScale Ps")
    peak = float(np.min(ps))
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_ylabel("Ps (fit units)")
    ax.set_title(
        f"P = {p_highlight:.2f}  peak Ps = {peak:.5f}\n"
        f"sum(Ps) = {float(np.sum(ps)):.4f}  "
        f"max|Ps - signal_model| = {float(np.max(np.abs(ps - sig))):.2e}"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- I+ / I- ---
    ax = axes[0, 1]
    ax.plot(f, ip, color="C0", lw=1.8, label="I+ GenerateDulyaFitScale")
    ax.plot(f, im, color="C3", lw=1.8, label="I- GenerateDulyaFitScale")
    ax.plot(f, ip_fit, color="C0", lw=1.0, ls="--", alpha=0.8, label="I+ from component_curves")
    ax.plot(f, im_fit, color="C3", lw=1.0, ls="--", alpha=0.8, label="I- from component_curves")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_ylabel("I+ / I- (fit units)")
    ax.set_title("Branch intensities (no P normalization)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Ps vs P at fit scale ---
    ax = axes[1, 0]
    for p0 in OVERLAY_P:
        ps_p, _, _ = GenerateDulyaLineshapeFitScale(float(p0), f, shape)
        ax.plot(
            f,
            ps_p,
            lw=1.4,
            label=f"P={p0:+.2f}  peak={float(np.min(ps_p)):.4f}  sum={float(np.sum(ps_p)):.3f}",
        )
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_xlabel("R")
    ax.set_ylabel("Ps (fit units)")
    ax.set_title("Fit-scale lineshape vs P (no sum-to-P rescale)")
    ax.legend(fontsize=6.5, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.3)

    # --- Peak & sum vs P ---
    ax = axes[1, 1]
    p_grid = np.linspace(-0.9, 0.9, 37)
    peaks = []
    sums = []
    fit_peaks = []
    for p0 in p_grid:
        ps_p, _, _ = GenerateDulyaLineshapeFitScale(float(p0), f, shape)
        sig_p = _fit_signal_model(float(p0), f, shape)
        peaks.append(float(np.min(ps_p)))
        sums.append(float(np.sum(ps_p)))
        fit_peaks.append(float(np.min(sig_p)))
    ax.plot(p_grid, peaks, "o-", ms=3, lw=1.2, label="peak Ps (GenerateDulyaFitScale)")
    ax.plot(p_grid, fit_peaks, "s--", ms=3, lw=1.0, label="peak signal_model")
    ax.set_xlabel("P")
    ax.set_ylabel("min(Ps)  [absorption peak]")
    ax.set_title("Peak height vs P at fit scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(p_grid, sums, ".", ms=4, color="0.45", alpha=0.8, label="sum(Ps)")
    ax2.set_ylabel("sum(I+ + I-)", color="0.45")
    ax2.tick_params(axis="y", labelcolor="0.45")

    fig.suptitle(
        "Dulya-fit lineshape at fit.py scale (no P normalization)\n"
        f"amp={shape['amp']:.4g}  R=[{F_MIN},{F_MAX}]  n={NUM_BINS}",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot fit-scale Dulya lineshape (no P norm)")
    p.add_argument("--p-highlight", type=float, default=DEFAULT_P)
    p.add_argument("--output", type=Path, default=OUT_PATH)
    args = p.parse_args()

    shape = shape_params_from_fit()
    f = np.asarray(FREQUENCY, dtype=float)
    P = float(args.p_highlight)

    ps, ip, im = GenerateDulyaLineshapeFitScale(P, f, shape)
    sig = _fit_signal_model(P, f, shape)
    curves = component_curves({**shape, "P": P}, f, exact_intensity=True, nphi=64)

    print("Shape params:", ", ".join(f"{k}={v:.6g}" for k, v in shape.items()))
    print(f"R grid: [{F_MIN}, {F_MAX}]  n={NUM_BINS}")
    print(f"\nP = {P:.3f}  (fit-scale, no normalization):")
    print(f"  peak Ps = {float(np.min(ps)):.6f}")
    print(f"  sum(I+ + I-) = {float(np.sum(ps)):.6f}")
    print(f"  sum(I+ - I-) = {float(np.sum(ip - im)):.6f}")
    print(f"  max|Ps - signal_model| = {float(np.max(np.abs(ps - sig))):.2e}")
    print(f"  max|Ps - component total| = {float(np.max(np.abs(ps - curves['total']))):.2e}")

    out = plot_fit_scale(p_highlight=P, output_path=args.output)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
