"""Predicted-vs-true P and Q, one panel per SNR bin, for each model.

Uses the same 5-wide SNR bins as the bias-vs-SNR figures. Bins with fewer
than ``MIN_BIN_COUNT`` events are merged into the neighboring bin so the
high-SNR tail is not a nearly empty panel.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from plot_bias_variance_vs_snr import MODELS, SNR_BIN_WIDTH, snr_bin_edges

ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ANALYSIS_DIR.parent / "results"

COLOR_P = "#2f6fed"
COLOR_Q = "#c45c26"
COLOR_IDEAL = "#1f2933"
MIN_BIN_COUNT = 25

OBSERVABLES = (
    ("P", r"$P_{\mathrm{total}}$", "true_P", "pred_P", COLOR_P),
    ("Q", r"$Q_{\mathrm{total}}$", "true_Q", "pred_Q", COLOR_Q),
)


def apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "figure.titlesize": 16,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.formatter.use_mathtext": True,
    })


def _style_ax(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", width=1.05, length=3.5, pad=2)
    ax.grid(True, which="major", linestyle="--", color="0.85", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.15)
    ax.set_facecolor("white")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    df = pd.read_csv(path)
    required = {"true_P", "pred_P", "true_Q", "pred_Q", "snr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.dropna(subset=list(required))


def snr_intervals(
    snr: np.ndarray,
    *,
    width: float = SNR_BIN_WIDTH,
    min_count: int = MIN_BIN_COUNT,
) -> list[tuple[float, float]]:
    """SNR intervals of ``width``, with sparse tail bins folded into a neighbor."""
    edges = snr_bin_edges(np.asarray(snr, dtype=float), width=width)
    counts = np.histogram(snr, bins=edges)[0]
    intervals: list[tuple[float, float, int]] = []
    i = 0
    n_bins = len(edges) - 1
    while i < n_bins:
        lo = float(edges[i])
        hi = float(edges[i + 1])
        n = int(counts[i])
        j = i
        while n < min_count and j + 1 < n_bins:
            j += 1
            hi = float(edges[j + 1])
            n += int(counts[j])
        if n < min_count and intervals:
            prev_lo, _prev_hi, prev_n = intervals[-1]
            intervals[-1] = (prev_lo, hi, prev_n + n)
        elif n > 0:
            intervals.append((lo, hi, n))
        i = j + 1
    return [(lo, hi) for lo, hi, _n in intervals]


def _fmt_stat(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    av = abs(value)
    if av == 0.0:
        return "0"
    if av >= 0.01:
        return f"{value:.4g}"
    return f"{value:.3e}"


def _fmt_r2(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.6f}"


def _mae_r2(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
    return mae, r2


def _axis_limits(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    pad = 0.05 * (hi - lo + 1e-8)
    return lo - pad, hi + pad


def _bin_label(lo: float, hi: float) -> str:
    return f"{lo:.0f}–{hi:.0f}"


def plot_model(
    df: pd.DataFrame,
    intervals: list[tuple[float, float]],
    *,
    model_label: str,
    out_stem: Path,
) -> tuple[list[Path], list[dict]]:
    n_bins = len(intervals)
    fig, axes = plt.subplots(
        len(OBSERVABLES),
        n_bins,
        figsize=(2.35 * n_bins, 6.15),
        sharex="row",
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    snr = df["snr"].to_numpy(dtype=float)
    summary_rows = []

    for row_i, (key, symbol, true_col, pred_col, color) in enumerate(OBSERVABLES):
        true_all = df[true_col].to_numpy(dtype=float)
        pred_all = df[pred_col].to_numpy(dtype=float)
        lims = _axis_limits(true_all, pred_all)
        for col, (lo, hi) in enumerate(intervals):
            ax = axes[row_i, col]
            mask = (snr >= lo) & (snr < hi if col < n_bins - 1 else snr <= hi)
            true = true_all[mask]
            pred = pred_all[mask]
            ax.scatter(true, pred, s=6, alpha=0.28, c=color, edgecolors="none", rasterized=True)
            ax.plot(lims, lims, color=COLOR_IDEAL, ls="--", lw=1.0, label="ideal", zorder=3)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")
            _style_ax(ax)
            mae, r2 = _mae_r2(pred, true) if true.size else (float("nan"), float("nan"))
            ax.set_title(f"SNR {_bin_label(lo, hi)}", fontsize=10, pad=4)
            ax.text(
                0.97,
                0.04,
                f"MAE {_fmt_stat(mae)}\n$R^2$ {_fmt_r2(r2)}\n$n={true.size}$",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#24303a",
                linespacing=1.25,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#d0d7de",
                    "alpha": 0.92,
                },
            )
            if row_i == len(OBSERVABLES) - 1:
                ax.set_xlabel(f"True {symbol}")
            if col == 0:
                ax.set_ylabel(f"Predicted {symbol}")
                ax.legend(loc="upper left", frameon=False, fontsize=8)
            summary_rows.append({
                "model": model_label,
                "observable": key,
                "snr_lo": lo,
                "snr_hi": hi,
                "n": int(true.size),
                "mae": mae,
                "r2": r2,
            })

    fig.suptitle(f"{model_label}: predicted vs true by SNR", fontsize=15)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext, dpi in (("png", 200), ("pdf", None)):
        path = out_stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.08}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ANALYSIS_DIR / "pred_vs_true_by_snr",
        help="Output directory for the binned predicted-vs-true figures",
    )
    parser.add_argument("--snr-bin-width", type=float, default=SNR_BIN_WIDTH)
    parser.add_argument("--min-count", type=int, default=MIN_BIN_COUNT)
    args = parser.parse_args()

    apply_pub_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for run_dir, label, _color in MODELS:
        frames.append(load_predictions(RESULTS_DIR / run_dir))
    # SNR is a property of the shared test split, so bin from one model.
    intervals = snr_intervals(
        frames[0]["snr"].to_numpy(dtype=float),
        width=args.snr_bin_width,
        min_count=args.min_count,
    )
    print(
        "SNR bins: " + ", ".join(_bin_label(lo, hi) for lo, hi in intervals),
        flush=True,
    )

    all_rows = []
    for (run_dir, label, _color), df in zip(MODELS, frames):
        slug = label.lower().replace(" ", "_")
        saved, rows = plot_model(
            df,
            intervals,
            model_label=label,
            out_stem=args.out_dir / f"{slug}_pred_vs_true_by_snr",
        )
        all_rows.extend(rows)
        for path in saved:
            print(f"Saved {path}", flush=True)
            dest = RESULTS_DIR / run_dir / f"pred_vs_true_by_snr{path.suffix}"
            shutil.copy2(path, dest)
            print(f"Saved {dest}", flush=True)

    csv_path = args.out_dir / "pred_vs_true_by_snr.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
