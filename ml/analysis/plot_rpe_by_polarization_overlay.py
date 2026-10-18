"""Overlay mean RPE by |P_total| band for multiple PQ model runs."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
ML_DIR = ANALYSIS_DIR.parent
RESULTS_DIR = ML_DIR / "results"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from analysis.pq_report import nonempty_polarization_range_rows

# Same runs as the fixed per-model rpe_by_polarization_range.png plots.
RUNS = [
    ("mlp/mlp_pq_results_v4", "MLP", "#2563eb"),
    ("lstm/lstm_result_ssrf_afp_combined_v5", "LSTM", "#16a34a"),
    ("cnn/cnn_pq_results_v5", "CNN", "#dc2626"),
    ("transformer/transformer_pq_results_v2", "Transformer", "#ea580c"),
    ("old_model/old_model_pq_results_v1", "Old model", "#9333ea"),
]

RUNS_NO_MLP = [run for run in RUNS if run[1] != "MLP"]


def _nonempty_range_rows(metrics_path: Path) -> list[dict]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return nonempty_polarization_range_rows(metrics)


def _band_key(row: dict) -> tuple[float, float]:
    return (float(row["lo"]), float(row["hi"]))


def load_aligned_stats(run_dir: Path, band_keys: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_band = {_band_key(r): r for r in _nonempty_range_rows(run_dir / "metrics.json")}
    mean_p, std_p, mean_q, std_q = [], [], [], []
    for key in band_keys:
        row = by_band.get(key)
        if row is None:
            raise KeyError(f"{run_dir}: missing band {key}")
        mean_p.append(float(row["P_rpe_mean"]))
        std_p.append(float(row["P_rpe_std"]))
        mean_q.append(float(row["Q_rpe_mean"]))
        std_q.append(float(row["Q_rpe_std"]))
    return (
        np.asarray(mean_p, dtype=float),
        np.asarray(std_p, dtype=float),
        np.asarray(mean_q, dtype=float),
        np.asarray(std_q, dtype=float),
    )


def _upper_sigma_yerr(std: np.ndarray) -> np.ndarray:
    """+1σ only — lower −σ is often clipped at 0 when std > mean (RPE is non-negative)."""
    std = np.asarray(std, dtype=float)
    return np.vstack([np.zeros_like(std), std])


def _style_ax(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.4, length=5)
    ax.grid(True, which="major", linestyle="--", color="0.85", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.6)
    ax.set_facecolor("white")


def _plot_runs_on_ax(
    ax: plt.Axes,
    runs: list[tuple[str, str, str]],
    *,
    x: np.ndarray,
    band_keys: list[tuple[float, float]],
    prefix: str,
    log_y: bool,
) -> None:
    lows: list[float] = []
    highs: list[float] = []
    for dirname, model_label, color in runs:
        run_dir = RESULTS_DIR / dirname
        mean_p, std_p, mean_q, std_q = load_aligned_stats(run_dir, band_keys)
        mean = mean_p if prefix == "P" else mean_q
        std = std_p if prefix == "P" else std_q
        yerr = _upper_sigma_yerr(std)
        lows.extend(mean.tolist())
        highs.extend((mean + std).tolist())
        ax.errorbar(
            x,
            mean,
            yerr=yerr,
            color=color,
            lw=2.0,
            ms=6.0,
            marker="o",
            mfc=color,
            mec="white",
            mew=0.8,
            capsize=3.0,
            elinewidth=1.1,
            label=model_label,
            zorder=3,
        )

    positive_lows = [v for v in lows if np.isfinite(v) and v > 0]
    positive_highs = [v for v in highs if np.isfinite(v) and v > 0]
    if not positive_highs:
        return
    if log_y:
        ax.set_yscale("log")
        y_lo = min(positive_lows) * 0.65
        y_hi = max(positive_highs) * 1.35
        ax.set_ylim(y_lo, y_hi)
    else:
        y_hi = max(positive_highs) * 1.12
        ax.set_ylim(0.0, max(y_hi, 1e-12))


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    ref_dir = RESULTS_DIR / RUNS[0][0]
    ref_rows = _nonempty_range_rows(ref_dir / "metrics.json")
    band_keys = [_band_key(r) for r in ref_rows]
    labels = [r["range"] for r in ref_rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.4), sharex="col")

    panel_specs = (
        (RUNS, True, "All models (log scale)"),
        (RUNS_NO_MLP, False, "Excluding MLP (linear scale)"),
    )
    for col, (runs, log_y, col_title) in enumerate(panel_specs):
        for row, prefix in enumerate(("P", "Q")):
            ax = axes[row, col]
            _plot_runs_on_ax(ax, runs, x=x, band_keys=band_keys, prefix=prefix, log_y=log_y)
            ylabel = rf"${prefix}$ RPE (\%)"
            ax.set_ylabel(ylabel, fontsize=13)
            if row == 0:
                ax.set_title(col_title, fontsize=13, pad=8)
            _style_ax(ax)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=10, loc="upper right")
            if row == 1 and col == 0:
                ax.text(
                    0.02,
                    0.03,
                    r"error bars: $+1\sigma$ (lower $-\sigma$ omitted; RPE $\geq 0$)",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8.5,
                    color="#24303a",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d0d7de", "alpha": 0.92},
                )

    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=40, ha="right")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=40, ha="right")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$|P_{\mathrm{total}}|$ band", fontsize=13)

    fig.patch.set_facecolor("white")
    fig.tight_layout()

    out_path = ANALYSIS_DIR / "rpe_by_polarization_overlay.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
