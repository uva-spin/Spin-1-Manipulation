"""Publication-ready bias and variance vs SNR for each |P| and |Q| bin."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ANALYSIS_DIR = Path(__file__).resolve().parent
ML_DIR = ANALYSIS_DIR.parent
RESULTS_DIR = ML_DIR / "results"

# Latest result run per architecture (matches weight_histo / val-loss overlay).
# Paths are relative to RESULTS_DIR.
MODELS = [
    ("lstm/lstm_result_v9", "LSTM", "#2ca02c"),
    # ("cnn/cnn_pq_results_v4", "CNN", "#d62728"),
    # ("mlp/mlp_pq_results_v3", "MLP", "#1f77b4"),
    # ("transformer/transformer_pq_results_v2", "Transformer", "#ff7f0e"),
    # ("old_model/old_model_pq_results_v1", "Old Polarization Model", "#9467bd"),
]

# 5% absolute-polarization bands (empty bands are skipped per observable).
POL_ABS_BANDS = tuple((lo / 100.0, (lo + 5) / 100.0) for lo in range(0, 95, 5))
MIN_BIN_COUNT = 25
SNR_BIN_WIDTH = 5.0


def apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "figure.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.formatter.use_mathtext": True,
    })


def _style_ax(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", width=1.35, length=5.0, pad=3)
    ax.grid(True, which="major", linestyle="--", color="0.85", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.6)
    ax.set_facecolor("white")


def pol_bin_label(lo: float, hi: float) -> str:
    return f"{lo * 100:.0f}–{hi * 100:.0f}%"


def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    df = pd.read_csv(path)
    required = {"true_P", "true_Q", "residual_P", "residual_Q", "snr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.copy()
    out["abs_P"] = np.abs(out["true_P"].to_numpy(dtype=float))
    out["abs_Q"] = np.abs(out["true_Q"].to_numpy(dtype=float))
    # Fractional polarization residual → percentage points.
    out["residual_P_pct"] = out["residual_P"].to_numpy(dtype=float) * 100.0
    out["residual_Q_pct"] = out["residual_Q"].to_numpy(dtype=float) * 100.0
    out["snr"] = out["snr"].to_numpy(dtype=float)
    return out.dropna(subset=["abs_P", "abs_Q", "residual_P_pct", "residual_Q_pct", "snr"])


def snr_bin_edges(snr: np.ndarray, *, width: float = SNR_BIN_WIDTH) -> np.ndarray:
    lo = np.floor(float(np.nanmin(snr)) / width) * width
    hi = np.ceil(float(np.nanmax(snr)) / width) * width
    if hi <= lo:
        hi = lo + width
    return np.arange(lo, hi + 0.5 * width, width)


def bias_variance_vs_snr(
    df: pd.DataFrame,
    *,
    abs_col: str,
    residual_col: str,
    pol_bands: tuple[tuple[float, float], ...] = POL_ABS_BANDS,
    snr_edges: np.ndarray | None = None,
    min_count: int = MIN_BIN_COUNT,
) -> list[dict]:
    """For each |obs| band, compute residual bias/variance (%) vs SNR bin centers.

    Variance here is the sample standard deviation of residuals (in %).
    """
    if snr_edges is None:
        snr_edges = snr_bin_edges(df["snr"].to_numpy())
    snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])
    rows: list[dict] = []

    for lo, hi in pol_bands:
        in_pol = (df[abs_col] >= lo) & (df[abs_col] < hi)
        if int(in_pol.sum()) < min_count:
            continue

        snr = df.loc[in_pol, "snr"].to_numpy()
        res = df.loc[in_pol, residual_col].to_numpy()
        snr_idx = np.digitize(snr, snr_edges) - 1

        xs, bias, variances, ns = [], [], [], []
        for i, center in enumerate(snr_centers):
            mask = snr_idx == i
            n = int(mask.sum())
            if n < min_count:
                continue
            xs.append(float(center))
            bias.append(float(np.mean(res[mask])))
            variances.append(float(np.std(res[mask], ddof=1)) if n > 1 else 0.0)
            ns.append(n)

        if not xs:
            continue
        rows.append({
            "lo": lo,
            "hi": hi,
            "label": pol_bin_label(lo, hi),
            "snr": np.asarray(xs, dtype=float),
            "bias": np.asarray(bias, dtype=float),
            "variance": np.asarray(variances, dtype=float),
            "n": np.asarray(ns, dtype=int),
            "n_pol": int(in_pol.sum()),
        })
    return rows


def _pol_colormap(n: int):
    if n <= 0:
        return []
    cmap = plt.get_cmap("viridis")
    if n == 1:
        return [cmap(0.55)]
    return [cmap(i / (n - 1)) for i in range(n)]


def _legend_handles(series: list[dict]) -> list[Line2D]:
    colors = _pol_colormap(len(series))
    return [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2.0,
            marker="o",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=row["label"],
        )
        for color, row in zip(colors, series)
    ]


def _draw_series(
    ax: plt.Axes,
    series: list[dict],
    *,
    key: str,
    ylabel: str,
    panel: str,
    title: str,
    show_zero: bool,
) -> None:
    colors = _pol_colormap(len(series))
    for color, row in zip(colors, series):
        ax.plot(
            row["snr"],
            row[key],
            color=color,
            lw=2.1,
            marker="o",
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.65,
            label=row["label"],
            zorder=3,
        )
    if show_zero:
        ax.axhline(0.0, color="0.45", lw=1.05, ls="--", zorder=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=7)
    ax.text(
        0.02,
        0.96,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        zorder=5,
    )
    _style_ax(ax)


def plot_model_bias_variance_vs_snr(
    series_p: list[dict],
    series_q: list[dict],
    *,
    model_label: str,
    out_stem: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.8),
        sharex=True,
        constrained_layout=False,
    )
    fig.patch.set_facecolor("white")

    _draw_series(
        axes[0, 0],
        series_p,
        key="bias",
        ylabel=r"Bias $\langle r_P\rangle$ (%)",
        panel="(a)",
        title=r"$P$ bias vs SNR",
        show_zero=True,
    )
    _draw_series(
        axes[0, 1],
        series_p,
        key="variance",
        ylabel=r"Variance $\sigma(r_P)$ (%)",
        panel="(b)",
        title=r"$P$ variance vs SNR",
        show_zero=False,
    )
    _draw_series(
        axes[1, 0],
        series_q,
        key="bias",
        ylabel=r"Bias $\langle r_Q\rangle$ (%)",
        panel="(c)",
        title=r"$Q$ bias vs SNR",
        show_zero=True,
    )
    _draw_series(
        axes[1, 1],
        series_q,
        key="variance",
        ylabel=r"Variance $\sigma(r_Q)$ (%)",
        panel="(d)",
        title=r"$Q$ variance vs SNR",
        show_zero=False,
    )

    for ax in axes[1, :]:
        ax.set_xlabel("SNR")

    # Outside legends so they do not cover data.
    axes[0, 1].legend(
        handles=_legend_handles(series_p),
        title=r"$|P|$ bin",
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        handlelength=1.6,
        labelspacing=0.35,
    )
    axes[1, 1].legend(
        handles=_legend_handles(series_q),
        title=r"$|Q|$ bin",
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        handlelength=1.6,
        labelspacing=0.35,
    )

    fig.suptitle(
        rf"{model_label}: polarization residual bias and variance vs SNR",
        fontsize=16,
        y=0.98,
        color="black",
    )
    fig.subplots_adjust(left=0.09, right=0.84, top=0.92, bottom=0.08, wspace=0.28, hspace=0.28)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext, dpi in (("png", 300), ("pdf", None)):
        path = out_stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.05}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ANALYSIS_DIR / "bias_std_vs_snr_pub",
        help="Output directory for publication plots",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=MIN_BIN_COUNT,
        help="Minimum events required in each (polarization, SNR) cell",
    )
    parser.add_argument(
        "--snr-bin-width",
        type=float,
        default=SNR_BIN_WIDTH,
        help="SNR bin width",
    )
    parser.add_argument(
        "--also-results-dir",
        action="store_true",
        help="Also write copies into each model's results directory",
    )
    args = parser.parse_args()

    apply_pub_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_snr = []
    frames: dict[str, pd.DataFrame] = {}
    for run_dir, label, _color in MODELS:
        df = load_predictions(RESULTS_DIR / run_dir)
        frames[label] = df
        all_snr.append(df["snr"].to_numpy())
    snr_edges = snr_bin_edges(np.concatenate(all_snr), width=args.snr_bin_width)

    for run_dir, label, _color in MODELS:
        df = frames[label]
        series_p = bias_variance_vs_snr(
            df,
            abs_col="abs_P",
            residual_col="residual_P_pct",
            snr_edges=snr_edges,
            min_count=args.min_count,
        )
        series_q = bias_variance_vs_snr(
            df,
            abs_col="abs_Q",
            residual_col="residual_Q_pct",
            snr_edges=snr_edges,
            min_count=args.min_count,
        )
        slug = label.lower().replace(" ", "_")
        out_stem = args.out_dir / f"{slug}_bias_std_vs_snr"
        saved = plot_model_bias_variance_vs_snr(
            series_p,
            series_q,
            model_label=label,
            out_stem=out_stem,
        )
        print(
            f"{label}: {len(series_p)} |P| bins, {len(series_q)} |Q| bins, "
            f"{len(df)} test events"
        )
        for path in saved:
            print(f"  Saved {path}")

        if args.also_results_dir:
            extra = plot_model_bias_variance_vs_snr(
                series_p,
                series_q,
                model_label=label,
                out_stem=RESULTS_DIR / run_dir / "bias_std_vs_snr",
            )
            for path in extra:
                print(f"  Saved {path}")


if __name__ == "__main__":
    main()
