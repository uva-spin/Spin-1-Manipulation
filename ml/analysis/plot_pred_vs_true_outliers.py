"""Outlier spectra from each model's predicted-vs-true panels, by SNR bin.

Uses the same SNR bins as ``plot_pred_vs_true_by_snr``. Within each bin, take
the top-N test events farthest from the ideal line on the P panel and on the Q
panel for every model, then draw those noisy input spectra. One figure is
written per SNR bin; each panel is labeled with the model, SNR, and the
predicted and true P and Q.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
ML_DIR = ANALYSIS_DIR.parent
RESULTS_DIR = ML_DIR / "results"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import lstm as lstm_pq
from plot_bias_variance_vs_snr import MODELS, SNR_BIN_WIDTH
from plot_model_example_comparisons import load_noisy_test_spectra
from plot_pred_vs_true_by_snr import MIN_BIN_COUNT, _bin_label, snr_intervals

SOURCE_NAME = lstm_pq.SOURCE_NAME
SPECTRUM_R_MIN = lstm_pq.SPECTRUM_R_MIN
SPECTRUM_R_MAX = lstm_pq.SPECTRUM_R_MAX
QUANTITIES = ("P", "Q")
DEFAULT_TOP_N = 1


def apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.labelsize": 13,
        "axes.titlesize": 11,
        "figure.titlesize": 16,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.formatter.use_mathtext": True,
    })


def _style_ax(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", width=1.35, length=4.5, pad=2)
    ax.grid(True, which="major", linestyle="--", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.4)
    ax.set_facecolor("white")


def load_predictions(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    df = pd.read_csv(path)
    required = {"event_idx", "true_P", "true_Q", "pred_P", "pred_Q", "source", "n_steps", "snr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.dropna(subset=list(required))


def _bin_mask(snr: np.ndarray, lo: float, hi: float, *, last: bool) -> np.ndarray:
    if last:
        return (snr >= lo) & (snr <= hi)
    return (snr >= lo) & (snr < hi)


def farthest_from_ideal(df: pd.DataFrame, quantity: str, *, top_n: int) -> pd.DataFrame:
    """Events with the largest |predicted − true| on one pred-vs-true panel."""
    if df.empty:
        return df.copy()
    residual = (df[f"pred_{quantity}"] - df[f"true_{quantity}"]).abs()
    n = min(top_n, len(df))
    top_idx = residual.nlargest(n).index
    out = df.loc[top_idx].copy()
    out["abs_residual"] = residual.loc[top_idx]
    return out


def select_outliers_by_snr(
    models: list[tuple[str, str, str]],
    intervals: list[tuple[float, float]],
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    n_bins = len(intervals)
    for run_dir, label, color in models:
        df = load_predictions(RESULTS_DIR / run_dir)
        snr = df["snr"].to_numpy(dtype=float)
        for bin_i, (lo, hi) in enumerate(intervals):
            mask = _bin_mask(snr, lo, hi, last=(bin_i == n_bins - 1))
            bin_df = df.loc[mask]
            for quantity in QUANTITIES:
                top = farthest_from_ideal(bin_df, quantity, top_n=top_n)
                for rank, (_, row) in enumerate(top.iterrows(), start=1):
                    src = int(row["source"])
                    rows.append({
                        "model": label,
                        "color": color,
                        "run_dir": run_dir,
                        "snr_lo": lo,
                        "snr_hi": hi,
                        "snr_bin": _bin_label(lo, hi),
                        "quantity": quantity,
                        "rank": rank,
                        "event_idx": int(row["event_idx"]),
                        "source": SOURCE_NAME.get(src, str(src)),
                        "n_steps": int(row["n_steps"]),
                        "snr": float(row["snr"]),
                        "true_P": float(row["true_P"]),
                        "pred_P": float(row["pred_P"]),
                        "true_Q": float(row["true_Q"]),
                        "pred_Q": float(row["pred_Q"]),
                        "abs_residual": float(row["abs_residual"]),
                    })
    return pd.DataFrame(rows)


def _panel_title(row: pd.Series) -> str:
    quantity = row["quantity"]
    rank = int(row["rank"])
    rank_txt = f" #{rank}" if int(row.get("_top_n", 1)) > 1 or rank > 1 else ""
    return (
        f"{row['model']} — ${quantity}$ outlier{rank_txt}\n"
        f"SNR $= {row['snr']:.2f}$\n"
        f"pred $P={row['pred_P']:.4f}$, true $P={row['true_P']:.4f}$\n"
        f"pred $Q={row['pred_Q']:.4f}$, true $Q={row['true_Q']:.4f}$"
    )


def plot_outliers_for_bin(
    selected: pd.DataFrame,
    spectra: dict[int, np.ndarray],
    out_stem: Path,
    *,
    snr_bin: str,
    top_n: int,
) -> list[Path]:
    n_models = len(MODELS)
    n_rows = top_n * len(QUANTITIES)
    fig, axes = plt.subplots(
        n_rows,
        n_models,
        figsize=(3.7 * n_models, 3.15 * n_rows + 0.8),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    n_bins = next(iter(spectra.values())).shape[0]
    freq = np.linspace(SPECTRUM_R_MIN, SPECTRUM_R_MAX, n_bins)

    for col, (_run, label, color) in enumerate(MODELS):
        model_rows = selected[selected["model"] == label]
        for q_i, quantity in enumerate(QUANTITIES):
            q_rows = (
                model_rows[model_rows["quantity"] == quantity]
                .sort_values("rank")
            )
            for rank_i in range(top_n):
                ax = axes[q_i * top_n + rank_i, col]
                if rank_i >= len(q_rows):
                    ax.set_axis_off()
                    continue
                row = q_rows.iloc[rank_i].copy()
                row["_top_n"] = top_n
                event_idx = int(row["event_idx"])
                ax.plot(freq, spectra[event_idx], color=color, lw=1.15)
                ax.set_xlim(float(freq[0]), float(freq[-1]))
                _style_ax(ax)
                ax.set_title(_panel_title(row), color=color, fontsize=8.5, pad=4)
                if q_i * top_n + rank_i == n_rows - 1:
                    ax.set_xlabel(r"$R$")
                if col == 0:
                    ax.set_ylabel("Amplitude")

    fig.suptitle(
        rf"SNR {snr_bin}: top-{top_n} $|$predicted $-$ true$|$ outliers",
        fontsize=15,
    )
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext, dpi in (("png", 200), ("pdf", None)):
        path = out_stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white", "pad_inches": 0.12}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def _slug_bin(lo: float, hi: float) -> str:
    return f"snr_{lo:.0f}-{hi:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectra",
        type=Path,
        default=lstm_pq.DEFAULT_SPECTRA_PATH,
        help="Path to spectra.npz used for training/evaluation",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ANALYSIS_DIR / "pred_vs_true_outliers",
        help="Output directory for per-SNR-bin figures and the selection table",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Farthest outliers per quantity per model per SNR bin (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument("--snr-bin-width", type=float, default=SNR_BIN_WIDTH)
    parser.add_argument("--min-count", type=int, default=MIN_BIN_COUNT)
    args = parser.parse_args()
    if args.top_n < 1:
        raise SystemExit("--top-n must be >= 1")

    apply_pub_style()
    frames = [load_predictions(RESULTS_DIR / run_dir) for run_dir, _label, _color in MODELS]
    intervals = snr_intervals(
        frames[0]["snr"].to_numpy(dtype=float),
        width=args.snr_bin_width,
        min_count=args.min_count,
    )
    print(
        "SNR bins: " + ", ".join(_bin_label(lo, hi) for lo, hi in intervals),
        flush=True,
    )

    selected = select_outliers_by_snr(MODELS, intervals, top_n=args.top_n)
    print("Selected outliers:", flush=True)
    print(selected.drop(columns=["color"]).to_string(index=False), flush=True)

    event_indices = selected["event_idx"].to_numpy(dtype=int)
    spectra = load_noisy_test_spectra(args.spectra, np.unique(event_indices))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "outlier_events.csv"
    selected.drop(columns=["color"]).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}", flush=True)

    for lo, hi in intervals:
        snr_bin = _bin_label(lo, hi)
        bin_rows = selected[
            (selected["snr_lo"] == lo) & (selected["snr_hi"] == hi)
        ]
        out_stem = out_dir / f"{_slug_bin(lo, hi)}_outliers"
        saved = plot_outliers_for_bin(
            bin_rows,
            spectra,
            out_stem,
            snr_bin=snr_bin,
            top_n=args.top_n,
        )
        for path in saved:
            print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
