"""Shared test-set example plots with a per-model P/Q comparison table.

Selects ~10 diverse events that appear in every listed model's
``test_predictions.csv``, plots the noisy input spectrum, and annotates a
table of true vs predicted P and Q, relative percent error, and SNR.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.table import Table

ANALYSIS_DIR = Path(__file__).resolve().parent
ML_DIR = ANALYSIS_DIR.parent
RESULTS_DIR = ML_DIR / "results"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import lstm as lstm_pq

# Latest result run per architecture (matches weight_histo / bias-vs-SNR).
# Paths are relative to RESULTS_DIR.
MODELS = [
    ("lstm/lstm_result_v9", "LSTM"),
    ("cnn/cnn_pq_results_v4", "CNN"),
    ("mlp/mlp_pq_results_v3", "MLP"),
    ("transformer/transformer_pq_results_v2", "Transformer"),
    ("old_model/old_model_pq_results_v1", "Old Polarization Model"),
]

SOURCE_NAME = lstm_pq.SOURCE_NAME
SPECTRUM_R_MIN = lstm_pq.SPECTRUM_R_MIN
SPECTRUM_R_MAX = lstm_pq.SPECTRUM_R_MAX
SEED = lstm_pq.SEED
NOISE_STD = lstm_pq.NOISE_STD
N_EXAMPLES_DEFAULT = 10


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
        "figure.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
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


def load_model_predictions(run_dir: Path, label: str) -> pd.DataFrame:
    path = run_dir / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    df = pd.read_csv(path)
    required = {
        "event_idx", "true_P", "true_Q", "pred_P", "pred_Q",
        "RPE_P_pct", "RPE_Q_pct", "snr",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.loc[:, list(required)].copy()
    out["model"] = label
    return out


def merge_model_predictions(models: list[tuple[str, str]]) -> pd.DataFrame:
    frames = []
    for run_dir, label in models:
        frames.append(load_model_predictions(RESULTS_DIR / run_dir, label))
    long = pd.concat(frames, ignore_index=True)

    # True values and SNR should match across models for a given event.
    truth = (
        long.groupby("event_idx", as_index=False)
        .agg(
            true_P=("true_P", "first"),
            true_Q=("true_Q", "first"),
            snr=("snr", "first"),
            n_models=("model", "nunique"),
        )
    )
    truth = truth[truth["n_models"] == len(models)].drop(columns=["n_models"])

    wide_parts = [truth]
    for _run_dir, label in models:
        sub = long.loc[long["model"] == label, ["event_idx", "pred_P", "pred_Q", "RPE_P_pct", "RPE_Q_pct"]]
        sub = sub.rename(
            columns={
                "pred_P": f"pred_P_{label}",
                "pred_Q": f"pred_Q_{label}",
                "RPE_P_pct": f"RPE_P_{label}",
                "RPE_Q_pct": f"RPE_Q_{label}",
            }
        )
        wide_parts.append(sub)
    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on="event_idx", how="inner")
    return wide


def select_example_events(wide: pd.DataFrame, n_examples: int, *, seed: int = SEED) -> np.ndarray:
    """Pick diverse events by |P|, SNR, and mean RPE across models."""
    n = len(wide)
    k = min(max(1, n_examples), n)
    if k >= n:
        return wide["event_idx"].to_numpy(dtype=int)

    abs_p = np.abs(wide["true_P"].to_numpy(dtype=float))
    snr = wide["snr"].to_numpy(dtype=float)
    rpe_cols = [c for c in wide.columns if c.startswith("RPE_P_")]
    mean_rpe = wide[rpe_cols].to_numpy(dtype=float).mean(axis=1)

    p_order = np.argsort(abs_p)
    snr_order = np.argsort(snr)
    rpe_order = np.argsort(mean_rpe)

    n_each = max(1, k // 3)
    picks = np.unique(
        np.concatenate([
            p_order[np.round(np.linspace(0, n - 1, n_each)).astype(int)],
            snr_order[np.round(np.linspace(0, n - 1, n_each)).astype(int)],
            rpe_order[np.round(np.linspace(0, n - 1, max(1, k - 2 * n_each))).astype(int)],
        ])
    )
    if picks.size < k:
        rng = np.random.default_rng(seed)
        extra = rng.choice(
            np.setdiff1d(np.arange(n), picks, assume_unique=False),
            size=k - picks.size,
            replace=False,
        )
        picks = np.concatenate([picks, extra])
    return np.sort(wide["event_idx"].to_numpy(dtype=int)[picks[:k]])


def load_noisy_test_spectra(spectra_path: Path, event_indices: np.ndarray) -> dict[int, np.ndarray]:
    """Rebuild the same noisy test inputs used during evaluation."""
    print(f"Loading spectra from {spectra_path} ...", flush=True)
    arrays = lstm_pq.load_lstm_npz(spectra_path)
    print("Rebuilding noisy test split (same seed/noise as training) ...", flush=True)
    _train_ds, _val_ds, test_ds, stats = lstm_pq.prepare_datasets(
        arrays, noise_std=NOISE_STD, seed=SEED,
    )
    del _train_ds, _val_ds, arrays
    noisy_ps = lstm_pq.test_input_ps(test_ds, stats)
    test_idx = np.asarray(stats["test_idx"], dtype=int)
    local_by_event = {int(e): i for i, e in enumerate(test_idx)}
    missing = [int(e) for e in event_indices if int(e) not in local_by_event]
    if missing:
        raise KeyError(f"Selected events not in reconstructed test split: {missing[:8]}")
    return {int(e): np.asarray(noisy_ps[local_by_event[int(e)]], dtype=np.float64) for e in event_indices}


def _fmt_pct(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{100.0 * value:.{digits}f}"


def _fmt_rpe(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _fmt_snr(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def build_comparison_table_data(row: pd.Series, model_labels: list[str]) -> tuple[list[str], list[list[str]]]:
    headers = ["Model", r"$P$ (%)", r"$Q$ (%)", r"RPE $P$ (%)", r"RPE $Q$ (%)", "SNR"]
    true_p = float(row["true_P"])
    true_q = float(row["true_Q"])
    snr = float(row["snr"])
    cells = [
        ["True", _fmt_pct(true_p), _fmt_pct(true_q), "—", "—", _fmt_snr(snr)],
    ]
    for label in model_labels:
        cells.append([
            label,
            _fmt_pct(float(row[f"pred_P_{label}"])),
            _fmt_pct(float(row[f"pred_Q_{label}"])),
            _fmt_rpe(float(row[f"RPE_P_{label}"])),
            _fmt_rpe(float(row[f"RPE_Q_{label}"])),
            _fmt_snr(snr),
        ])
    return headers, cells


def plot_example_with_table(
    freq: np.ndarray,
    ps: np.ndarray,
    row: pd.Series,
    model_labels: list[str],
    out_path: Path,
    *,
    meta: dict | None = None,
) -> None:
    headers, cells = build_comparison_table_data(row, model_labels)
    event_idx = int(row["event_idx"])
    title_bits = [f"event {event_idx}"]
    if meta:
        if "source" in meta:
            title_bits.append(str(meta["source"]))
        if "n_steps" in meta:
            title_bits.append(f"n_steps={int(meta['n_steps'])}")

    fig = plt.figure(figsize=(9.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")

    ax.plot(freq, ps, color="#1f2933", lw=1.7)
    ax.set_xlabel(r"$R$")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(float(freq[0]), float(freq[-1]))
    ax.set_title(", ".join(title_bits))
    _style_ax(ax)

    nrows = len(cells) + 1
    ncols = len(headers)
    table = Table(ax_tbl, bbox=[0.02, 0.05, 0.96, 0.9])
    width = 1.0 / ncols
    height = 1.0 / nrows
    for j, header in enumerate(headers):
        table.add_cell(
            0, j, width, height, text=header, loc="center",
            facecolor="#e8eef5", edgecolor="0.55",
        )
    for i, row_cells in enumerate(cells, start=1):
        face = "#f7f7f7" if i % 2 else "white"
        for j, text in enumerate(row_cells):
            cell = table.add_cell(
                i, j, width, height, text=text, loc="center",
                facecolor=face, edgecolor="0.55",
            )
            if j == 0:
                cell.get_text().set_fontweight("bold")
    for (i, j), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        cell.get_text().set_fontsize(10 if i == 0 else 9.5)
    ax_tbl.add_table(table)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_summary_csv(wide_sel: pd.DataFrame, model_labels: list[str], path: Path) -> Path:
    rows = []
    for _, row in wide_sel.iterrows():
        event_idx = int(row["event_idx"])
        true_p = float(row["true_P"])
        true_q = float(row["true_Q"])
        snr = float(row["snr"])
        rows.append({
            "event_idx": event_idx,
            "model": "True",
            "true_P": true_p,
            "true_Q": true_q,
            "pred_P": true_p,
            "pred_Q": true_q,
            "RPE_P_pct": np.nan,
            "RPE_Q_pct": np.nan,
            "snr": snr,
        })
        for label in model_labels:
            rows.append({
                "event_idx": event_idx,
                "model": label,
                "true_P": true_p,
                "true_Q": true_q,
                "pred_P": float(row[f"pred_P_{label}"]),
                "pred_Q": float(row[f"pred_Q_{label}"]),
                "RPE_P_pct": float(row[f"RPE_P_{label}"]),
                "RPE_Q_pct": float(row[f"RPE_Q_{label}"]),
                "snr": snr,
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def load_event_meta(spectra_path: Path, event_indices: np.ndarray) -> dict[int, dict]:
    with np.load(spectra_path, allow_pickle=False) as raw:
        source = np.asarray(raw["source"]).reshape(-1) if "source" in raw.files else None
        n_steps = np.asarray(raw["n_steps"]).reshape(-1)
        p0 = np.asarray(raw["p0"]).reshape(-1) if "p0" in raw.files else None
    meta = {}
    for e in event_indices:
        gi = int(e)
        entry = {"n_steps": int(n_steps[gi])}
        if source is not None:
            entry["source"] = SOURCE_NAME.get(int(source[gi]), str(int(source[gi])))
        if p0 is not None:
            entry["p0"] = float(p0[gi])
        meta[gi] = entry
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ~10 shared test examples with a multi-model P/Q table.",
    )
    parser.add_argument(
        "--spectra",
        type=Path,
        default=lstm_pq.DEFAULT_SPECTRA_PATH,
        help="Path to spectra.npz used for training/evaluation",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ANALYSIS_DIR / "example_comparisons_pub",
        help="Output directory for plots and summary CSV",
    )
    parser.add_argument("--n-examples", type=int, default=N_EXAMPLES_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    apply_pub_style()
    model_labels = [label for _run, label in MODELS]
    print("Loading per-model test predictions ...", flush=True)
    wide = merge_model_predictions(MODELS)
    print(f"Shared test events across {len(MODELS)} models: {len(wide)}", flush=True)

    event_indices = select_example_events(wide, args.n_examples, seed=args.seed)
    wide_sel = wide[wide["event_idx"].isin(event_indices)].sort_values("event_idx").reset_index(drop=True)
    print(f"Selected {len(wide_sel)} events: {event_indices.tolist()}", flush=True)

    spectra_by_event = load_noisy_test_spectra(args.spectra, event_indices)
    meta_by_event = load_event_meta(args.spectra, event_indices)

    # Frequency axis from first spectrum length.
    n_bins = next(iter(spectra_by_event.values())).shape[0]
    freq = np.linspace(SPECTRUM_R_MIN, SPECTRUM_R_MAX, n_bins)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for _, row in wide_sel.iterrows():
        event_idx = int(row["event_idx"])
        meta = meta_by_event.get(event_idx, {})
        steps = int(meta.get("n_steps", 0))
        out_path = out_dir / f"test_example_{event_idx:05d}_nsteps{steps:04d}.png"
        plot_example_with_table(
            freq,
            spectra_by_event[event_idx],
            row,
            model_labels,
            out_path,
            meta=meta,
        )
        saved.append(out_path)
        print(f"Saved {out_path}", flush=True)

    csv_path = save_summary_csv(wide_sel, model_labels, out_dir / "example_comparison_table.csv")
    print(f"Saved comparison table -> {csv_path}", flush=True)
    print(f"Done: {len(saved)} example plots in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
