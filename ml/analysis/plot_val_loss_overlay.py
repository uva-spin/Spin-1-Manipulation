"""Overlay validation loss for CNN, Seq2Seq, and MLP PQ runs."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ANALYSIS_DIR.parent / "results"

RUNS = [
    ("cnn/cnn_pq_results_v4", "CNN", "red"),
    ("lstm/lstm_result_v9", "LSTM", "green"),
    ("mlp/mlp_pq_results_v3", "MLP", "blue"),
    ("old_model/old_model_pq_results_v1", "Old Polarization Model", "purple"),
    ("transformer/transformer_pq_results_v2", "Transformer", "orange"),
]


def load_val_loss(run_dir: Path) -> np.ndarray:
    history = json.loads((run_dir / "history.json").read_text())
    return np.asarray(history["val_loss"], dtype=float)


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.6,
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    for dirname, label, color in RUNS:
        val = load_val_loss(RESULTS_DIR / dirname)
        epochs = np.arange(1, val.size + 1)
        ax.plot(epochs, val, color=color, lw=2.2, label=label)
        best_i = int(np.nanargmin(val))
        ax.scatter(
            [epochs[best_i]], [val[best_i]],
            s=48, color=color, zorder=5, edgecolors="white", linewidths=0.9,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel(r"Validation loss ($P + Q$)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=13, width=1.4, length=5)
    ax.grid(True, which="both", linestyle="--", color="0.85", linewidth=0.9)
    ax.legend(frameon=False, fontsize=13, loc="upper right")

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.6)

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    out_path = ANALYSIS_DIR / "val_loss_overlay_cnn_seq2seq_mlp.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
