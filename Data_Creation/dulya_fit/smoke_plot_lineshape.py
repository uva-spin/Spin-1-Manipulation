"""Smoke-plot full Dulya-fit lineshapes at several polarizations."""

from __future__ import annotations

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
from lineshape import GenerateDulyaLineshape, shape_params_from_fit  # noqa: E402

SMOKE_P = (-0.9, -0.5, -0.25, 0.25, 0.5, 0.9)
OUT_PATH = DATA_DIR / "smoke_dulya_full_lineshape.png"


def main() -> None:
    shape = shape_params_from_fit()
    f = np.asarray(FREQUENCY, dtype=float)
    print("Shape params:", ", ".join(f"{k}={v:.6g}" for k, v in shape.items()))
    print(f"R grid: [{F_MIN}, {F_MAX}]  n={NUM_BINS}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for p in SMOKE_P:
        ps, ip, im = GenerateDulyaLineshape(float(p), f, shape)
        axes[0].plot(f, ps, lw=1.6, label=f"P={p:+.2f}  ΣPs={np.sum(ps):+.3f}")
        axes[1].plot(f, ip, lw=1.2, label=f"I+ P={p:+.2f}")
        axes[1].plot(f, im, lw=1.2, ls="--", label=f"I- P={p:+.2f}")

    axes[0].axhline(0.0, color="0.5", lw=0.8)
    axes[0].set_ylabel("Ps = I+ + I-")
    axes[0].set_title("Dulya-fit full lineshape smoke test (frozen shape, vary P)")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_xlabel("R")
    axes[1].set_ylabel("I+ / I-")
    axes[1].legend(loc="upper right", fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
