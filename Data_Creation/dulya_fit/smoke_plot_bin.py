"""Plot a per-bin Dulya smoke shard (ssRF or AFP) for visual check."""

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

from common import AFP_SHARD_DIR, DATA_DIR, SSRF_SHARD_DIR  # noqa: E402
from afp_bin_traj import load_shard as load_afp_shard  # noqa: E402
from ssrf_bin_traj import load_shard as load_ssrf_shard  # noqa: E402


def _load_ssrf(path: Path) -> dict:
    return load_ssrf_shard(path)


def _load_afp(path: Path) -> dict:
    return load_afp_shard(path)


def plot_shard(mode: str, path: Path, out: Path) -> None:
    shard = _load_ssrf(path) if mode == "ssrf" else _load_afp(path)
    p_values = np.asarray(shard["p_values"], dtype=float)
    n_steps = np.asarray(shard["n_steps"], dtype=int)
    ps = np.asarray(shard["ps"], dtype=float)
    ps_m = np.asarray(shard["ps_m"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for j, p0 in enumerate(p_values):
        n = int(n_steps[j])
        if n <= 0:
            continue
        t = np.arange(n)
        axes[0].plot(t, ps[j, :n], lw=1.4, label=f"P={p0:+.2f}")
        axes[1].plot(t, ps_m[j, :n], lw=1.4, label=f"P={p0:+.2f}")

    axes[0].set_ylabel("Ps (center)")
    axes[0].set_title(
        f"Dulya {mode} bin {int(shard['bin_idx'])}  "
        f"R={float(shard['R']):.3f}  mirror={int(shard['mirror_idx'])}"
    )
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("step")
    axes[1].set_ylabel("Ps (mirror)")
    axes[1].legend(loc="best", fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("ssrf", "afp"), default="ssrf")
    p.add_argument("--bin-idx", type=int, default=172)
    args = p.parse_args()

    if args.mode == "ssrf":
        path = SSRF_SHARD_DIR / f"ssrf_bin_{args.bin_idx:04d}.npz"
    else:
        path = AFP_SHARD_DIR / f"afp_bin_{args.bin_idx:04d}.npz"
    out = DATA_DIR / f"smoke_dulya_{args.mode}_bin_{args.bin_idx:04d}.png"
    plot_shard(args.mode, path, out)


if __name__ == "__main__":
    main()
