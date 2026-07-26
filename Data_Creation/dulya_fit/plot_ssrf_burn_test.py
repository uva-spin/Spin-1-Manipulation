"""
ssRF burn smoke test: full-spectrum before/after + burn/mirror trajectories.

Examples:
  python Data_Creation/dulya_fit/plot_ssrf_burn_test.py --gamma-rf 5.0
  python Data_Creation/dulya_fit/plot_ssrf_burn_test.py --gamma-rf 5.0 --p 0.48 --burn-bin 228
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

from common import DATA_DIR, FREQUENCY, MAX_BURN_STEPS  # noqa: E402
from generate import _shade_burn_mirror, _ssrf_support_and_mirrors  # noqa: E402
from lineshape import GenerateDulyaLineshape, shape_params_from_fit  # noqa: E402
from manipulate import run_event  # noqa: E402
from ssrf_bin_traj import mirror_bin_idx, run_one_polarization  # noqa: E402


def plot_ssrf_burn_test(
    *,
    gamma_rf: float,
    polarization: float = 0.48,
    burn_bin: int = 228,
    n_steps: int = MAX_BURN_STEPS,
    max_traj_steps: int = 2000,
    output_path: Path | None = None,
) -> Path:
    shape = shape_params_from_fit()
    f = np.asarray(FREQUENCY, dtype=float)

    row = run_event(
        polarization=float(polarization),
        mode="ssrf",
        burn_bin=int(burn_bin),
        gamma_rf=float(gamma_rf),
        n_steps=int(n_steps),
        shape_params=shape,
    )
    row["frequency"] = f

    traj = run_one_polarization(
        int(burn_bin),
        float(polarization),
        gamma_rf=float(gamma_rf),
        max_steps=int(max_traj_steps),
        shape_params=shape,
    )
    mirror_idx = mirror_bin_idx(len(f), int(burn_bin))

    _, ip_u, im_u = GenerateDulyaLineshape(float(polarization), f, shape)
    ps_u = np.asarray(ip_u + im_u, dtype=float)
    ps_f = np.asarray(row["Ps"], dtype=float)
    support, mirrors, burn, mirror = _ssrf_support_and_mirrors(row, len(f))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.2, 1.0), hspace=0.32, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(f, ps_u, color="0.65", lw=1.4, label="Ps unmanipulated")
    ax0.plot(f, ps_f, color="C0", lw=2.0, label="Ps after ssRF")
    _shade_burn_mirror(ax0, f, support, mirrors, burn, mirror, label=True)
    ax0.set_xlim(float(f.min()), float(f.max()))
    ax0.set_ylabel("Ps")
    ax0.set_xlabel("R")
    ax0.set_title(
        f"ssRF burn  P={polarization:.2f}  gamma_rf={gamma_rf}  "
        f"burn R={float(row['burn_freq']):.3f}  steps={int(row['burn_step'])}  "
        f"Ps/Ps0={float(row['ps_ratio']):.3f}  "
        f"mirror/burn={float(row['mirror_over_burn_area']):.3f}"
    )
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    if not traj["skipped"]:
        n = int(traj["n_steps"])
        t = np.arange(n)
        ax1.plot(t, traj["ps"][:n], "C3.-", ms=3, lw=1.6, label="Ps @ burn")
        ax1.plot(t, traj["iplus"][:n], "C0--", ms=2, lw=1.0, alpha=0.8, label="I+ burn")
        ax1.plot(t, traj["iminus"][:n], "C4--", ms=2, lw=1.0, alpha=0.8, label="I- burn")
        ax2.plot(t, traj["ps_m"][:n], "C2.-", ms=3, lw=1.6, label="Ps @ mirror")
        ax2.plot(t, traj["iplus_m"][:n], "C0--", ms=2, lw=1.0, alpha=0.8, label="I+ mirror")
        ax2.plot(t, traj["iminus_m"][:n], "C4--", ms=2, lw=1.0, alpha=0.8, label="I- mirror")
        ax1.set_title(f"Burn bin {burn_bin}  ({n} steps, {traj.get('stop_reason', '')})")
        ax2.set_title(f"Mirror bin {mirror_idx}")
    else:
        ax1.text(
            0.5,
            0.5,
            "trajectory skipped",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )

    for ax in (ax1, ax2):
        ax.set_xlabel("step")
        ax.set_ylabel("intensity (fit scale)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Dulya-fit ssRF burn test @ gamma_rf={gamma_rf}",
        fontsize=12,
        y=0.98,
    )

    if output_path is None:
        output_path = DATA_DIR / f"ssrf_burn_gamma{gamma_rf:.1f}_P{polarization:.2f}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    p = argparse.ArgumentParser(description="Plot ssRF burn test at a given gamma_rf")
    p.add_argument("--gamma-rf", type=float, default=1.0)
    p.add_argument("--p", type=float, default=0.48, dest="polarization")
    p.add_argument("--burn-bin", type=int, default=228)
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--max-traj-steps", type=int, default=3000)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    out = plot_ssrf_burn_test(
        gamma_rf=float(args.gamma_rf),
        polarization=float(args.polarization),
        burn_bin=int(args.burn_bin),
        n_steps=int(args.n_steps),
        max_traj_steps=int(args.max_traj_steps),
        output_path=args.output,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
