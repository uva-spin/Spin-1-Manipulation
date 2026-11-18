#!/usr/bin/env python3
"""Reproduce ssRF-beta-style tensor design panels (intensity + RF command U)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from physics.rf.kinetic_defaults import (
    SSRF_BETA_DESIGN_N_BINS,
    SSRF_BETA_DESIGN_R_MAX,
    SSRF_BETA_DESIGN_R_MIN,
)
from physics.rf.optimal_profile import (
    DATA_GEN_SETTINGS,
    SSRF_BETA_OPTIMIZER_SETTINGS,
    run_optimal_profile_polarization,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0", type=float, default=0.45)
    parser.add_argument("--fast", action="store_true", help="Use DATA_GEN_SETTINGS")
    parser.add_argument("--out", type=Path, default=Path("/opt/cursor/artifacts/ssrf_beta_tensor_demo.png"))
    args = parser.parse_args()
    settings = DATA_GEN_SETTINGS if args.fast else SSRF_BETA_OPTIMIZER_SETTINGS
    traj = run_optimal_profile_polarization(
        args.p0,
        n_steps=0,
        n_bins=SSRF_BETA_DESIGN_N_BINS,
        r_min=SSRF_BETA_DESIGN_R_MIN,
        r_max=SSRF_BETA_DESIGN_R_MAX,
        settings=settings,
    )
    r = np.asarray(traj["frequency"], dtype=float)
    end = int(traj["program_end_step"])
    ip0, im0, tot0 = _plot_units(traj, 0)
    ip1, im1, tot1 = _plot_units(traj, end)
    q0, q1 = float(traj["q_initial"]), float(traj["q_final"])
    T = float(traj["duration"])
    rates = np.asarray(traj["rates"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(r, tot0, color="#1f77b4", lw=1.6, label="input I+ + I-")
    ax.plot(r, tot1, color="#ff7f0e", lw=1.6, label="after the pulse")
    ax.set_xlim(-6, 6)
    ax.set_ylabel("intensity")
    ax.set_xlabel("R")
    ax.set_title(
        f"Spin-1-Manipulation: Q {q0:.3f} → {q1:.3f} (ΔQ {q1 - q0:+.3f}), T = {T:g}"
    )
    ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    ax.plot(r, rates, color="#1e5631", lw=1.6)
    ax.set_xlim(-6, 6)
    ax.set_ylabel("RF command U")
    ax.set_xlabel("R")
    ax.set_title(f"optimized profile, total power {rates.sum():.3g} U")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


def _plot_units(traj, frame: int):
    from physics.rf.profile_control import make_ideal_model

    model = make_ideal_model(float(traj["polarization"]))
    iplus = np.asarray(traj["iplus_full"][frame])
    iminus = np.asarray(traj["iminus_full"][frame])
    scale = 1.0 / (max(abs(model.display_cal), 1e-30) * model.params.plot_divisor)
    tot = (iplus + iminus) * scale
    return iplus * scale, iminus * scale, tot


if __name__ == "__main__":
    main()