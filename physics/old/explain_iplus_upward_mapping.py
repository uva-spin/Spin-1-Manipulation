"""
Demonstrate why BurnLookupMapper can increase I+ at the burn bin.

The mapper pools all polarizations at a fixed burn bin, then maps a target Ps
to the nearest stored Ps and returns that row's (Iplus, Iminus). Ps alone does
not fix the I+/I- split: many trajectories share nearly the same Ps with very
different channel values. A tiny burn from P=0.45 can therefore snap to a
heavily burned point from a higher-P trajectory where I+ is larger — i.e. it
jumps onto a different initial-polarization burn trajectory instead of advancing
along the current one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
_PHYSICS_DIR = REPO_ROOT / "physics"
_DATA_CREATION = REPO_ROOT / "Data_Creation"
if str(_PHYSICS_DIR) not in sys.path:
    sys.path.insert(0, str(_PHYSICS_DIR))

from burn_lookup_realtime import BurnTrajectoryConfig

from burnLookupMapper import BurnLookupMapper
from Lineshape import GenerateVectorLineshape

P = 0.45
BURN_FREQ = -0.92
BURN_AMP = 1.42e-4
NUM_BINS = 500
OUTPUT = REPO_ROOT / "results" / "current" / "lineshape" / "explain_iplus_upward_mapping.png"
LINESHAPE_OUTPUT = (
    REPO_ROOT / "results" / "current" / "lineshape" / "explain_iplus_upward_mapping_lineshape.png"
)


def nearest_row(df: pd.DataFrame, target_ps: float) -> pd.Series:
    idx = int(np.argmin(np.abs(df["ps_at_burn_bin"].to_numpy(float) - target_ps)))
    return df.iloc[idx]


def main() -> None:
    cfg = BurnTrajectoryConfig(n_bins=NUM_BINS)
    f = cfg.f
    burn_bin_idx = int(np.argmin(np.abs(f - BURN_FREQ)))

    lookup_path = _DATA_CREATION / "burn_lookup_table.pkl"
    df = pd.read_pickle(lookup_path)
    bin_df = df[df["burn_bin_idx"] == burn_bin_idx]

    traj45 = bin_df[np.isclose(bin_df["P"], P)].sort_values("burn_step")
    if traj45.empty:
        raise ValueError(f"No lookup rows for P={P} at burn_bin_idx={burn_bin_idx}")

    mapper = BurnLookupMapper(f, P, burn_bin_idx, bin_df)
    mapper.compute_lookup_index()

    step0 = traj45.iloc[0]
    ps_before = float(step0["ps_at_burn_bin"])
    iplus_before = float(step0["Iplus"])

    ps_after = ps_before - abs(BURN_AMP)

    matched_all = nearest_row(bin_df, ps_after)

    matched_p = float(matched_all["P"])
    matched_step = int(matched_all["burn_step"])
    traj_current = traj45
    traj_matched = bin_df[np.isclose(bin_df["P"], matched_p)].sort_values("burn_step")
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(
        traj_current["ps_at_burn_bin"],
        traj_current["Iplus"],
        "o-",
        ms=3,
        lw=1.5,
        label=fr"$I_+$ on current trajectory ($P={P:.2f}$)",
        color="tab:red",
    )
    ax.plot(
        traj_current["ps_at_burn_bin"],
        traj_current["Iminus"],
        "o-",
        ms=3,
        lw=1.5,
        label=fr"$I_-$ on current trajectory ($P={P:.2f}$)",
        color="tab:blue",
    )
    ax.plot(
        traj_matched["ps_at_burn_bin"],
        traj_matched["Iplus"],
        "o-",
        ms=3,
        lw=1.2,
        label=fr"$I_+$ on matched trajectory ($P={matched_p:+.2f}$)",
        color="tab:orange",
        alpha=0.9,
    )
    ax.plot(
        traj_matched["ps_at_burn_bin"],
        traj_matched["Iminus"],
        "o-",
        ms=3,
        lw=1.2,
        label=fr"$I_-$ on matched trajectory ($P={matched_p:+.2f}$)",
        color="tab:purple",
        alpha=0.9,
    )
    ax.axvline(ps_before, color="black", ls="--", lw=1, alpha=0.5, label="start $P_s$")
    ax.axvline(ps_after, color="green", ls=":", lw=1.2, label="$P_s$ after tiny burn")
    # ax.scatter(
    #     [float(matched_all["ps_at_burn_bin"])],
    #     [float(matched_all["Iplus"])],
    #     s=110,
    #     c="tab:orange",
    #     marker="*",
    #     zorder=5,
    #     edgecolors="black",
    #     linewidths=0.4,
    # )
    # ax.scatter(
    #     [float(matched_all["ps_at_burn_bin"])],
    #     [float(matched_all["Iminus"])],
    #     s=110,
    #     c="tab:purple",
    #     marker="*",
    #     zorder=5,
    #     edgecolors="black",
    #     linewidths=0.4,
    # )
    ax.scatter(
        [ps_before],
        [iplus_before],
        s=70,
        c="tab:red",
        marker="o",
        zorder=6,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_xlabel(r"$P_s$ at burn bin")
    ax.set_ylabel("intensity")
    ax.set_title(
        "Burn Trajectories with Different Initial Polarizations"
    )
    ax.legend(loc="upper right", fontsize=6.5)
    ax.grid(True, alpha=0.3)
    ps_lo = min(
        float(traj_current["ps_at_burn_bin"].min()),
        float(traj_matched["ps_at_burn_bin"].min()),
    )
    ps_hi = max(ps_before, float(traj_matched["ps_at_burn_bin"].max()))
    ps_pad = 0.05 * (ps_hi - ps_lo)
    ax.set_xlim(ps_lo - ps_pad, ps_hi + ps_pad)

    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {OUTPUT}")

    _, iplus0, iminus0 = GenerateVectorLineshape(P, f)
    signal0 = iplus0 + iminus0
    _ps_burned, iplus_burned, iminus_burned = mapper.apply_bin_burn(
        signal0, iplus0, iminus0, burn_bin_idx, BURN_AMP
    )
    
    print(f"Difference in ps_burned - ( i+ + i- ) at burn bin: {_ps_burned[burn_bin_idx] - (iplus_burned[burn_bin_idx] + iminus_burned[burn_bin_idx]):.6e}")
    
    fig_ls, ax_ls = plt.subplots(figsize=(10, 5))
    ax_ls.plot(f, signal0, color="black", alpha=0.35, lw=1.2, label=fr"$P_s$ before ($P={P:.2f}$)")
    ax_ls.plot(f, iplus0, "--", color="tab:red", alpha=0.45, lw=1.0, label=r"$I_+$ before")
    ax_ls.plot(f, iminus0, "--", color="tab:blue", alpha=0.45, lw=1.0, label=r"$I_-$ before")
    ax_ls.plot(
        f,
        iplus_burned + iminus_burned,
        color="black",
        lw=2.0,
        label=fr"$P_s$ after burn (amp={BURN_AMP:.2e})",
    )
    ax_ls.plot(
        f,
        iplus_burned,
        color="tab:orange",
        lw=1.8,
        label=fr"$I_+$ after (mapped, bin={iplus_burned[burn_bin_idx]:.4g})",
        linestyle="--",
    )
    ax_ls.plot(
        f,
        iminus_burned,
        color="tab:purple",
        lw=1.8,
        label=fr"$I_-$ after (mapped, bin={iminus_burned[burn_bin_idx]:.4g})",
        linestyle="--",
    )
    ax_ls.axvline(
        f[burn_bin_idx],
        color="gray",
        ls=":",
        lw=1.2,
        alpha=0.8,
        label=fr"burn bin ($f={BURN_FREQ:.2f}$ MHz)",
    )
    ax_ls.set_xlabel("Frequency (MHz)")
    ax_ls.set_ylabel("Amplitude")
    ax_ls.set_title(
        rf"Burned lineshape after nearest-$P_s$ mapping at $f={BURN_FREQ:.2f}$ MHz "
        rf"($P={P:.2f}$, matched $P={matched_p:+.2f}$ step {matched_step})"
    )
    ax_ls.legend(loc="upper right", fontsize=8)
    ax_ls.grid(True, alpha=0.3)
    fig_ls.tight_layout()
    fig_ls.savefig(LINESHAPE_OUTPUT, dpi=150)
    plt.close(fig_ls)
    print(f"Saved lineshape figure to {LINESHAPE_OUTPUT}")

if __name__ == "__main__":
    main()
