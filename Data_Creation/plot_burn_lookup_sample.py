"""
Plot a sample burn trajectory from burn_lookup_table.pkl.

By default, selects one (P, burn_bin_idx) trajectory and compares the unburned
state (burn_step=0) with the final burned state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LOOKUP = SCRIPT_DIR / "burn_lookup_table_smoke.pkl"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "current" / "data_creation" / "burn_lookup_example.png"

NUM_BINS = 500
F_MHZ = np.linspace(-3.0, 3.0, NUM_BINS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a sample burn trajectory from burn_lookup_table.pkl."
    )
    parser.add_argument(
        "--lookup-table",
        type=Path,
        default=DEFAULT_LOOKUP,
        help="Path to burn_lookup_table.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path",
    )
    parser.add_argument(
        "--polarization",
        type=float,
        default=None,
        help="Trajectory polarization P (default: median P in table)",
    )
    parser.add_argument(
        "--burn-bin-idx",
        type=int,
        default=None,
        help="Burn bin index (default: median for chosen P; with --random, fixes bin and randomizes P)",
    )
    parser.add_argument(
        "--burn-step",
        type=int,
        default=None,
        help="Plot this burn step instead of the final step (still compares to step 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --random (omit for a different P each run)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomize P (and burn_bin_idx if not given) instead of using medians",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving",
    )
    return parser.parse_args()


def _choose_trajectory_keys(
    df: pd.DataFrame,
    *,
    polarization: float | None,
    burn_bin_idx: int | None,
    random_pick: bool,
    seed: int | None,
) -> tuple[float, int]:
    if polarization is not None and burn_bin_idx is not None:
        return float(polarization), int(burn_bin_idx)

    keys = df[["P", "burn_bin_idx"]].drop_duplicates()
    if random_pick:
        rng = np.random.default_rng(seed)
        if burn_bin_idx is not None:
            candidates = keys[keys["burn_bin_idx"] == burn_bin_idx]
            if candidates.empty:
                raise ValueError(f"No trajectories found for burn_bin_idx={burn_bin_idx}")
            row = candidates.iloc[int(rng.integers(0, len(candidates)))]
            return float(row["P"]), int(burn_bin_idx)

        row = keys.iloc[int(rng.integers(0, len(keys)))]
        return float(row["P"]), int(row["burn_bin_idx"])

    if polarization is None:
        polarization = float(np.median(keys["P"].to_numpy(dtype=float)))
    p_matches = keys[np.isclose(keys["P"].astype(float), float(polarization))]
    if p_matches.empty:
        raise ValueError(f"No trajectories found for P={polarization}")

    if burn_bin_idx is None:
        burn_bin_idx = int(np.median(p_matches["burn_bin_idx"].to_numpy(dtype=int)))
    return float(polarization), int(burn_bin_idx)


def load_trajectory(
    df: pd.DataFrame,
    polarization: float,
    burn_bin_idx: int,
) -> pd.DataFrame:
    mask = np.isclose(df["P"].astype(float), polarization) & (df["burn_bin_idx"] == burn_bin_idx)
    traj = df.loc[mask].sort_values("burn_step")
    if traj.empty:
        raise ValueError(
            f"No rows for P={polarization} and burn_bin_idx={burn_bin_idx}"
        )
    return traj


def _row_spectrum(row: pd.Series, burn_bin_idx: int) -> tuple[float, float, float]:
    """Return Ps, Iplus, and Iminus at the burn bin from scalar or legacy rows."""
    arr_ps = np.asarray(row["Ps"], dtype=float)
    arr_iplus = np.asarray(row["Iplus"], dtype=float)
    arr_iminus = np.asarray(row["Iminus"], dtype=float)
    if arr_ps.size == 1:
        return float(arr_ps), float(arr_iplus), float(arr_iminus)
    burn_idx = int(burn_bin_idx)
    return float(arr_ps[burn_idx]), float(arr_iplus[burn_idx]), float(arr_iminus[burn_idx])


def _has_full_spectrum(row: pd.Series) -> bool:
    return np.asarray(row["Ps"]).size > 1


def plot_burn_trajectory(
    trajectory: pd.DataFrame,
    *,
    burn_step: int | None,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    unburned = trajectory.loc[trajectory["burn_step"] == 0].iloc[0]
    if burn_step is None:
        burned = trajectory.iloc[-1]
    else:
        matches = trajectory.loc[trajectory["burn_step"] == burn_step]
        if matches.empty:
            available = trajectory["burn_step"].to_numpy(dtype=int).tolist()
            raise ValueError(
                f"burn_step={burn_step} not found; available steps: {available[:10]}..."
            )
        burned = matches.iloc[0]

    burn_freq = float(unburned["burn_freq"])
    polarization = float(unburned["P"])
    burn_bin_idx = int(unburned["burn_bin_idx"])
    final_step = int(burned["burn_step"])

    if _has_full_spectrum(unburned):
        ps_unburned = np.asarray(unburned["Ps"], dtype=float)
        ps_burned = np.asarray(burned["Ps"], dtype=float)
        iplus_unburned = np.asarray(unburned["Iplus"], dtype=float)
        iplus_burned = np.asarray(burned["Iplus"], dtype=float)
        iminus_unburned = np.asarray(unburned["Iminus"], dtype=float)
        iminus_burned = np.asarray(burned["Iminus"], dtype=float)
        burn_bin_only = False
    else:
        ps_unburned, iplus_unburned, iminus_unburned = _row_spectrum(unburned, burn_bin_idx)
        ps_burned, iplus_burned, iminus_burned = _row_spectrum(burned, burn_bin_idx)
        burn_bin_only = True

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.2),
        sharex=True,
        gridspec_kw={"wspace": 0.28},
        layout="constrained",
    )

    ax_ps, ax_sig = axes
    if burn_bin_only:
        ax_ps.scatter([burn_freq], [ps_unburned], color="gray", marker="o", s=45, label=r"$P_s$ unburned")
        ax_ps.scatter([burn_freq], [ps_burned], color="black", marker="o", s=55, label=rf"$P_s$ step {final_step}")
        ax_sig.scatter([burn_freq], [iplus_unburned], color="tab:red", marker="o", s=45, label=r"$I_+$ unburned")
        ax_sig.scatter([burn_freq], [iplus_burned], color="tab:red", marker="s", s=55, label=rf"$I_+$ step {final_step}")
        ax_sig.scatter([burn_freq], [iminus_unburned], color="tab:blue", marker="o", s=45, label=r"$I_-$ unburned")
        ax_sig.scatter([burn_freq], [iminus_burned], color="tab:blue", marker="s", s=55, label=rf"$I_-$ step {final_step}")
    else:
        ax_ps.plot(F_MHZ, ps_unburned, color="gray", alpha=0.65, linestyle="--", linewidth=1.0, label=r"$P_s$ unburned")
        ax_ps.plot(F_MHZ, ps_burned, color="black", linestyle="-", linewidth=1.25, label=rf"$P_s$ step {final_step}")
        ax_sig.plot(F_MHZ, iplus_unburned, color="tab:red", alpha=0.45, linestyle="--", linewidth=1.0, label=r"$I_+$ unburned")
        ax_sig.plot(F_MHZ, iplus_burned, color="tab:red", linestyle="-", linewidth=1.25, alpha=0.9, label=rf"$I_+$ step {final_step}")
        ax_sig.plot(F_MHZ, iminus_unburned, color="tab:blue", alpha=0.45, linestyle="--", linewidth=1.0, label=r"$I_-$ unburned")
        ax_sig.plot(F_MHZ, iminus_burned, color="tab:blue", linestyle="-", linewidth=1.25, alpha=0.9, label=rf"$I_-$ step {final_step}")
    ax_ps.axvline(burn_freq, color="green", alpha=0.55, linestyle=":", linewidth=1.2, label="burn freq")
    ax_ps.set_ylabel(r"$P_s$")
    ax_ps.grid(True, alpha=0.3)
    ax_ps.legend(loc="upper right", fontsize=8)
    ax_ps.set_title(
        f"P = {polarization:.3f}, burn bin = {burn_bin_idx}, "
        f"f_burn = {burn_freq:.3f} MHz, steps = {final_step}",
        fontsize=10,
    )

    ax_sig.axvline(burn_freq, color="green", alpha=0.55, linestyle=":", linewidth=1.2)
    ax_sig.set_ylabel(r"$I_\pm$")
    ax_sig.grid(True, alpha=0.3)
    ax_sig.legend(loc="upper right", fontsize=8)
    ax_sig.set_title("Iplus / Iminus lineshape", fontsize=10)

    for ax in axes:
        ax.set_xlabel("frequency (MHz)")

    fig.suptitle(
        "Burn lookup sample: burn-bin values only"
        if burn_bin_only
        else "Burn lookup sample: dashed = unburned (step 0), solid = burned",
        fontsize=12,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    lookup_path = Path(args.lookup_table)
    if not lookup_path.is_file():
        raise FileNotFoundError(f"Lookup table not found: {lookup_path}")

    print(f"Loading {lookup_path} ...")
    df = pd.read_pickle(lookup_path)
    print(f"Loaded {len(df)} rows")

    polarization, burn_bin_idx = _choose_trajectory_keys(
        df,
        polarization=args.polarization,
        burn_bin_idx=args.burn_bin_idx,
        random_pick=args.random,
        seed=args.seed,
    )
    trajectory = load_trajectory(df, polarization, burn_bin_idx)
    print(
        f"Plotting trajectory P={polarization:.3f}, burn_bin_idx={burn_bin_idx}, "
        f"steps={len(trajectory)}"
    )

    plot_burn_trajectory(
        trajectory,
        burn_step=args.burn_step,
        output_path=args.output,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
