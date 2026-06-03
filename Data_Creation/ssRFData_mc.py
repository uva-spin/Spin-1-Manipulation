"""
Monte Carlo training data generation for ssRF burn correction.

Samples random initial polarizations, burn parameters, and AFP sweeps, then
accepts only events whose final total vector polarization (after burn + AFP)
falls within a target band, e.g. 0.50–0.55 (not percent).

Final vector polarization:
    P_final = sum(Ps_burned)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "current" / "data_creation_mc"

sys.path.insert(0, str(REPO_ROOT))

from physics.afp import AFP
from physics.lineshape.Lineshape import GenerateVectorLineshape
from ssRFMapper import ssRFMapper

# Bins within this distance of the spectrum center (n_bins // 2) skip AFP.
AFP_CENTER_EXCLUSION_BINS = 5


def _afp_center_forbidden_indices(
    n_bins: int, margin: int = AFP_CENTER_EXCLUSION_BINS
) -> frozenset[int]:
    centre = n_bins // 2
    lo = max(0, centre - margin)
    hi = min(n_bins - 1, centre + margin)
    return frozenset(range(lo, hi + 1))


def _random_afp_bin_range(n_bins: int, min_width: int, max_width: int) -> tuple[int, int]:
    min_width = int(np.clip(min_width, 1, n_bins))
    max_width = int(np.clip(max_width, min_width, n_bins))
    width = np.random.randint(min_width, max_width + 1)
    start = np.random.randint(0, n_bins - width + 1)
    return start, start + width


def _apply_afp_sweep(
    Iplus: np.ndarray,
    Iminus: np.ndarray,
    bin_range: tuple[int, int],
    efficiency: float,
    center_margin: int = AFP_CENTER_EXCLUSION_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    total_area = np.sum(Iplus + Iminus)
    afp = AFP.from_intensities(Iplus, Iminus)
    start, stop = int(bin_range[0]), int(bin_range[1])
    forbidden = _afp_center_forbidden_indices(len(Iplus), center_margin)
    subset = [i for i in range(start, stop) if i not in forbidden]
    if subset:
        afp.perform_afp(subset_indices=subset, efficiency=efficiency)
    Iplus_afp, Iminus_afp = afp.to_intensities()

    Iplus[:] = Iplus_afp * total_area
    Iminus[:] = Iminus_afp * total_area
    return Iplus, Iminus


def _final_vector_pol(Ps: np.ndarray) -> float:
    return float(np.sum(Ps))


def _build_viz_sample(f: np.ndarray, row: dict) -> dict:
    """Build unburned/burned population + lineshape dict for plotting."""
    Ps_base, Iplus_base, Iminus_base = GenerateVectorLineshape(row["P_initial"], f)
    rp0, rz0, rm0 = AFP.intensities_to_populations(
        np.asarray(Iplus_base, dtype=float),
        np.asarray(Iminus_base, dtype=float),
    )
    rho_plus, rho_zero, rho_minus = AFP.intensities_to_populations(
        np.asarray(row["Iplus"], dtype=float),
        np.asarray(row["Iminus"], dtype=float),
    )
    return {
        "P": row["P_initial"],
        "P_initial": row["P_initial"],
        "P_final": row["P_final"],
        "x0": row["x0"],
        "amp": row["amp"],
        "amp_second": row.get("amp_second", 0.0),
        "x0_second": row["x0_second"],
        "afp_sweep_injected": row["afp_sweep_injected"],
        "afp_bin_start": row["afp_bin_start"],
        "afp_bin_stop": row["afp_bin_stop"],
        "Ps_unburned": np.asarray(Ps_base, dtype=float),
        "Iplus_unburned": np.asarray(Iplus_base, dtype=float),
        "Iminus_unburned": np.asarray(Iminus_base, dtype=float),
        "Ps": np.asarray(row["Ps"], dtype=float),
        "Iplus": np.asarray(row["Iplus"], dtype=float),
        "Iminus": np.asarray(row["Iminus"], dtype=float),
        "rho_plus_unburned": rp0,
        "rho_zero_unburned": rz0,
        "rho_minus_unburned": rm0,
        "rho_plus": rho_plus,
        "rho_zero": rho_zero,
        "rho_minus": rho_minus,
    }


def _save_population_burn_figure(
    f_arr: np.ndarray,
    samples: list,
    output_path: Path,
    suptitle_extra: str = "",
    dpi: int = 200,
) -> None:
    """n rows x 2 cols: [level populations | lineshape: Ps and I+/- unburned vs burned]."""
    n = len(samples)
    if n == 0:
        return
    row_h = 3.35
    fig_h = max(4.0, row_h * n + 0.9)
    fig_w = 11.8
    fig, axes = plt.subplots(
        n,
        2,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=False,
        gridspec_kw={"width_ratios": [1.15, 1.05], "wspace": 0.28, "hspace": 0.32},
        layout="constrained",
    )
    if n == 1:
        axes = np.asarray(axes).reshape(1, -1)
    title_fs = 10

    def _burn_vlines(ax, sample):
        ax.axvline(sample["x0"], color="green", alpha=0.45, linestyle=":", linewidth=1)
        ax.axvline(-sample["x0"], color="purple", alpha=0.45, linestyle=":", linewidth=1)
        x0_2 = sample.get("x0_second", np.nan)
        if pd.notna(x0_2):
            ax.axvline(float(x0_2), color="darkorange", alpha=0.45, linestyle=":", linewidth=1)
            ax.axvline(-float(x0_2), color="coral", alpha=0.45, linestyle=":", linewidth=1)

    def _afp_spans(ax, sample):
        if not sample.get("afp_sweep_injected", False):
            return
        start = int(sample["afp_bin_start"])
        stop = int(sample["afp_bin_stop"])
        if start < 0 or stop <= start:
            return
        ax.axvspan(f_arr[start], f_arr[stop - 1], color="gold", alpha=0.16)
        mirror_start = len(f_arr) - stop
        mirror_stop = len(f_arr) - start
        if mirror_start != start or mirror_stop != stop:
            ax.axvspan(f_arr[mirror_start], f_arr[mirror_stop - 1], color="gold", alpha=0.08)

    for row_idx, sample in enumerate(samples):
        ax_pop = axes[row_idx, 0]
        ax_sig = axes[row_idx, 1]

        ax_pop.plot(
            f_arr, sample["rho_plus_unburned"], color="tab:red", linestyle="--", alpha=0.55, linewidth=1.0
        )
        ax_pop.plot(
            f_arr, sample["rho_zero_unburned"], color="tab:gray", linestyle="--", alpha=0.55, linewidth=1.0
        )
        ax_pop.plot(
            f_arr, sample["rho_minus_unburned"], color="tab:blue", linestyle="--", alpha=0.55, linewidth=1.0
        )
        ax_pop.plot(f_arr, sample["rho_plus"], color="tab:red", linestyle="-", linewidth=1.25)
        ax_pop.plot(f_arr, sample["rho_zero"], color="tab:gray", linestyle="-", linewidth=1.25)
        ax_pop.plot(f_arr, sample["rho_minus"], color="tab:blue", linestyle="-", linewidth=1.25)
        _afp_spans(ax_pop, sample)
        _burn_vlines(ax_pop, sample)
        ax_pop.set_ylabel("population (norm.)")
        ax_pop.grid(True, alpha=0.3)
        ax_pop.set_title(
            f"P_init = {sample['P_initial']:.3f}, P_final = {sample['P_final']:.3f}, "
            f"x0 = {sample['x0']:.2f}, AFP = {sample['afp_bin_start']}:{sample['afp_bin_stop']}",
            fontsize=title_fs,
        )

        ax_sig.plot(
            f_arr, sample["Ps_unburned"], color="gray", alpha=0.6, linestyle="--", linewidth=1.0
        )
        ax_sig.plot(f_arr, sample["Ps"], color="black", linestyle="-", linewidth=1.25)
        ax_sig.plot(
            f_arr, sample["Iplus_unburned"], color="tab:red", alpha=0.4, linestyle="--", linewidth=1.0
        )
        ax_sig.plot(f_arr, sample["Iplus"], color="tab:red", linestyle="-", linewidth=1.25, alpha=0.9)
        ax_sig.plot(
            f_arr, sample["Iminus_unburned"], color="tab:blue", alpha=0.4, linestyle="--", linewidth=1.0
        )
        ax_sig.plot(f_arr, sample["Iminus"], color="tab:blue", linestyle="-", linewidth=1.25, alpha=0.9)
        _afp_spans(ax_sig, sample)
        _burn_vlines(ax_sig, sample)
        ax_sig.set_ylabel(r"$P_s$, $I_\pm$")
        ax_sig.grid(True, alpha=0.3)
        ax_sig.set_title("lineshape", fontsize=title_fs)

    for ax in axes[-1, :]:
        ax.set_xlabel("frequency")

    title = "Spin-1 populations and lineshape vs frequency (MC ss-RF burn)"
    if suptitle_extra:
        title = f"{title} - {suptitle_extra}"
    fig.suptitle(title, fontsize=13)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def _save_grid_plot(
    f_arr: np.ndarray,
    samples: list,
    output_path: Path,
    P_final_bounds: tuple[float, float],
    dpi: int = 600,
) -> None:
    grid_size = int(np.ceil(np.sqrt(len(samples))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = np.atleast_2d(axes).flatten()

    for idx, data in enumerate(samples):
        if idx >= len(axes):
            break
        axes[idx].plot(f_arr, data["Ps_unburned"], color="gray", alpha=0.5, linestyle="--", linewidth=0.8)
        axes[idx].plot(f_arr, data["Iplus_unburned"], color="red", alpha=0.2, linestyle="--", linewidth=1)
        axes[idx].plot(f_arr, data["Iminus_unburned"], color="blue", alpha=0.2, linestyle="--", linewidth=1)
        axes[idx].plot(f_arr, data["Ps"], alpha=1.0, linewidth=1, color="black")
        axes[idx].plot(f_arr, data["Iplus"], alpha=0.3, linestyle="-", linewidth=2, color="red")
        axes[idx].plot(f_arr, data["Iminus"], alpha=0.3, linestyle="-", linewidth=2, color="blue")
        if data["afp_sweep_injected"]:
            axes[idx].axvspan(
                f_arr[data["afp_bin_start"]],
                f_arr[data["afp_bin_stop"] - 1],
                color="gold",
                alpha=0.16,
            )
        axes[idx].axvline(data["x0"], color="green", alpha=0.5, linestyle=":", linewidth=1)
        axes[idx].axvline(-data["x0"], color="purple", alpha=0.5, linestyle=":", linewidth=1)
        axes[idx].set_title(
            f"P_init = {data['P_initial']:.3f}, P_final = {data['P_final']:.3f}, "
            f"x0 = {data['x0']:.2f}, amp = {data['amp']:.2e}, "
            f"x0_2 = {data['x0_second']:.2f}, AFP = {data['afp_bin_start']}:{data['afp_bin_stop']}",
            fontsize=10,
        )
        axes[idx].grid(True, alpha=0.3)

    for idx in range(len(samples), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"MC events with final vector pol in [{P_final_bounds[0]:.3f}, {P_final_bounds[1]:.3f}]\n"
        "dashed = unburned, solid = after burn + AFP",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def _sample_and_manipulate(
    mapper: ssRFMapper,
    f: np.ndarray,
    *,
    P_initial_bounds: tuple[float, float],
    burn_locations: np.ndarray,
    amp_min: float,
    amp_max: float,
    second_burn_probability: float,
    afp_sweep_probability: float,
    afp_min_sweep_bins: int,
    afp_max_sweep_bins: int,
    afp_efficiency: float,
    num_bins: int,
) -> dict:
    P_initial = np.random.uniform(P_initial_bounds[0], P_initial_bounds[1])
    Ps_base, Iplus_base, Iminus_base = GenerateVectorLineshape(P_initial, f)
    true_Qs = float(np.sum(Iplus_base - Iminus_base))

    x0 = np.random.choice(burn_locations)
    amp = np.random.uniform(amp_min, amp_max)
    inject_second_burn = np.random.rand() < second_burn_probability
    x0_second = np.random.choice(burn_locations) if inject_second_burn else np.nan
    amp_second = np.random.uniform(amp_min, amp_max) if inject_second_burn else 0.0
    inject_afp_sweep = np.random.rand() < afp_sweep_probability
    afp_bin_start, afp_bin_stop = (-1, -1)

    Ps_burned = np.asarray(Ps_base, dtype=float).copy()
    Iplus_burned = np.asarray(Iplus_base, dtype=float).copy()
    Iminus_burned = np.asarray(Iminus_base, dtype=float).copy()

    mapper.x0 = x0
    mapper.amp = amp
    mapper.apply_ssRF(Ps_burned, Iplus_burned, Iminus_burned, return_burn_info=False)

    if inject_second_burn:
        mapper.x0 = x0_second
        mapper.amp = amp_second
        mapper.apply_ssRF(Ps_burned, Iplus_burned, Iminus_burned, return_burn_info=False)

    if inject_afp_sweep:
        afp_bin_start, afp_bin_stop = _random_afp_bin_range(
            num_bins,
            afp_min_sweep_bins,
            afp_max_sweep_bins,
        )
        Iplus_burned, Iminus_burned = _apply_afp_sweep(
            Iplus_burned,
            Iminus_burned,
            bin_range=(afp_bin_start, afp_bin_stop),
            efficiency=afp_efficiency,
        )
        Ps_burned[:] = Iplus_burned + Iminus_burned

    P_final = _final_vector_pol(Ps_burned)

    return {
        "P_initial": P_initial,
        "P": P_initial,
        "P_final": P_final,
        "burn_injected": True,
        "x0": x0,
        "amp": amp,
        "second_burn_injected": inject_second_burn,
        "x0_second": x0_second,
        "amp_second": amp_second,
        "afp_sweep_injected": inject_afp_sweep,
        "afp_bin_start": afp_bin_start,
        "afp_bin_stop": afp_bin_stop,
        "afp_sweep_width": afp_bin_stop - afp_bin_start,
        "afp_freq_start": f[afp_bin_start] if inject_afp_sweep else np.nan,
        "afp_freq_stop": f[afp_bin_stop - 1] if inject_afp_sweep else np.nan,
        "afp_efficiency": afp_efficiency,
        "Ps": Ps_burned,
        "Iminus": Iminus_burned,
        "Iplus": Iplus_burned,
        "true_Ps": P_final,
        "true_Qs": true_Qs,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MC-generate ssRF training events with target final vector polarization."
    )
    parser.add_argument(
        "--P-final-min",
        type=float,
        default=0.50,
        help="Minimum final total vector polarization (fraction, e.g. 0.40), inclusive",
    )
    parser.add_argument(
        "--P-final-max",
        type=float,
        default=0.55,
        help="Maximum final total vector polarization (fraction, e.g. 0.45), inclusive",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of accepted MC events to collect",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Cap on MC trials (default: 200 * num_samples)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle path (default: results/.../testing_data_mc_Pmin-Pmax.pkl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--plot-histogram",
        action="store_true",
        help="Save histogram of accepted P_final (fraction) distribution",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip grid_plot.png and population_burn_example.png",
    )
    parser.add_argument(
        "--num-viz-examples",
        type=int,
        default=8,
        help="Number of accepted events to plot (spread across the dataset)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.P_final_min > args.P_final_max:
        raise ValueError(
            f"--P-final-min ({args.P_final_min}) must be <= --P-final-max ({args.P_final_max})"
        )

    if args.seed is not None:
        np.random.seed(args.seed)

    # --- Lineshape grid (match ssRFData.py) ---
    num_bins = 249
    f = np.linspace(-3, 3, num_bins)
    sigma = 0.16
    gamma = 0.05

    # Broad initial-P sampling; burns/AFP move polarization into the target band.
    P_initial_bounds = (0.5, 0.7)
    x0_negative = np.linspace(-1.5, -0.3, 50)
    x0_positive = np.linspace(0.3, 1.5, 50)
    burn_locations = np.concatenate([x0_negative, x0_positive])

    amp_min, amp_max = 5e-4, 5e-2
    second_burn_probability = 0.5
    afp_sweep_probability = 0.5
    afp_min_sweep_bins = 8
    afp_max_sweep_bins = num_bins // 3
    afp_efficiency = 1.0

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        pmin = f"{args.P_final_min:.2f}".rstrip("0").rstrip(".")
        pmax = f"{args.P_final_max:.2f}".rstrip("0").rstrip(".")
        output_path = output_dir / f"testing_data_mc_{pmin}-{pmax}.pkl"

    max_attempts = args.max_attempts
    if max_attempts is None:
        max_attempts = max(args.num_samples * 500, args.num_samples + 1)

    lookup_path = SCRIPT_DIR / "lookup_table.pkl"
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"Lookup table not found at {lookup_path}. Run lookup_table.py to create it."
        )
    mapping_data = pd.read_pickle(lookup_path)

    base_mapper = ssRFMapper(f, sigma, gamma, x0=0.0, amp=1e-2)
    base_mapper.compute_lookup_tables(mapping_data)
    mapper = ssRFMapper(f, sigma, gamma, x0=0.0, amp=1e-4)
    mapper.signal_to_iplus_lookup = base_mapper.signal_to_iplus_lookup
    mapper.signal_to_iminus_lookup = base_mapper.signal_to_iminus_lookup

    accepted: list[dict] = []
    collected_for_viz: list[dict] = []
    n_viz = 0 if args.no_viz else max(1, int(args.num_viz_examples))
    if n_viz > 0:
        viz_accept_indices = frozenset(
            int(round(float(i)))
            for i in np.linspace(0, max(args.num_samples - 1, 0), min(n_viz, args.num_samples))
        )
    else:
        viz_accept_indices = frozenset()

    attempts = 0
    pbar = tqdm.tqdm(total=args.num_samples, desc="Accepted MC events")

    while len(accepted) < args.num_samples and attempts < max_attempts:
        attempts += 1
        row = _sample_and_manipulate(
            mapper,
            f,
            P_initial_bounds=P_initial_bounds,
            burn_locations=burn_locations,
            amp_min=amp_min,
            amp_max=amp_max,
            second_burn_probability=second_burn_probability,
            afp_sweep_probability=afp_sweep_probability,
            afp_min_sweep_bins=afp_min_sweep_bins,
            afp_max_sweep_bins=afp_max_sweep_bins,
            afp_efficiency=afp_efficiency,
            num_bins=num_bins,
        )
        if args.P_final_min <= row["P_final"] <= args.P_final_max:
            accept_idx = len(accepted)
            accepted.append(row)
            if accept_idx in viz_accept_indices:
                collected_for_viz.append(_build_viz_sample(f, row))
            pbar.update(1)

    pbar.close()

    if len(accepted) < args.num_samples:
        raise RuntimeError(
            f"Only collected {len(accepted)}/{args.num_samples} accepted events "
            f"after {attempts} attempts (acceptance "
            f"{100.0 * len(accepted) / max(attempts, 1):.2f}%). "
            "Widen --P-final bounds, relax manipulation settings, or raise --max-attempts."
        )

    training_data = pd.DataFrame(accepted)
    training_data.to_pickle(output_path)

    acceptance_rate = 100.0 * len(accepted) / attempts
    p_final = training_data["P_final"].to_numpy()
    p_initial = training_data["P_initial"].to_numpy()

    print(f"\nSaved {len(training_data)} MC samples to {output_path}")
    print(f"  Target final vector pol: [{args.P_final_min}, {args.P_final_max}]")
    print(f"  MC attempts: {attempts}, acceptance rate: {acceptance_rate:.2f}%")
    print(
        f"  P_final: min={p_final.min():.3f}, max={p_final.max():.3f}, "
        f"mean={p_final.mean():.3f}"
    )
    print(
        f"  P_initial: min={p_initial.min():.3f}, max={p_initial.max():.3f}, "
        f"mean={p_initial.mean():.3f}"
    )

    if args.plot_histogram:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(p_final, bins=30, color="steelblue", edgecolor="white", alpha=0.9)
        ax.axvline(args.P_final_min, color="crimson", linestyle="--", label="target min")
        ax.axvline(args.P_final_max, color="crimson", linestyle="--", label="target max")
        ax.set_xlabel("final total vector polarization")
        ax.set_ylabel("count")
        ax.set_title(
            f"MC accepted events (n={len(accepted)}, "
            f"acceptance {acceptance_rate:.1f}%)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        hist_path = output_path.with_suffix(".P_final_hist.png")
        fig.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Histogram: {hist_path}")

    if collected_for_viz:
        plot_dir = output_path.parent
        p_bounds = (args.P_final_min, args.P_final_max)
        _save_grid_plot(
            f,
            collected_for_viz,
            plot_dir / "grid_plot.png",
            P_final_bounds=p_bounds,
        )
        n_pop = min(len(collected_for_viz), n_viz)
        pop_chunk = collected_for_viz[:n_pop]
        extra = (
            f"{len(pop_chunk)} example(s), "
            f"P_final in [{args.P_final_min:.3f}, {args.P_final_max:.3f}]"
        )
        _save_population_burn_figure(
            f,
            pop_chunk,
            plot_dir / "population_burn_example.png",
            suptitle_extra=extra,
        )
    elif not args.no_viz:
        print("  Visualization: no examples collected for plotting.")


if __name__ == "__main__":
    main()
