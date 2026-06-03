from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.lineshape.ssRFMapper import ssRFMapper


def q_polarization(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def q_at_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    ## calcuate Q in theta space:
    # return float(iplus[bin_idx] - iminus[bin_idx])
    iplus_theta = iplus[bin_idx] + iplus[len(iplus)-bin_idx - 1]
    iminus_theta = iminus[bin_idx] + iminus[len(iminus)-bin_idx - 1]
    return float(iplus_theta - iminus_theta)


def total_signal_area(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus + iminus))


@dataclass
class BurnConfig:
    num_bins: int = 249
    f_min: float = -3.0
    f_max: float = 3.0
    sigma: float = 0.16
    gamma: float = 0.05
    amp_min: float = 5e-4
    amp_max: float = 1e-2
    n_amp_steps: int = 100
    lookup_path: Path | None = None

    def __post_init__(self) -> None:
        self.lookup_path = REPO_ROOT / "Data_Creation" / "lookup_table.pkl"

    @property
    def f(self) -> np.ndarray:
        return np.linspace(self.f_min, self.f_max, self.num_bins)

    @property
    def amp_values(self) -> np.ndarray:
        return np.linspace(self.amp_min, self.amp_max, self.n_amp_steps)


def load_mapper(config: BurnConfig) -> ssRFMapper:
    lookup_path = Path(config.lookup_path)
    mapping_data = pd.read_pickle(lookup_path)
    mapper = ssRFMapper(config.f, config.sigma, config.gamma, x0=0.0, amp=1e-3)
    mapper.compute_lookup_tables(mapping_data)
    return mapper


def find_best_amp_for_bin(
    ps: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    bin_idx: int,
    amp_values: np.ndarray,
    mapper: ssRFMapper,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None:
    
    """Return (best_amp, best_q_bin, ps, iplus, iminus) after a single optimal burn, or None."""
    baseline_q_bin = q_at_bin(iplus, iminus, bin_idx)
    best_amp = 0.0
    best_q_bin = baseline_q_bin
    best_ps: np.ndarray | None = None
    best_iplus: np.ndarray | None = None
    best_iminus: np.ndarray | None = None

    for amp in amp_values:
        if amp < 0.0:
            continue
        ps_try = ps.copy()
        iplus_try = iplus.copy()
        iminus_try = iminus.copy()
        mapper.apply_bin_burn(ps_try, iplus_try, iminus_try, bin_idx, float(amp))
        q_try_bin = q_at_bin(iplus_try, iminus_try, bin_idx)
        if q_try_bin > best_q_bin:
            # print(f"New best amp: {amp}, delta_q_bin: {q_try_bin - baseline_q_bin}")
            best_q_bin = q_try_bin
            best_amp = float(amp)
            best_ps = ps_try
            best_iplus = iplus_try
            best_iminus = iminus_try

    if best_amp <= 0.0 or best_ps is None or best_iplus is None or best_iminus is None:
        return None
    return best_amp, best_q_bin, best_ps, best_iplus, best_iminus


def optimize_binwise_incremental(
    config: BurnConfig,
    polarization: float,
    mapper: ssRFMapper,
) -> dict:
    f = config.f
    _, iplus, iminus = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus, dtype=float)
    iminus = np.asarray(iminus, dtype=float)
    ps = iplus + iminus

    initial_q = q_polarization(iplus, iminus)
    current_q = initial_q
    initial_iplus_area = float(np.sum(iplus))
    initial_iminus_area = float(np.sum(iminus))
    current_iplus_area = initial_iplus_area
    current_iminus_area = initial_iminus_area
    initial_area = total_signal_area(iplus, iminus)
    current_area = initial_area
    trace: list[dict] = [
        {
            "step": 0,
            "q": initial_q,
            "iplus_area": initial_iplus_area,
            "iminus_area": initial_iminus_area,
            "area": initial_area,
            "action": None,
        }
    ]
    step = 0

    for bin_idx in tqdm.tqdm(range(len(f)), desc="Optimizing bins"):
        q_bin_before = q_at_bin(iplus, iminus, bin_idx)
        best = find_best_amp_for_bin(
            ps, iplus, iminus, bin_idx, config.amp_values, mapper
        )

        if best is None:
            trace.append(
                {
                    "step": step,
                    "bin_idx": bin_idx,
                    "f": float(f[bin_idx]),
                    "amp": 0.0,
                    "reward": 0.0,
                    "q_bin": q_bin_before,
                    "q_bin_gain": 0.0,
                    "q": current_q,
                    "q_gain": current_q - initial_q,
                    "iplus_area": current_iplus_area,
                    "iplus_area_change": current_iplus_area - initial_iplus_area,
                    "iminus_area": current_iminus_area,
                    "iminus_area_change": current_iminus_area - initial_iminus_area,
                    "area": current_area,
                    "area_gain": current_area - initial_area,
                    "skipped": True,
                    "skip_reason": "no_amp_improves_q_at_bin",
                }
            )
            continue

        best_amp, best_q_bin, ps, iplus, iminus = best
        iplus_area_before = current_iplus_area
        iminus_area_before = current_iminus_area
        current_q_bin = best_q_bin
        current_q = q_polarization(iplus, iminus)
        current_iplus_area = float(np.sum(iplus))
        current_iminus_area = float(np.sum(iminus))
        current_area = total_signal_area(iplus, iminus)
        q_bin_gain = current_q_bin - q_bin_before
        step += 1
        trace.append(
            {
                "step": step,
                "bin_idx": bin_idx,
                "f": float(f[bin_idx]),
                "amp": best_amp,
                "reward": q_bin_gain,
                "q_bin_reward": q_bin_gain,
                "q_bin": current_q_bin,
                "q_bin_gain": current_q_bin - q_bin_before,
                "iplus_reduction": iplus_area_before - current_iplus_area,
                "iminus_reduction": iminus_area_before - current_iminus_area,
                "q": current_q,
                "q_gain": current_q - initial_q,
                "iplus_area": current_iplus_area,
                "iplus_area_change": current_iplus_area - initial_iplus_area,
                "iminus_area": current_iminus_area,
                "iminus_area_change": current_iminus_area - initial_iminus_area,
                "area": current_area,
                "area_gain": current_area - initial_area,
            }
        )

    return {
        "polarization": polarization,
        "initial_q": initial_q,
        "final_q": current_q,
        "initial_iplus_area": initial_iplus_area,
        "final_iplus_area": current_iplus_area,
        "initial_iminus_area": initial_iminus_area,
        "final_iminus_area": current_iminus_area,
        "initial_area": initial_area,
        "final_area": current_area,
        "trace": trace,
        "iplus_unburned": np.asarray(GenerateVectorLineshape(polarization, f)[1], dtype=float),
        "iminus_unburned": np.asarray(GenerateVectorLineshape(polarization, f)[2], dtype=float),
        "iplus": iplus,
        "iminus": iminus,
        "f": f.copy(),
    }


def plot_greedy_burns(result: dict, output_path: Path) -> None:
    f = result["f"]
    iplus = result["iplus"]
    iminus = result["iminus"]
    iplus0 = result["iplus_unburned"]
    iminus0 = result["iminus_unburned"]
    q_profile = iplus - iminus
    q_profile0 = iplus0 - iminus0

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].step(
        f, iplus0 + iminus0, color="black", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$P_s$ (unburned)",
    )
    axes[0].step(
        f, iplus0, color="tab:red", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$I_+$ (unburned)",
    )
    axes[0].step(
        f, iminus0, color="tab:blue", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$I_-$ (unburned)",
    )
    axes[0].step(f, iplus + iminus, label=r"$P_s = I_+ + I_-$", color="black")
    axes[0].step(f, iplus, label=r"$I_+$", color="tab:red")
    axes[0].step(f, iminus, label=r"$I_-$", color="tab:blue")
    for row in result["trace"][1:]:
        if row.get("amp", 0.0) > 0.0:
            axes[0].axvline(row["f"], color="green", alpha=0.3, linestyle=":")
            axes[0].axvline(-row["f"], color="purple", alpha=0.2, linestyle=":")
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].step(
        f, q_profile0, color="tab:purple", linestyle="--", alpha=0.55, linewidth=1.0,
        label=r"$Q$ (unburned)",
    )
    axes[1].step(f, q_profile, color="tab:purple", label=r"$Q = I_+ - I_-$")
    axes[1].set_xlabel("frequency")
    axes[1].set_ylabel("Q profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    delta_q = result["final_q"] - result["initial_q"]
    delta_iplus = result["final_iplus_area"] - result["initial_iplus_area"]
    delta_iminus = result["final_iminus_area"] - result["initial_iminus_area"]
    delta_area = result["final_area"] - result["initial_area"]
    title = (
        f"P={result['polarization']:.3f}  "
        f"Q: {result['initial_q']:.4f} -> {result['final_q']:.4f} ({delta_q:+.4f})  "
        f"I+: {delta_iplus:+.4f}  I-: {delta_iminus:+.4f}  area: {delta_area:+.4f}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bin-wise optimizer: one burn per frequency bin at the amplitude "
            "that maximizes Q at that bin."
        )
    )
    parser.add_argument("--polarization", type=float, default=0.45)
    parser.add_argument("--amp-max", type=float, default=1e-5)
    parser.add_argument(
        "--amp-steps",
        type=int,
        default=165,
        help="Number of candidate amplitudes in [0, amp-max] per bin.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "current" / "binwise_incremental",
    )
    args = parser.parse_args()

    config = BurnConfig(
        amp_max=args.amp_max,
        n_amp_steps=args.amp_steps
    )
    mapper = load_mapper(config)
    result = optimize_binwise_incremental(config, args.polarization, mapper)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_greedy_burns(
        result, out_dir / f"incremental_policy_P{args.polarization:.2f}.png"
    )

    print(f"Bin-wise optimization (one burn per bin) at P={args.polarization * 100:.2f}%:")
    print(f"  start: Q={result['initial_q'] * 100:.5f}%")
    print(f"  start I+ area: {result['initial_iplus_area']:.8f}")
    print(f"  start I- area: {result['initial_iminus_area']:.8f}")
    print(f"  start area: {result['initial_area']:.8f}")
    for row in result["trace"][1:]:
        if row.get("amp", 0.0) <= 0.0:
            continue
        print(
            f"  burn {row['step']}: bin={row['bin_idx']}, f={row['f']:.3f}, "
            f"amp={row['amp']:.4e}, Q_bin_gain={row['reward']:.5e}, "
            f"I+ reduction={row['iplus_reduction']:.5e}, "
            f"I- reduction={row['iminus_reduction']:.5e}, "
            f"Q_bin={row['q_bin']:.5e}, Q_total={row['q'] * 100:.5f}%"
        )
    print(f"  total Q gain: {(result['final_q'] - result['initial_q']) * 100:.5f}%")
    print(f"  total I+ change: {result['final_iplus_area'] - result['initial_iplus_area']:.8f}")
    print(f"  total I- change: {result['final_iminus_area'] - result['initial_iminus_area']:.8f}")
    print(f"  total area gain: {result['final_area'] - result['initial_area']:.8f}")
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
