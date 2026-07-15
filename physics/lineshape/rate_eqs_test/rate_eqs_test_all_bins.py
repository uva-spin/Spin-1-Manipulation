"""Sequential rate-eq burns over all bins, optimizing Q per theta bin (R + -R)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_LINESHAPE_DIR = Path(__file__).resolve().parent.parent
if str(_LINESHAPE_DIR) not in sys.path:
    sys.path.insert(0, str(_LINESHAPE_DIR))

from Lineshape import GenerateVectorLineshape

P = 0.60
NUM_BINS = 500
XI = 0.6
T_MAX = 5.0
N_T = 200
OUT_DIR = Path(__file__).resolve().parent


def mirror_bin_idx(n_bins: int, bin_idx: int) -> int:
    return int(n_bins) - 1 - int(bin_idx)


def intensity_after_burn(t: float | np.ndarray, prefactor: float) -> float | np.ndarray:
    return prefactor * np.exp(np.asarray(t, dtype=float) * (1.0 - 2.0 * XI))


def q_at_theta_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    """Tensor polarization Q in the theta bin: total Q at R and -R."""
    mirror_idx = mirror_bin_idx(len(iplus), bin_idx)
    q_r = float(iplus[bin_idx] - iminus[bin_idx])
    q_mirror = float(iplus[mirror_idx] - iminus[mirror_idx])
    return q_r + q_mirror


def q_total(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def apply_burn_at_t(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one burn/mirror transfer at burn time t; only those two bins change."""
    mirror_idx = mirror_bin_idx(len(iplus), burn_idx)
    iplus_i = float(iplus[burn_idx])
    iminus_i = float(iminus[burn_idx])
    iplus_f = float(iplus[mirror_idx])
    iminus_f = float(iminus[mirror_idx])

    iplus_burn = float(intensity_after_burn(t, iplus_i))
    iminus_burn = float(intensity_after_burn(t, iminus_i))
    # Mirror gains the depleted burn amount (crossed, 2:1).
    iplus_mirror = iplus_f + (iminus_i - iminus_burn) / 2.0
    iminus_mirror = iminus_f + (iplus_i - iplus_burn) / 2.0

    iplus_out = np.array(iplus, dtype=float, copy=True)
    iminus_out = np.array(iminus, dtype=float, copy=True)
    iplus_out[burn_idx] = iplus_burn
    iminus_out[burn_idx] = iminus_burn
    iplus_out[mirror_idx] = iplus_mirror
    iminus_out[mirror_idx] = iminus_mirror
    return iplus_out, iminus_out


def find_best_t_for_bin(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    t_grid: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """Maximize Q(R) + Q(-R) over burn time t. Skip if no improvement."""
    baseline_q = q_at_theta_bin(iplus, iminus, burn_idx)
    best_t = 0.0
    best_q = baseline_q
    best_iplus: np.ndarray | None = None
    best_iminus: np.ndarray | None = None

    for t in t_grid:
        t_val = float(t)
        if t_val <= 0.0:
            continue
        iplus_try, iminus_try = apply_burn_at_t(iplus, iminus, burn_idx, t_val)
        q_try = q_at_theta_bin(iplus_try, iminus_try, burn_idx)
        if q_try > best_q:
            best_q = q_try
            best_t = t_val
            best_iplus = iplus_try
            best_iminus = iminus_try

    if best_t <= 0.0 or best_iplus is None or best_iminus is None:
        return None
    return best_t, best_q, best_iplus, best_iminus


def optimize_all_bins(
    polarization: float = P,
    num_bins: int = NUM_BINS,
    t_max: float = T_MAX,
    n_t: int = N_T,
) -> dict:
    f = np.linspace(-3.0, 3.0, num_bins)
    _, iplus0, iminus0 = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    iplus_unburned = iplus.copy()
    iminus_unburned = iminus.copy()

    t_grid = np.linspace(0.0, t_max, n_t)
    initial_q = q_total(iplus, iminus)
    trace: list[dict] = []
    applied = 0
    skipped = 0

    for burn_idx in range(num_bins):
        mirror_idx = mirror_bin_idx(num_bins, burn_idx)
        q_bin_before = q_at_theta_bin(iplus, iminus, burn_idx)
        best = find_best_t_for_bin(iplus, iminus, burn_idx, t_grid)

        if best is None:
            skipped += 1
            trace.append(
                {
                    "bin_idx": burn_idx,
                    "mirror_idx": mirror_idx,
                    "f": float(f[burn_idx]),
                    "t": 0.0,
                    "q_bin_before": q_bin_before,
                    "q_bin_after": q_bin_before,
                    "q_bin_gain": 0.0,
                    "q_total": q_total(iplus, iminus),
                    "skipped": True,
                }
            )
            continue

        best_t, best_q_bin, iplus, iminus = best
        applied += 1
        q_bin_gain = best_q_bin - q_bin_before
        current_q = q_total(iplus, iminus)
        trace.append(
            {
                "bin_idx": burn_idx,
                "mirror_idx": mirror_idx,
                "f": float(f[burn_idx]),
                "t": best_t,
                "q_bin_before": q_bin_before,
                "q_bin_after": best_q_bin,
                "q_bin_gain": q_bin_gain,
                "q_total": current_q,
                "skipped": False,
            }
        )
        print(
            f"burn bin={burn_idx:3d} R={f[burn_idx]:+.4f} "
            f"t={best_t:.4f}  Q_bin {q_bin_before:.6e} -> {best_q_bin:.6e} "
            f"(+{q_bin_gain:.6e})  Q_total={current_q:.6e}"
        )

    final_q = q_total(iplus, iminus)
    return {
        "polarization": polarization,
        "f": f,
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus": iplus,
        "iminus": iminus,
        "initial_q": initial_q,
        "final_q": final_q,
        "n_applied": applied,
        "n_skipped": skipped,
        "trace": trace,
    }


def plot_result(result: dict, output_path: Path) -> None:
    f = result["f"]
    iplus = result["iplus"]
    iminus = result["iminus"]
    iplus0 = result["iplus_unburned"]
    iminus0 = result["iminus_unburned"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
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
    axes[0].step(f, iplus + iminus, color="black", label=r"$P_s$")
    axes[0].step(f, iplus, color="tab:red", label=r"$I_+$")
    axes[0].step(f, iminus, color="tab:blue", label=r"$I_-$")
    for row in result["trace"]:
        if not row["skipped"]:
            axes[0].axvline(row["f"], color="green", alpha=0.15, linestyle=":")
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].step(
        f, iplus0 - iminus0, color="tab:purple", linestyle="--", alpha=0.55,
        label=r"$Q$ (unburned)",
    )
    axes[1].step(f, iplus - iminus, color="tab:purple", label=r"$Q = I_+ - I_-$")
    axes[1].set_xlabel(r"$R$")
    axes[1].set_ylabel("Q profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    delta_q = result["final_q"] - result["initial_q"]
    fig.suptitle(
        f"P={result['polarization']:.2f}  "
        f"Q: {result['initial_q']:.6f} -> {result['final_q']:.6f} ({delta_q:+.6f})  "
        f"applied={result['n_applied']}  skipped={result['n_skipped']}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_q_gains(result: dict, output_path: Path) -> None:
    applied = [row for row in result["trace"] if not row["skipped"]]
    if not applied:
        return

    f_vals = [row["f"] for row in applied]
    gains = [row["q_bin_gain"] for row in applied]
    t_vals = [row["t"] for row in applied]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].stem(f_vals, gains, basefmt=" ")
    axes[0].set_ylabel(r"$\Delta Q_{\theta}$")
    axes[0].set_title("Per-bin Q gain (applied burns only)")
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(f_vals, t_vals, basefmt=" ", linefmt="C1-", markerfmt="C1o")
    axes[1].set_xlabel(r"burn $R$")
    axes[1].set_ylabel(r"optimal $t$")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    result = optimize_all_bins()
    lineshape_path = OUT_DIR / "rate_eqs_test_all_bins_lineshape.png"
    gains_path = OUT_DIR / "rate_eqs_test_all_bins_gains.png"
    plot_result(result, lineshape_path)
    plot_q_gains(result, gains_path)

    print()
    print(f"P={result['polarization']}")
    print(f"bins applied: {result['n_applied']}  skipped: {result['n_skipped']}")
    print(f"Q total: {result['initial_q']:.8f} -> {result['final_q']:.8f}")
    print(f"Q gain:  {result['final_q'] - result['initial_q']:+.8f}")
    print(f"Saved {lineshape_path.name}, {gains_path.name}")


if __name__ == "__main__":
    main()
