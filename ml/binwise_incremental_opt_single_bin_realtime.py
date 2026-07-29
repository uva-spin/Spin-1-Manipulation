from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime_v2 import Spin1Params
from physics.ssrf_realtime_v2.rate_equations_realtime import (
    build_model_for_intensities,
    burn_preserves_branch_order,
    burn_preserves_ps_sign,
    configure_single_bin_ssrf,
)


def q_polarization(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def q_at_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    iplus_theta = iplus[bin_idx] + iplus[len(iplus) - bin_idx - 1]
    iminus_theta = iminus[bin_idx] + iminus[len(iminus) - bin_idx - 1]
    return float(iplus_theta - iminus_theta)
    # return float(abs(iplus[bin_idx] - iminus[bin_idx]))


def total_signal_area(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus + iminus))


def freq_to_bin_idx(f: np.ndarray, f_target: float) -> int:
    return int(np.argmin(np.abs(f - float(f_target))))


@dataclass
class BurnConfig:
    num_bins: int = 500
    f_min: float = -3.0
    f_max: float = 3.0
    dt: float = 0.015
    steps: int = 20
    max_steps: int = 1000
    amp_min: float = 0.0
    amp_max: float = 50.0
    n_amp_steps: int = 100

    def __post_init__(self) -> None:
        if self.amp_min > self.amp_max:
            raise ValueError(
                f"amp_min ({self.amp_min}) must be <= amp_max ({self.amp_max})"
            )
        if self.n_amp_steps < 1:
            raise ValueError(f"n_amp_steps must be >= 1, got {self.n_amp_steps}")

    @property
    def f(self) -> np.ndarray:
        return np.linspace(self.f_min, self.f_max, self.num_bins)

    @property
    def rf_amp_values(self) -> np.ndarray:
        if self.n_amp_steps == 1:
            return np.array([self.amp_max], dtype=float)
        return np.linspace(self.amp_min, self.amp_max, self.n_amp_steps)

    def spin1_params(self, polarization: float) -> Spin1Params:
        return Spin1Params(
            n_bins=self.num_bins,
            r_min=self.f_min,
            r_max=self.f_max,
            p0=float(polarization),
            dnp_enabled=False,
            t1_rate=0.0,
            gamma_rf=1.0,
            dt=self.dt,
            steps=self.steps,
            diffusion_scale=0.0,
            relax_enabled=False,
        )


def apply_rate_equation_burn(
    iplus: np.ndarray,
    iminus: np.ndarray,
    bin_idx: int,
    rf_amp: float,
    polarization: float,
    params: Spin1Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if rf_amp <= 0.0:
        return None

    step_params = replace(
        params,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
    )
    model = build_model_for_intensities(
        iplus,
        iminus,
        params=step_params,
        p0=polarization,
    )
    configure_single_bin_ssrf(
        model,
        int(bin_idx),
        float(rf_amp),
        apply_demo_recovery=False,
    )

    iplus_cur = np.asarray(iplus, dtype=float).copy()
    iminus_cur = np.asarray(iminus, dtype=float).copy()
    iplus_new = iplus_cur.copy()
    iminus_new = iminus_cur.copy()

    for _ in range(model.params.steps):
        state_before = model.n.copy()
        model.step_once(dt=params.dt, rf_on=True, dnp_on=False)
        iplus_new, iminus_new, _ = model.physical_intensities()
        if not burn_preserves_ps_sign(iplus_cur, iminus_cur, iplus_new, iminus_new, bin_idx):
            model.n = state_before
            iplus_new, iminus_new = iplus_cur, iminus_cur
            break
        if not burn_preserves_branch_order(iplus, iminus, iplus_new, iminus_new, bin_idx):
            model.n = state_before
            iplus_new, iminus_new = iplus_cur, iminus_cur
            break
        iplus_cur = np.asarray(iplus_new, dtype=float).copy()
        iminus_cur = np.asarray(iminus_new, dtype=float).copy()

    iplus_new = np.asarray(iplus_new, dtype=float)
    iminus_new = np.asarray(iminus_new, dtype=float)

    if not burn_preserves_ps_sign(iplus, iminus, iplus_new, iminus_new, bin_idx):
        return None

    ps_before = float(iplus[bin_idx] + iminus[bin_idx])
    ps_after = float(iplus_new[bin_idx] + iminus_new[bin_idx])
    if ps_after == ps_before:
        return None

    return iplus_new + iminus_new, iplus_new, iminus_new


def find_best_rf_amp_for_bin(
    ps: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    bin_idx: int,
    rf_amp_values: np.ndarray,
    polarization: float,
    params: Spin1Params,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None:
    baseline_q_bin = q_at_bin(iplus, iminus, bin_idx)
    best_rf_amp = 0.0
    best_q_bin = baseline_q_bin
    best_ps: np.ndarray | None = None
    best_iplus: np.ndarray | None = None
    best_iminus: np.ndarray | None = None

    for rf_amp in rf_amp_values:
        burned = apply_rate_equation_burn(
            iplus, iminus, bin_idx, float(rf_amp), polarization, params
        )
        if burned is None:
            continue
        ps_try, iplus_try, iminus_try = burned
        q_try_bin = q_at_bin(iplus_try, iminus_try, bin_idx)
        print(f"rf_amp: {rf_amp}, q_try_bin: {q_try_bin}, best_q_bin: {best_q_bin}")
        if q_try_bin > best_q_bin:
            print(
                f"New best RF amp: {rf_amp:.6e}, "
                f"delta_q_bin: {q_try_bin - baseline_q_bin:.6e}"
            )
            best_q_bin = q_try_bin
            best_rf_amp = float(rf_amp)
            best_ps = ps_try
            best_iplus = iplus_try
            best_iminus = iminus_try

    if best_rf_amp <= 0.0 or best_ps is None or best_iplus is None or best_iminus is None:
        return None
    return best_rf_amp, best_q_bin, best_ps, best_iplus, best_iminus


def optimize_binwise_incremental(
    config: BurnConfig,
    polarization: float,
) -> dict:
    f = config.f
    spin1_params = config.spin1_params(polarization)
    _, iplus, iminus = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus, dtype=float)
    iminus = np.asarray(iminus, dtype=float)
    ps = iplus + iminus

    initial_q = q_polarization(iplus, iminus)
    initial_iplus_area = float(np.sum(iplus))
    initial_iminus_area = float(np.sum(iminus))
    initial_area = total_signal_area(iplus, iminus)
    current_iplus_area = initial_iplus_area
    current_iminus_area = initial_iminus_area
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

    f_target = -0.92
    bin_idx = freq_to_bin_idx(f, f_target)

    q_bin_before = q_at_bin(iplus, iminus, bin_idx)
    best = find_best_rf_amp_for_bin(
        ps,
        iplus,
        iminus,
        bin_idx,
        config.rf_amp_values,
        polarization,
        spin1_params,
    )
    if best is None:
        raise RuntimeError(
            f"No valid rate-equation burn found at bin_idx={bin_idx}, f={f[bin_idx]:.4f}"
        )

    best_rf_amp, best_q_bin, ps, iplus, iminus = best
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
            "rf_amp": best_rf_amp,
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


def plot_unburned_signal(
    f: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    polarization: float,
    output_path: Path,
    *,
    f_target: float | None = None,
) -> None:
    ps = iplus + iminus
    q_profile = iplus - iminus

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].step(f, ps, label=r"$P_s = I_+ + I_-$", color="black")
    axes[0].step(f, iplus, label=r"$I_+$", color="tab:red")
    axes[0].step(f, iminus, label=r"$I_-$", color="tab:blue")
    if f_target is not None:
        axes[0].axvline(f_target, color="green", alpha=0.4, linestyle=":", label="burn target")
        axes[0].axvline(-f_target, color="purple", alpha=0.25, linestyle=":")
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].step(f, q_profile, color="tab:purple", label=r"$Q = I_+ - I_-$")
    axes[1].set_xlabel("frequency")
    axes[1].set_ylabel("Q profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    q_total = q_polarization(iplus, iminus)
    fig.suptitle(
        f"Unburned lineshape  P={polarization:.3f}  "
        f"Q_total={q_total * 100:.4f}%  area={total_signal_area(iplus, iminus):.4f}"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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
        if row.get("rf_amp", 0.0) > 0.0:
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
    polarization = 0.45
    f_target = -0.92
    config = BurnConfig()
    out_dir = REPO_ROOT / "results" / "current" / "binwise_incremental_realtime"
    out_dir.mkdir(parents=True, exist_ok=True)

    f = config.f
    _, iplus0, iminus0 = GenerateVectorLineshape(polarization, f)
    iplus0 = np.asarray(iplus0, dtype=float)
    iminus0 = np.asarray(iminus0, dtype=float)
    plot_unburned_signal(
        f,
        iplus0,
        iminus0,
        polarization,
        out_dir / f"unburned_P{polarization:.2f}.png",
        f_target=f_target,
    )

    result = optimize_binwise_incremental(config, polarization)
    plot_greedy_burns(
        result, out_dir / f"incremental_policy_P{polarization:.2f}.png"
    )

    print(
        f"Bin-wise optimization (ssrf_realtime_v2, one burn per bin) at "
        f"P={polarization * 100:.2f}%:"
    )
    print(f"  start: Q={result['initial_q'] * 100:.5f}%")
    print(f"  start I+ area: {result['initial_iplus_area']:.8f}")
    print(f"  start I- area: {result['initial_iminus_area']:.8f}")
    print(f"  start area: {result['initial_area']:.8f}")
    for row in result["trace"][1:]:
        if row.get("rf_amp", 0.0) <= 0.0:
            continue
        print(
            f"  burn {row['step']}: bin={row['bin_idx']}, f={row['f']:.3f}, "
            f"rf_amp={row['rf_amp']:.4e}, Q_bin_gain={row['reward']:.5e}, "
            f"I+ reduction={row['iplus_reduction']:.5e}, "
            f"I- reduction={row['iminus_reduction']:.5e}, "
            f"Q_bin={row['q_bin']:.5e}, Q_total={row['q'] * 100:.5f}%"
        )
    print(f"  total Q gain: {(result['final_q'] - result['initial_q']) * 100:.5f}%")
    print(f" total P change: {(result['final_area'] - result['initial_area']) * 100:.5f}%")
    print(f"  total I+ change: {result['final_iplus_area'] - result['initial_iplus_area']:.8f}")
    print(f"  total I- change: {result['final_iminus_area'] - result['initial_iminus_area']:.8f}")
    print(f"  total area gain: {result['final_area'] - result['initial_area']:.8f}")
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
