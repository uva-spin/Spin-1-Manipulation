from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SPIN1_ROOT = REPO_ROOT / "physics" / "spin1_ssrf_realtime"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SPIN1_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIN1_ROOT))

from physics.ssrf_realtime.rate_equations_realtime import burn_preserves_ps_sign
from ssrf_realtime.model import Spin1Model, Spin1Params


def normalize_vector_lineshape(
    iplus: np.ndarray,
    iminus: np.ndarray,
    polarization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale branches so sum(I+ + I-) == P, matching Lineshape.GenerateVectorLineshape."""
    p_summed = float(np.sum(iplus + iminus))
    if p_summed <= 0.0:
        return iplus, iminus
    delta_p = float(polarization) / p_summed
    return iplus * delta_p, iminus * delta_p


def event_scale_from_spectrum(
    iplus: np.ndarray,
    iminus: np.ndarray,
    polarization: float,
) -> float:
    """Fixed scale factor mapping Spin1Model display units to Lineshape.py event units."""
    p_summed = float(np.sum(iplus + iminus))
    if p_summed <= 0.0:
        return 1.0
    return float(polarization) / p_summed


def q_polarization(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def q_at_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    iplus_theta = iplus[bin_idx] + iplus[len(iplus) - bin_idx - 1]
    iminus_theta = iminus[bin_idx] + iminus[len(iminus) - bin_idx - 1]
    return float(iplus_theta - iminus_theta)


def total_signal_area(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus + iminus))


def freq_to_bin_idx(f: np.ndarray, f_target: float) -> int:
    return int(np.argmin(np.abs(f - float(f_target))))


def raw_model_spectrum(model: Spin1Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, iplus, iminus, _ = model.spectrum()
    return (
        np.asarray(f, dtype=float),
        np.asarray(iplus, dtype=float),
        np.asarray(iminus, dtype=float),
    )


def model_spectrum(model: Spin1Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, iplus, iminus = raw_model_spectrum(model)
    event_scale = getattr(model, "_event_scale", None)
    if event_scale is None:
        event_scale = event_scale_from_spectrum(iplus, iminus, model.params.p0)
        model._event_scale = event_scale
    return f, iplus * event_scale, iminus * event_scale


def clone_model(model: Spin1Model) -> Spin1Model:
    """Copy model state so burn trials do not mutate the working trajectory."""
    trial = Spin1Model(copy.deepcopy(model.params))
    trial.n = model.n.copy()
    trial.t = float(model.t)
    if hasattr(model, "_event_scale"):
        trial._event_scale = float(model._event_scale)
    return trial


@dataclass
class BurnConfig:
    num_bins: int = 500
    f_min: float = -3.0
    f_max: float = 3.0
    dt: float = 0.0015
    steps: int = 500
    gamma_min: float = 0.0
    gamma_max: float = 1.0
    n_gamma_steps: int = 10
    line_gamma: float = 0.05
    line_asym: float = 0.04

    def __post_init__(self) -> None:
        if self.gamma_min > self.gamma_max:
            raise ValueError(
                f"gamma_min ({self.gamma_min}) must be <= gamma_max ({self.gamma_max})"
            )
        if self.n_gamma_steps < 1:
            raise ValueError(f"n_gamma_steps must be >= 1, got {self.n_gamma_steps}")

    @property
    def f(self) -> np.ndarray:
        return np.linspace(self.f_min, self.f_max, self.num_bins)

    @property
    def gamma_rf_values(self) -> np.ndarray:
        if self.n_gamma_steps == 1:
            return np.array([self.gamma_max], dtype=float)
        return np.linspace(self.gamma_min, self.gamma_max, self.n_gamma_steps)

    def spin1_params(self, polarization: float) -> Spin1Params:
        return Spin1Params(
            n_bins=self.num_bins,
            r_min=self.f_min,
            r_max=self.f_max,
            line_gamma=self.line_gamma,
            line_asym=self.line_asym,
            p0=float(polarization),
            rf_enabled=True,
            gamma_rf=Spin1Params.gamma_rf,
            dnp_enabled=False,
            dt=self.dt,
        )


def build_model(config: BurnConfig, polarization: float) -> Spin1Model:
    """Create a Spin1Model and calibrate Lineshape.py event normalization."""
    model = Spin1Model(config.spin1_params(polarization))
    _, iplus_raw, iminus_raw = raw_model_spectrum(model)
    model._event_scale = event_scale_from_spectrum(
        iplus_raw, iminus_raw, polarization
    )
    return model


def apply_spin1_burn(
    model: Spin1Model,
    burn_R: float,
    gamma_rf: float,
    n_steps: int,
) -> Spin1Model | None:
    """Burn with spin1_ssrf_realtime dynamics via model.step, matching burn.py."""
    if gamma_rf <= 0.0 or n_steps <= 0:
        return None

    f_before, iplus_before, iminus_before = model_spectrum(model)
    bin_idx = freq_to_bin_idx(f_before, burn_R)
    ps_before = float(iplus_before[bin_idx] + iminus_before[bin_idx])

    burned = clone_model(model)
    burned.params.rf_burn_R = float(burn_R)
    burned.params.gamma_rf = float(gamma_rf)
    burned.params.rf_enabled = True
    burned.params.dnp_enabled = False
    burned.step(n_steps, rf_on=True, dnp_on=False)

    _, iplus_after, iminus_after = model_spectrum(burned)
    ps_after = float(iplus_after[bin_idx] + iminus_after[bin_idx])
    if ps_after == ps_before:
        return None
    if not burn_preserves_ps_sign(
        iplus_before, iminus_before, iplus_after, iminus_after, bin_idx
    ):
        return None

    return burned


def find_best_gamma_rf_for_bin(
    model: Spin1Model,
    burn_R: float,
    bin_idx: int,
    gamma_rf_values: np.ndarray,
    n_steps: int,
) -> tuple[float, float, Spin1Model] | None:
    f, iplus, iminus = model_spectrum(model)
    baseline_q_bin = q_at_bin(iplus, iminus, bin_idx)
    best_gamma_rf = 0.0
    best_q_bin = baseline_q_bin
    best_model: Spin1Model | None = None

    for gamma_rf in gamma_rf_values:
        burned = apply_spin1_burn(model, burn_R, float(gamma_rf), n_steps)
        if burned is None:
            continue
        _, iplus_try, iminus_try = model_spectrum(burned)
        q_try_bin = q_at_bin(iplus_try, iminus_try, bin_idx)
        print(f"gamma_rf: {gamma_rf}, q_try_bin: {q_try_bin}, best_q_bin: {best_q_bin}")
        if q_try_bin > best_q_bin:
            print(
                f"New best gamma_rf: {gamma_rf:.6e}, "
                f"delta_q_bin: {q_try_bin - baseline_q_bin:.6e}"
            )
            best_q_bin = q_try_bin
            best_gamma_rf = float(gamma_rf)
            best_model = burned

    if best_gamma_rf <= 0.0 or best_model is None:
        return None
    return best_gamma_rf, best_q_bin, best_model


def optimize_binwise_incremental(
    config: BurnConfig,
    polarization: float,
    *,
    f_target: float = -0.92,
) -> dict:
    model = build_model(config, polarization)
    f, iplus, iminus = model_spectrum(model)
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

    bin_idx = freq_to_bin_idx(f, f_target)
    burn_R = float(f[bin_idx])

    q_bin_before = q_at_bin(iplus, iminus, bin_idx)
    best = find_best_gamma_rf_for_bin(
        model,
        burn_R,
        bin_idx,
        config.gamma_rf_values,
        config.steps,
    )
    if best is None:
        raise RuntimeError(
            f"No valid spin1 burn found at bin_idx={bin_idx}, f={burn_R:.4f}"
        )

    best_gamma_rf, best_q_bin, model = best
    f, iplus, iminus = model_spectrum(model)
    ps = iplus + iminus
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
            "f": burn_R,
            "gamma_rf": best_gamma_rf,
            "steps": config.steps,
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

    unburned = build_model(config, polarization)
    _, iplus0, iminus0 = model_spectrum(unburned)

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
        "iplus_unburned": iplus0,
        "iminus_unburned": iminus0,
        "iplus": iplus,
        "iminus": iminus,
        "f": f.copy(),
        "model": model,
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
        f"Unburned lineshape (spin1)  P={polarization:.3f}  "
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
        if row.get("gamma_rf", 0.0) > 0.0:
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
        f"P={result['polarization']*100:.2f}%  "
        f"Q: {result['initial_q']*100:.4f}% -> {result['final_q']*100:.4f}% ({delta_q*100:.4f}%)  "
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
    out_dir = REPO_ROOT / "results" / "current" / "binwise_incremental_spin1"
    out_dir.mkdir(parents=True, exist_ok=True)

    unburned = build_model(config, polarization)
    f, iplus0, iminus0 = model_spectrum(unburned)
    plot_unburned_signal(
        f,
        iplus0,
        iminus0,
        polarization,
        out_dir / f"unburned_P{polarization:.2f}.png",
        f_target=f_target,
    )

    result = optimize_binwise_incremental(config, polarization, f_target=f_target)
    plot_greedy_burns(
        result, out_dir / f"incremental_policy_P{polarization:.2f}.png"
    )

    print(
        f"Bin-wise optimization (spin1_ssrf_realtime, one burn per bin) at "
        f"P={polarization * 100:.2f}%:"
    )
    print(f"  start: Q={result['initial_q'] * 100:.5f}%")
    print(f"  start I+ area: {result['initial_iplus_area']:.8f}")
    print(f"  start I- area: {result['initial_iminus_area']:.8f}")
    print(f"  start area: {result['initial_area']:.8f}")
    for row in result["trace"][1:]:
        if row.get("gamma_rf", 0.0) <= 0.0:
            continue
        print(
            f"  burn {row['step']}: bin={row['bin_idx']}, f={row['f']:.3f}, "
            f"gamma_rf={row['gamma_rf']:.4e}, steps={row['steps']}, "
            f"Q_bin_gain={row['reward']:.5e}, "
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
