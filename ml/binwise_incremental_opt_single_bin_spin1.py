"""
Bin-wise Q optimization with a generic vector lineshape + ssrf_realtime_v2.

Starts from ``GenerateVectorLineshape`` at P=0.45, applies physical-Voigt (or
single-bin) ssRF with shared spectral recovery (event-shape ``n_ref`` so
unburned bins stay put). Selects the ``gamma_rf`` that maximizes level tensor Q.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DULYA_V2 = REPO_ROOT / "Data_Creation" / "dulya_fit_v2"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DULYA_V2) not in sys.path:
    sys.path.insert(0, str(DULYA_V2))

from common import (  # noqa: E402
    DIFFUSION_SCALE,
    DT,
    NUM_BINS,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE_PHYSICAL_VOIGT,
    RF_MODE_SINGLE_BIN,
)

RF_GAUSSIAN_FWHM_R = 0.0
RF_LORENTZIAN_FWHM_R = 0.0


from model_bridge import (  # noqa: E402
    build_spin1_model,
    configure_ssrf_burn,
    euler_n_sub,
    level_pq,
)
from physics.lineshape.Lineshape import GenerateVectorLineshape  # noqa: E402
from physics.ssrf_realtime_v2.model import Spin1Model  # noqa: E402
from physics.ssrf_realtime_v2.rate_equations_realtime import (  # noqa: E402
    burn_preserves_ps_sign,
)

DEFAULT_POLARIZATION = 0.45


def q_polarization(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def q_at_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    mirror = len(iplus) - int(bin_idx) - 1
    iplus_theta = float(iplus[bin_idx] + iplus[mirror])
    iminus_theta = float(iminus[bin_idx] + iminus[mirror])
    return iplus_theta - iminus_theta


def total_signal_area(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus + iminus))


def freq_to_bin_idx(f: np.ndarray, f_target: float) -> int:
    return int(np.argmin(np.abs(f - float(f_target))))


def model_spectrum(model: Spin1Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R, I+, I-) in event / vector-lineshape intensity units."""
    ip, im, _ = model.physical_intensities()
    return (
        np.asarray(model.Rplus, dtype=float).copy(),
        np.asarray(ip, dtype=float),
        np.asarray(im, dtype=float),
    )


def clone_model(model: Spin1Model) -> Spin1Model:
    """Copy model state so burn trials do not mutate the working trajectory."""
    trial = Spin1Model(copy.deepcopy(model.params))
    trial.n = np.asarray(model.n, dtype=float).copy()
    trial.n_ref = np.asarray(model.n_ref, dtype=float).copy()
    trial.n_initial = np.asarray(model.n_initial, dtype=float).copy()
    trial.t = float(model.t)
    trial.display_cal = float(model.display_cal)
    trial._populations_from_intensities = bool(model._populations_from_intensities)
    trial._recovery_boltzmann_P = model._recovery_boltzmann_P
    trial._force_boltzmann_recovery = bool(model._force_boltzmann_recovery)
    trial._active_idx = None if model._active_idx is None else np.asarray(model._active_idx).copy()
    trial.n_plus = float(model.n_plus)
    trial.n_zero = float(model.n_zero)
    trial.n_minus = float(model.n_minus)
    trial.n_plus_initial = float(model.n_plus_initial)
    trial.n_zero_initial = float(model.n_zero_initial)
    trial.n_minus_initial = float(model.n_minus_initial)
    return trial


@dataclass
class BurnConfig:
    num_bins: int = NUM_BINS
    f_min: float = -3.0
    f_max: float = 3.0
    dt: float = DT
    steps: int = 3000
    gamma_min: float = 0.0
    gamma_max: float = 2.0
    n_gamma_steps: int = 11
    rf_mode: str = RF_MODE_PHYSICAL_VOIGT
    diffusion_scale: float = DIFFUSION_SCALE
    gaussian_fwhm_R: float = RF_GAUSSIAN_FWHM_R
    lorentzian_fwhm_R: float = RF_LORENTZIAN_FWHM_R

    def __post_init__(self) -> None:
        if self.gamma_min > self.gamma_max:
            raise ValueError(
                f"gamma_min ({self.gamma_min}) must be <= gamma_max ({self.gamma_max})"
            )
        if self.n_gamma_steps < 1:
            raise ValueError(f"n_gamma_steps must be >= 1, got {self.n_gamma_steps}")
        if self.rf_mode not in (RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN):
            raise ValueError(f"unknown rf_mode={self.rf_mode!r}")

    @property
    def f(self) -> np.ndarray:
        return np.linspace(self.f_min, self.f_max, self.num_bins)

    @property
    def gamma_rf_values(self) -> np.ndarray:
        if self.n_gamma_steps == 1:
            return np.array([self.gamma_max], dtype=float)
        return np.linspace(self.gamma_min, self.gamma_max, self.n_gamma_steps)


def build_model(config: BurnConfig, polarization: float) -> Spin1Model:
    """Generic vector lineshape loaded into ssrf_realtime_v2 with shared recovery."""
    P = float(polarization)
    _, iplus, iminus = GenerateVectorLineshape(P, config.f)
    model = build_spin1_model(
        np.asarray(iplus, dtype=float),
        np.asarray(iminus, dtype=float),
        polarization=P,
        num_bins=config.num_bins,
        dt=config.dt,
        rf_enabled=False,
        relax_enabled=True,
        diffusion_scale=config.diffusion_scale,
        rf_gaussian_fwhm_R=config.gaussian_fwhm_R,
        rf_lorentzian_fwhm_R=config.lorentzian_fwhm_R,
        r_min=config.f_min,
        r_max=config.f_max,
    )
    # ssRF hole-filling: keep event n_ref; Boltzmann P = initial P if P drifts.
    model.set_recovery_boltzmann_P(float(model.n_plus - model.n_minus))
    return model


def apply_spin1_burn(
    model: Spin1Model,
    burn_R: float,
    gamma_rf: float,
    n_steps: int,
    *,
    rf_mode: str,
    gaussian_fwhm_R: float,
    lorentzian_fwhm_R: float,
) -> Spin1Model | None:
    """Burn with ssrf_realtime_v2 + shared Dulya recovery via configure_ssrf_burn."""
    if gamma_rf <= 0.0 or n_steps <= 0:
        return None

    f_before, iplus_before, iminus_before = model_spectrum(model)
    bin_idx = freq_to_bin_idx(f_before, burn_R)
    ps_before = float(iplus_before[bin_idx] + iminus_before[bin_idx])

    burned = clone_model(model)
    configure_ssrf_burn(
        burned,
        bin_idx,
        float(gamma_rf),
        rf_mode=rf_mode,
        gaussian_fwhm_R=gaussian_fwhm_R,
        lorentzian_fwhm_R=lorentzian_fwhm_R,
    )
    burned.params.rf_burn_R = float(burn_R)
    burned.params.rf_enabled = True
    burned.params.dnp_enabled = False

    n_sub, dt_sub = euler_n_sub(float(gamma_rf), float(burned.params.dt))
    for _ in range(int(n_steps)):
        for _ in range(n_sub):
            burned.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)

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
    config: BurnConfig,
) -> tuple[float, float, float, Spin1Model] | None:
    """Sweep gamma_rf; maximize level tensor Q (Boltzmann / spin-1 Q)."""
    _, iplus, iminus = model_spectrum(model)
    baseline_q_fit = q_polarization(iplus, iminus)
    baseline_q_bin = q_at_bin(iplus, iminus, bin_idx)
    baseline_level_q = level_pq(model)[1]
    best_gamma_rf = 0.0
    best_q_fit = baseline_q_fit
    best_q_bin = baseline_q_bin
    best_level_q = baseline_level_q
    best_model: Spin1Model | None = None

    for gamma_rf in config.gamma_rf_values:
        burned = apply_spin1_burn(
            model,
            burn_R,
            float(gamma_rf),
            config.steps,
            rf_mode=config.rf_mode,
            gaussian_fwhm_R=config.gaussian_fwhm_R,
            lorentzian_fwhm_R=config.lorentzian_fwhm_R,
        )
        if burned is None:
            continue
        _, iplus_try, iminus_try = model_spectrum(burned)
        q_fit_try = q_polarization(iplus_try, iminus_try)
        q_try_bin = q_at_bin(iplus_try, iminus_try, bin_idx)
        _, level_q_try = level_pq(burned)
        print(
            f"gamma_rf: {gamma_rf:.4f}  Q_level={level_q_try:.6g}  "
            f"Q_fit={q_fit_try:.6g}  Q_bin={q_try_bin:.6g}  "
            f"best_Q_level={best_level_q:.6g}"
        )
        # Optimize physical tensor Q; tie-break on local spectral bin Q.
        better = (level_q_try > best_level_q + 1e-15) or (
            abs(level_q_try - best_level_q) <= 1e-15 and q_try_bin > best_q_bin
        )
        if better:
            print(
                f"New best gamma_rf: {gamma_rf:.6e}, "
                f"delta_Q_level: {level_q_try - baseline_level_q:.6e}, "
                f"delta_Q_fit: {q_fit_try - baseline_q_fit:.6e}"
            )
            best_q_fit = q_fit_try
            best_q_bin = q_try_bin
            best_level_q = level_q_try
            best_gamma_rf = float(gamma_rf)
            best_model = burned

    if best_gamma_rf <= 0.0 or best_model is None:
        return None
    return best_gamma_rf, best_q_fit, best_q_bin, best_model


def optimize_binwise_incremental(
    config: BurnConfig,
    polarization: float,
    *,
    f_target: float = -0.92,
) -> dict:
    model = build_model(config, polarization)
    f, iplus, iminus = model_spectrum(model)
    p_initial, q_initial_level = level_pq(model)

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
            "q_level": q_initial_level,
            "p_level": p_initial,
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
    best = find_best_gamma_rf_for_bin(model, burn_R, bin_idx, config)
    if best is None:
        raise RuntimeError(
            f"No valid v2 burn found at bin_idx={bin_idx}, f={burn_R:.4f}"
        )

    best_gamma_rf, best_q_fit, best_q_bin, model = best
    f, iplus, iminus = model_spectrum(model)
    p_final, q_final_level = level_pq(model)
    iplus_area_before = current_iplus_area
    iminus_area_before = current_iminus_area
    current_q_bin = best_q_bin
    current_q = best_q_fit
    current_iplus_area = float(np.sum(iplus))
    current_iminus_area = float(np.sum(iminus))
    current_area = total_signal_area(iplus, iminus)
    q_bin_gain = current_q_bin - q_bin_before
    q_level_gain = q_final_level - q_initial_level
    step += 1
    trace.append(
        {
            "step": step,
            "bin_idx": bin_idx,
            "f": burn_R,
            "gamma_rf": best_gamma_rf,
            "steps": config.steps,
            "rf_mode": config.rf_mode,
            "reward": q_level_gain,
            "q_bin_reward": q_bin_gain,
            "q_bin": current_q_bin,
            "q_bin_gain": q_bin_gain,
            "iplus_reduction": iplus_area_before - current_iplus_area,
            "iminus_reduction": iminus_area_before - current_iminus_area,
            "q": current_q,
            "q_gain": current_q - initial_q,
            "q_level": q_final_level,
            "q_level_gain": q_level_gain,
            "p_level": p_final,
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
        "rf_mode": config.rf_mode,
        "physics_model": "ssrf_realtime_v2",
        "lineshape_model": "GenerateVectorLineshape",
        "initial_q": initial_q,
        "final_q": current_q,
        "p_initial": p_initial,
        "q_initial_level": q_initial_level,
        "p_final": p_final,
        "q_final_level": q_final_level,
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
    axes[1].set_xlabel("R")
    axes[1].set_ylabel("Q profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    q_total = q_polarization(iplus, iminus)
    fig.suptitle(
        f"Unburned vector lineshape (v2)  P={polarization:.3f}  "
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
    axes[1].set_xlabel("R")
    axes[1].set_ylabel("Q profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    delta_q = result["final_q"] - result["initial_q"]
    delta_iplus = result["final_iplus_area"] - result["initial_iplus_area"]
    delta_iminus = result["final_iminus_area"] - result["initial_iminus_area"]
    delta_area = result["final_area"] - result["initial_area"]
    title = (
        f"Vector+v2 {result.get('rf_mode', '')}  P={result['polarization']*100:.2f}%  "
        f"Q_initial: {result['q_initial_level']*100:.2f}% -> {result['q_final_level']*100:.2f}%  "
        f"Q_gain={delta_q*100:.2f}%  "
        f"I+: {delta_iplus:+.4f}  I-: {delta_iminus:+.4f}  area: {delta_area:+.4f}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    polarization = float(DEFAULT_POLARIZATION)
    f_target = -0.92
    config = BurnConfig()
    out_dir = REPO_ROOT / "results" / "current" / "binwise_incremental_spin1_v2"
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
        f"Bin-wise Q opt (ssrf_realtime_v2 + GenerateVectorLineshape, "
        f"rf_mode={config.rf_mode}) at P={polarization * 100:.2f}%:"
    )
    print(
        f"  start: Q_spec={result['initial_q'] * 100:.5f}%  "
        f"P_level={result['p_initial']:.6g}  Q_level={result['q_initial_level']:.6g}"
    )
    print(f"  start I+ area: {result['initial_iplus_area']:.8f}")
    print(f"  start I- area: {result['initial_iminus_area']:.8f}")
    print(f"  start area: {result['initial_area']:.8f}")
    for row in result["trace"][1:]:
        if row.get("gamma_rf", 0.0) <= 0.0:
            continue
        print(
            f"  burn {row['step']}: bin={row['bin_idx']}, f={row['f']:.3f}, "
            f"gamma_rf={row['gamma_rf']:.4e}, steps={row['steps']}, "
            f"Q_level_gain={row['reward']:.5e}, "
            f"Q_bin_gain={row['q_bin_reward']:.5e}, "
            f"I+ reduction={row['iplus_reduction']:.5e}, "
            f"I- reduction={row['iminus_reduction']:.5e}, "
            f"Q_bin={row['q_bin']:.5e}, Q_spec={row['q'] * 100:.5f}%"
        )
    print(
        f"  final: Q_spec={result['final_q'] * 100:.5f}%  "
        f"P_level={result['p_final']:.6g}  Q_level={result['q_final_level']:.6g}"
    )
    print(
        f"  total Q_level gain: "
        f"{(result['q_final_level'] - result['q_initial_level']) * 100:.5f}% "
        f"(spec dQ={(result['final_q'] - result['initial_q']) * 100:.5f}%)"
    )
    print(f" total P change: {(result['final_area'] - result['initial_area']) * 100:.5f}%")
    print(f"  total I+ change: {result['final_iplus_area'] - result['initial_iplus_area']:.8f}")
    print(f"  total I- change: {result['final_iminus_area'] - result['initial_iminus_area']:.8f}")
    print(f"  total area gain: {result['final_area'] - result['initial_area']:.8f}")
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
