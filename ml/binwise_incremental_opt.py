"""
Bin-wise Q optimization over spectral bins with ssrf_realtime_v2.

Burn trials rollback integration steps when I± or Ps would cross zero at the
burn or RF-mirror bin. Only those two bins are committed; neighbor diffusion
spillover is discarded so burns do not propagate across the R grid beyond the
ssRF mirror pair. Voigt RF widths are zero (single-bin limit).
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm

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
    commit_touched_bins_only,
    configure_ssrf_burn,
    euler_n_sub,
    level_pq,
    mirror_bin_idx,
)
from physics.lineshape.Lineshape import GenerateVectorLineshape  # noqa: E402
from physics.ssrf_realtime_v2.conversions import physical_intensities_to_packet_n  # noqa: E402
from physics.ssrf_realtime_v2.model import Spin1Model  # noqa: E402
from physics.ssrf_realtime_v2.rate_equations_realtime import (  # noqa: E402
    _value_crosses_zero,
    burn_preserves_ps_sign,
)

DEFAULT_POLARIZATION = 0.45


def q_polarization(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def theta_q_profile(iplus: np.ndarray, iminus: np.ndarray) -> np.ndarray:
    """Theta-space Q at each bin: Q_theta(i) = Q_spec(i) + Q_spec(mirror(i))."""
    spectral_q = np.asarray(iplus, dtype=float) - np.asarray(iminus, dtype=float)
    mirror_idx = np.arange(len(spectral_q))[::-1]
    return spectral_q + spectral_q[mirror_idx]


def q_at_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    return float(theta_q_profile(iplus, iminus)[bin_idx])


def candidate_burn_bins(
    iplus: np.ndarray,
    iminus: np.ndarray,
    f: np.ndarray | None = None,
    *,
    only_negative_initial_q: bool = True,
    use_theta_q: bool = False,
    burn_r_abs_min: float = 0.0,
) -> np.ndarray:
    """Bins to try burning; default filters spectral Q = I+ - I- < 0 at t=0."""
    n = len(iplus)
    spectral_q = np.asarray(iplus, dtype=float) - np.asarray(iminus, dtype=float)
    if not only_negative_initial_q:
        cands = np.arange(n, dtype=int)
    elif use_theta_q:
        mirror_idx = np.arange(n)[::-1]
        unique = np.arange(n) <= mirror_idx
        negative = theta_q_profile(iplus, iminus) < 0.0
        cands = np.flatnonzero(unique & negative)
    else:
        cands = np.flatnonzero(spectral_q < 0.0)
    if f is not None and burn_r_abs_min > 0.0:
        ff = np.asarray(f, dtype=float)
        cands = cands[np.abs(ff[cands]) >= float(burn_r_abs_min)]
    return cands


def ssrf_touched_bins(n_bins: int, bin_idx: int) -> list[int]:
    """Burn bin plus RF mirror; incremental commit stays on this pair only."""
    mirror = mirror_bin_idx(n_bins, int(bin_idx))
    if mirror == int(bin_idx):
        return [int(bin_idx)]
    return [int(bin_idx), mirror]


def sync_model_from_spectrum(
    model: Spin1Model,
    iplus: np.ndarray,
    iminus: np.ndarray,
    *,
    n_ref: np.ndarray,
    recovery_P: float,
) -> None:
    """Reload packet state from intensities while keeping the event ``n_ref``."""
    model.n = physical_intensities_to_packet_n(
        np.asarray(iplus, dtype=float),
        np.asarray(iminus, dtype=float),
        model.mu,
        display_cal=model.display_cal,
        dR=model.dR,
    )
    model.n_ref = np.asarray(n_ref, dtype=float).copy()
    model._populations_from_intensities = True
    model.set_recovery_boltzmann_P(float(recovery_P))
    model._sync_level_populations(capture_initial=False)
    model.params.rf_enabled = False
    model.params.gamma_rf = 0.0
    model.t = 0.0


def commit_burn_to_spectrum(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_sim: np.ndarray,
    iminus_sim: np.ndarray,
    bin_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply burn trial at burn+mirror only; discard neighbor diffusion spillover."""
    touched = ssrf_touched_bins(len(iplus), bin_idx)
    return commit_touched_bins_only(iplus, iminus, iplus_sim, iminus_sim, touched)


def total_signal_area(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus + iminus))


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
    n_gamma_coarse: int = 5
    n_gamma_fine: int = 5
    gamma_early_stop: int = 2
    only_negative_initial_q: bool = True
    q_filter_use_theta: bool = False
    burn_r_abs_min: float = 0.02
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
        vals = np.linspace(self.gamma_min, self.gamma_max, self.n_gamma_steps)
        return vals[vals > 0.0]


def gamma_search_values(
    config: BurnConfig,
    *,
    center: float | None = None,
    half_width: float | None = None,
    n_steps: int | None = None,
) -> np.ndarray:
    """Positive gamma_rf grid; skip zero (no burn)."""
    n = config.n_gamma_coarse if center is None else (n_steps or config.n_gamma_fine)
    if center is None:
        lo, hi = config.gamma_min, config.gamma_max
    else:
        hw = half_width if half_width is not None else 0.25 * (config.gamma_max - config.gamma_min)
        lo = max(config.gamma_min, float(center) - hw)
        hi = min(config.gamma_max, float(center) + hw)
    if n <= 1 or lo >= hi:
        vals = np.array([hi if center is not None else config.gamma_max], dtype=float)
    else:
        vals = np.linspace(lo, hi, n)
    return vals[vals > 0.0]


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
    model.set_recovery_boltzmann_P(float(model.n_plus - model.n_minus))
    return model


def apply_spin1_burn(
    model: Spin1Model,
    bin_idx: int,
    gamma_rf: float,
    n_steps: int,
    *,
    rf_mode: str,
    gaussian_fwhm_R: float,
    lorentzian_fwhm_R: float,
    burn_R: float | None = None,
) -> Spin1Model | None:
    """Burn with rollback if I± or Ps cross zero at burn / mirror bins."""
    if gamma_rf <= 0.0 or n_steps <= 0:
        return None

    f_before, iplus_before, iminus_before = model_spectrum(model)
    bin_idx = int(bin_idx)
    n_bins = len(iplus_before)
    touched = ssrf_touched_bins(n_bins, bin_idx)
    if burn_R is None:
        burn_R = float(f_before[bin_idx])
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

    ip_prev = {k: float(iplus_before[k]) for k in touched}
    im_prev = {k: float(iminus_before[k]) for k in touched}

    n_sub, dt_sub = euler_n_sub(float(gamma_rf), float(burned.params.dt))
    steps_done = 0
    for _ in range(int(n_steps)):
        state_before = burned.n.copy()
        for _ in range(n_sub):
            burned.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)

        _, ip_step, im_step = model_spectrum(burned)
        sign_ok = True
        for idx in touched:
            for b, a in (
                (ip_prev[idx], float(ip_step[idx])),
                (im_prev[idx], float(im_step[idx])),
                (ip_prev[idx] + im_prev[idx], float(ip_step[idx] + im_step[idx])),
            ):
                if _value_crosses_zero(b, a):
                    sign_ok = False
                    break
            if not sign_ok:
                break
        if not sign_ok:
            burned.n = state_before
            break

        for idx in touched:
            ip_prev[idx] = float(ip_step[idx])
            im_prev[idx] = float(im_step[idx])
        steps_done += 1

    if steps_done == 0:
        return None

    _, iplus_after, iminus_after = model_spectrum(burned)
    ps_after = float(iplus_after[bin_idx] + iminus_after[bin_idx])
    if ps_after == ps_before:
        return None
    if not burn_preserves_ps_sign(
        iplus_before, iminus_before, iplus_after, iminus_after, bin_idx
    ):
        return None

    return burned


def _evaluate_gamma_rf(
    model: Spin1Model,
    iplus: np.ndarray,
    iminus: np.ndarray,
    bin_idx: int,
    gamma_rf: float,
    config: BurnConfig,
    *,
    n_ref: np.ndarray,
    recovery_P: float,
    baseline_q_fit: float,
    baseline_q_bin: float,
    baseline_level_q: float,
    best_q_fit: float,
    best_q_bin: float,
    best_level_q: float,
    best_gamma_rf: float,
    best_iplus: np.ndarray | None,
    best_iminus: np.ndarray | None,
) -> tuple[float, float, float, float, np.ndarray | None, np.ndarray | None, bool]:
    """Try one gamma on committed burn+mirror bins; return updated bests."""
    burned = apply_spin1_burn(
        model,
        bin_idx,
        float(gamma_rf),
        config.steps,
        rf_mode=config.rf_mode,
        gaussian_fwhm_R=config.gaussian_fwhm_R,
        lorentzian_fwhm_R=config.lorentzian_fwhm_R,
    )
    if burned is None:
        return (
            best_q_fit,
            best_q_bin,
            best_level_q,
            best_gamma_rf,
            best_iplus,
            best_iminus,
            False,
        )

    _, ip_sim, im_sim = model_spectrum(burned)
    ip_try, im_try = commit_burn_to_spectrum(iplus, iminus, ip_sim, im_sim, bin_idx)

    trial = clone_model(model)
    sync_model_from_spectrum(
        trial, ip_try, im_try, n_ref=n_ref, recovery_P=recovery_P
    )
    q_fit_try = q_polarization(ip_try, im_try)
    q_try_bin = q_at_bin(ip_try, im_try, bin_idx)
    _, level_q_try = level_pq(trial)
    better = (level_q_try > best_level_q + 1e-15) or (
        abs(level_q_try - best_level_q) <= 1e-15 and q_try_bin > best_q_bin
    )
    if better:
        return (
            q_fit_try,
            q_try_bin,
            level_q_try,
            float(gamma_rf),
            ip_try,
            im_try,
            True,
        )
    return (
        best_q_fit,
        best_q_bin,
        best_level_q,
        best_gamma_rf,
        best_iplus,
        best_iminus,
        False,
    )


def find_best_gamma_rf_for_bin(
    model: Spin1Model,
    iplus: np.ndarray,
    iminus: np.ndarray,
    bin_idx: int,
    config: BurnConfig,
    *,
    n_ref: np.ndarray,
    recovery_P: float,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray] | None:
    """Coarse-to-fine gamma_rf sweep; maximize level tensor Q on committed burns."""
    baseline_q_fit = q_polarization(iplus, iminus)
    baseline_q_bin = q_at_bin(iplus, iminus, bin_idx)
    baseline_level_q = level_pq(model)[1]
    best_gamma_rf = 0.0
    best_q_fit = baseline_q_fit
    best_q_bin = baseline_q_bin
    best_level_q = baseline_level_q
    best_iplus: np.ndarray | None = None
    best_iminus: np.ndarray | None = None

    stale = 0
    for gamma_rf in gamma_search_values(config):
        (
            best_q_fit,
            best_q_bin,
            best_level_q,
            best_gamma_rf,
            best_iplus,
            best_iminus,
            improved,
        ) = _evaluate_gamma_rf(
            model,
            iplus,
            iminus,
            bin_idx,
            gamma_rf,
            config,
            n_ref=n_ref,
            recovery_P=recovery_P,
            baseline_q_fit=baseline_q_fit,
            baseline_q_bin=baseline_q_bin,
            baseline_level_q=baseline_level_q,
            best_q_fit=best_q_fit,
            best_q_bin=best_q_bin,
            best_level_q=best_level_q,
            best_gamma_rf=best_gamma_rf,
            best_iplus=best_iplus,
            best_iminus=best_iminus,
        )
        if improved:
            stale = 0
        elif best_gamma_rf > 0.0:
            stale += 1
            if stale >= config.gamma_early_stop:
                break

    if best_gamma_rf > 0.0 and config.n_gamma_fine > 1:
        stale = 0
        for gamma_rf in gamma_search_values(config, center=best_gamma_rf):
            if abs(float(gamma_rf) - best_gamma_rf) < 1e-15:
                continue
            (
                best_q_fit,
                best_q_bin,
                best_level_q,
                best_gamma_rf,
                best_iplus,
                best_iminus,
                improved,
            ) = _evaluate_gamma_rf(
                model,
                iplus,
                iminus,
                bin_idx,
                gamma_rf,
                config,
                n_ref=n_ref,
                recovery_P=recovery_P,
                baseline_q_fit=baseline_q_fit,
                baseline_q_bin=baseline_q_bin,
                baseline_level_q=baseline_level_q,
                best_q_fit=best_q_fit,
                best_q_bin=best_q_bin,
                best_level_q=best_level_q,
                best_gamma_rf=best_gamma_rf,
                best_iplus=best_iplus,
                best_iminus=best_iminus,
            )
            if improved:
                stale = 0
            else:
                stale += 1
                if stale >= config.gamma_early_stop:
                    break

    if best_gamma_rf <= 0.0 or best_iplus is None or best_iminus is None:
        return None
    return best_gamma_rf, best_q_fit, best_q_bin, best_level_q, best_iplus, best_iminus


def optimize_binwise_incremental(
    config: BurnConfig,
    polarization: float,
) -> dict:
    model = build_model(config, polarization)
    f, iplus, iminus = model_spectrum(model)
    iplus_unburned = iplus.copy()
    iminus_unburned = iminus.copy()
    n_ref_event = model.n_ref.copy()
    recovery_P = float(model.n_plus - model.n_minus)
    p_initial, q_initial_level = level_pq(model)

    burn_bins = candidate_burn_bins(
        iplus_unburned,
        iminus_unburned,
        f,
        only_negative_initial_q=config.only_negative_initial_q,
        use_theta_q=config.q_filter_use_theta,
        burn_r_abs_min=config.burn_r_abs_min,
    )

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
            "q_level": q_initial_level,
            "p_level": p_initial,
            "iplus_area": initial_iplus_area,
            "iminus_area": initial_iminus_area,
            "area": initial_area,
            "action": None,
            "n_candidate_bins": int(len(burn_bins)),
        }
    ]
    step = 0
    n_skipped_no_improvement = 0

    for bin_idx in tqdm.tqdm(burn_bins, desc="Optimizing bins", total=len(burn_bins)):
        bin_idx = int(bin_idx)
        burn_R = float(f[bin_idx])
        q_bin_before = q_at_bin(iplus, iminus, bin_idx)
        _, q_level_before = level_pq(model)
        best = find_best_gamma_rf_for_bin(
            model,
            iplus,
            iminus,
            bin_idx,
            config,
            n_ref=n_ref_event,
            recovery_P=recovery_P,
        )

        if best is None:
            n_skipped_no_improvement += 1
            continue

        best_gamma_rf, best_q_fit, best_q_bin, best_level_q, iplus, iminus = best
        sync_model_from_spectrum(
            model,
            iplus,
            iminus,
            n_ref=n_ref_event,
            recovery_P=recovery_P,
        )
        f, _, _ = model_spectrum(model)
        p_after, _ = level_pq(model)
        iplus_area_before = current_iplus_area
        iminus_area_before = current_iminus_area
        current_q_bin = best_q_bin
        current_q = best_q_fit
        current_iplus_area = float(np.sum(iplus))
        current_iminus_area = float(np.sum(iminus))
        current_area = total_signal_area(iplus, iminus)
        q_bin_gain = current_q_bin - q_bin_before
        q_level_gain = best_level_q - q_level_before
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
                "q_level": best_level_q,
                "q_level_gain": q_level_gain,
                "p_level": p_after,
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

    p_final, q_final_level = level_pq(model)

    return {
        "polarization": polarization,
        "rf_mode": config.rf_mode,
        "physics_model": "ssrf_realtime_v2",
        "lineshape_model": "GenerateVectorLineshape",
        "diffusion_scale": config.diffusion_scale,
        "only_negative_initial_q": config.only_negative_initial_q,
        "n_candidate_bins": int(len(burn_bins)),
        "n_skipped_no_improvement": n_skipped_no_improvement,
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
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
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
) -> None:
    ps = iplus + iminus
    q_profile = iplus - iminus

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].step(f, ps, label=r"$P_s = I_+ + I_-$", color="black")
    axes[0].step(f, iplus, label=r"$I_+$", color="tab:red")
    axes[0].step(f, iminus, label=r"$I_-$", color="tab:blue")
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
        f"All-bin v2 {result.get('rf_mode', '')}  P={result['polarization']*100:.2f}%  "
        f"Q_level: {result['q_initial_level']*100:.2f}% -> {result['q_final_level']*100:.2f}%  "
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
        out_dir / f"unburned_all_bins_P{polarization:.2f}.png",
    )

    result = optimize_binwise_incremental(config, polarization)
    plot_greedy_burns(
        result, out_dir / f"incremental_policy_all_bins_P{polarization:.2f}.png"
    )

    n_burns = sum(1 for row in result["trace"][1:] if row.get("gamma_rf", 0.0) > 0.0)

    print(
        f"Bin-wise Q opt (ssrf_realtime_v2 + GenerateVectorLineshape, "
        f"rf_mode={config.rf_mode}, diffusion_scale={config.diffusion_scale}, "
        f"voigt_width=0, only_Q<0 bins={config.only_negative_initial_q}) "
        f"at P={polarization * 100:.2f}%:"
    )
    print(
        f"  start: Q_spec={result['initial_q'] * 100:.5f}%  "
        f"P_level={result['p_initial']:.6g}  Q_level={result['q_initial_level']:.6g}"
    )
    print(f"  start I+ area: {result['initial_iplus_area']:.8f}")
    print(f"  start I- area: {result['initial_iminus_area']:.8f}")
    print(f"  start area: {result['initial_area']:.8f}")
    print(
        f"  candidate bins: {result['n_candidate_bins']}  "
        f"burns applied: {n_burns}  "
        f"no-improvement skips: {result['n_skipped_no_improvement']}"
    )
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
