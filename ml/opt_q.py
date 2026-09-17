import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RIVANNA = REPO_ROOT / "Data_Creation" / "rivanna"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RIVANNA) not in sys.path:
    sys.path.insert(0, str(RIVANNA))

from bin_setup import equilibrium_lineshape, get_shape_params, spin1_scale_factors
from burn_selection import is_manipulation_shard_bin
from common import (
    DIFFUSION_SCALE,
    F_MAX,
    F_MIN,
    NUM_BINS,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE_PHYSICAL_VOIGT,
)
from model_bridge import (
    build_spin1_model,
    configure_ssrf_burn,
    euler_n_sub,
    full_spectrum_intensities,
)

DT = 0.0055
GAUSSIAN_FWHM_R = RF_GAUSSIAN_FWHM_R
LORENTZIAN_FWHM_R = RF_LORENTZIAN_FWHM_R
MAX_BURN_STEPS = 500
# Per-bin γ_RF search grid (applied power / rate strength).
GAMMA_RF_MIN = 0.0
GAMMA_RF_MAX = 100.0
N_GAMMA_STEPS = 100


def q_polarization(iplus, iminus):
    return np.sum(iplus - iminus)


def q_at_bin(iplus, iminus, bin_idx):
    return iplus[bin_idx] - iminus[bin_idx]


def total_signal_area(iplus, iminus):
    return np.sum(iplus + iminus)


def initially_negative_q_bins(iplus, iminus):
    """Burn-window bins where unburned spectral Q = I+ - I- is negative."""
    q_profile = iplus - iminus
    return [
        i
        for i in np.flatnonzero(q_profile < 0.0)
        if is_manipulation_shard_bin(i)
    ]


def equilibrium_spin1_intensities(polarization, f):
    """Dulya equilibrium lineshape scaled into Spin1 intensity units."""
    _, ip_fit, im_fit = equilibrium_lineshape(
        polarization,
        f,
        get_shape_params(),
    )
    to_spin1, _ = spin1_scale_factors(polarization, ip_fit, im_fit)
    return ip_fit * to_spin1, im_fit * to_spin1


class BurnConfig:
    def __init__(self):
        self.num_bins = NUM_BINS
        self.f_min = F_MIN
        self.f_max = F_MAX
        self.dt = DT
        self.max_steps = MAX_BURN_STEPS
        self.gamma_rf_min = GAMMA_RF_MIN
        self.gamma_rf_max = GAMMA_RF_MAX
        self.n_gamma_steps = N_GAMMA_STEPS
        self.gaussian_fwhm_R = GAUSSIAN_FWHM_R
        self.lorentzian_fwhm_R = LORENTZIAN_FWHM_R
        self.diffusion_scale = DIFFUSION_SCALE
        self.rf_mode = RF_MODE_PHYSICAL_VOIGT

    @property
    def f(self):
        return np.linspace(self.f_min, self.f_max, self.num_bins)

    @property
    def gamma_rf_values(self):
        if self.n_gamma_steps < 1:
            raise ValueError(f"n_gamma_steps must be >= 1, got {self.n_gamma_steps}")
        if self.gamma_rf_min > self.gamma_rf_max:
            raise ValueError(
                f"gamma_rf_min ({self.gamma_rf_min}) must be <= "
                f"gamma_rf_max ({self.gamma_rf_max})"
            )
        if self.n_gamma_steps == 1:
            return np.array([self.gamma_rf_max])
        return np.linspace(self.gamma_rf_min, self.gamma_rf_max, self.n_gamma_steps)


def apply_ssrf_burn_trajectory(
    iplus,
    iminus,
    bin_idx,
    polarization,
    config,
    n_steps,
    gamma_rf,
):
    """Apply exactly ``n_steps`` ssRF macro-steps at ``gamma_rf``.

    Starts from the supplied spin1-scale intensities (so sequential opt burns can
    chain), using create_dae Voigt / dt / diffusion settings.
    """
    if n_steps <= 0 or gamma_rf <= 0.0:
        return None

    model = build_spin1_model(
        iplus,
        iminus,
        polarization=polarization,
        num_bins=config.num_bins,
        dt=config.dt,
        rf_enabled=True,
        relax_enabled=True,
        diffusion_scale=config.diffusion_scale,
        rf_gaussian_fwhm_R=config.gaussian_fwhm_R,
        rf_lorentzian_fwhm_R=config.lorentzian_fwhm_R,
        r_min=config.f_min,
        r_max=config.f_max,
    )
    configure_ssrf_burn(
        model,
        bin_idx,
        gamma_rf,
        rf_mode=config.rf_mode,
        gaussian_fwhm_R=config.gaussian_fwhm_R,
        lorentzian_fwhm_R=config.lorentzian_fwhm_R,
    )
    n_sub, dt_sub = euler_n_sub(gamma_rf, config.dt)
    iplus_full = np.empty((n_steps + 1, config.num_bins))
    iminus_full = np.empty((n_steps + 1, config.num_bins))
    ip0, im0, _ = full_spectrum_intensities(model)
    iplus_full[0] = ip0
    iminus_full[0] = im0
    for k in range(1, n_steps + 1):
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
        ip_k, im_k, _ = full_spectrum_intensities(model)
        iplus_full[k] = ip_k
        iminus_full[k] = im_k
    return iplus_full, iminus_full, n_sub


def _best_frame_for_trajectory(
    iplus_full,
    iminus_full,
    bin_idx,
    baseline_q_bin,
    baseline_q_tot,
    max_steps,
):
    """Pick trajectory frame that best raises total Q (and local bin Q)."""
    best_q_tot = baseline_q_tot
    best_q_bin = baseline_q_bin
    best_steps = 0
    best_iplus = None
    best_iminus = None
    for k in range(1, max_steps + 1):
        q_bin = q_at_bin(iplus_full[k], iminus_full[k], bin_idx)
        q_tot = q_polarization(iplus_full[k], iminus_full[k])
        if q_bin <= baseline_q_bin or q_tot <= baseline_q_tot:
            continue
        if q_tot > best_q_tot or (q_tot == best_q_tot and q_bin > best_q_bin):
            best_q_tot = q_tot
            best_q_bin = q_bin
            best_steps = k
            best_iplus = iplus_full[k].copy()
            best_iminus = iminus_full[k].copy()
    if best_steps <= 0 or best_iplus is None or best_iminus is None:
        return None
    return best_q_tot, best_q_bin, best_steps, best_iplus, best_iminus


def find_best_burn_for_bin(iplus, iminus, bin_idx, polarization, config):
    """Search γ_RF × n_steps for the burn that most improves total Q.

    For each candidate power, run a fixed-length create_dae-style trajectory and
    keep the best valid frame (local bin Q and total Q both improve). Across
    powers, prefer the larger total-Q gain.
    """
    baseline_q_bin = q_at_bin(iplus, iminus, bin_idx)
    baseline_q_tot = q_polarization(iplus, iminus)

    best_q_tot = baseline_q_tot
    best_q_bin = baseline_q_bin
    best_gamma = 0.0
    best_steps = 0
    best_n_sub = 0
    best_iplus = None
    best_iminus = None

    for gamma_rf in config.gamma_rf_values:
        traj = apply_ssrf_burn_trajectory(
            iplus,
            iminus,
            bin_idx,
            polarization,
            config,
            config.max_steps,
            gamma_rf,
        )
        if traj is None:
            continue
        iplus_full, iminus_full, n_sub = traj
        picked = _best_frame_for_trajectory(
            iplus_full,
            iminus_full,
            bin_idx,
            baseline_q_bin,
            baseline_q_tot,
            config.max_steps,
        )
        if picked is None:
            continue
        q_tot, q_bin, steps, ip_try, im_try = picked
        if q_tot > best_q_tot or (
            q_tot == best_q_tot and q_bin > best_q_bin
        ):
            best_q_tot = q_tot
            best_q_bin = q_bin
            best_gamma = gamma_rf
            best_steps = steps
            best_n_sub = n_sub
            best_iplus = ip_try
            best_iminus = im_try

    if best_gamma <= 0.0 or best_iplus is None or best_iminus is None:
        return None

    ps = best_iplus + best_iminus
    return (
        best_gamma,
        best_q_bin,
        ps,
        best_iplus,
        best_iminus,
        best_steps,
        "best_gamma_n_steps",
        best_n_sub,
    )


def optimize_binwise_incremental(config, polarization):
    f = config.f
    iplus, iminus = equilibrium_spin1_intensities(polarization, f)
    iplus_unburned = iplus.copy()
    iminus_unburned = iminus.copy()

    target_bins = initially_negative_q_bins(iplus_unburned, iminus_unburned)
    if not target_bins:
        raise RuntimeError("No bins with initially negative spectral Q to optimize")

    initial_q = q_polarization(iplus, iminus)
    initial_iplus_area = np.sum(iplus)
    initial_iminus_area = np.sum(iminus)
    initial_area = total_signal_area(iplus, iminus)
    current_iplus_area = initial_iplus_area
    current_iminus_area = initial_iminus_area
    current_area = initial_area
    current_q = initial_q
    trace = [
        {
            "step": 0,
            "q": initial_q,
            "iplus_area": initial_iplus_area,
            "iminus_area": initial_iminus_area,
            "area": initial_area,
            "action": None,
            "n_target_bins": len(target_bins),
        }
    ]
    step = 0

    for bin_idx in target_bins:
        q_bin_before = q_at_bin(iplus, iminus, bin_idx)
        best = find_best_burn_for_bin(
            iplus,
            iminus,
            bin_idx,
            polarization,
            config,
        )
        if best is None:
            continue

        best_rf_amp, best_q_bin, _ps, iplus, iminus, steps_done, stop_reason, n_sub = best
        iplus_area_before = current_iplus_area
        iminus_area_before = current_iminus_area
        current_q_bin = best_q_bin
        current_q = q_polarization(iplus, iminus)
        current_iplus_area = np.sum(iplus)
        current_iminus_area = np.sum(iminus)
        current_area = total_signal_area(iplus, iminus)
        q_bin_gain = current_q_bin - q_bin_before
        step += 1
        trace.append(
            {
                "step": step,
                "bin_idx": bin_idx,
                "f": f[bin_idx],
                "rf_amp": best_rf_amp,
                "n_steps": steps_done,
                "n_steps_budget": config.max_steps,
                "n_sub": n_sub,
                "stop_reason": stop_reason,
                "reward": q_bin_gain,
                "q_bin_reward": q_bin_gain,
                "q_bin": current_q_bin,
                "q_bin_gain": q_bin_gain,
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
        "target_bins": target_bins,
        "n_burns": step,
        "trace": trace,
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus": iplus,
        "iminus": iminus,
        "f": f.copy(),
    }


def plot_unburned_signal(
    f,
    iplus,
    iminus,
    polarization,
    output_path,
    *,
    negative_q_bins=None,
):
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
    axes[1].axhline(0.0, color="gray", alpha=0.5, linewidth=0.8)
    if negative_q_bins:
        axes[1].fill_between(
            f,
            q_profile,
            0.0,
            where=q_profile < 0.0,
            color="tab:orange",
            alpha=0.25,
            label=rf"$Q<0$ bins ({len(negative_q_bins)})",
        )
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


def plot_greedy_burns(result, output_path):
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


def plot_optimal_power_profile(result, output_path):
    """Plot per-bin optimal γ_RF (and n_steps) vs frequency."""
    burns = [row for row in result["trace"][1:] if row.get("rf_amp", 0.0) > 0.0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    if burns:
        rs = np.array([row["f"] for row in burns])
        gammas = np.array([row["rf_amp"] for row in burns])
        steps = np.array([row["n_steps"] for row in burns])
        axes[0].stem(rs, gammas, linefmt="C2-", markerfmt="C2o", basefmt="k-")
        axes[1].stem(rs, steps, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    axes[0].set_ylabel(r"optimal $\gamma_{\mathrm{rf}}$")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"Optimal ssRF power profile  P={result['polarization']:.3f}  "
        f"n_burns={result['n_burns']}  "
        f"Q: {result['initial_q'] * 100:.3f}% -> {result['final_q'] * 100:.3f}%"
    )
    axes[1].set_ylabel("optimal n_steps")
    axes[1].set_xlabel("frequency R")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    polarization = 0.45
    config = BurnConfig()
    out_dir = REPO_ROOT / "results" / "current" / "binwise_incremental_realtime"
    out_dir.mkdir(parents=True, exist_ok=True)

    f = config.f
    iplus0, iminus0 = equilibrium_spin1_intensities(polarization, f)
    negative_q_bins = initially_negative_q_bins(iplus0, iminus0)
    plot_unburned_signal(
        f,
        iplus0,
        iminus0,
        polarization,
        out_dir / f"unburned_P{polarization:.2f}.png",
        negative_q_bins=negative_q_bins,
    )

    gamma_vals = config.gamma_rf_values
    print(
        f"Physical Voigt bin-wise Q optimization at P={polarization * 100:.2f}% "
        f"({len(negative_q_bins)} initially Q<0 burn-window bins, "
        f"gamma_rf∈[{gamma_vals[0]:.3g}, {gamma_vals[-1]:.3g}] "
        f"×{len(gamma_vals)}, max_steps={config.max_steps}, "
        f"Gauss FWHM={config.gaussian_fwhm_R:.3f}, "
        f"Lorentz FWHM={config.lorentzian_fwhm_R:.3f}, "
        f"diffusion_scale={config.diffusion_scale:.3f}, "
        f"dt={config.dt:.6g})..."
    )
    result = optimize_binwise_incremental(config, polarization)
    plot_greedy_burns(
        result, out_dir / f"incremental_policy_P{polarization:.2f}.png"
    )
    plot_optimal_power_profile(
        result, out_dir / f"optimal_power_profile_P{polarization:.2f}.png"
    )

    print(
        f"Applied {result['n_burns']} Voigt burns "
        f"(of {len(result['target_bins'])} initially Q<0 bins):"
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
            f"gamma_rf={row['rf_amp']:.4g}, n_steps={row['n_steps']}/{row['n_steps_budget']} "
            f"(n_sub={row['n_sub']}, stop={row['stop_reason']}), "
            f"Q_bin_gain={row['reward']:.5e}, "
            f"I+ reduction={row['iplus_reduction']:.5e}, "
            f"I- reduction={row['iminus_reduction']:.5e}, "
            f"Q_bin={row['q_bin']:.5e}, Q_total={row['q'] * 100:.5f}%"
        )
    print("  per-bin optimal power profile:")
    for row in result["trace"][1:]:
        if row.get("rf_amp", 0.0) <= 0.0:
            continue
        print(
            f"    bin={row['bin_idx']:4d}  R={row['f']:+.4f}  "
            f"gamma_rf={row['rf_amp']:7.3f}  "
            f"n_steps={row['n_steps']:4d}/{row['n_steps_budget']}"
        )
    print(f"  final: Q={result['final_q'] * 100:.5f}%")
    print(f"  total Q gain: {(result['final_q'] - result['initial_q']) * 100:.5f}%")
    print(f"  total P change: {(result['final_area'] - result['initial_area']) * 100:.5f}%")
    print(f"  total I+ change: {result['final_iplus_area'] - result['initial_iplus_area']:.8f}")
    print(f"  total I- change: {result['final_iminus_area'] - result['initial_iminus_area']:.8f}")
    print(f"  total area gain: {result['final_area'] - result['initial_area']:.8f}")
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
