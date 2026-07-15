import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime import Spin1Params
from physics.ssrf_realtime.rate_equations_realtime import (
    build_model_for_intensities,
    burn_preserves_ps_sign,
)
from physics.lineshape.rate_eqs_test.rate_eqs_test_ssrf_all_bins_gamma_opt import (
    AFP_BIN_RANGE,
    AFP_CENTER_EXCLUSION_BINS,
    AFP_EFFICIENCY,
    AFP_ENABLED,
    apply_afp_sweep,
)

P = .40
NUM_BINS = 500
GAMMA_RF = 0.5
DT = 0.005
N_STEPS = 100
OUT_DIR = Path(__file__).resolve().parent


def mirror_bin_idx(n_bins: int, bin_idx: int) -> int:
    return int(n_bins) - 1 - int(bin_idx)


def rf_burn_profile(q: np.ndarray, gamma_max: float) -> np.ndarray:
    """Per-bin RF burn rate from Q: zero for Q>=0, up to gamma_max at deepest Q<0."""
    q = np.asarray(q, dtype=float)
    q_min = float(np.min(q))
    if q_min >= 0.0:
        return np.zeros_like(q)
    scale = np.clip(q / q_min, 0.0, 1.0)
    return float(gamma_max) * scale


def q_at_theta_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    mirror_idx = mirror_bin_idx(len(iplus), bin_idx)
    q_r = float(iplus[bin_idx] - iminus[bin_idx])
    q_mirror = float(iplus[mirror_idx] - iminus[mirror_idx])
    return q_r + q_mirror


def lineshape_area(iplus: np.ndarray, iminus: np.ndarray, f: np.ndarray) -> float:
    return float(np.trapezoid(np.asarray(iplus) + np.asarray(iminus), f))


def q_total(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def p_total(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus + iminus))


def _rf_only_params(n_bins: int, r_min: float, r_max: float, polarization: float) -> Spin1Params:
    return Spin1Params(
        n_bins=n_bins,
        r_min=r_min,
        r_max=r_max,
        p0=polarization,
        initial_polarization=polarization,
        gamma_rf=0.0,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
        dnp_enabled=False,
        t1_rate=0.0,
        dt=DT,
        steps=1,
    )


def try_one_rf_step(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    gamma_rf: float,
    *,
    f: np.ndarray,
    polarization: float,
    area0: float,
    max_area_loss: float,
    dt: float = DT,
) -> dict | None:
    """Apply one RF step at burn_idx. Return metrics if Q_total rises under budget."""
    if gamma_rf <= 0.0 or max_area_loss <= 0.0:
        return None

    area_now = lineshape_area(iplus, iminus, f)
    loss_already = area0 - area_now
    if loss_already >= max_area_loss:
        return None

    q_total_before = q_total(iplus, iminus)
    q_bin_before = q_at_theta_bin(iplus, iminus, burn_idx)

    params = _rf_only_params(len(f), float(f[0]), float(f[-1]), polarization)
    model = build_model_for_intensities(
        iplus,
        iminus,
        params=params,
        rf_burn_R=float(f[burn_idx]),
        initial_polarization=polarization,
    )
    model.params.gamma_rf = float(gamma_rf)
    model.step_once(dt=dt, rf_on=True, dnp_on=False)
    iplus_new, iminus_new, _ = model.physical_intensities()
    iplus_new = np.asarray(iplus_new, dtype=float)
    iminus_new = np.asarray(iminus_new, dtype=float)

    if not burn_preserves_ps_sign(iplus, iminus, iplus_new, iminus_new, burn_idx):
        return None

    area_try = lineshape_area(iplus_new, iminus_new, f)
    area_loss_try = area0 - area_try
    if area_loss_try > max_area_loss:
        return None

    q_total_after = q_total(iplus_new, iminus_new)
    q_total_gain = q_total_after - q_total_before
    if q_total_gain <= 0.0:
        return None

    q_bin_after = q_at_theta_bin(iplus_new, iminus_new, burn_idx)
    return {
        "burn_idx": int(burn_idx),
        "gamma_rf": float(gamma_rf),
        "q_total_before": q_total_before,
        "q_total_after": q_total_after,
        "q_total_gain": q_total_gain,
        "q_bin_before": q_bin_before,
        "q_bin_after": q_bin_after,
        "q_bin_gain": q_bin_after - q_bin_before,
        "area_before": area_now,
        "area_after": area_try,
        "area_loss": area_now - area_try,
        "area_loss_total": float(area_loss_try),
        "iplus": iplus_new.copy(),
        "iminus": iminus_new.copy(),
    }


def optimize_all_bins(
    polarization: float = P,
    num_bins: int = NUM_BINS,
    gamma_max: float = GAMMA_RF,
    n_steps: int = N_STEPS,
    dt: float = DT,
) -> dict:
    """N_STEPS passes over bins; each pass applies one RF step per bin if ΔQ_total > 0."""
    f = np.linspace(-3.0, 3.0, num_bins)
    _, iplus0, iminus0 = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    iplus_unburned = iplus.copy()
    iminus_unburned = iminus.copy()

    initial_q = q_total(iplus, iminus)
    initial_p = p_total(iplus, iminus)
    area0 = lineshape_area(iplus, iminus, f)
    q0 = iplus_unburned - iminus_unburned
    # Fixed burn profile from unburned Q; integral sets the global area budget.
    gamma_profile0 = rf_burn_profile(q0, gamma_max)
    max_area_loss = float(np.trapezoid(gamma_profile0, f))
    candidate_bins = [i for i in range(num_bins) if float(gamma_profile0[i]) > 0.0]

    trace: list[dict] = []
    applied = 0
    steps_per_bin = np.zeros(num_bins, dtype=int)

    for step in range(1, int(n_steps) + 1):
        applied_this_pass = 0
        for burn_idx in candidate_bins:
            trial = try_one_rf_step(
                iplus,
                iminus,
                burn_idx,
                float(gamma_profile0[burn_idx]),
                f=f,
                polarization=polarization,
                area0=area0,
                max_area_loss=max_area_loss,
                dt=dt,
            )
            if trial is None:
                continue

            mirror_idx = mirror_bin_idx(num_bins, burn_idx)
            iplus = trial["iplus"]
            iminus = trial["iminus"]
            applied += 1
            applied_this_pass += 1
            steps_per_bin[burn_idx] += 1
            current_q = float(trial["q_total_after"])
            current_p = p_total(iplus, iminus)
            t_burn = steps_per_bin[burn_idx] * dt
            trace.append(
                {
                    "step": step,
                    "bin_idx": burn_idx,
                    "mirror_idx": mirror_idx,
                    "f": float(f[burn_idx]),
                    "steps": int(steps_per_bin[burn_idx]),
                    "t": t_burn,
                    "gamma_rf": float(trial["gamma_rf"]),
                    "q_bin_before": float(trial["q_bin_before"]),
                    "q_bin_after": float(trial["q_bin_after"]),
                    "q_bin_gain": float(trial["q_bin_gain"]),
                    "q_total_gain": float(trial["q_total_gain"]),
                    "area_before": float(trial["area_before"]),
                    "area_after": float(trial["area_after"]),
                    "area_loss": float(trial["area_loss"]),
                    "area_loss_total": float(trial["area_loss_total"]),
                    "q_total": current_q,
                    "p_total": current_p,
                    "skipped": False,
                }
            )

        if applied_this_pass == 0:
            print(f"pass {step:3d}/{n_steps}: no bin improves Q; stopping")
            break
        print(
            f"pass {step:3d}/{n_steps}: applied {applied_this_pass:4d} bin-steps  "
            f"Q_total={q_total(iplus, iminus):.6e}  "
            f"A_loss={area0 - lineshape_area(iplus, iminus, f):.6e}/"
            f"{max_area_loss:.6e}"
        )

    n_bins_burned = int(np.count_nonzero(steps_per_bin > 0))
    n_skipped = int(num_bins - n_bins_burned)
    iplus_pre_afp, iminus_pre_afp = iplus.copy(), iminus.copy()
    afp_subset: list[int] = []
    if AFP_ENABLED:
        burned = [i for i in range(num_bins) if int(steps_per_bin[i]) > 0]
        subset = None
        if AFP_BIN_RANGE is None and burned:
            subset = sorted(
                {int(i) for i in burned}
                | {mirror_bin_idx(num_bins, int(i)) for i in burned}
            )
        iplus, iminus, afp_subset = apply_afp_sweep(
            iplus,
            iminus,
            bin_range=AFP_BIN_RANGE,
            subset_indices=subset,
            efficiency=AFP_EFFICIENCY,
            center_margin=AFP_CENTER_EXCLUSION_BINS,
        )
    final_q = q_total(iplus, iminus)
    final_p = p_total(iplus, iminus)
    final_area = lineshape_area(iplus, iminus, f)
    return {
        "polarization": polarization,
        "f": f,
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus_pre_afp": iplus_pre_afp,
        "iminus_pre_afp": iminus_pre_afp,
        "iplus": iplus,
        "iminus": iminus,
        "gamma_profile0": gamma_profile0,
        "max_area_loss": max_area_loss,
        "area0": area0,
        "area_final": final_area,
        "area_loss_total": area0 - final_area,
        "initial_q": initial_q,
        "final_q": final_q,
        "initial_p": initial_p,
        "final_p": final_p,
        "q_pre_afp": q_total(iplus_pre_afp, iminus_pre_afp),
        "p_pre_afp": p_total(iplus_pre_afp, iminus_pre_afp),
        "n_applied": applied,
        "n_skipped": n_skipped,
        "n_bins_burned": n_bins_burned,
        "steps_per_bin": steps_per_bin,
        "trace": trace,
        "afp_enabled": bool(AFP_ENABLED),
        "afp_efficiency": float(AFP_EFFICIENCY),
        "afp_subset": afp_subset,
        "afp_center_margin": int(AFP_CENTER_EXCLUSION_BINS),
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
        f"SSRF P={result['polarization']*100:.0f}%  "
        f"Q: {result['initial_q']*100:.2f}% -> {result['final_q']*100:.2f}% ({delta_q*100:+.2f}%)  "
        f"A_loss={result['area_loss_total']:.4g}/{result['max_area_loss']:.4g}  "
        f"applied={result['n_applied']}  skipped={result['n_skipped']}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_q_gains(result: dict, output_path: Path) -> None:
    applied = [row for row in result["trace"] if not row["skipped"]]
    if not applied:
        return

    # Aggregate multi-step burns that landed on the same bin across N_STEPS.
    by_bin: dict[int, dict] = {}
    for row in applied:
        idx = int(row["bin_idx"])
        if idx not in by_bin:
            by_bin[idx] = {
                "f": float(row["f"]),
                "q_bin_gain": 0.0,
                "q_total_gain": 0.0,
                "area_loss": 0.0,
                "steps": 0,
            }
        by_bin[idx]["q_bin_gain"] += float(row["q_bin_gain"])
        by_bin[idx]["q_total_gain"] += float(row.get("q_total_gain", 0.0))
        by_bin[idx]["area_loss"] += float(row["area_loss"])
        by_bin[idx]["steps"] = max(by_bin[idx]["steps"], int(row["steps"]))

    rows = sorted(by_bin.values(), key=lambda r: r["f"])
    f_vals = [r["f"] for r in rows]
    gains = [r["q_total_gain"] for r in rows]
    t_vals = [r["steps"] * DT for r in rows]
    area_vals = [r["area_loss"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].stem(f_vals, gains, basefmt=" ")
    axes[0].set_ylabel(r"$\sum\Delta Q_{\mathrm{total}}$")
    axes[0].set_title("Per-bin total-Q gain (N_STEPS passes × all bins)")
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(f_vals, t_vals, basefmt=" ", linefmt="C1-", markerfmt="C1o")
    axes[1].set_ylabel(r"burn $t$ on bin")
    axes[1].grid(True, alpha=0.3)

    axes[2].stem(f_vals, area_vals, basefmt=" ", linefmt="C2-", markerfmt="C2o")
    axes[2].axhline(
        result["max_area_loss"],
        color="gray",
        linestyle=":",
        label=r"$A_{\max}=\int\gamma\,\mathrm{d}R$",
    )
    axes[2].set_xlabel(r"burn $R$")
    axes[2].set_ylabel(r"$\Delta A$ (bin)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_burn_profile(result: dict, output_path: Path) -> None:
    f = result["f"]
    q0 = result["iplus_unburned"] - result["iminus_unburned"]
    gamma0 = result["gamma_profile0"]
    max_area_loss = result["max_area_loss"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(f, q0, label=r"$Q$ (unburned)")
    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[0].set_ylabel(r"$Q$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f, gamma0, label=r"$\gamma_{\mathrm{RF}}(Q)$")
    axes[1].axhline(GAMMA_RF, color="gray", linestyle=":", label=r"$\gamma_{\max}$")
    axes[1].fill_between(f, gamma0, alpha=0.25, color="tab:blue")
    axes[1].set_xlabel(r"$R$")
    axes[1].set_ylabel(r"$\gamma_{\mathrm{RF}}$")
    axes[1].set_title(
        rf"$A_{{\max}}=\int\gamma\,\mathrm{{d}}R={max_area_loss:.6g}$  "
        rf"(used {result['area_loss_total']:.6g})"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    result = optimize_all_bins()
    lineshape_path = OUT_DIR / "rate_eqs_test_ssrf_all_bins_lineshape.png"
    gains_path = OUT_DIR / "rate_eqs_test_ssrf_all_bins_gains.png"
    profile_path = OUT_DIR / "rate_eqs_test_ssrf_all_bins_burn_profile.png"
    plot_result(result, lineshape_path)
    plot_q_gains(result, gains_path)
    plot_burn_profile(result, profile_path)

    print()
    print(f"P0={result['polarization']}")
    if result.get("afp_enabled"):
        print(
            f"AFP: efficiency={result['afp_efficiency']}  "
            f"bins={len(result.get('afp_subset', []))}  "
            f"Q pre→post: {result['q_pre_afp']:.6f} -> {result['final_q']:.6f}"
        )
    print(
        f"RF steps applied: {result['n_applied']}/{N_STEPS}  "
        f"bins burned: {result['n_bins_burned']}  "
        f"bins unused: {result['n_skipped']}"
    )
    print(f"P total: {result['initial_p']:.8f} -> {result['final_p']:.8f}")
    print(f"Q total: {result['initial_q']:.8f} -> {result['final_q']:.8f}")
    print(f"Q gain:  {result['final_q'] - result['initial_q']:+.8f}")
    print(
        f"Area loss: {result['area_loss_total']:.8f} / "
        f"{result['max_area_loss']:.8f} (budget = ∫ gamma dR)"
    )
    print(f"Saved {lineshape_path.name}, {gains_path.name}, {profile_path.name}")


if __name__ == "__main__":
    main()
