"""
SSRF all-bins burn: fixed DT, ≤N_STEPS; bisect min gamma_rf to null local Q(R).

Burn-down uses RF + sameθ + neighbor diffusion (DNP/T1 off). Neighbor spillover
is discarded on commit (burn + RF-mirror bins only). Optional AFP sweep (physics.afp)
runs after all burns, matching Data_Creation/ssRFData_mc intensity rescaling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.afp import AFP
from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime import Spin1Params
from physics.ssrf_realtime.rate_equations_realtime import build_model_for_intensities

P = 0.40
NUM_BINS = 500
DT = 0.05
N_STEPS = 500
Q_ABS_TOL = 1e-10
Q_FRAC_TOL = 1e-7
GAMMA_HI_INIT = 5.0
GAMMA_MAX = GAMMA_HI_INIT
N_BISECT = 12
MAX_GDT = 0.05
MAX_NSUB = 20
EVOLVE_WINDOW_RADIUS = 8

# Post-burn AFP (physics.afp.AFP); None bin_range → full grid minus center exclusion.
AFP_ENABLED = True
AFP_EFFICIENCY = 1.0
AFP_CENTER_EXCLUSION_BINS = 5
AFP_BIN_RANGE: tuple[int, int] | None = None

OUT_DIR = Path(__file__).resolve().parent
D_SAME_PLUS0 = 0.25
D_SAME_0MINUS = 0.15
D_SPEC_PLUS0 = 1.5
D_SPEC_0MINUS = 0.8


def max_euler_gamma(dt: float) -> float:
    return float(MAX_NSUB) * float(MAX_GDT) / max(float(dt), 1e-30)


def mirror_bin_idx(n_bins: int, bin_idx: int) -> int:
    return int(n_bins) - 1 - int(bin_idx)


def q_at_r_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    return float(iplus[int(bin_idx)] - iminus[int(bin_idx)])


def lineshape_area(iplus: np.ndarray, iminus: np.ndarray, f: np.ndarray) -> float:
    return float(np.trapezoid(np.asarray(iplus) + np.asarray(iminus), f))


def q_total(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def p_total(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus + iminus))


def q_target_tol(q_before: float) -> float:
    return max(float(Q_ABS_TOL), float(Q_FRAC_TOL) * abs(float(q_before)))


def _crossed_zero(before: float, after: float) -> bool:
    if before > 0.0:
        return after <= 0.0
    if before < 0.0:
        return after >= 0.0
    return after != 0.0


def physical_ip_im_at_bin(model, bin_idx: int) -> tuple[float, float]:
    """Physical I±(R) at one bin from packet state."""
    bin_idx = int(bin_idx)
    m = mirror_bin_idx(len(model.n), bin_idx)
    scale = float(model.display_cal) / float(model.dR)
    n = model.n
    return (
        scale * float(n[bin_idx, 0] - n[bin_idx, 1]),
        scale * float(n[m, 1] - n[m, 2]),
    )


def q_from_packet(model, burn_idx: int) -> float:
    ip, im = physical_ip_im_at_bin(model, burn_idx)
    return ip - im


def enrich_trial_totals(
    trial: dict,
    iplus_before: np.ndarray,
    iminus_before: np.ndarray,
    f: np.ndarray,
) -> dict:
    if "area_loss" in trial:
        return trial
    iplus_cur, iminus_cur = trial["iplus"], trial["iminus"]
    qt0, qt1 = q_total(iplus_before, iminus_before), q_total(iplus_cur, iminus_cur)
    a0, a1 = lineshape_area(iplus_before, iminus_before, f), lineshape_area(iplus_cur, iminus_cur, f)
    out = dict(trial)
    out.update(
        q_total_before=qt0,
        q_total_after=qt1,
        q_total_gain=qt1 - qt0,
        area_before=a0,
        area_after=a1,
        area_loss=a0 - a1,
    )
    return out


def apply_afp_sweep(
    iplus: np.ndarray,
    iminus: np.ndarray,
    *,
    bin_range: tuple[int, int] | None = None,
    subset_indices: list[int] | None = None,
    efficiency: float = AFP_EFFICIENCY,
    center_margin: int = AFP_CENTER_EXCLUSION_BINS,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    AFP on I± via physics.afp.AFP.

    AFP.intensities_to_populations re-normalizes Σρ, so raw to_intensities()×Σ(I++I−)
    globally rescales the spectrum. Match ssRF commit_burn_bins_only: recover intensity
    units from a pre-AFP round-trip, then write only sweep bins and their mirrors.

    Returns (Iplus, Iminus, subset_used).
    """
    iplus = np.asarray(iplus, dtype=float)
    iminus = np.asarray(iminus, dtype=float)
    n = len(iplus)
    total_area = float(np.sum(iplus + iminus))
    if total_area <= 0.0:
        return iplus.copy(), iminus.copy(), []

    if subset_indices is not None:
        subset = [int(i) for i in subset_indices]
    elif bin_range is not None:
        start, stop = int(bin_range[0]), int(bin_range[1])
        subset = list(range(start, stop))
    else:
        subset = list(range(n))

    if center_margin > 0:
        c = n // 2
        forbidden = set(range(max(0, c - center_margin), min(n, c + center_margin + 1)))
        subset = [i for i in subset if i not in forbidden]

    afp = AFP.from_intensities(iplus, iminus)
    ip_rt, im_rt = afp.to_intensities()
    rt_sum = float(np.sum(np.asarray(ip_rt, dtype=float) + np.asarray(im_rt, dtype=float)))
    scale = total_area / rt_sum if abs(rt_sum) > 1e-30 else total_area

    if subset:
        afp.perform_afp(subset_indices=subset, efficiency=float(efficiency), show_progress=False)
    ip_new, im_new = afp.to_intensities()
    ip_new = np.asarray(ip_new, dtype=float) * scale
    im_new = np.asarray(im_new, dtype=float) * scale

    # Commit only AFP-touched packets (and mirrors), like commit_burn_bins_only.
    out_ip = iplus.copy()
    out_im = iminus.copy()
    touched: set[int] = set()
    for i in subset:
        touched.add(int(i))
        touched.add(mirror_bin_idx(n, int(i)))
    for k in touched:
        out_ip[k] = float(ip_new[k])
        out_im[k] = float(im_new[k])
    return out_ip, out_im, subset


def _burn_params(n_bins: int, r_min: float, r_max: float, polarization: float) -> Spin1Params:
    return Spin1Params(
        n_bins=n_bins,
        r_min=r_min,
        r_max=r_max,
        p0=polarization,
        initial_polarization=polarization,
        gamma_rf=0.0,
        d_same_plus0=D_SAME_PLUS0,
        d_same_0minus=D_SAME_0MINUS,
        d_spec_plus0=D_SPEC_PLUS0,
        d_spec_0minus=D_SPEC_0MINUS,
        dnp_enabled=False,
        t1_rate=0.0,
        dt=DT,
        steps=1,
    )


def commit_burn_bins_only(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_sim: np.ndarray,
    iminus_sim: np.ndarray,
    burn_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    burn_idx = int(burn_idx)
    mirror_idx = mirror_bin_idx(len(iplus), burn_idx)
    iplus_out = np.asarray(iplus, dtype=float).copy()
    iminus_out = np.asarray(iminus, dtype=float).copy()
    for idx in (burn_idx, mirror_idx):
        iplus_out[idx] = float(iplus_sim[idx])
        iminus_out[idx] = float(iminus_sim[idx])
    return iplus_out, iminus_out


def apply_rf_burn(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    gamma_rf: float,
    *,
    f: np.ndarray,
    polarization: float,
    dt: float = DT,
    n_steps: int = N_STEPS,
    q_tol: float | None = None,
    model=None,
    n0: np.ndarray | None = None,
    light: bool = False,
) -> dict | None:
    if gamma_rf <= 0.0 or n_steps <= 0:
        return None

    burn_idx = int(burn_idx)
    q_before = q_at_r_bin(iplus, iminus, burn_idx)
    tol = float(q_tol) if q_tol is not None else q_target_tol(q_before)

    if model is None or n0 is None:
        params = _burn_params(len(f), float(f[0]), float(f[-1]), polarization)
        model = build_model_for_intensities(
            iplus, iminus, params=params, rf_burn_R=float(f[burn_idx]), initial_polarization=polarization
        )
        n0 = model.n.copy()
    else:
        model.params.rf_burn_R = float(f[burn_idx])

    model.params.gamma_rf = float(gamma_rf)
    model.n = n0.copy()
    model.t = 0.0
    model.set_evolution_window(EVOLVE_WINDOW_RADIUS)

    mirror_idx = mirror_bin_idx(len(iplus), burn_idx)
    ip_prev = {burn_idx: float(iplus[burn_idx]), mirror_idx: float(iplus[mirror_idx])}
    im_prev = {burn_idx: float(iminus[burn_idx]), mirror_idx: float(iminus[mirror_idx])}

    g = float(gamma_rf)
    dt_f = float(dt)
    n_sub = min(max(1, int(np.ceil(abs(g) * dt_f / MAX_GDT))), int(MAX_NSUB)) if g else 1
    dt_sub = dt_f / float(n_sub)
    steps_done = 0

    for _ in range(int(n_steps)):
        if abs(q_from_packet(model, burn_idx)) <= tol:
            break
        state_before = model.n.copy()
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)

        ip_new, im_new = {}, {}
        sign_ok = True
        for idx in (burn_idx, mirror_idx):
            ip_new[idx], im_new[idx] = physical_ip_im_at_bin(model, idx)
            for b, a in (
                (ip_prev[idx], ip_new[idx]),
                (im_prev[idx], im_new[idx]),
                (ip_prev[idx] + im_prev[idx], ip_new[idx] + im_new[idx]),
            ):
                if _crossed_zero(b, a):
                    sign_ok = False
                    break
            if not sign_ok:
                break
        if not sign_ok:
            model.n = state_before
            break
        ip_prev, im_prev = ip_new, im_new
        steps_done += 1

    if steps_done == 0:
        return None

    iplus_sim, iminus_sim, _ = model.physical_intensities()
    iplus_cur, iminus_cur = commit_burn_bins_only(
        iplus, iminus, np.asarray(iplus_sim, dtype=float), np.asarray(iminus_sim, dtype=float), burn_idx
    )
    q_after = float(iplus_cur[burn_idx] - iminus_cur[burn_idx])
    out = {
        "burn_idx": burn_idx,
        "gamma_rf": g,
        "n_steps": int(steps_done),
        "t_burn": float(steps_done) * dt_f,
        "q_before": q_before,
        "q_after": q_after,
        "q_gain": q_after - q_before,
        "iplus": iplus_cur,
        "iminus": iminus_cur,
    }
    return out if light else enrich_trial_totals(out, iplus, iminus, f)


def find_gamma_to_null_q_r(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    *,
    f: np.ndarray,
    polarization: float,
    dt: float = DT,
    n_steps: int = N_STEPS,
    gamma_hi: float = GAMMA_HI_INIT,
    n_bisect: int = N_BISECT,
    gamma_guess: float | None = None,
) -> dict | None:
    q_before = q_at_r_bin(iplus, iminus, burn_idx)
    if q_before >= 0.0:
        return None

    tol = q_target_tol(q_before)
    params = _burn_params(len(f), float(f[0]), float(f[-1]), polarization)
    model = build_model_for_intensities(
        iplus, iminus, params=params, rf_burn_R=float(f[burn_idx]), initial_polarization=polarization
    )
    n0 = model.n.copy()
    model.set_evolution_window(EVOLVE_WINDOW_RADIUS)

    def trial_at(gamma: float) -> dict | None:
        return apply_rf_burn(
            iplus, iminus, burn_idx, gamma,
            f=f, polarization=polarization, dt=dt, n_steps=n_steps,
            q_tol=tol, model=model, n0=n0, light=True,
        )

    def meets(trial: dict | None) -> bool:
        return trial is not None and abs(float(trial["q_after"])) <= tol

    expand_cap = max(float(gamma_hi), 2.0 * max_euler_gamma(dt))
    hi = min(float(gamma_hi), expand_cap)
    hi_trial = None

    if gamma_guess is not None and gamma_guess > 0.0:
        warm = trial_at(min(float(gamma_guess), expand_cap))
        if meets(warm):
            hi_trial, hi = warm, float(warm["gamma_rf"])
        else:
            hi = min(max(hi, float(gamma_guess)), expand_cap)

    if hi_trial is None:
        hi_trial = trial_at(hi)
        last_ok = hi_trial
        for _ in range(10):
            if hi_trial is None:
                hi *= 0.5
                if hi < 1e-12:
                    return None
                hi_trial = trial_at(hi)
                if hi_trial is not None:
                    last_ok = hi_trial
                continue
            if meets(hi_trial) or hi >= expand_cap * (1.0 - 1e-12):
                break
            last_ok = hi_trial
            hi = min(hi * 2.0, expand_cap)
            nxt = trial_at(hi)
            if nxt is None:
                hi_trial = last_ok
                break
            hi_trial = nxt

    if hi_trial is None:
        return None
    if not meets(hi_trial):
        return enrich_trial_totals(hi_trial, iplus, iminus, f)

    lo_ok, hi_ok, best = 0.0, float(hi_trial["gamma_rf"]), hi_trial
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo_ok + hi_ok)
        trial = trial_at(mid)
        if meets(trial):
            hi_ok, best = mid, trial
        else:
            lo_ok = mid
    return enrich_trial_totals(best, iplus, iminus, f)


def _skip_trace(burn_idx: int, mirror_idx: int, f: np.ndarray, q_before: float, iplus, iminus) -> dict:
    return {
        "bin_idx": burn_idx,
        "mirror_idx": mirror_idx,
        "f": float(f[burn_idx]),
        "gamma_rf": 0.0,
        "n_steps": 0,
        "t_burn": 0.0,
        "q_before": q_before,
        "q_after": q_before,
        "q_gain": 0.0,
        "q_total": q_total(iplus, iminus),
        "p_total": p_total(iplus, iminus),
        "area_loss": 0.0,
        "skipped": True,
    }


def optimize_all_bins(
    polarization: float = P,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    n_steps: int = N_STEPS,
    afp: bool = AFP_ENABLED,
    afp_efficiency: float = AFP_EFFICIENCY,
    afp_bin_range: tuple[int, int] | None = AFP_BIN_RANGE,
    afp_center_margin: int = AFP_CENTER_EXCLUSION_BINS,
) -> dict:
    f = np.linspace(-3.0, 3.0, num_bins)
    _, iplus0, iminus0 = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    iplus_unburned, iminus_unburned = iplus.copy(), iminus.copy()
    q0 = iplus_unburned - iminus_unburned
    candidates = [i for i in range(num_bins) if float(q0[i]) < 0.0]

    initial_q, initial_p = q_total(iplus, iminus), p_total(iplus, iminus)
    area0 = lineshape_area(iplus, iminus, f)
    gamma_profile = np.zeros(num_bins)
    steps_profile = np.zeros(num_bins, dtype=int)
    trace: list[dict] = []
    applied = skipped = 0
    gamma_guess = None

    pbar = tqdm.tqdm(candidates, desc="gamma-opt bins", unit="bin")
    for burn_idx in pbar:
        mirror_idx = mirror_bin_idx(num_bins, burn_idx)
        q_before = q_at_r_bin(iplus, iminus, burn_idx)
        if q_before >= 0.0:
            skipped += 1
            trace.append(_skip_trace(burn_idx, mirror_idx, f, q_before, iplus, iminus))
            continue

        trial = find_gamma_to_null_q_r(
            iplus, iminus, burn_idx, f=f, polarization=polarization,
            dt=dt, n_steps=n_steps, gamma_guess=gamma_guess,
        )
        if trial is None:
            skipped += 1
            trace.append(_skip_trace(burn_idx, mirror_idx, f, q_before, iplus, iminus))
            continue

        iplus, iminus = trial["iplus"], trial["iminus"]
        gamma_profile[burn_idx] = float(trial["gamma_rf"])
        steps_profile[burn_idx] = int(trial["n_steps"])
        gamma_guess = float(trial["gamma_rf"])
        applied += 1
        pbar.set_postfix(
            idx=burn_idx, R=f"{f[burn_idx]:+.3f}", gamma=f"{trial['gamma_rf']:.4g}",
            steps=f"{trial['n_steps']}/{n_steps}", Q=f"{trial['q_after']:.2e}",
            applied=applied, refresh=False,
        )
        trace.append({
            "bin_idx": burn_idx, "mirror_idx": mirror_idx, "f": float(f[burn_idx]),
            "gamma_rf": float(trial["gamma_rf"]), "n_steps": int(trial["n_steps"]),
            "t_burn": float(trial["t_burn"]), "q_before": float(trial["q_before"]),
            "q_after": float(trial["q_after"]), "q_gain": float(trial["q_gain"]),
            "q_total_gain": float(trial["q_total_gain"]), "q_total": float(trial["q_total_after"]),
            "p_total": p_total(iplus, iminus), "area_loss": float(trial["area_loss"]),
            "skipped": False,
        })

    iplus_pre_afp, iminus_pre_afp = iplus.copy(), iminus.copy()
    afp_subset: list[int] = []
    if afp:
        subset = None
        if afp_bin_range is None:
            burned = [i for i in range(num_bins) if float(gamma_profile[i]) > 0.0]
            if burned:
                subset = sorted(
                    {int(i) for i in burned}
                    | {mirror_bin_idx(num_bins, int(i)) for i in burned}
                )
        iplus, iminus, afp_subset = apply_afp_sweep(
            iplus, iminus,
            bin_range=afp_bin_range,
            subset_indices=subset,
            efficiency=afp_efficiency,
            center_margin=afp_center_margin,
        )

    return {
        "polarization": polarization,
        "dt": dt,
        "n_steps": n_steps,
        "f": f,
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus_pre_afp": iplus_pre_afp,
        "iminus_pre_afp": iminus_pre_afp,
        "iplus": iplus,
        "iminus": iminus,
        "gamma_profile": gamma_profile,
        "steps_profile": steps_profile,
        "q0": q0,
        "area0": area0,
        "area_final": lineshape_area(iplus, iminus, f),
        "area_loss_total": area0 - lineshape_area(iplus, iminus, f),
        "initial_q": initial_q,
        "final_q": q_total(iplus, iminus),
        "initial_p": initial_p,
        "final_p": p_total(iplus, iminus),
        "q_pre_afp": q_total(iplus_pre_afp, iminus_pre_afp),
        "p_pre_afp": p_total(iplus_pre_afp, iminus_pre_afp),
        "n_applied": applied,
        "n_skipped": skipped,
        "n_candidates": len(candidates),
        "trace": trace,
        "afp_enabled": bool(afp),
        "afp_efficiency": float(afp_efficiency),
        "afp_subset": afp_subset,
        "afp_center_margin": int(afp_center_margin),
    }


def save_gamma_profile(result: dict, output_path: Path) -> None:
    q_final = result["iplus"] - result["iminus"]
    data = np.column_stack([
        result["f"], result["gamma_profile"], result["steps_profile"], result["q0"], q_final
    ])
    np.savetxt(output_path, data, delimiter=",", header="R,gamma_rf,n_steps,Q_unburned,Q_final", comments="")


def _afp_spans(ax, result: dict) -> None:
    subset = result.get("afp_subset") or []
    if not subset:
        return
    f = result["f"]
    lo, hi = min(subset), max(subset)
    ax.axvspan(f[lo], f[hi], color="gold", alpha=0.12, label="AFP sweep")


def plot_result(result: dict, output_path: Path) -> None:
    f = result["f"]
    iplus, iminus = result["iplus"], result["iminus"]
    iplus0, iminus0 = result["iplus_unburned"], result["iminus_unburned"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax_data, color, label in (
        (iplus0 + iminus0, "black", r"$P_s$"),
        (iplus0, "tab:red", r"$I_+$"),
        (iminus0, "tab:blue", r"$I_-$"),
    ):
        axes[0].step(f, ax_data, color=color, linestyle="--", alpha=0.55, linewidth=1.0, label=f"{label} (unburned)")
    axes[0].step(f, iplus + iminus, color="black", label=r"$P_s$")
    axes[0].step(f, iplus, color="tab:red", label=r"$I_+$")
    axes[0].step(f, iminus, color="tab:blue", label=r"$I_-$")
    if result.get("afp_enabled") and "iplus_pre_afp" in result:
        axes[0].step(
            f, result["iplus_pre_afp"] + result["iminus_pre_afp"],
            color="gray", alpha=0.5, linewidth=0.9, label=r"$P_s$ (pre-AFP)",
        )
    _afp_spans(axes[0], result)
    for row in result["trace"]:
        if not row["skipped"]:
            axes[0].axvline(row["f"], color="green", alpha=0.15, linestyle=":")
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].step(f, iplus0 - iminus0, color="tab:purple", linestyle="--", alpha=0.55, label=r"$Q$ (unburned)")
    axes[1].step(f, iplus - iminus, color="tab:purple", label=r"$Q = I_+ - I_-$")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel(r"$R$")
    axes[1].set_ylabel("Q profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    afp_tag = " + AFP" if result.get("afp_enabled") else ""
    delta_q = result["final_q"] - result["initial_q"]
    fig.suptitle(
        f"SSRF gamma-opt{afp_tag}  P={result['polarization']*100:.0f}%  "
        f"dt={result['dt']}  n_steps≤{result['n_steps']}  "
        f"Q: {result['initial_q']*100:.2f}% -> {result['final_q']*100:.2f}% "
        f"({delta_q*100:+.2f}%)  applied={result['n_applied']}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_gamma_profile(result: dict, output_path: Path) -> None:
    f, gamma, q0 = result["f"], result["gamma_profile"], result["q0"]
    q_final = result["iplus"] - result["iminus"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(f, q0, label=r"$Q$ (unburned)")
    axes[0].plot(f, q_final, label=r"$Q$ (after)")
    if result.get("afp_enabled") and "iplus_pre_afp" in result:
        axes[0].plot(
            f, result["iplus_pre_afp"] - result["iminus_pre_afp"],
            color="gray", alpha=0.7, label=r"$Q$ (pre-AFP)",
        )
    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[0].set_ylabel(r"$Q(R)$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f, gamma, color="tab:orange", label=r"$\gamma_{\mathrm{RF}}(R)$")
    axes[1].fill_between(f, gamma, alpha=0.25, color="tab:orange")
    _afp_spans(axes[1], result)
    axes[1].set_ylabel(r"$\gamma_{\mathrm{RF}}$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    applied = [r for r in result["trace"] if not r["skipped"]]
    if applied:
        fv = [r["f"] for r in applied]
        axes[2].stem(fv, [r["q_before"] for r in applied], linefmt="C0-", markerfmt="C0o", basefmt=" ", label=r"$Q_R$ before")
        axes[2].stem(fv, [r["q_after"] for r in applied], linefmt="C1-", markerfmt="C1o", basefmt=" ", label=r"$Q_R$ after")
    axes[2].axhline(0.0, color="black", linestyle="--")
    axes[2].set_xlabel(r"$R$")
    axes[2].set_ylabel(r"$Q$ at burn bin")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    afp_tag = ", AFP on" if result.get("afp_enabled") else ""
    fig.suptitle(
        rf"Optimized $\gamma_{{\mathrm{{RF}}}}(R)$ at fixed $dt={result['dt']}$, "
        rf"$\leq{result['n_steps']}$ steps{afp_tag}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_q_gains(result: dict, output_path: Path) -> None:
    applied = [r for r in result["trace"] if not r["skipped"]]
    if not applied:
        return
    fv = [r["f"] for r in applied]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].stem(fv, [r["q_gain"] for r in applied], basefmt=" ")
    axes[0].set_ylabel(r"$\Delta Q(R)$")
    axes[0].set_title(f"Per-bin local-Q change (dt={result['dt']}, ≤{result['n_steps']} steps)")
    axes[0].grid(True, alpha=0.3)
    axes[1].stem(fv, [r.get("q_total_gain", 0.0) for r in applied], basefmt=" ", linefmt="C1-", markerfmt="C1o")
    axes[1].set_ylabel(r"$\Delta Q_{\mathrm{total}}$")
    axes[1].grid(True, alpha=0.3)
    axes[2].stem(fv, [r["gamma_rf"] for r in applied], basefmt=" ", linefmt="C2-", markerfmt="C2o")
    axes[2].set_ylabel(r"$\gamma_{\mathrm{RF}}$")
    axes[2].grid(True, alpha=0.3)
    axes[3].stem(fv, [r["n_steps"] for r in applied], basefmt=" ", linefmt="C3-", markerfmt="C3o")
    axes[3].axhline(result["n_steps"], color="gray", linestyle=":", label=r"$N_{\mathrm{steps}}$ max")
    axes[3].set_xlabel(r"burn $R$")
    axes[3].set_ylabel(r"steps used")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    result = optimize_all_bins()
    stem = "rate_eqs_test_ssrf_all_bins_gamma_opt"
    plot_result(result, OUT_DIR / f"{stem}_lineshape.png")
    plot_q_gains(result, OUT_DIR / f"{stem}_gains.png")
    plot_gamma_profile(result, OUT_DIR / f"{stem}_burn_profile.png")
    save_gamma_profile(result, OUT_DIR / f"{stem}_gamma_profile.csv")

    q_final = result["iplus"] - result["iminus"]
    burned = result["gamma_profile"] > 0.0
    if np.any(burned):
        max_abs_q = float(np.max(np.abs(q_final[burned])))
        mean_abs_q = float(np.mean(np.abs(q_final[burned])))
        steps_used = result["steps_profile"][burned]
        mean_steps, max_steps = float(np.mean(steps_used)), int(np.max(steps_used))
        g_min = float(np.min(result["gamma_profile"][burned]))
    else:
        max_abs_q = mean_abs_q = mean_steps = float("nan")
        max_steps, g_min = 0, 0.0

    print()
    print(f"P0={result['polarization']}  dt={result['dt']}  n_steps≤{result['n_steps']}")
    print(
        f"dynamics: sameθ d+0={D_SAME_PLUS0}, d0-={D_SAME_0MINUS}; "
        f"neighbors d+0={D_SPEC_PLUS0}, d0-={D_SPEC_0MINUS}; commit burn/mirror; DNP off"
    )
    if result["afp_enabled"]:
        print(
            f"AFP: efficiency={result['afp_efficiency']}  "
            f"bins={len(result['afp_subset'])}  center_excl=±{result['afp_center_margin']}  "
            f"Q pre→post AFP: {result['q_pre_afp']:.6f} -> {result['final_q']:.6f}"
        )
    print(f"RF bins applied: {result['n_applied']}/{result['n_candidates']}  skipped={result['n_skipped']}")
    print(f"P total: {result['initial_p']:.8f} -> {result['final_p']:.8f}")
    print(f"Q total: {result['initial_q']:.8f} -> {result['final_q']:.8f}")
    print(f"Q gain:  {result['final_q'] - result['initial_q']:+.8f}")
    print(f"Area loss: {result['area_loss_total']:.8f}")
    print(f"|Q(R)| on burned bins: mean={mean_abs_q:.3e}  max={max_abs_q:.3e}")
    print(f"steps used: mean={mean_steps:.1f}  max={max_steps}/{result['n_steps']}")
    print(f"gamma_rf(R): min={g_min:.6g}  max={float(np.max(result['gamma_profile'])):.6g}")
    print(f"Saved {stem}_lineshape.png, {stem}_gains.png, {stem}_burn_profile.png, {stem}_gamma_profile.csv")


if __name__ == "__main__":
    main()
