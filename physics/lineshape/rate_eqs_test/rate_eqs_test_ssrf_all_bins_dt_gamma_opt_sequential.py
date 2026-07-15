"""
SSRF all-bins burn: joint per-bin (dt, gamma_rf) sequential search under a hard
step budget.

This is the dt+gamma joint variant of ``rate_eqs_test_ssrf_all_bins_gamma_opt``
(which uses a fixed static ``DT`` and only bisects minimum ``gamma_rf``).

Unlike rate_eqs_test_ssrf_all_bins.py, there is no a-priori RF profile constraint.
For each negative-Q R bin, search a dt grid and bisect the smallest gamma_rf
(≤ GAMMA_MAX) that drives local |Q(R)| below tolerance within at most N_STEPS
steps (ps-sign preserved). Among feasible (dt, gamma) pairs, pick the one that
minimizes a joint cost favoring small gamma_rf and small dt. Gamma is never
allowed above GAMMA_MAX (no upward expansion).

Burn-down simulation matches ``spin1_ssrf_realtime`` / ``physics.ssrf_realtime``
with DNP off: RF plus sameθ recovery (``d_same_*``) and spectral neighbor
diffusion (``d_spec_*``). Neighbor diffusion is used only while choosing
(dt, gamma) and integrating the burn; when the burn is committed to the running
lineshape, only the burn and RF-mirror bins are updated so neighbor spillover
does not accumulate in the final spectrum. T1 and DNP remain disabled.
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

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime import Spin1Params
from physics.ssrf_realtime.rate_equations_realtime import (
    build_model_for_intensities,
    burn_preserves_ps_sign,
)

P = 0.40
NUM_BINS = 500
N_STEPS = 100
# Absolute near-zero target for local Q(R). A relative fraction would force
# the same gamma for every bin (same residual ratio after fixed time).
Q_ABS_TOL = 1e-10
# Hard upper bound on RF amplitude — never expand above this.
GAMMA_MAX = 5.0
GAMMA_HI_INIT = GAMMA_MAX  # alias for callers / SLURM
N_BISECT = 12
# Cap |gamma_rf| * dt_sub for Euler stability when recovery requires large gamma.
MAX_GDT = 0.05
# Per-bin dt search (log grid). Both dt and gamma are minimized jointly.
DT_MIN = 1e-3
DT_MAX = 0.5
N_DT = 16
DT = 0.05  # default mid-range reference; per-bin opt searches the grid
OUT_DIR = Path(__file__).resolve().parent

# Material-scale recovery / neighborhood rates from spin1_ssrf_realtime defaults.
D_SAME_PLUS0 = 0.25
D_SAME_0MINUS = 0.15
D_SPEC_PLUS0 = 1.5
D_SPEC_0MINUS = 0.8


def mirror_bin_idx(n_bins: int, bin_idx: int) -> int:
    return int(n_bins) - 1 - int(bin_idx)


def q_at_r_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    """Local tensor polarization at a single R bin (not theta-paired)."""
    return float(iplus[int(bin_idx)] - iminus[int(bin_idx)])


def lineshape_area(iplus: np.ndarray, iminus: np.ndarray, f: np.ndarray) -> float:
    return float(np.trapezoid(np.asarray(iplus) + np.asarray(iminus), f))


def q_total(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def p_total(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus + iminus))


def q_target_tol(q_before: float) -> float:
    """Absolute |Q| floor; deeper bins may need larger dt within fixed n_steps."""
    del q_before
    return float(Q_ABS_TOL)


def joint_minimize_cost(gamma_rf: float, dt: float) -> float:
    """Equal-weight normalized cost: smaller gamma and smaller dt preferred."""
    return float(gamma_rf) / float(GAMMA_MAX) + float(dt) / float(DT_MAX)


def _burn_params(
    n_bins: int,
    r_min: float,
    r_max: float,
    polarization: float,
    dt: float = DT,
) -> Spin1Params:
    """RF + sameθ recovery + neighbor diffusion; DNP/T1 off."""
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
        dt=float(dt),
        steps=1,
    )


def commit_burn_bins_only(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_sim: np.ndarray,
    iminus_sim: np.ndarray,
    burn_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep the pre-burn lineshape, but write simulated intensities only at the
    burn bin and its RF mirror. Neighbor-diffusion changes elsewhere are dropped.
    """
    burn_idx = int(burn_idx)
    mirror_idx = mirror_bin_idx(len(iplus), burn_idx)
    iplus_out = np.asarray(iplus, dtype=float).copy()
    iminus_out = np.asarray(iminus, dtype=float).copy()
    iplus_sim = np.asarray(iplus_sim, dtype=float)
    iminus_sim = np.asarray(iminus_sim, dtype=float)
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
    """
    Up to ``n_steps`` full-model steps at burn_idx with fixed ``dt``.

    Each step includes RF plus sameθ recovery and neighbor diffusion (no DNP)
    so burn-down feels neighbor refill. Returned ``iplus`` / ``iminus`` commit
    only the burn and RF-mirror bins onto the input lineshape (neighbor
    spillover is not written back). Stops early when |Q(R)| <= q_tol (if given)
    or when the next macro-step would flip ps-sign. Returns None if gamma_rf
    <= 0 or the first step is invalid.
    """
    if gamma_rf <= 0.0 or n_steps <= 0:
        return None

    burn_idx = int(burn_idx)
    q_before = q_at_r_bin(iplus, iminus, burn_idx)
    tol = float(q_tol) if q_tol is not None else q_target_tol(q_before)

    if model is None or n0 is None:
        params = _burn_params(len(f), float(f[0]), float(f[-1]), polarization, dt=dt)
        model = build_model_for_intensities(
            iplus,
            iminus,
            params=params,
            rf_burn_R=float(f[burn_idx]),
            initial_polarization=polarization,
        )
        n0 = model.n.copy()
    else:
        model.params.rf_burn_R = float(f[burn_idx])
        model.params.dt = float(dt)

    model.params.gamma_rf = float(gamma_rf)
    model.n = n0.copy()
    model.t = 0.0

    iplus_sim = np.asarray(iplus, dtype=float).copy()
    iminus_sim = np.asarray(iminus, dtype=float).copy()
    steps_done = 0
    g = float(gamma_rf)
    dt_f = float(dt)
    n_sub = max(1, int(np.ceil(abs(g) * dt_f / MAX_GDT))) if g != 0.0 else 1
    dt_sub = dt_f / float(n_sub)

    for _ in range(int(n_steps)):
        q_now = float(iplus_sim[burn_idx] - iminus_sim[burn_idx])
        if abs(q_now) <= tol:
            break

        state_before = model.n.copy()
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False)
        iplus_new, iminus_new, _ = model.physical_intensities()
        iplus_new = np.asarray(iplus_new, dtype=float)
        iminus_new = np.asarray(iminus_new, dtype=float)

        if not burn_preserves_ps_sign(iplus_sim, iminus_sim, iplus_new, iminus_new, burn_idx):
            model.n = state_before
            break

        iplus_sim = iplus_new
        iminus_sim = iminus_new
        steps_done += 1

    if steps_done == 0:
        return None

    # Neighbor diffusion shaped the burn-down; only keep burn/mirror updates.
    iplus_cur, iminus_cur = commit_burn_bins_only(
        iplus, iminus, iplus_sim, iminus_sim, burn_idx
    )
    q_after = float(iplus_cur[burn_idx] - iminus_cur[burn_idx])
    out = {
        "burn_idx": burn_idx,
        "gamma_rf": g,
        "dt": dt_f,
        "n_steps": int(steps_done),
        "t_burn": float(steps_done) * dt_f,
        "q_before": q_before,
        "q_after": q_after,
        "q_gain": q_after - q_before,
        "iplus": iplus_cur,
        "iminus": iminus_cur,
    }
    if not light:
        q_total_before = q_total(iplus, iminus)
        area_before = lineshape_area(iplus, iminus, f)
        area_after = lineshape_area(iplus_cur, iminus_cur, f)
        q_total_after = q_total(iplus_cur, iminus_cur)
        out.update(
            {
                "q_total_before": q_total_before,
                "q_total_after": q_total_after,
                "q_total_gain": q_total_after - q_total_before,
                "area_before": area_before,
                "area_after": area_after,
                "area_loss": area_before - area_after,
            }
        )
    return out


def find_gamma_to_null_q_r(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    *,
    f: np.ndarray,
    polarization: float,
    dt: float = DT,
    n_steps: int = N_STEPS,
    gamma_hi: float = GAMMA_MAX,
    n_bisect: int = N_BISECT,
    gamma_guess: float | None = None,
) -> dict | None:
    """
    Find the smallest gamma_rf ≤ ``gamma_hi`` (hard-capped at GAMMA_MAX) such
    that |Q(R)| at burn_idx reaches ~0 within ``n_steps`` of ``dt``.

    Never expands gamma above ``min(gamma_hi, GAMMA_MAX)``. If tol is unreachable
    at that cap, returns the capped trial (best effort) when any steps succeed.
    ``gamma_guess`` is only an upper-bound warm-start from a previous bin.
    """
    q_before = q_at_r_bin(iplus, iminus, burn_idx)
    if q_before >= 0.0:
        return None

    gamma_cap = min(float(gamma_hi), float(GAMMA_MAX))
    if gamma_cap <= 0.0:
        return None

    tol = q_target_tol(q_before)
    params = _burn_params(len(f), float(f[0]), float(f[-1]), polarization, dt=dt)
    model = build_model_for_intensities(
        iplus,
        iminus,
        params=params,
        rf_burn_R=float(f[burn_idx]),
        initial_polarization=polarization,
    )
    n0 = model.n.copy()

    def trial_at(gamma: float, *, light: bool = True) -> dict | None:
        return apply_rf_burn(
            iplus,
            iminus,
            burn_idx,
            gamma,
            f=f,
            polarization=polarization,
            dt=dt,
            n_steps=n_steps,
            q_tol=tol,
            model=model,
            n0=n0,
            light=light,
        )

    def meets_tol(trial: dict | None) -> bool:
        return trial is not None and abs(float(trial["q_after"])) <= tol

    # Establish a feasible upper bound within the hard cap (warm-start if valid).
    hi = gamma_cap
    hi_trial: dict | None = None
    if gamma_guess is not None and 0.0 < float(gamma_guess) <= gamma_cap:
        warm = trial_at(float(gamma_guess))
        if meets_tol(warm):
            hi_trial = warm
            hi = float(warm["gamma_rf"])

    if hi_trial is None:
        hi_trial = trial_at(hi)
        # If the cap itself is unsafe (ps-sign), shrink — never expand above cap.
        shrink = 0
        while hi_trial is None and shrink < 12:
            hi *= 0.5
            if hi < 1e-12:
                return None
            hi_trial = trial_at(hi)
            shrink += 1

    if hi_trial is None:
        return None

    if not meets_tol(hi_trial):
        # Tol unreachable at/below the hard gamma cap for this dt.
        return trial_at(float(hi_trial["gamma_rf"]), light=False)

    # Bisect for the smallest gamma that still meets |Q| <= tol within n_steps.
    lo_ok = 0.0
    hi_ok = float(hi_trial["gamma_rf"])
    best_meet = hi_trial
    for _ in range(int(n_bisect)):
        mid = 0.5 * (lo_ok + hi_ok)
        trial = trial_at(mid)
        if meets_tol(trial):
            hi_ok = mid
            best_meet = trial
        else:
            lo_ok = mid
    return trial_at(float(best_meet["gamma_rf"]), light=False)


def find_dt_gamma_to_null_q_r(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    *,
    f: np.ndarray,
    polarization: float,
    n_steps: int = N_STEPS,
    gamma_hi: float = GAMMA_MAX,
    n_bisect: int = N_BISECT,
    dt_min: float = DT_MIN,
    dt_max: float = DT_MAX,
    n_dt: int = N_DT,
    gamma_guess: float | None = None,
    dt_guess: float | None = None,
) -> dict | None:
    """
    Joint per-bin search over (dt, gamma_rf) with ≤ ``n_steps`` and gamma ≤ cap.

    For each dt on a log grid, find the minimal gamma_rf that nulls |Q(R)|.
    Among tolerance-meeting pairs, return the one minimizing
    ``gamma/GAMMA_MAX + dt/DT_MAX`` (both small). If none meet tol, return the
    best-effort pair (lowest |q_after|, then lowest joint cost).
    """
    q_before = q_at_r_bin(iplus, iminus, burn_idx)
    if q_before >= 0.0:
        return None

    tol = q_target_tol(q_before)
    dt_grid = np.geomspace(float(dt_min), float(dt_max), int(n_dt))
    # Prefer trying a warm-start dt first, then larger→smaller so gamma warm-starts drop.
    order = list(dt_grid[::-1])
    if dt_guess is not None and float(dt_guess) > 0.0:
        order = [float(dt_guess)] + [d for d in order if abs(d - float(dt_guess)) > 1e-15]

    successes: list[dict] = []
    fallbacks: list[dict] = []
    g_warm = float(gamma_guess) if gamma_guess is not None and gamma_guess > 0.0 else None

    for dt in order:
        trial = find_gamma_to_null_q_r(
            iplus,
            iminus,
            burn_idx,
            f=f,
            polarization=polarization,
            dt=float(dt),
            n_steps=n_steps,
            gamma_hi=gamma_hi,
            n_bisect=n_bisect,
            gamma_guess=g_warm,
        )
        if trial is None:
            continue
        g_warm = float(trial["gamma_rf"])
        if abs(float(trial["q_after"])) <= tol:
            successes.append(trial)
        else:
            fallbacks.append(trial)

    def pick_joint_min(cands: list[dict]) -> dict:
        return min(
            cands,
            key=lambda t: (
                joint_minimize_cost(float(t["gamma_rf"]), float(t["dt"])),
                float(t["gamma_rf"]),
                float(t["dt"]),
                float(t["t_burn"]),
            ),
        )

    if successes:
        return pick_joint_min(successes)
    if fallbacks:
        return min(
            fallbacks,
            key=lambda t: (
                abs(float(t["q_after"])),
                joint_minimize_cost(float(t["gamma_rf"]), float(t["dt"])),
                float(t["gamma_rf"]),
                float(t["dt"]),
            ),
        )
    return None


def optimize_all_bins(
    polarization: float = P,
    num_bins: int = NUM_BINS,
    n_steps: int = N_STEPS,
    dt_min: float = DT_MIN,
    dt_max: float = DT_MAX,
    n_dt: int = N_DT,
    gamma_hi: float = GAMMA_MAX,
) -> dict:
    """Joint (dt, gamma) RF burn (≤n_steps, γ≤cap) per negative-Q R bin."""
    f = np.linspace(-3.0, 3.0, num_bins)
    _, iplus0, iminus0 = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    iplus_unburned = iplus.copy()
    iminus_unburned = iminus.copy()

    q0 = iplus_unburned - iminus_unburned
    candidate_bins = [i for i in range(num_bins) if float(q0[i]) < 0.0]

    initial_q = q_total(iplus, iminus)
    initial_p = p_total(iplus, iminus)
    area0 = lineshape_area(iplus, iminus, f)

    gamma_profile = np.zeros(num_bins, dtype=float)
    dt_profile = np.zeros(num_bins, dtype=float)
    steps_profile = np.zeros(num_bins, dtype=int)
    trace: list[dict] = []
    applied = 0
    skipped = 0
    gamma_guess: float | None = None
    dt_guess: float | None = None

    pbar = tqdm.tqdm(candidate_bins, desc="dt/gamma-opt bins", unit="bin")
    for burn_idx in pbar:
        mirror_idx = mirror_bin_idx(num_bins, burn_idx)
        q_before = q_at_r_bin(iplus, iminus, burn_idx)
        if q_before >= 0.0:
            skipped += 1
            pbar.set_postfix(idx=burn_idx, status="skip+", applied=applied, refresh=False)
            trace.append(
                {
                    "bin_idx": burn_idx,
                    "mirror_idx": mirror_idx,
                    "f": float(f[burn_idx]),
                    "gamma_rf": 0.0,
                    "dt": 0.0,
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
            )
            continue

        trial = find_dt_gamma_to_null_q_r(
            iplus,
            iminus,
            burn_idx,
            f=f,
            polarization=polarization,
            n_steps=n_steps,
            gamma_hi=gamma_hi,
            dt_min=dt_min,
            dt_max=dt_max,
            n_dt=n_dt,
            gamma_guess=gamma_guess,
            dt_guess=dt_guess,
        )
        if trial is None:
            skipped += 1
            pbar.set_postfix(idx=burn_idx, status="skip", applied=applied, refresh=False)
            trace.append(
                {
                    "bin_idx": burn_idx,
                    "mirror_idx": mirror_idx,
                    "f": float(f[burn_idx]),
                    "gamma_rf": 0.0,
                    "dt": 0.0,
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
            )
            continue

        iplus = trial["iplus"]
        iminus = trial["iminus"]
        gamma_profile[burn_idx] = float(trial["gamma_rf"])
        dt_profile[burn_idx] = float(trial["dt"])
        steps_profile[burn_idx] = int(trial["n_steps"])
        gamma_guess = float(trial["gamma_rf"])
        dt_guess = float(trial["dt"])
        applied += 1
        pbar.set_postfix(
            idx=burn_idx,
            R=f"{f[burn_idx]:+.3f}",
            gamma=f"{trial['gamma_rf']:.4g}",
            dt=f"{trial['dt']:.3g}",
            steps=f"{trial['n_steps']}/{n_steps}",
            Q=f"{trial['q_after']:.2e}",
            applied=applied,
            refresh=False,
        )
        trace.append(
            {
                "bin_idx": burn_idx,
                "mirror_idx": mirror_idx,
                "f": float(f[burn_idx]),
                "gamma_rf": float(trial["gamma_rf"]),
                "dt": float(trial["dt"]),
                "n_steps": int(trial["n_steps"]),
                "t_burn": float(trial["t_burn"]),
                "q_before": float(trial["q_before"]),
                "q_after": float(trial["q_after"]),
                "q_gain": float(trial["q_gain"]),
                "q_total_gain": float(trial["q_total_gain"]),
                "q_total": float(trial["q_total_after"]),
                "p_total": p_total(iplus, iminus),
                "area_loss": float(trial["area_loss"]),
                "skipped": False,
            }
        )

    final_q = q_total(iplus, iminus)
    final_p = p_total(iplus, iminus)
    final_area = lineshape_area(iplus, iminus, f)
    return {
        "polarization": polarization,
        "dt_min": float(dt_min),
        "dt_max": float(dt_max),
        "n_dt": int(n_dt),
        "gamma_max": float(min(gamma_hi, GAMMA_MAX)),
        "n_steps": n_steps,
        "f": f,
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus": iplus,
        "iminus": iminus,
        "gamma_profile": gamma_profile,
        "dt_profile": dt_profile,
        "steps_profile": steps_profile,
        "q0": q0,
        "area0": area0,
        "area_final": final_area,
        "area_loss_total": area0 - final_area,
        "initial_q": initial_q,
        "final_q": final_q,
        "initial_p": initial_p,
        "final_p": final_p,
        "n_applied": applied,
        "n_skipped": skipped,
        "n_candidates": len(candidate_bins),
        "trace": trace,
    }


def save_gamma_profile(result: dict, output_path: Path) -> None:
    f = result["f"]
    gamma = result["gamma_profile"]
    dt = result["dt_profile"]
    steps = result["steps_profile"]
    q0 = result["q0"]
    q_final = result["iplus"] - result["iminus"]
    data = np.column_stack([f, gamma, dt, steps, q0, q_final])
    header = "R,gamma_rf,dt,n_steps,Q_unburned,Q_final"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")


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
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel(r"$R$")
    axes[1].set_ylabel("Q profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    delta_q = result["final_q"] - result["initial_q"]
    fig.suptitle(
        f"SSRF dt/gamma-opt + recovery (neighbors in burn-down only) "
        f"P={result['polarization']*100:.0f}%  "
        f"γ≤{result['gamma_max']}  n_steps≤{result['n_steps']}  "
        f"Q: {result['initial_q']*100:.2f}% -> {result['final_q']*100:.2f}% "
        f"({delta_q*100:+.2f}%)  applied={result['n_applied']}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_gamma_profile(result: dict, output_path: Path) -> None:
    f = result["f"]
    gamma = result["gamma_profile"]
    dt = result["dt_profile"]
    q0 = result["q0"]
    q_final = result["iplus"] - result["iminus"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(f, q0, label=r"$Q$ (unburned)")
    axes[0].plot(f, q_final, label=r"$Q$ (after)")
    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[0].set_ylabel(r"$Q(R)$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(f, gamma, color="tab:orange", label=r"$\gamma_{\mathrm{RF}}(R)$")
    axes[1].fill_between(f, gamma, alpha=0.25, color="tab:orange")
    axes[1].axhline(result["gamma_max"], color="gray", linestyle=":", label=r"$\gamma$ max")
    axes[1].set_ylabel(r"$\gamma_{\mathrm{RF}}$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(f, dt, color="tab:green", label=r"$dt(R)$")
    axes[2].fill_between(f, dt, alpha=0.25, color="tab:green")
    axes[2].set_ylabel(r"$dt$")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    applied = [row for row in result["trace"] if not row["skipped"]]
    if applied:
        f_vals = [r["f"] for r in applied]
        q_before = [r["q_before"] for r in applied]
        q_after = [r["q_after"] for r in applied]
        axes[3].stem(f_vals, q_before, linefmt="C0-", markerfmt="C0o", basefmt=" ",
                     label=r"$Q_R$ before")
        axes[3].stem(f_vals, q_after, linefmt="C1-", markerfmt="C1o", basefmt=" ",
                     label=r"$Q_R$ after")
    axes[3].axhline(0.0, color="black", linestyle="--")
    axes[3].set_xlabel(r"$R$")
    axes[3].set_ylabel(r"$Q$ at burn bin")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(
        rf"Joint min $(\gamma_{{\mathrm{{RF}}}}, dt)$ per bin "
        rf"(neighbors in burn-down only; commit burn/mirror), "
        rf"$\gamma\leq{result['gamma_max']}$, $\leq{result['n_steps']}$ steps"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_q_gains(result: dict, output_path: Path) -> None:
    applied = [row for row in result["trace"] if not row["skipped"]]
    if not applied:
        return

    f_vals = [r["f"] for r in applied]
    q_gain = [r["q_gain"] for r in applied]
    q_tot_gain = [r.get("q_total_gain", 0.0) for r in applied]
    gamma_vals = [r["gamma_rf"] for r in applied]
    dt_vals = [r["dt"] for r in applied]
    n_steps_vals = [r["n_steps"] for r in applied]

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    axes[0].stem(f_vals, q_gain, basefmt=" ")
    axes[0].set_ylabel(r"$\Delta Q(R)$")
    axes[0].set_title(
        f"Per-bin local-Q change (γ≤{result['gamma_max']}, ≤{result['n_steps']} steps, "
        "joint dt/gamma-opt; neighbors in burn-down only)"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].stem(f_vals, q_tot_gain, basefmt=" ", linefmt="C1-", markerfmt="C1o")
    axes[1].set_ylabel(r"$\Delta Q_{\mathrm{total}}$")
    axes[1].grid(True, alpha=0.3)

    axes[2].stem(f_vals, gamma_vals, basefmt=" ", linefmt="C2-", markerfmt="C2o")
    axes[2].axhline(result["gamma_max"], color="gray", linestyle=":", label=r"$\gamma$ max")
    axes[2].set_ylabel(r"$\gamma_{\mathrm{RF}}$")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].stem(f_vals, dt_vals, basefmt=" ", linefmt="C4-", markerfmt="C4o")
    axes[3].set_ylabel(r"$dt$")
    axes[3].grid(True, alpha=0.3)

    axes[4].stem(f_vals, n_steps_vals, basefmt=" ", linefmt="C3-", markerfmt="C3o")
    axes[4].axhline(result["n_steps"], color="gray", linestyle=":", label=r"$N_{\mathrm{steps}}$ max")
    axes[4].set_xlabel(r"burn $R$")
    axes[4].set_ylabel(r"steps used")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    result = optimize_all_bins()
    stem = "rate_eqs_test_ssrf_all_bins_dt_gamma_opt_sequential"
    lineshape_path = OUT_DIR / f"{stem}_lineshape.png"
    gains_path = OUT_DIR / f"{stem}_gains.png"
    profile_path = OUT_DIR / f"{stem}_burn_profile.png"
    csv_path = OUT_DIR / f"{stem}_gamma_profile.csv"

    plot_result(result, lineshape_path)
    plot_q_gains(result, gains_path)
    plot_gamma_profile(result, profile_path)
    save_gamma_profile(result, csv_path)

    q_final = result["iplus"] - result["iminus"]
    burned = result["gamma_profile"] > 0.0
    if np.any(burned):
        max_abs_q = float(np.max(np.abs(q_final[burned])))
        mean_abs_q = float(np.mean(np.abs(q_final[burned])))
        steps_used = result["steps_profile"][burned]
        mean_steps = float(np.mean(steps_used))
        max_steps = int(np.max(steps_used))
    else:
        max_abs_q = mean_abs_q = mean_steps = float("nan")
        max_steps = 0

    print()
    print(
        f"P0={result['polarization']}  γ≤{result['gamma_max']}  "
        f"n_steps≤{result['n_steps']}  "
        f"dt∈[{result['dt_min']:g},{result['dt_max']:g}]×{result['n_dt']}"
    )
    print(
        f"dynamics: sameθ d+0={D_SAME_PLUS0}, d0-={D_SAME_0MINUS}; "
        f"neighbors d+0={D_SPEC_PLUS0}, d0-={D_SPEC_0MINUS} (burn-down only); "
        f"commit burn/mirror bins; DNP off"
    )
    print(
        f"RF bins applied: {result['n_applied']}/{result['n_candidates']}  "
        f"skipped={result['n_skipped']}"
    )
    print(f"P total: {result['initial_p']:.8f} -> {result['final_p']:.8f}")
    print(f"Q total: {result['initial_q']:.8f} -> {result['final_q']:.8f}")
    print(f"Q gain:  {result['final_q'] - result['initial_q']:+.8f}")
    print(f"Area loss: {result['area_loss_total']:.8f}")
    print(f"|Q(R)| on burned bins: mean={mean_abs_q:.3e}  max={max_abs_q:.3e}")
    print(f"steps used: mean={mean_steps:.1f}  max={max_steps}/{result['n_steps']}")
    if np.any(burned):
        dt_b = result["dt_profile"][burned]
        g_b = result["gamma_profile"][burned]
        print(f"dt(R):     min={float(np.min(dt_b)):.6g}  max={float(np.max(dt_b)):.6g}")
        print(f"gamma_rf:  min={float(np.min(g_b)):.6g}  max={float(np.max(g_b)):.6g}")
    else:
        print("dt(R)/gamma_rf: (no burns)")
    print(
        f"Saved {lineshape_path.name}, {gains_path.name}, "
        f"{profile_path.name}, {csv_path.name}"
    )


if __name__ == "__main__":
    main()
