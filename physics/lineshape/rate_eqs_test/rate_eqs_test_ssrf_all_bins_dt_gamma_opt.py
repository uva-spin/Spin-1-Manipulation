"""
SSRF all-bins burn: joint (dt, gamma_rf) search to null local Q(R) per bin.

For each negative-Q R bin on the unburned lineshape, sweep a dt grid and
bisect the smallest gamma_rf that drives |Q(R)| below tolerance within
N_STEPS Euler steps (ps-sign preserved). Results are visualized as a 3D
scatter of (Q, dt, gamma).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime import Spin1Params
from physics.ssrf_realtime.model import MINUS, PLUS, ZERO
from physics.ssrf_realtime.rate_equations_realtime import build_model_for_intensities

P = 0.40
NUM_BINS = 500
N_STEPS = 100
Q_ABS_TOL = 1e-10
GAMMA_HI_INIT = 5.0
N_BISECT = 24
# Log-spaced dt grid: small dt needs larger gamma; large dt needs less.
DT_MIN = 1e-3
DT_MAX = 0.5
N_DT = 16
OUT_DIR = Path(__file__).resolve().parent


def mirror_bin_idx(n_bins: int, bin_idx: int) -> int:
    return int(n_bins) - 1 - int(bin_idx)


def q_at_r_bin(iplus: np.ndarray, iminus: np.ndarray, bin_idx: int) -> float:
    return float(iplus[int(bin_idx)] - iminus[int(bin_idx)])


def q_total(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus - iminus))


def p_total(iplus: np.ndarray, iminus: np.ndarray) -> float:
    return float(np.sum(iplus + iminus))


def q_target_tol(q_before: float) -> float:
    del q_before
    return float(Q_ABS_TOL)


def _rf_only_params(
    n_bins: int,
    r_min: float,
    r_max: float,
    polarization: float,
    dt: float,
) -> Spin1Params:
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
        dt=float(dt),
        steps=1,
    )


def _sign_ok(before: float, after: float) -> bool:
    if before > 0.0:
        return after > 0.0
    if before < 0.0:
        return after < 0.0
    return after == 0.0


def _rf_euler_two_packets(
    n: np.ndarray,
    mu: np.ndarray,
    kp: int,
    km: int,
    gamma_rf: float,
    dt: float,
) -> None:
    gdt = float(gamma_rf) * float(dt)
    j_p = gdt * (n[kp, PLUS] - n[kp, ZERO])
    n[kp, PLUS] -= j_p
    n[kp, ZERO] += j_p
    j_m = gdt * (n[km, ZERO] - n[km, MINUS])
    n[km, ZERO] -= j_m
    n[km, MINUS] += j_m
    for k in (kp, km):
        row = np.maximum(n[k], 1e-30)
        row *= float(mu[k]) / float(row.sum())
        n[k] = row


def _sync_branch_intensities(
    n: np.ndarray,
    iplus: np.ndarray,
    iminus: np.ndarray,
    kp: int,
    km: int,
    scale: float,
) -> None:
    n_bins = n.shape[0]
    for k in (kp, km):
        iplus[k] = scale * (n[k, PLUS] - n[k, ZERO])
        iminus[n_bins - 1 - k] = scale * (n[k, ZERO] - n[k, MINUS])


def apply_rf_burn(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    gamma_rf: float,
    *,
    f: np.ndarray,
    polarization: float,
    dt: float,
    n_steps: int = N_STEPS,
    q_tol: float | None = None,
    model=None,
    n0: np.ndarray | None = None,
) -> dict | None:
    """Up to ``n_steps`` RF Euler steps; stop early at |Q|<=tol or ps-sign flip."""
    if gamma_rf <= 0.0 or n_steps <= 0 or dt <= 0.0:
        return None

    burn_idx = int(burn_idx)
    mirror_idx = mirror_bin_idx(len(iplus), burn_idx)
    q_before = q_at_r_bin(iplus, iminus, burn_idx)
    tol = float(q_tol) if q_tol is not None else q_target_tol(q_before)

    if model is None or n0 is None:
        params = _rf_only_params(len(f), float(f[0]), float(f[-1]), polarization, dt)
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

    kp, km = model.branch_indices(float(f[burn_idx]))
    if kp is None or km is None:
        return None

    n = n0.copy()
    scale = float(model.display_cal) / float(model.dR)
    mu = model.mu
    iplus_cur = np.asarray(iplus, dtype=float).copy()
    iminus_cur = np.asarray(iminus, dtype=float).copy()
    steps_done = 0
    g = float(gamma_rf)
    dt_f = float(dt)

    for _ in range(int(n_steps)):
        q_now = float(iplus_cur[burn_idx] - iminus_cur[burn_idx])
        if abs(q_now) <= tol:
            break

        n_kp = n[kp].copy()
        n_km = n[km].copy()
        ip_b = float(iplus_cur[burn_idx])
        im_b = float(iminus_cur[burn_idx])
        ip_m = float(iplus_cur[mirror_idx])
        im_m = float(iminus_cur[mirror_idx])

        _rf_euler_two_packets(n, mu, kp, km, g, dt_f)
        _sync_branch_intensities(n, iplus_cur, iminus_cur, kp, km, scale)

        ok = (
            _sign_ok(ip_b, float(iplus_cur[burn_idx]))
            and _sign_ok(im_b, float(iminus_cur[burn_idx]))
            and _sign_ok(ip_b + im_b, float(iplus_cur[burn_idx] + iminus_cur[burn_idx]))
            and _sign_ok(ip_m, float(iplus_cur[mirror_idx]))
            and _sign_ok(im_m, float(iminus_cur[mirror_idx]))
            and _sign_ok(ip_m + im_m, float(iplus_cur[mirror_idx] + iminus_cur[mirror_idx]))
        )
        if not ok:
            n[kp] = n_kp
            n[km] = n_km
            iplus_cur[burn_idx] = ip_b
            iminus_cur[burn_idx] = im_b
            iplus_cur[mirror_idx] = ip_m
            iminus_cur[mirror_idx] = im_m
            break
        steps_done += 1

    if steps_done == 0:
        return None

    q_after = float(iplus_cur[burn_idx] - iminus_cur[burn_idx])
    return {
        "burn_idx": burn_idx,
        "gamma_rf": g,
        "dt": dt_f,
        "n_steps": int(steps_done),
        "t_burn": float(steps_done) * dt_f,
        "q_before": q_before,
        "q_after": q_after,
        "q_gain": q_after - q_before,
    }


def find_gamma_for_dt(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    dt: float,
    *,
    f: np.ndarray,
    polarization: float,
    n_steps: int = N_STEPS,
    gamma_hi: float = GAMMA_HI_INIT,
    n_bisect: int = N_BISECT,
    gamma_guess: float | None = None,
) -> dict | None:
    """Smallest gamma_rf that nulls |Q(R)| within n_steps at fixed dt."""
    q_before = q_at_r_bin(iplus, iminus, burn_idx)
    if q_before >= 0.0:
        return None

    tol = q_target_tol(q_before)
    params = _rf_only_params(len(f), float(f[0]), float(f[-1]), polarization, dt)
    model = build_model_for_intensities(
        iplus,
        iminus,
        params=params,
        rf_burn_R=float(f[burn_idx]),
        initial_polarization=polarization,
    )
    n0 = model.n.copy()

    def trial_at(gamma: float) -> dict | None:
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
        )

    def meets_tol(trial: dict | None) -> bool:
        return trial is not None and abs(float(trial["q_after"])) <= tol

    hi_trial: dict | None = None
    hi = float(gamma_hi)
    if gamma_guess is not None and gamma_guess > 0.0:
        warm = trial_at(float(gamma_guess))
        if meets_tol(warm):
            hi_trial = warm
            hi = float(warm["gamma_rf"])
        else:
            hi = max(hi, float(gamma_guess))

    if hi_trial is None:
        hi_trial = trial_at(hi)
        expand = 0
        while expand < 10:
            if hi_trial is None:
                hi *= 0.5
                if hi < 1e-12:
                    return None
                hi_trial = trial_at(hi)
                expand += 1
                continue
            if meets_tol(hi_trial):
                break
            hi *= 2.0
            hi_trial = trial_at(hi)
            expand += 1

    if not meets_tol(hi_trial):
        return None

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
    return best_meet


def sweep_dt_gamma_all_bins(
    polarization: float = P,
    num_bins: int = NUM_BINS,
    n_steps: int = N_STEPS,
    dt_min: float = DT_MIN,
    dt_max: float = DT_MAX,
    n_dt: int = N_DT,
) -> dict:
    """
    Independent per-bin (dt, gamma) sweep on the unburned lineshape.

    For each Q<0 bin and each dt on a log grid, find the minimum gamma that
    nulls local Q within ``n_steps``.
    """
    f = np.linspace(-3.0, 3.0, num_bins)
    _, iplus0, iminus0 = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    q0 = iplus - iminus
    candidate_bins = [i for i in range(num_bins) if float(q0[i]) < 0.0]
    dt_grid = np.geomspace(float(dt_min), float(dt_max), int(n_dt))

    rows: list[dict] = []
    bin_summary: list[dict] = []

    for k, burn_idx in enumerate(candidate_bins):
        q_before = float(q0[burn_idx])
        r_val = float(f[burn_idx])
        gamma_guess: float | None = None
        dt_ok: list[float] = []
        gamma_ok: list[float] = []
        steps_ok: list[int] = []

        # Sweep large->small dt so warm-start gamma decreases as dt grows first,
        # then increases as dt shrinks — still a useful upper bound.
        for dt in dt_grid[::-1]:
            trial = find_gamma_for_dt(
                iplus,
                iminus,
                burn_idx,
                float(dt),
                f=f,
                polarization=polarization,
                n_steps=n_steps,
                gamma_guess=gamma_guess,
            )
            if trial is None:
                rows.append(
                    {
                        "bin_idx": burn_idx,
                        "R": r_val,
                        "Q": q_before,
                        "dt": float(dt),
                        "gamma_rf": np.nan,
                        "n_steps": 0,
                        "t_burn": 0.0,
                        "q_after": q_before,
                        "success": False,
                    }
                )
                continue

            g = float(trial["gamma_rf"])
            gamma_guess = g
            dt_ok.append(float(dt))
            gamma_ok.append(g)
            steps_ok.append(int(trial["n_steps"]))
            rows.append(
                {
                    "bin_idx": burn_idx,
                    "R": r_val,
                    "Q": q_before,
                    "dt": float(dt),
                    "gamma_rf": g,
                    "n_steps": int(trial["n_steps"]),
                    "t_burn": float(trial["t_burn"]),
                    "q_after": float(trial["q_after"]),
                    "success": True,
                }
            )

        if dt_ok:
            dt_arr = np.asarray(dt_ok, dtype=float)
            g_arr = np.asarray(gamma_ok, dtype=float)
            i_min_g = int(np.argmin(g_arr))
            i_min_dt = int(np.argmin(dt_arr))
            bin_summary.append(
                {
                    "bin_idx": burn_idx,
                    "R": r_val,
                    "Q": q_before,
                    "gamma_min": float(g_arr[i_min_g]),
                    "dt_at_gamma_min": float(dt_arr[i_min_g]),
                    "dt_min": float(dt_arr[i_min_dt]),
                    "gamma_at_dt_min": float(g_arr[i_min_dt]),
                    "n_dt_success": len(dt_ok),
                }
            )
            print(
                f"bin {k + 1:3d}/{len(candidate_bins)}: idx={burn_idx:3d}  "
                f"R={r_val:+.4f}  Q={q_before:.3e}  "
                f"gamma_min={g_arr[i_min_g]:.4g}@dt={dt_arr[i_min_g]:.3g}  "
                f"dt_min={dt_arr[i_min_dt]:.3g}@gamma={g_arr[i_min_dt]:.4g}  "
                f"ok={len(dt_ok)}/{n_dt}"
            )
        else:
            bin_summary.append(
                {
                    "bin_idx": burn_idx,
                    "R": r_val,
                    "Q": q_before,
                    "gamma_min": np.nan,
                    "dt_at_gamma_min": np.nan,
                    "dt_min": np.nan,
                    "gamma_at_dt_min": np.nan,
                    "n_dt_success": 0,
                }
            )
            print(
                f"bin {k + 1:3d}/{len(candidate_bins)}: idx={burn_idx:3d}  "
                f"R={r_val:+.4f}  Q={q_before:.3e}  FAILED all dt"
            )

    return {
        "polarization": polarization,
        "n_steps": n_steps,
        "dt_grid": dt_grid,
        "f": f,
        "q0": q0,
        "iplus": iplus,
        "iminus": iminus,
        "initial_q": q_total(iplus, iminus),
        "initial_p": p_total(iplus, iminus),
        "n_candidates": len(candidate_bins),
        "rows": rows,
        "bin_summary": bin_summary,
    }


def save_sweep_csv(result: dict, output_path: Path) -> None:
    rows = result["rows"]
    if not rows:
        return
    data = np.array(
        [
            [
                r["bin_idx"],
                r["R"],
                r["Q"],
                r["dt"],
                r["gamma_rf"],
                r["n_steps"],
                r["t_burn"],
                r["q_after"],
                float(r["success"]),
            ]
            for r in rows
        ],
        dtype=float,
    )
    header = "bin_idx,R,Q,dt,gamma_rf,n_steps,t_burn,q_after,success"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")


def save_bin_summary_csv(result: dict, output_path: Path) -> None:
    summary = result["bin_summary"]
    if not summary:
        return
    data = np.array(
        [
            [
                s["bin_idx"],
                s["R"],
                s["Q"],
                s["gamma_min"],
                s["dt_at_gamma_min"],
                s["dt_min"],
                s["gamma_at_dt_min"],
                s["n_dt_success"],
            ]
            for s in summary
        ],
        dtype=float,
    )
    header = (
        "bin_idx,R,Q,gamma_min,dt_at_gamma_min,dt_min,gamma_at_dt_min,n_dt_success"
    )
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")


def plot_q_dt_gamma_3d(result: dict, output_path: Path) -> None:
    """3D scatter: Q vs dt vs gamma_rf for successful (dt, gamma) pairs."""
    ok = [r for r in result["rows"] if r["success"]]
    if not ok:
        return

    q_vals = np.array([r["Q"] for r in ok], dtype=float)
    dt_vals = np.array([r["dt"] for r in ok], dtype=float)
    g_vals = np.array([r["gamma_rf"] for r in ok], dtype=float)
    r_vals = np.array([r["R"] for r in ok], dtype=float)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        q_vals,
        dt_vals,
        g_vals,
        c=r_vals,
        cmap="coolwarm",
        s=18,
        alpha=0.85,
        depthshade=True,
    )
    ax.set_xlabel(r"$Q(R)$ (unburned)")
    ax.set_ylabel(r"$dt$")
    ax.set_zlabel(r"$\gamma_{\mathrm{RF}}$ (min)")
    ax.set_title(
        rf"Min $\gamma_{{\mathrm{{RF}}}}$ vs $dt$ to null $Q(R)$ "
        rf"(P={result['polarization']*100:.0f}%, ≤{result['n_steps']} steps)"
    )
    # Default azim=-60; rotate view 90° left → azim=30.
    ax.view_init(elev=30, azim=30)
    cbar = fig.colorbar(sc, ax=ax, pad=0.12, shrink=0.75)
    cbar.set_label(r"burn $R$")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_q_dt_gamma_surface(result: dict, output_path: Path) -> None:
    """
    3D surface-style view: for each bin, the (dt, gamma) curve colored by Q.

    Uses plot of polylines in (Q, dt, gamma) space — one curve per bin.
    """
    by_bin: dict[int, list[dict]] = defaultdict(list)
    for r in result["rows"]:
        if r["success"]:
            by_bin[int(r["bin_idx"])].append(r)

    if not by_bin:
        return

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for _burn_idx, pts in by_bin.items():
        pts_sorted = sorted(pts, key=lambda p: p["dt"])
        q = float(pts_sorted[0]["Q"])
        dt = [p["dt"] for p in pts_sorted]
        g = [p["gamma_rf"] for p in pts_sorted]
        qq = [q] * len(dt)
        ax.plot(qq, dt, g, alpha=0.55, linewidth=1.2)

    # Highlight per-bin extrema: min-gamma and min-dt points.
    summary = [s for s in result["bin_summary"] if s["n_dt_success"] > 0]
    if summary:
        q_s = [s["Q"] for s in summary]
        ax.scatter(
            q_s,
            [s["dt_at_gamma_min"] for s in summary],
            [s["gamma_min"] for s in summary],
            c="tab:orange",
            s=28,
            marker="o",
            depthshade=True,
            label=r"min $\gamma$ per bin",
        )
        ax.scatter(
            q_s,
            [s["dt_min"] for s in summary],
            [s["gamma_at_dt_min"] for s in summary],
            c="tab:green",
            s=28,
            marker="^",
            depthshade=True,
            label=r"min $dt$ per bin",
        )
        ax.legend(loc="upper left")

    ax.set_xlabel(r"$Q(R)$ (unburned)")
    ax.set_ylabel(r"$dt$")
    ax.set_zlabel(r"$\gamma_{\mathrm{RF}}$")
    ax.set_title(
        rf"Per-bin $(dt,\gamma)$ curves to null $Q$ "
        rf"(P={result['polarization']*100:.0f}%, ≤{result['n_steps']} steps)"
    )
    # Default azim=-60; rotate view 90° left → azim=30.
    ax.view_init(elev=30, azim=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_minima_vs_q(result: dict, output_path: Path) -> None:
    """2D companion: min gamma and min dt vs Q for each bin."""
    summary = [s for s in result["bin_summary"] if s["n_dt_success"] > 0]
    if not summary:
        return

    q_vals = np.array([s["Q"] for s in summary], dtype=float)
    g_min = np.array([s["gamma_min"] for s in summary], dtype=float)
    dt_min = np.array([s["dt_min"] for s in summary], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].scatter(q_vals, g_min, s=18, c="tab:orange")
    axes[0].set_ylabel(r"min $\gamma_{\mathrm{RF}}$")
    axes[0].set_title(
        rf"Minimum $\gamma$ / $dt$ needed to null $Q(R)$ "
        rf"(≤{result['n_steps']} steps)"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(q_vals, dt_min, s=18, c="tab:green")
    axes[1].set_xlabel(r"$Q(R)$ (unburned)")
    axes[1].set_ylabel(r"min $dt$")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    result = sweep_dt_gamma_all_bins()
    stem = "rate_eqs_test_ssrf_all_bins_dt_gamma_opt"
    scatter_path = OUT_DIR / f"{stem}_3d.png"
    curves_path = OUT_DIR / f"{stem}_3d_curves.png"
    minima_path = OUT_DIR / f"{stem}_minima_vs_q.png"
    sweep_csv = OUT_DIR / f"{stem}_sweep.csv"
    summary_csv = OUT_DIR / f"{stem}_bin_summary.csv"

    plot_q_dt_gamma_3d(result, scatter_path)
    plot_q_dt_gamma_surface(result, curves_path)
    plot_minima_vs_q(result, minima_path)
    save_sweep_csv(result, sweep_csv)
    save_bin_summary_csv(result, summary_csv)

    ok_rows = [r for r in result["rows"] if r["success"]]
    summary_ok = [s for s in result["bin_summary"] if s["n_dt_success"] > 0]
    print()
    print(
        f"P0={result['polarization']}  n_steps≤{result['n_steps']}  "
        f"dt grid=[{result['dt_grid'][0]:.3g}, {result['dt_grid'][-1]:.3g}] "
        f"x {len(result['dt_grid'])}"
    )
    print(
        f"Candidate bins: {result['n_candidates']}  "
        f"bins with ≥1 success: {len(summary_ok)}  "
        f"successful (dt,gamma) pairs: {len(ok_rows)}"
    )
    if summary_ok:
        g_mins = [s["gamma_min"] for s in summary_ok]
        dt_mins = [s["dt_min"] for s in summary_ok]
        print(
            f"Across bins: gamma_min in [{min(g_mins):.4g}, {max(g_mins):.4g}]  "
            f"dt_min in [{min(dt_mins):.3g}, {max(dt_mins):.3g}]"
        )
    print(
        f"Saved {scatter_path.name}, {curves_path.name}, {minima_path.name}, "
        f"{sweep_csv.name}, {summary_csv.name}"
    )


if __name__ == "__main__":
    main()
