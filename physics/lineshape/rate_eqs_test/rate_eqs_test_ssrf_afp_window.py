"""
ssRF (optional) then instantaneous AFP, then relaxation time steps.

AFP is armed and fired via ``apply_pending_afp`` *before* any Euler ``dt``
advance. All subsequent ``step`` calls are relaxation/diffusion only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime.model import MINUS, PLUS, ZERO
from physics.ssrf_realtime.rate_equations_realtime import build_model_for_intensities
import physics.lineshape.rate_eqs_test.rate_eqs_test_ssrf_all_bins_gamma_opt as gopt

P = 0.50
NUM_BINS = 500
GAMMA_RF = 0.0
DT = 0.005
N_STEPS = 1000

# AFP sweep frequencies only (mirrors update inside each fire — do not list them).
# None → same bins as ssRF candidates (Q<0 on the unburned profile).
AFP_BIN_RANGE: tuple[int, int] | None = None
AFP_EFFICIENCY = 1.0
AFP_CENTER_MARGIN = 0
RELAXATION_ON = True

OUT_DIR = Path(__file__).resolve().parent
STEM = "rate_eqs_test_ssrf_afp_window"


def _resolve_afp_subset(
    n_bins: int,
    *,
    ssrf_bins: np.ndarray,
    bin_range: tuple[int, int] | None,
    center_margin: int,
) -> list[int]:
    if bin_range is not None:
        start, stop = int(bin_range[0]), int(bin_range[1])
        subset = list(range(start, stop))
    else:
        subset = [int(i) for i in ssrf_bins]
    if center_margin > 0:
        c = n_bins // 2
        forbidden = set(range(max(0, c - center_margin), min(n_bins, c + center_margin + 1)))
        subset = [i for i in subset if i not in forbidden]
    return subset


def run_event(
    *,
    polarization: float = P,
    num_bins: int = NUM_BINS,
    gamma_rf: float = GAMMA_RF,
    dt: float = DT,
    n_steps: int = N_STEPS,
    afp_efficiency: float = AFP_EFFICIENCY,
    afp_center_margin: int = AFP_CENTER_MARGIN,
    afp_bin_range: tuple[int, int] | None = AFP_BIN_RANGE,
) -> dict:
    f = np.linspace(-3.0, 3.0, num_bins)
    _, iplus0, iminus0 = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    iplus_unburned, iminus_unburned = iplus.copy(), iminus.copy()

    q_signal = iplus_unburned - iminus_unburned
    SSRF_BINS = np.flatnonzero(q_signal < 0)

    params = gopt._burn_params(num_bins, float(f[0]), float(f[-1]), polarization)
    model = build_model_for_intensities(
        iplus, iminus, params=params, initial_polarization=polarization
    )
    pops_before = model.level_populations()

    # Optional per-bin ssRF (skipped entirely when gamma_rf <= 0).
    burn_trace: list[dict] = []
    for burn_idx in SSRF_BINS:
        trial = gopt.apply_rf_burn(
            iplus,
            iminus,
            int(burn_idx),
            float(gamma_rf),
            f=f,
            polarization=polarization,
            dt=dt,
            n_steps=n_steps,
            light=False,
        )
        if trial is None:
            burn_trace.append(
                {
                    "bin_idx": int(burn_idx),
                    "R": float(f[burn_idx]),
                    "gamma_rf": float(gamma_rf),
                    "n_steps": 0,
                    "q_before": gopt.q_at_r_bin(iplus, iminus, int(burn_idx)),
                    "q_after": gopt.q_at_r_bin(iplus, iminus, int(burn_idx)),
                    "skipped": True,
                }
            )
            continue
        iplus, iminus = trial["iplus"], trial["iminus"]
        burn_trace.append(
            {
                "bin_idx": int(burn_idx),
                "R": float(f[burn_idx]),
                "gamma_rf": float(trial["gamma_rf"]),
                "n_steps": int(trial["n_steps"]),
                "q_before": float(trial["q_before"]),
                "q_after": float(trial["q_after"]),
                "skipped": False,
            }
        )

    AFP_BINS = _resolve_afp_subset(
        num_bins,
        ssrf_bins=SSRF_BINS,
        bin_range=afp_bin_range,
        center_margin=int(afp_center_margin),
    )
    afp_touched = gopt.afp_touched_bins(num_bins, AFP_BINS)

    print(f"SSRF_BINS (Q<0):  n={len(SSRF_BINS)}  span=[{SSRF_BINS[0]},{SSRF_BINS[-1]}]")
    print(f"AFP sweep:        n={len(AFP_BINS)}  {AFP_BINS[:3]}...{AFP_BINS[-3:] if AFP_BINS else []}")
    if afp_touched:
        print(
            f"AFP touched (∪mirrors): n={len(afp_touched)}  "
            f"R∈[{f[afp_touched[0]]:+.3f},{f[afp_touched[-1]]:+.3f}]"
        )
    else:
        print("AFP touched: []")

    iplus_pre_afp, iminus_pre_afp = iplus.copy(), iminus.copy()
    model.load_from_physical_intensities(iplus, iminus)

    # Configure: instantaneous AFP first (no dt), then relaxation time steps.
    model.params.rf_enabled = False
    model.params.gamma_rf = 0.0
    model.params.dt = float(dt)
    model.params.afp_enabled = True
    model.params.afp_efficiency = float(afp_efficiency)
    model.params.afp_center_margin = int(afp_center_margin)
    model.params.afp_subset_indices = list(AFP_BINS)
    model.arm_afp(AFP_BINS)

    if RELAXATION_ON:
        model.params.d_same_plus0 = float(gopt.D_SAME_PLUS0)
        model.params.d_same_0minus = float(gopt.D_SAME_0MINUS)
        model.params.d_spec_plus0 = float(gopt.D_SPEC_PLUS0)
        model.params.d_spec_0minus = float(gopt.D_SPEC_0MINUS)
    else:
        model.params.d_same_plus0 = 0.0
        model.params.d_same_0minus = 0.0
        model.params.d_spec_plus0 = 0.0
        model.params.d_spec_0minus = 0.0

    # AFP map before any Euler / relaxation steps.
    afp_subset = model.apply_pending_afp()
    iplus_post_afp, iminus_post_afp, _ = model.physical_intensities()

    n_steps = max(0, int(n_steps))
    relax_steps = int(n_steps)
    if relax_steps > 0:
        model.step(n_steps=relax_steps, rf_on=False, dnp_on=False)

    iplus, iminus, _ = model.physical_intensities()
    pops_after = model.level_populations()

    afp_lo = int(AFP_BINS[0]) if AFP_BINS else 0
    afp_hi = int(AFP_BINS[-1]) + 1 if AFP_BINS else 0
    ssrf_lo = int(SSRF_BINS[0]) if len(SSRF_BINS) else 0
    ssrf_hi = int(SSRF_BINS[-1]) if len(SSRF_BINS) else 0

    return {
        "polarization": polarization,
        "f": f,
        "dt": dt,
        "n_steps": n_steps,
        "relax_steps": relax_steps,
        "gamma_rf": float(gamma_rf),
        "ssrf_bins": SSRF_BINS,
        "ssrf_bin_range": (ssrf_lo, ssrf_hi),
        "afp_bin_range": (afp_lo, afp_hi),
        "afp_subset": afp_subset,
        "afp_touched": afp_touched,
        "afp_efficiency": float(afp_efficiency),
        "d_same_plus0": float(model.params.d_same_plus0),
        "d_same_0minus": float(model.params.d_same_0minus),
        "d_spec_plus0": float(model.params.d_spec_plus0),
        "d_spec_0minus": float(model.params.d_spec_0minus),
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus_pre_afp": iplus_pre_afp,
        "iminus_pre_afp": iminus_pre_afp,
        "iplus_post_afp": np.asarray(iplus_post_afp, dtype=float),
        "iminus_post_afp": np.asarray(iminus_post_afp, dtype=float),
        "iplus": iplus,
        "iminus": iminus,
        "burn_trace": burn_trace,
        "q_unburned": gopt.q_total(iplus_unburned, iminus_unburned),
        "p_unburned": gopt.p_total(iplus_unburned, iminus_unburned),
        "q_pre_afp": gopt.q_total(iplus_pre_afp, iminus_pre_afp),
        "p_pre_afp": gopt.p_total(iplus_pre_afp, iminus_pre_afp),
        "q_post_afp": gopt.q_total(iplus_post_afp, iminus_post_afp),
        "p_post_afp": gopt.p_total(iplus_post_afp, iminus_post_afp),
        "q_final": gopt.q_total(iplus, iminus),
        "p_final": gopt.p_total(iplus, iminus),
        "model": model,
        "pops_before": pops_before,
        "pops_after": pops_after,
    }


def _shade_afp(ax, result: dict) -> None:
    f = result["f"]
    subset = result.get("afp_subset") or []
    if not subset:
        return
    n = len(f)
    lo, hi = min(subset), max(subset)
    ax.axvspan(f[lo], f[hi], color="gold", alpha=0.18, label="AFP sweep")
    m_lo, m_hi = n - 1 - hi, n - 1 - lo
    if m_lo != lo or m_hi != hi:
        ax.axvspan(f[m_lo], f[m_hi], color="gold", alpha=0.08, label="AFP mirrors")


def plot_event(result: dict, output_path: Path) -> None:
    f = result["f"]
    ip0, im0 = result["iplus_unburned"], result["iminus_unburned"]
    ip1, im1 = result["iplus_pre_afp"], result["iminus_pre_afp"]
    ip_a, im_a = result["iplus_post_afp"], result["iminus_post_afp"]
    ip2, im2 = result["iplus"], result["iminus"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].step(f, ip0 + im0, color="black", linestyle="--", alpha=0.5, label=r"$P_s$ unburned")
    # axes[0].step(f, ip0, color="tab:red", label=r"$I_+$ unburned")
    # axes[0].step(f, im0, color="tab:blue", label=r"$I_-$ unburned")
    axes[0].step(f, ip1 + im1, color="tab:orange", alpha=0.85, label=r"$P_s$ pre-AFP")
    axes[0].step(f, ip_a + im_a, color="tab:green", alpha=0.75, label=r"$P_s$ after AFP (pre-relax)")
    axes[0].step(f, ip_a, color="tab:red", label=r"$I_+$ after AFP (pre-relax)")
    axes[0].step(f, im_a, color="tab:blue", label=r"$I_-$ after AFP (pre-relax)")
    axes[0].step(f, ip2 + im2, color="black", label=r"$P_s$ after relax")
    _shade_afp(axes[0], result)
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # axes[1].step(f, ip0 - im0, color="tab:purple", linestyle="--", alpha=0.5, label=r"$Q$ unburned")
    axes[1].step(f, ip1 - im1, color="tab:orange", alpha=0.85, label=r"$Q$ pre-AFP")
    axes[1].step(f, ip_a - im_a, color="tab:green", alpha=0.75, label=r"$Q$ after AFP (step 0)")
    axes[1].step(f, ip2 - im2, color="tab:purple", label=r"$Q$ after relax")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    _shade_afp(axes[1], result)
    axes[1].set_xlabel(r"$R$")
    axes[1].set_ylabel("Q profile")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    s0, s1 = result["ssrf_bin_range"]
    a0, a1 = result["afp_bin_range"]
    fig.suptitle(
        f"ssRF [{s0},{s1}] γ={result['gamma_rf']}  |  "
        f"AFP before relax [{a0},{a1})  |  {result['relax_steps']} relax steps  "
        f"Q: {result['q_pre_afp']:.4f} → {result['q_post_afp']:.4f} → {result['q_final']:.4f}  |  \n"
        f"P: {result['p_pre_afp']:.4f} → {result['p_post_afp']:.4f} → {result['p_final']:.4f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    result = run_event()
    out = OUT_DIR / f"{STEM}_lineshape.png"
    plot_event(result, out)

    applied = [t for t in result["burn_trace"] if not t["skipped"]]
    print()
    print(f"P0={result['polarization']}  bins={len(result['f'])}")
    s0, s1 = result["ssrf_bin_range"]
    print(
        f"ssRF: bins [{s0}, {s1}]  "
        f"γ={result['gamma_rf']}  dt={result['dt']}  n_steps≤{result['n_steps']}  "
        f"applied={len(applied)}/{len(result['ssrf_bins'])}"
    )
    a0, a1 = result["afp_bin_range"]
    print(
        f"AFP: applied before any time steps  sweep [{a0}, {a1})  n={len(result['afp_subset'])}  "
        f"touched (∪mirrors)={len(result['afp_touched'])}  "
        f"efficiency={result['afp_efficiency']}"
    )
    print(
        f"relaxation: {result['relax_steps']} steps after AFP  dt={result['dt']}  "
        f"d_same=({result['d_same_plus0']}, {result['d_same_0minus']})  "
        f"d_spec=({result['d_spec_plus0']}, {result['d_spec_0minus']})"
    )
    print(
        f"P: {result['p_unburned']:.6f} → pre {result['p_pre_afp']:.6f} → "
        f"AFP {result['p_post_afp']:.6f} → relax {result['p_final']:.6f}"
    )
    print(
        f"Q: {result['q_unburned']:.6f} → pre {result['q_pre_afp']:.6f} → "
        f"AFP {result['q_post_afp']:.6f} → relax {result['q_final']:.6f}"
    )
    d_relax_p = result["p_final"] - result["p_post_afp"]
    d_relax_q = result["q_final"] - result["q_post_afp"]
    print(f"Δ from relax only:  ΔP={d_relax_p:+.6f}  ΔQ={d_relax_q:+.6f}")

    before, after = result["pops_before"], result["pops_after"]
    model = result["model"]
    print(
        f"populations before: n+={before['n_plus']:.6f}  n0={before['n_zero']:.6f}  "
        f"n-={before['n_minus']:.6f}  n+-n-={before['P']:.6f}"
    )
    print(
        f"populations after:  n+={after['n_plus']:.6f}  n0={after['n_zero']:.6f}  "
        f"n-={after['n_minus']:.6f}  n+-n-={after['n_plus'] - after['n_minus']:.6f}"
    )
    print(f"Final Q: {model.n_plus - 2.0 * model.n_zero + model.n_minus:.6f}")
    print(f"Saved {out.name}")

    f = result["f"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f, model.n[:, PLUS], label="n+ (packet)")
    ax.plot(f, model.n[:, ZERO], label="n0 (packet)")
    ax.plot(f, model.n[:, MINUS], label="n- (packet)")
    _shade_afp(ax, result)
    ax.set_xlabel(r"$R$")
    ax.set_ylabel("packet population")
    ax.set_title(
        rf"level totals: $n_+={model.n_plus:.4f}$, $n_0={model.n_zero:.4f}$, "
        rf"$n_-={model.n_minus:.4f}$  ($n_+-n_-={model.n_plus - model.n_minus:.4f}$)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / f"{STEM}_populations.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
