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
DT = 0.05
N_STEPS = 100

AFP_EFFICIENCY = 1.0
AFP_CENTER_MARGIN = 0
RELAXATION_ON = True

OUT_DIR = Path(__file__).resolve().parent
STEM = "rate_eqs_test_ssrf_afp_window"


def run_event(
    *,
    polarization: float = P,
    num_bins: int = NUM_BINS,
    gamma_rf: float = GAMMA_RF,
    dt: float = DT,
    n_steps: int = N_STEPS,
    afp_efficiency: float = AFP_EFFICIENCY,
    afp_center_margin: int = AFP_CENTER_MARGIN,
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

    # AFP only where Q < 0 on the post-ssRF profile.
    q_pre_afp_profile = iplus - iminus
    AFP_BINS = np.flatnonzero(q_pre_afp_profile < 0)

    print(f"SSRF_BINS (Q<0):  {SSRF_BINS}")
    print(f"AFP_BINS  (Q<0):  {AFP_BINS}")

    iplus_pre_afp, iminus_pre_afp = iplus.copy(), iminus.copy()
    model.load_from_physical_intensities(iplus, iminus)

    # First "step": instantaneous AFP sweep (no rate term in derivative).
    model.params.rf_enabled = False
    model.params.gamma_rf = 0.0
    model.params.dt = float(dt)
    model.params.afp_efficiency = float(afp_efficiency)
    model.params.afp_center_margin = int(afp_center_margin)
    model.params.afp_subset_indices = AFP_BINS.tolist()
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

    afp_subset = model.afp_sweep(
        subset_indices=AFP_BINS.tolist(),
        efficiency=float(afp_efficiency),
        center_margin=int(afp_center_margin),
    )

    # Remaining steps: spin diffusion / relaxation only.
    relax_steps = max(0, int(n_steps) - 1)
    if relax_steps > 0:
        model.step(n_steps=relax_steps, rf_on=False, dnp_on=False)

    iplus, iminus, _ = model.physical_intensities()
    pops_after = model.level_populations()

    afp_lo = int(AFP_BINS[0]) if len(AFP_BINS) else 0
    afp_hi = int(AFP_BINS[-1]) + 1 if len(AFP_BINS) else 0
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
        "afp_efficiency": float(afp_efficiency),
        "d_same_plus0": float(model.params.d_same_plus0),
        "d_same_0minus": float(model.params.d_same_0minus),
        "d_spec_plus0": float(model.params.d_spec_plus0),
        "d_spec_0minus": float(model.params.d_spec_0minus),
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus_pre_afp": iplus_pre_afp,
        "iminus_pre_afp": iminus_pre_afp,
        "iplus": iplus,
        "iminus": iminus,
        "burn_trace": burn_trace,
        "q_unburned": gopt.q_total(iplus_unburned, iminus_unburned),
        "p_unburned": gopt.p_total(iplus_unburned, iminus_unburned),
        "q_pre_afp": gopt.q_total(iplus_pre_afp, iminus_pre_afp),
        "p_pre_afp": gopt.p_total(iplus_pre_afp, iminus_pre_afp),
        "q_final": gopt.q_total(iplus, iminus),
        "p_final": gopt.p_total(iplus, iminus),
        "model": model,
        "pops_before": pops_before,
        "pops_after": pops_after,
    }


def plot_event(result: dict, output_path: Path) -> None:
    f = result["f"]
    ip0, im0 = result["iplus_unburned"], result["iminus_unburned"]
    ip1, im1 = result["iplus_pre_afp"], result["iminus_pre_afp"]
    ip2, im2 = result["iplus"], result["iminus"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].step(f, ip0 + im0, color="black", linestyle="--", alpha=0.5, label=r"$P_s$ unburned")
    axes[0].step(f, ip1 + im1, color="tab:orange", alpha=0.85, label=r"$P_s$ after ssRF")
    axes[0].step(f, ip2 + im2, color="black", label=r"$P_s$ after AFP+relax")
    axes[0].step(f, ip2, color="tab:red", label=r"$I_+$ final")
    axes[0].step(f, im2, color="tab:blue", label=r"$I_-$ final")

    s0, s1 = result["ssrf_bin_range"]
    a0, a1 = result["afp_bin_range"]
    axes[0].axvspan(f[a0], f[a1 - 1], color="gold", alpha=0.18, label="AFP window")
    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].step(f, ip0 - im0, color="tab:purple", linestyle="--", alpha=0.5, label=r"$Q$ unburned")
    axes[1].step(f, ip1 - im1, color="tab:orange", alpha=0.85, label=r"$Q$ after ssRF")
    axes[1].step(f, ip2 - im2, color="tab:purple", label=r"$Q$ after AFP+relax")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].axvspan(f[a0], f[a1 - 1], color="gold", alpha=0.18)
    axes[1].set_xlabel(r"$R$")
    axes[1].set_ylabel("Q profile")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"ssRF bins [{s0},{s1}] γ={result['gamma_rf']}  +  "
        f"AFP [{a0},{a1}) + {result['relax_steps']} relax steps  "
        f"Q: {result['q_unburned']:.4f} → {result['q_pre_afp']:.4f} → {result['q_final']:.4f}"
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
        f"AFP: instant sweep on [{a0}, {a1})  n={len(result['afp_subset'])}  "
        f"efficiency={result['afp_efficiency']}  (Q<0 only)"
    )
    print(
        f"relaxation: {result['relax_steps']} steps  dt={result['dt']}  "
        f"d_same=({result['d_same_plus0']}, {result['d_same_0minus']})  "
        f"d_spec=({result['d_spec_plus0']}, {result['d_spec_0minus']})"
    )
    print(f"P: {result['p_unburned']:.6f} → {result['p_pre_afp']:.6f} → {result['p_final']:.6f}")
    print(f"Q: {result['q_unburned']:.6f} → {result['q_pre_afp']:.6f} → {result['q_final']:.6f}")

    before, after = result["pops_before"], result["pops_after"]
    model = result["model"]
    print(
        f"populations before: n+={before['n_plus']:.6f}  n0={before['n_zero']:.6f}  "
        f"n-={before['n_minus']:.6f}  n+-n-={before['P']:.6f}"
    )
    print(
        f"populations after:  n+={after['n_plus']:.6f}  n0={after['n_zero']:.6f}  "
        f"n-={after['n_minus']:.6f}  n+-n-={after['P']:.6f}"
    )

    print(f"Final Q: {model.n_plus - 2.0 * model.n_zero + model.n_minus:.6f}")
    print(f"Saved {out.name}")

    f = result["f"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f, model.n[:, PLUS], label="n+ (packet)")
    ax.plot(f, model.n[:, ZERO], label="n0 (packet)")
    ax.plot(f, model.n[:, MINUS], label="n- (packet)")
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
