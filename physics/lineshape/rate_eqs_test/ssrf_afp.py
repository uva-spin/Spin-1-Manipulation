import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.lineshape.rate_eqs_test.ssrf_bin_traj import (
    SIGMA_BINS,
    VOIGT_GAMMA_BINS,
    freeze_rf_profile,
    make_voigt_rf_profile,
    ssrf_touched_bins as traj_ssrf_touched_bins,
)
from physics.ssrf_realtime.model import MINUS, PLUS, ZERO, Spin1Model, Spin1Params

P = 0.50
NUM_BINS = 500
DT = 0.005
N_STEPS = 0

RF_ON = True
GAMMA_RF = 0.0
# Single-bin Voigt burn preview (smooth threshold-based support around BURN_BIN).
# None -> deepest Q<0 bin.
BURN_BIN: int | None = None
RF_SIGMA_BINS = SIGMA_BINS
RF_VOIGT_GAMMA_BINS = VOIGT_GAMMA_BINS

AFP_ON = True
AFP_BINS: list[int] | None = list(np.arange(170,200))
AFP_EFFICIENCY = 1.0
AFP_CENTER_MARGIN = 0

RELAXATION_ON = True

OUT_DIR = Path(__file__).resolve().parent
STEM = "rate_eqs_test_ssrf_afp_window"


def mirror_bin_idx(n_bins: int, bin_idx: int) -> int:
    return int(n_bins) - 1 - int(bin_idx)


def ssrf_touched_bins(n_bins: int, subset: list[int] | np.ndarray) -> list[int]:
    """Intensity/packet bins ssRF changes: each burn index i also updates mirror(i)."""
    return traj_ssrf_touched_bins(n_bins, subset)


def commit_ssrf_bins_only(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_sim: np.ndarray,
    iminus_sim: np.ndarray,
    touched: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep pre-RF intensities everywhere except RF-touched bins.

    Same pattern as AFP: simulation may spill via spectral diffusion / row
    renormalization; only write burn ∪ mirror bins so the rest of the
    lineshape is not globally shifted.
    """
    out_ip = np.asarray(iplus, dtype=float).copy()
    out_im = np.asarray(iminus, dtype=float).copy()
    ip_sim = np.asarray(iplus_sim, dtype=float)
    im_sim = np.asarray(iminus_sim, dtype=float)
    for k in touched:
        out_ip[k] = float(ip_sim[k])
        out_im[k] = float(im_sim[k])
    return out_ip, out_im


def resolve_burn_bin(q_signal: np.ndarray, burn_bin: int | None) -> int:
    """Pick burn center: explicit index, else deepest Q<0 bin."""
    if burn_bin is not None:
        return int(burn_bin)
    q = np.asarray(q_signal, dtype=float)
    neg = np.flatnonzero(q < 0.0)
    if neg.size == 0:
        return int(np.argmin(q))
    return int(neg[np.argmin(q[neg])])


def run_event(
    *,
    polarization: float = P,
    num_bins: int = NUM_BINS,
    gamma_rf: float = GAMMA_RF,
    dt: float = DT,
    n_steps: int = N_STEPS,
    afp_efficiency: float = AFP_EFFICIENCY,
    afp_center_margin: int = AFP_CENTER_MARGIN,
    afp_bins: list[int] | None = AFP_BINS,
    burn_bin: int | None = BURN_BIN,
    sigma_bins: float = RF_SIGMA_BINS,
    voigt_gamma_bins: float = RF_VOIGT_GAMMA_BINS,
    half_width: int | None = None,
) -> dict:
    f = np.linspace(-3.0, 3.0, num_bins)
    _, iplus0, iminus0 = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    iplus_unburned, iminus_unburned = iplus.copy(), iminus.copy()

    q_signal = iplus_unburned - iminus_unburned
    burn_idx = resolve_burn_bin(q_signal, burn_bin)
    profile, ssrf_subset = make_voigt_rf_profile(
        num_bins,
        burn_idx,
        float(gamma_rf),
        sigma=float(sigma_bins),
        lorentz_gamma=float(voigt_gamma_bins),
        half_width=half_width,
    )
    SSRF_BINS = np.asarray(ssrf_subset, dtype=int)

    params = Spin1Params(
        p0=polarization,
        q0=0.0,
        p_dnp_sat=polarization,
        # t1_p_eq=1.0,
        dnp_enabled=False,
        rf_enabled=RF_ON,
        relax_enabled=RELAXATION_ON,
    )

    model = Spin1Model(params)

    pops_before = model.level_populations()

    AFP_BINS = model._resolve_afp_subset(
        num_bins,
        subset_indices=afp_bins,
        center_margin=int(afp_center_margin),
    )

    iplus_pre_afp, iminus_pre_afp = iplus.copy(), iminus.copy()
    model.load_from_physical_intensities(iplus, iminus)

    touched = ssrf_touched_bins(num_bins, ssrf_subset)

    model.params.gamma_rf = gamma_rf
    model.params.dt = float(dt)
    model.params.ssrf_subset_indices = [int(i) for i in ssrf_subset]
    model.params.rf_burn_R = float(f[burn_idx])
    model.params.afp_enabled = bool(AFP_ON)
    model.params.afp_efficiency = float(afp_efficiency)
    model.params.afp_center_margin = int(afp_center_margin)
    model.params.afp_subset_indices = list(afp_bins) if afp_bins else None
    freeze_rf_profile(model, profile)
    # Restrict Euler + relaxation updates to RF-touched packets (burn ∪ mirrors).
    model._active_idx = np.asarray(touched, dtype=int) if touched else None

    n_steps = max(0, int(n_steps))
    model.step(n_steps=n_steps)

    ip_afp, im_afp = model.ip_afp, model.im_afp
    if ip_afp is None or im_afp is None:
        ip_afp, im_afp = iplus_pre_afp.copy(), iminus_pre_afp.copy()

    iplus_sim, iminus_sim, _ = model.physical_intensities()
    # Baseline = post-AFP (if AFP ran) else unburned; then overlay RF-touched only.
    if AFP_ON and ip_afp is not None:
        base_ip = np.asarray(ip_afp, dtype=float)
        base_im = np.asarray(im_afp, dtype=float)
    else:
        base_ip, base_im = iplus_unburned, iminus_unburned
    iplus, iminus = commit_ssrf_bins_only(
        base_ip, base_im, iplus_sim, iminus_sim, touched
    )
    # Reload committed intensities so packet / population views match the plot.
    model.load_from_physical_intensities(iplus, iminus)
    pops_after = model.level_populations()

    afp_lo = int(afp_bins[0]) if afp_bins else 0
    afp_hi = int(afp_bins[-1]) + 1 if afp_bins else 0
    ssrf_lo = int(SSRF_BINS[0]) if len(SSRF_BINS) else burn_idx
    ssrf_hi = int(SSRF_BINS[-1]) if len(SSRF_BINS) else burn_idx
    support_half_width = max(burn_idx - ssrf_lo, ssrf_hi - burn_idx)

    return {
        "polarization": polarization,
        "f": f,
        "dt": dt,
        "n_steps": n_steps,
        "gamma_rf": float(gamma_rf),
        "burn_bin": burn_idx,
        "mirror_bin": mirror_bin_idx(num_bins, burn_idx),
        "ssrf_bins": SSRF_BINS,
        "ssrf_bin_range": (ssrf_lo, ssrf_hi),
        "support_half_width": int(support_half_width),
        "half_width": half_width,
        "sigma_bins": float(sigma_bins),
        "voigt_gamma_bins": float(voigt_gamma_bins),
        "afp_bin_range": (afp_lo, afp_hi),
        "afp_subset": afp_bins,
        "afp_efficiency": float(afp_efficiency),
        "d_same_plus0": float(model.params.d_same_plus0),
        "d_same_0minus": float(model.params.d_same_0minus),
        "d_spec_plus0": float(model.params.d_spec_plus0),
        "d_spec_0minus": float(model.params.d_spec_0minus),
        "iplus_unburned": iplus_unburned,
        "iminus_unburned": iminus_unburned,
        "iplus_pre_afp": iplus_pre_afp,
        "iminus_pre_afp": iminus_pre_afp,
        "iplus_post_afp": ip_afp,
        "iminus_post_afp": im_afp,
        "iplus": iplus,
        "iminus": iminus,
        "q_unburned": np.sum(iplus_unburned - iminus_unburned),
        "p_unburned": np.sum(iplus_unburned + iminus_unburned),
        "q_pre_afp": np.sum(iplus_pre_afp - iminus_pre_afp),
        "p_pre_afp": np.sum(iplus_pre_afp + iminus_pre_afp),
        "q_post_afp": np.sum(ip_afp - im_afp),
        "p_post_afp": np.sum(ip_afp + im_afp),
        "q_final": np.sum(iplus - iminus),
        "p_final": np.sum(iplus + iminus),
        "model": model,
        "pops_before": pops_before,
        "pops_after": pops_after,
    }



def plot_event(result: dict, output_path: Path) -> None:
    f = result["f"]
    ip0, im0 = result["iplus_unburned"], result["iminus_unburned"]
    ip1, im1 = result["iplus_pre_afp"], result["iminus_pre_afp"]
    ip_a, im_a = result["iplus_post_afp"], result["iminus_post_afp"]
    ip2, im2 = result["iplus"], result["iminus"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].step(f, ip0 + im0, color="black", linestyle="--", alpha=0.5, label=r"$P_s$ unburned")
    axes[0].step(f, ip0, color="tab:red", label=r"$I_+$ unburned", linestyle="--")
    axes[0].step(f, im0, color="tab:blue", label=r"$I_-$ unburned", linestyle="--")
    if AFP_ON:
        axes[0].step(f, ip1 + im1, color="tab:orange", alpha=0.85, label=r"$P_s$ pre-AFP")
        axes[0].step(f, ip_a + im_a, color="tab:green", alpha=0.75, label=r"$P_s$ after AFP (pre-relax)")
        axes[0].step(f, ip_a, color="tab:red", label=r"$I_+$ after AFP (pre-relax)")
        axes[0].step(f, im_a, color="tab:blue", label=r"$I_-$ after AFP (pre-relax)")
    axes[0].step(f, ip2 + im2, color="black", label=r"$P_s$ after relax")
    axes[0].step(f, ip2, color="tab:red", label=r"$I_+$ after ssrf")
    axes[0].step(f, im2, color="tab:blue", label=r"$I_-$ after ssrf")

    axes[0].set_ylabel("intensity")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # axes[1].step(f, ip0 - im0, color="tab:purple", linestyle="--", alpha=0.5, label=r"$Q$ unburned")
    if AFP_ON:
        axes[1].step(f, ip1 - im1, color="tab:orange", alpha=0.85, label=r"$Q$ pre-AFP")
        axes[1].step(f, ip_a - im_a, color="tab:green", alpha=0.75, label=r"$Q$ after AFP (step 0)")
    axes[1].step(f, ip2 - im2, color="tab:purple", label=r"$Q$ after relax")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel(r"$R$")
    axes[1].set_ylabel("Q profile")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].grid(True, alpha=0.3)

    s0, s1 = result["ssrf_bin_range"]
    a0, a1 = result["afp_bin_range"]
    burn = result.get("burn_bin", s0)
    mir = result.get("mirror_bin", mirror_bin_idx(len(f), int(burn)))
    fig.suptitle(
        f"ssRF Voigt burn={burn} mirror={mir} support=[{s0},{s1}] "
        f"gamma_rf={result['gamma_rf']} +/-{result.get('support_half_width', '?')}  |  "
        f"AFP [{a0},{a1})  |  {result['n_steps']} steps  "
        f"Q: {result['q_pre_afp']:.4f} -> {result['q_final']:.4f}  |  "
        f"P: {result['p_pre_afp']:.4f} -> {result['p_final']:.4f}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    result = run_event()
    out = OUT_DIR / f"{STEM}_lineshape.png"
    plot_event(result, out)

    print()
    print(f"P0={result['polarization']}  bins={len(result['f'])}")
    s0, s1 = result["ssrf_bin_range"]
    print(
        f"ssRF Voigt: burn={result['burn_bin']} mirror={result['mirror_bin']}  "
        f"support=[{s0}, {s1}]  +/-{result['support_half_width']}  "
        f"sigma={result['sigma_bins']}  voigt_gamma={result['voigt_gamma_bins']}  "
        f"gamma_rf={result['gamma_rf']}  dt={result['dt']}  n_steps={result['n_steps']}"
    )
    a0, a1 = result["afp_bin_range"]
    print(
        f"relaxation: {result['n_steps']} steps  dt={result['dt']}  "
        f"d_same=({result['d_same_plus0']}, {result['d_same_0minus']})  "
        f"d_spec=({result['d_spec_plus0']}, {result['d_spec_0minus']})"
    )
    print(
        f"P: {result['p_unburned']:.6f} -> pre {result['p_pre_afp']:.6f} -> "
        f"AFP {result['p_post_afp']:.6f} -> relax {result['p_final']:.6f}"
    )
    print(
        f"Q: {result['q_unburned']:.6f} -> pre {result['q_pre_afp']:.6f} -> "
        f"AFP {result['q_post_afp']:.6f} -> relax {result['q_final']:.6f}"
    )
    d_relax_p = result["p_final"] - result["p_post_afp"]
    d_relax_q = result["q_final"] - result["q_post_afp"]
    print(f"d from relax only:  dP={d_relax_p:+.6f}  dQ={d_relax_q:+.6f}")

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
