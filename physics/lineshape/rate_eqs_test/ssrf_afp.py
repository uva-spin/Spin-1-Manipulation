import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.lineshape.rate_eqs_test.ssrf_bin_traj import SIGMA_BINS, VOIGT_GAMMA_BINS, freeze_rf_profile, make_voigt_rf_profile, ssrf_touched_bins as traj_ssrf_touched_bins
from physics.ssrf_realtime.model import MINUS, PLUS, ZERO, Spin1Model, Spin1Params
P = 0.5
NUM_BINS = 500
DT = 0.005
N_STEPS = 0
RF_ON = True
GAMMA_RF = 0.0
BURN_BIN = None
RF_SIGMA_BINS = SIGMA_BINS
RF_VOIGT_GAMMA_BINS = VOIGT_GAMMA_BINS
AFP_ON = True
AFP_BINS = list(np.arange(170, 200))
AFP_EFFICIENCY = 1.0
AFP_CENTER_MARGIN = 0
RELAXATION_ON = True
OUT_DIR = Path(__file__).resolve().parent
STEM = 'rate_eqs_test_ssrf_afp_window'

def mirror_bin_idx(n_bins, bin_idx):
    return n_bins - 1 - bin_idx

def ssrf_touched_bins(n_bins, subset):
    """Intensity/packet bins ssRF changes: each burn index i also updates mirror(i)."""
    return traj_ssrf_touched_bins(n_bins, subset)

def commit_ssrf_bins_only(iplus, iminus, iplus_sim, iminus_sim, touched):
    """
    Keep pre-RF intensities everywhere except RF-touched bins.

    Same pattern as AFP: simulation may spill via spectral diffusion / row
    renormalization; only write burn ∪ mirror bins so the rest of the
    lineshape is not globally shifted.
    """
    out_ip = np.asarray(iplus).copy()
    out_im = np.asarray(iminus).copy()
    ip_sim = np.asarray(iplus_sim)
    im_sim = np.asarray(iminus_sim)
    for k in touched:
        out_ip[k] = ip_sim[k]
        out_im[k] = im_sim[k]
    return (out_ip, out_im)

def resolve_burn_bin(q_signal, burn_bin):
    """Pick burn center: explicit index, else deepest Q<0 bin."""
    if burn_bin is not None:
        return burn_bin
    q = np.asarray(q_signal)
    neg = np.flatnonzero(q < 0.0)
    if neg.size == 0:
        return np.argmin(q)
    return neg[np.argmin(q[neg])]

def run_event(*, polarization=P, num_bins=NUM_BINS, gamma_rf=GAMMA_RF, dt=DT, n_steps=N_STEPS, afp_efficiency=AFP_EFFICIENCY, afp_center_margin=AFP_CENTER_MARGIN, afp_bins=AFP_BINS, burn_bin=BURN_BIN, sigma_bins=RF_SIGMA_BINS, voigt_gamma_bins=RF_VOIGT_GAMMA_BINS, half_width=None):
    f = np.linspace(-3.0, 3.0, num_bins)
    (_, iplus0, iminus0) = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus0).copy()
    iminus = np.asarray(iminus0).copy()
    (iplus_unburned, iminus_unburned) = (iplus.copy(), iminus.copy())
    q_signal = iplus_unburned - iminus_unburned
    burn_idx = resolve_burn_bin(q_signal, burn_bin)
    (profile, ssrf_subset) = make_voigt_rf_profile(num_bins, burn_idx, gamma_rf, sigma=sigma_bins, lorentz_gamma=voigt_gamma_bins, half_width=half_width)
    SSRF_BINS = np.asarray(ssrf_subset, dtype=int)
    params = Spin1Params(p0=polarization, q0=0.0, p_dnp_sat=polarization, dnp_enabled=False, rf_enabled=RF_ON, relax_enabled=RELAXATION_ON)
    model = Spin1Model(params)
    pops_before = model.level_populations()
    AFP_BINS = model._resolve_afp_subset(num_bins, subset_indices=afp_bins, center_margin=afp_center_margin)
    (iplus_pre_afp, iminus_pre_afp) = (iplus.copy(), iminus.copy())
    model.load_from_physical_intensities(iplus, iminus)
    touched = ssrf_touched_bins(num_bins, ssrf_subset)
    model.params.gamma_rf = gamma_rf
    model.params.dt = dt
    model.params.ssrf_subset_indices = [i for i in ssrf_subset]
    model.params.rf_burn_R = f[burn_idx]
    model.params.afp_enabled = AFP_ON
    model.params.afp_efficiency = afp_efficiency
    model.params.afp_center_margin = afp_center_margin
    model.params.afp_subset_indices = list(afp_bins) if afp_bins else None
    freeze_rf_profile(model, profile)
    model._active_idx = np.asarray(touched, dtype=int) if touched else None
    n_steps = max(0, n_steps)
    model.step(n_steps=n_steps)
    (ip_afp, im_afp) = (model.ip_afp, model.im_afp)
    if ip_afp is None or im_afp is None:
        (ip_afp, im_afp) = (iplus_pre_afp.copy(), iminus_pre_afp.copy())
    (iplus_sim, iminus_sim, _) = model.physical_intensities()
    if AFP_ON and ip_afp is not None:
        base_ip = np.asarray(ip_afp)
        base_im = np.asarray(im_afp)
    else:
        (base_ip, base_im) = (iplus_unburned, iminus_unburned)
    (iplus, iminus) = commit_ssrf_bins_only(base_ip, base_im, iplus_sim, iminus_sim, touched)
    model.load_from_physical_intensities(iplus, iminus)
    pops_after = model.level_populations()
    afp_lo = afp_bins[0] if afp_bins else 0
    afp_hi = afp_bins[-1] + 1 if afp_bins else 0
    ssrf_lo = SSRF_BINS[0] if len(SSRF_BINS) else burn_idx
    ssrf_hi = SSRF_BINS[-1] if len(SSRF_BINS) else burn_idx
    support_half_width = max(burn_idx - ssrf_lo, ssrf_hi - burn_idx)
    return {'polarization': polarization, 'f': f, 'dt': dt, 'n_steps': n_steps, 'gamma_rf': gamma_rf, 'burn_bin': burn_idx, 'mirror_bin': mirror_bin_idx(num_bins, burn_idx), 'ssrf_bins': SSRF_BINS, 'ssrf_bin_range': (ssrf_lo, ssrf_hi), 'support_half_width': support_half_width, 'half_width': half_width, 'sigma_bins': sigma_bins, 'voigt_gamma_bins': voigt_gamma_bins, 'afp_bin_range': (afp_lo, afp_hi), 'afp_subset': afp_bins, 'afp_efficiency': afp_efficiency, 'd_same_plus0': model.params.d_same_plus0, 'd_same_0minus': model.params.d_same_0minus, 'd_spec_plus0': model.params.d_spec_plus0, 'd_spec_0minus': model.params.d_spec_0minus, 'iplus_unburned': iplus_unburned, 'iminus_unburned': iminus_unburned, 'iplus_pre_afp': iplus_pre_afp, 'iminus_pre_afp': iminus_pre_afp, 'iplus_post_afp': ip_afp, 'iminus_post_afp': im_afp, 'iplus': iplus, 'iminus': iminus, 'q_unburned': np.sum(iplus_unburned - iminus_unburned), 'p_unburned': np.sum(iplus_unburned + iminus_unburned), 'q_pre_afp': np.sum(iplus_pre_afp - iminus_pre_afp), 'p_pre_afp': np.sum(iplus_pre_afp + iminus_pre_afp), 'q_post_afp': np.sum(ip_afp - im_afp), 'p_post_afp': np.sum(ip_afp + im_afp), 'q_final': np.sum(iplus - iminus), 'p_final': np.sum(iplus + iminus), 'model': model, 'pops_before': pops_before, 'pops_after': pops_after}

def plot_event(result, output_path):
    f = result['f']
    (ip0, im0) = (result['iplus_unburned'], result['iminus_unburned'])
    (ip1, im1) = (result['iplus_pre_afp'], result['iminus_pre_afp'])
    (ip_a, im_a) = (result['iplus_post_afp'], result['iminus_post_afp'])
    (ip2, im2) = (result['iplus'], result['iminus'])
    (fig, axes) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].step(f, ip0 + im0, color='black', linestyle='--', alpha=0.5, label='$P_s$ unburned')
    axes[0].step(f, ip0, color='tab:red', label='$I_+$ unburned', linestyle='--')
    axes[0].step(f, im0, color='tab:blue', label='$I_-$ unburned', linestyle='--')
    if AFP_ON:
        axes[0].step(f, ip1 + im1, color='tab:orange', alpha=0.85, label='$P_s$ pre-AFP')
        axes[0].step(f, ip_a + im_a, color='tab:green', alpha=0.75, label='$P_s$ after AFP (pre-relax)')
        axes[0].step(f, ip_a, color='tab:red', label='$I_+$ after AFP (pre-relax)')
        axes[0].step(f, im_a, color='tab:blue', label='$I_-$ after AFP (pre-relax)')
    axes[0].step(f, ip2 + im2, color='black', label='$P_s$ after relax')
    axes[0].step(f, ip2, color='tab:red', label='$I_+$ after ssrf')
    axes[0].step(f, im2, color='tab:blue', label='$I_-$ after ssrf')
    axes[0].set_ylabel('intensity')
    axes[0].legend(loc='upper right', fontsize=7)
    axes[0].grid(True, alpha=0.3)
    if AFP_ON:
        axes[1].step(f, ip1 - im1, color='tab:orange', alpha=0.85, label='$Q$ pre-AFP')
        axes[1].step(f, ip_a - im_a, color='tab:green', alpha=0.75, label='$Q$ after AFP (step 0)')
    axes[1].step(f, ip2 - im2, color='tab:purple', label='$Q$ after relax')
    axes[1].axhline(0.0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('$R$')
    axes[1].set_ylabel('Q profile')
    axes[1].legend(loc='upper right', fontsize=7)
    axes[1].grid(True, alpha=0.3)
    (s0, s1) = result['ssrf_bin_range']
    (a0, a1) = result['afp_bin_range']
    burn = result.get('burn_bin', s0)
    mir = result.get('mirror_bin', mirror_bin_idx(len(f), burn))
    fig.suptitle(f"ssRF Voigt burn={burn} mirror={mir} support=[{s0},{s1}] gamma_rf={result['gamma_rf']} +/-{result.get('support_half_width', '?')}  |  AFP [{a0},{a1})  |  {result['n_steps']} steps  Q: {result['q_pre_afp']:.4f} -> {result['q_final']:.4f}  |  P: {result['p_pre_afp']:.4f} -> {result['p_final']:.4f}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    result = run_event()
    out = OUT_DIR / f'{STEM}_lineshape.png'
    plot_event(result, out)
    print()
    print(f"P0={result['polarization']}  bins={len(result['f'])}")
    (s0, s1) = result['ssrf_bin_range']
    print(f"ssRF Voigt: burn={result['burn_bin']} mirror={result['mirror_bin']}  support=[{s0}, {s1}]  +/-{result['support_half_width']}  sigma={result['sigma_bins']}  voigt_gamma={result['voigt_gamma_bins']}  gamma_rf={result['gamma_rf']}  dt={result['dt']}  n_steps={result['n_steps']}")
    (a0, a1) = result['afp_bin_range']
    print(f"relaxation: {result['n_steps']} steps  dt={result['dt']}  d_same=({result['d_same_plus0']}, {result['d_same_0minus']})  d_spec=({result['d_spec_plus0']}, {result['d_spec_0minus']})")
    print(f"P: {result['p_unburned']:.6f} -> pre {result['p_pre_afp']:.6f} -> AFP {result['p_post_afp']:.6f} -> relax {result['p_final']:.6f}")
    print(f"Q: {result['q_unburned']:.6f} -> pre {result['q_pre_afp']:.6f} -> AFP {result['q_post_afp']:.6f} -> relax {result['q_final']:.6f}")
    d_relax_p = result['p_final'] - result['p_post_afp']
    d_relax_q = result['q_final'] - result['q_post_afp']
    print(f'd from relax only:  dP={d_relax_p:+.6f}  dQ={d_relax_q:+.6f}')
    (before, after) = (result['pops_before'], result['pops_after'])
    model = result['model']
    print(f"populations before: n+={before['n_plus']:.6f}  n0={before['n_zero']:.6f}  n-={before['n_minus']:.6f}  n+-n-={before['P']:.6f}")
    print(f"populations after:  n+={after['n_plus']:.6f}  n0={after['n_zero']:.6f}  n-={after['n_minus']:.6f}  n+-n-={after['n_plus'] - after['n_minus']:.6f}")
    print(f'Final Q: {model.n_plus - 2.0 * model.n_zero + model.n_minus:.6f}')
    print(f'Saved {out.name}')
    f = result['f']
    (fig, ax) = plt.subplots(figsize=(10, 5))
    ax.plot(f, model.n[:, PLUS], label='n+ (packet)')
    ax.plot(f, model.n[:, ZERO], label='n0 (packet)')
    ax.plot(f, model.n[:, MINUS], label='n- (packet)')
    ax.set_xlabel('$R$')
    ax.set_ylabel('packet population')
    ax.set_title(f'level totals: $n_+={model.n_plus:.4f}$, $n_0={model.n_zero:.4f}$, $n_-={model.n_minus:.4f}$  ($n_+-n_-={model.n_plus - model.n_minus:.4f}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / f'{STEM}_populations.png', dpi=150)
    plt.close(fig)
if __name__ == '__main__':
    main()
