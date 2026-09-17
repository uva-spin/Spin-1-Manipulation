"""
Demo plots for ssrf_realtime: baseline, ssRF burn, and AFP.

Run from repo root:
  python physics/ssrf_realtime/tests/plot_v2_manipulation_demo.py
  python physics/ssrf_realtime/tests/plot_v2_manipulation_demo.py -m single
  python physics/ssrf_realtime/tests/plot_v2_manipulation_demo.py -m voigt
  python physics/ssrf_realtime/tests/plot_v2_manipulation_demo.py -m both
  python physics/ssrf_realtime/tests/plot_v2_manipulation_demo.py --neighbor-bins
  python physics/ssrf_realtime/tests/plot_v2_manipulation_demo.py --compare-ssrf-modes
  python physics/ssrf_realtime/tests/plot_v2_manipulation_demo.py --compare-ssrf-rf-only
  python physics/ssrf_realtime/tests/plot_v2_manipulation_demo.py --compare-voigt-renorm
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime import Spin1Model, Spin1Params
from physics.ssrf_realtime.rate_equations_realtime import burn_preserves_branch_order, configure_discrete_bins_ssrf, configure_single_bin_ssrf, configure_voigt_ssrf, verify_burn_response
from physics.ssrf_realtime.rf_profile import HALF_WIDTH, SIGMA_BINS, VOIGT_GAMMA_BINS, make_voigt_rf_profile
OUTPUT_DIR = TESTS_DIR / 'output'
P = 0.48
N_BINS = 500
BURN_R = -0.92
GAMMA_RF = 1.0
DT = 0.0015
CAPACITY_POWER = 1.0
MAX_SSRF_STEPS = 1000
AFP_SUBSET = list(range(170, 200))
VOIGT_HALF_WIDTH = 15
NEIGHBOR_BINS = list(range(173, 177))

def _burn_params(*, rf_only=False):
    params = Spin1Params(p0=P, n_bins=N_BINS, r_min=-3.0, r_max=3.0, capacity_rate_power=CAPACITY_POWER, rf_burn_R=BURN_R, gamma_rf=GAMMA_RF, dt=DT)
    if rf_only:
        params = replace(params, d_same_plus0=0.0, d_same_0minus=0.0, d_spec_plus0=0.0, d_spec_0minus=0.0)
    return params

def voigt_peak_gamma_for_sum_match(*, n_bins=N_BINS, burn_idx, target_sum=GAMMA_RF):
    """Scale Voigt peak so ``sum(profile) == target_sum`` (single-bin budget)."""
    (profile, _) = make_voigt_rf_profile(n_bins, burn_idx, GAMMA_RF, sigma=SIGMA_BINS, lorentz_gamma=VOIGT_GAMMA_BINS, half_width=VOIGT_HALF_WIDTH)
    profile_sum = profile.sum()
    if profile_sum <= 0.0:
        return target_sum
    return target_sum * GAMMA_RF / profile_sum

def load_equilibrium_model(*, rf_only=False):
    f = np.linspace(-3.0, 3.0, N_BINS)
    (_, ip, im) = GenerateVectorLineshape(P, f)
    model = Spin1Model(_burn_params(rf_only=rf_only))
    model.load_from_physical_intensities(ip, im)
    return (model, f, ip, im)

def plot_baseline(f, ip, im, out):
    (fig, ax) = plt.subplots(figsize=(8.0, 4.0), layout='constrained')
    ax.plot(f, ip, color='tab:red', linewidth=1.2, label='$I_+$')
    ax.plot(f, im, color='tab:blue', linewidth=1.2, label='$I_-$')
    ax.plot(f, ip + im, color='black', linewidth=0.9, linestyle='--', alpha=0.7, label='$P_s$')
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('intensity')
    ax.set_title(f'v2 baseline (no manipulation)  |  P = {P:.2f}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def _configure_ssrf(model, burn_idx, mode, *, gamma_rf=None, voigt_sum_normalize=False):
    peak_gamma = GAMMA_RF if gamma_rf is None else gamma_rf
    if mode == 'voigt' and voigt_sum_normalize:
        peak_gamma = voigt_peak_gamma_for_sum_match(burn_idx=burn_idx, target_sum=peak_gamma)
    model.params.rf_enabled = True
    model.params.gamma_rf = peak_gamma
    if mode == 'single':
        configure_single_bin_ssrf(model, burn_idx, peak_gamma)
        return [burn_idx]
    if mode == 'voigt':
        return configure_voigt_ssrf(model, burn_idx, peak_gamma, sigma=SIGMA_BINS, voigt_gamma=VOIGT_GAMMA_BINS, half_width=VOIGT_HALF_WIDTH, full_spectrum_recovery=True)
    raise ValueError(f'unknown ssRF mode {mode!r}')

def _run_ssrf_burn(mode, *, max_steps=None, early_stop=True, gamma_rf=None, rf_only=False, voigt_sum_normalize=False):
    """Run one ssRF burn demo and return final state plus burn-bin traces."""
    steps_limit = MAX_SSRF_STEPS if max_steps is None else max_steps
    (model, f, ip0, im0) = load_equilibrium_model(rf_only=rf_only)
    burn_idx = model.burn_index(BURN_R)
    mirror_idx = len(f) - 1 - burn_idx
    support = _configure_ssrf(model, burn_idx, mode, gamma_rf=gamma_rf, voigt_sum_normalize=voigt_sum_normalize)
    peak_gamma = model.params.gamma_rf
    traces = {'iplus_burn': [], 'iminus_burn': [], 'iplus_mirror': [], 'iminus_mirror': []}
    (ip_ref, im_ref, _) = model.physical_intensities()
    for key in traces:
        idx = burn_idx if 'burn' in key else mirror_idx
        arr = ip_ref if 'iplus' in key else im_ref
        traces[key].append(arr[idx])
    steps_applied = 0
    (ip, im) = (ip_ref, im_ref)
    for step in range(steps_limit):
        model.step_once(dt=DT, rf_on=True, dnp_on=False)
        (ip, im, _) = model.physical_intensities()
        traces['iplus_burn'].append(ip[burn_idx])
        traces['iminus_burn'].append(im[burn_idx])
        traces['iplus_mirror'].append(ip[mirror_idx])
        traces['iminus_mirror'].append(im[mirror_idx])
        steps_applied = step + 1
        if early_stop and step > 5:
            prev = traces['iplus_burn'][-2]
            cur = traces['iplus_burn'][-1]
            if abs(cur - prev) < 1e-08 * max(abs(prev), 1e-12):
                break
    branch_ok = burn_preserves_branch_order(ip0, im0, ip, im, burn_idx)
    return {'mode': mode, 'f': f, 'ip0': ip0, 'im0': im0, 'ip': ip, 'im': im, 'burn_idx': burn_idx, 'mirror_idx': mirror_idx, 'support': support, 'steps_applied': steps_applied, 'max_steps': steps_limit, 'gamma_rf': peak_gamma, 'profile_sum': np.sum(model.params.rf_profile), 'branch_ok': branch_ok, 'im_ge_ip': im[burn_idx] >= ip[burn_idx], 'rf_profile': np.asarray(model.params.rf_profile), 'traces': {k: np.asarray(v) for (k, v) in traces.items()}, 'rf_only': rf_only, 'voigt_sum_normalize': voigt_sum_normalize}

def run_ssrf_demo(f, ip0, im0, out_lineshape, out_trajectory, *, mode, out_profile=None):
    run = _run_ssrf_burn(mode)
    f = run['f']
    ip0 = run['ip0']
    im0 = run['im0']
    ip = run['ip']
    im = run['im']
    burn_idx = run['burn_idx']
    mirror_idx = run['mirror_idx']
    support = run['support']
    steps_applied = run['steps_applied']
    branch_order_ok = run['branch_ok']
    im_ge_ip = run['im_ge_ip']
    rf_profile = run['rf_profile']
    traces = run['traces']
    check = verify_burn_response(ip0, im0, ip, im, burn_idx, rtol=0.12)
    d_ip_burn = check['d_iplus_burn']
    d_im_burn = check['d_iminus_burn']
    d_ip_mirror = check['d_iplus_mirror']
    d_im_mirror = check['d_iminus_mirror']
    ratio_ip_mirror_over_im_burn = abs(d_ip_mirror) / max(abs(d_im_burn), 1e-30)
    ratio_im_mirror_over_ip_burn = abs(d_im_mirror) / max(abs(d_ip_burn), 1e-30)
    mode_label = 'single-bin' if mode == 'single' else 'Voigt multi-bin'
    print(f'\nssRF burn checks ({mode_label}):')
    print(f'  RF center idx={burn_idx}  support={support[0]}..{support[-1]} ({len(support)} bins)')
    print(f'  sigma={SIGMA_BINS}  voigt_gamma={VOIGT_GAMMA_BINS}  half_width={VOIGT_HALF_WIDTH}  peak_gamma={GAMMA_RF}')
    print(f'  steps applied: {steps_applied}')
    print(f'  branch order preserved: {branch_order_ok}')
    print(f'  I- >= I+ at burn: {im_ge_ip}  (I-={im[burn_idx]:.6e}, I+={ip[burn_idx]:.6e})')
    print(f'  burn idx={burn_idx}, mirror idx={mirror_idx}')
    print(f'  dI+_burn={d_ip_burn:.6e}, dI-_burn={d_im_burn:.6e}')
    print(f'  dI+_mirror={d_ip_mirror:.6e}, dI-_mirror={d_im_mirror:.6e}')
    print(f'  |dI+_mirror|/|dI-_burn| = {ratio_ip_mirror_over_im_burn:.4f} (target ~0.5)')
    print(f'  |dI-_mirror|/|dI+_burn| = {ratio_im_mirror_over_ip_burn:.4f} (target ~0.5)')
    print(f"  amp_burn/amp_mirror = {check['ratios']['amp_burn_over_amp_mirror']:.4f} (target ~2.0)")
    if out_profile is not None:
        (fig, ax) = plt.subplots(figsize=(8.0, 3.2), layout='constrained')
        ax.plot(f, rf_profile, color='green', linewidth=1.4, label='$\\Gamma_{\\mathrm{RF}}(R)$')
        ax.axvline(BURN_R, color='black', linestyle=':', alpha=0.5, label='burn center')
        ax.axvspan(f[support[0]], f[support[-1]], color='green', alpha=0.12, label='RF support')
        ax.set_xlabel('frequency (MHz)')
        ax.set_ylabel('RF rate $\\Gamma_{\\mathrm{RF}}$')
        ax.set_title(f'Voigt RF profile  |  sigma={SIGMA_BINS}  gamma={VOIGT_GAMMA_BINS}  half_width={VOIGT_HALF_WIDTH}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        out_profile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_profile, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out_profile}')
    (fig, ax) = plt.subplots(figsize=(8.0, 4.0), layout='constrained')
    ax.plot(f, ip0, color='tab:red', alpha=0.35, linestyle='--', linewidth=1.0, label='$I_+$ before')
    ax.plot(f, im0, color='tab:blue', alpha=0.35, linestyle='--', linewidth=1.0, label='$I_-$ before')
    ax.plot(f, ip, color='tab:red', linewidth=1.2, label='$I_+$ after ssRF')
    ax.plot(f, im, color='tab:blue', linewidth=1.2, label='$I_-$ after ssRF')
    ax.axvline(BURN_R, color='green', linestyle=':', alpha=0.6, label='RF center')
    if mode == 'voigt':
        ax.axvspan(f[support[0]], f[support[-1]], color='green', alpha=0.08)
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('intensity')
    ax.set_title(f'ssRF burn ({mode_label})  |  {steps_applied} steps  |  gamma_rf={GAMMA_RF}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    out_lineshape.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_lineshape, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_lineshape}')
    steps = np.arange(len(traces['iplus_burn']))
    (fig, ax) = plt.subplots(figsize=(8.0, 4.2), layout='constrained')
    ax.plot(steps, traces['iplus_burn'], color='tab:red', label='$I_+$ burn')
    ax.plot(steps, traces['iminus_burn'], color='tab:blue', label='$I_-$ burn')
    ax.plot(steps, traces['iplus_mirror'], color='tab:orange', linestyle='--', label='$I_+$ mirror')
    ax.plot(steps, traces['iminus_mirror'], color='tab:cyan', linestyle='--', label='$I_-$ mirror')
    ax.set_xlabel('RF integration step')
    ax.set_ylabel('intensity')
    ax.set_title(f'ssRF burn-location intensities ({mode_label})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    fig.savefig(out_trajectory, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_trajectory}')

def run_afp_demo(model, f, out):
    model.params.afp_subset_indices = AFP_SUBSET
    (ip_before, im_before, _) = model.physical_intensities()
    P_before = model.level_populations()['P']
    model.afp_sweep()
    (ip_after, im_after, _) = model.physical_intensities()
    P_after = model.level_populations()['P']
    dP = P_after - P_before
    print('\nAFP checks:')
    print(f'  afp_subset = [{AFP_SUBSET[0]}, ..., {AFP_SUBSET[-1]}]  ({len(AFP_SUBSET)} bins)')
    print(f'  P_before = {P_before:.8f}')
    print(f'  P_after  = {P_after:.8f}')
    print(f'  |dP|     = {abs(dP):.3e}')
    ps_before = ip_before + im_before
    ps_after = ip_after + im_after
    (fig, ax) = plt.subplots(figsize=(8.0, 4.0), layout='constrained')
    ax.plot(f, ps_before, color='black', linewidth=1.25, linestyle='--', alpha=0.75, label='$P_s = I_+ + I_-$ before AFP')
    ax.plot(f, ps_after, color='tab:green', linewidth=1.35, label='$P_s = I_+ + I_-$ after AFP')
    if AFP_SUBSET:
        ax.axvspan(f[AFP_SUBSET[0]], f[AFP_SUBSET[-1]], color='C5', alpha=0.15, label='AFP sweep window')
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('total intensity $P_s$')
    ax.set_title('Total lineshape before / after AFP')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(f'v2 AFP  |  P: {P_before:.4f} -> {P_after:.4f}  |  |dP|={abs(dP):.2e}', fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def _run_voigt_burn(*, capacity_weighting='center_shared'):
    (model, f, ip0, im0) = load_equilibrium_model()
    burn_idx = model.burn_index(BURN_R)
    mirror_idx = len(f) - 1 - burn_idx
    configure_voigt_ssrf(model, burn_idx, GAMMA_RF, sigma=SIGMA_BINS, voigt_gamma=VOIGT_GAMMA_BINS, half_width=VOIGT_HALF_WIDTH, full_spectrum_recovery=True, capacity_weighting=capacity_weighting)
    traces = {'iplus_burn': [], 'iminus_burn': [], 'iplus_mirror': [], 'iminus_mirror': []}
    (ip_ref, im_ref, _) = model.physical_intensities()
    for key in traces:
        idx = burn_idx if 'burn' in key else mirror_idx
        arr = ip_ref if 'iplus' in key else im_ref
        traces[key].append(arr[idx])
    for _ in range(MAX_SSRF_STEPS):
        model.step_once(dt=DT, rf_on=True, dnp_on=False)
        (ip, im, _) = model.physical_intensities()
        traces['iplus_burn'].append(ip[burn_idx])
        traces['iminus_burn'].append(im[burn_idx])
        traces['iplus_mirror'].append(ip[mirror_idx])
        traces['iminus_mirror'].append(im[mirror_idx])
    branch_ok = burn_preserves_branch_order(ip0, im0, ip, im, burn_idx)
    return {'f': f, 'ip0': ip0, 'im0': im0, 'ip': ip, 'im': im, 'burn_idx': burn_idx, 'mirror_idx': mirror_idx, 'branch_ok': branch_ok, 'im_ge_ip': im[burn_idx] >= ip[burn_idx], 'traces': {k: np.asarray(v) for (k, v) in traces.items()}}

def run_neighbor_bins_demo(bins=NEIGHBOR_BINS, *, out_lineshape=None, out_trajectory=None, out_profile=None):
    """Apply flat single-bin-style RF to neighboring bins (default 173, 174).

    Example rates: full ``GAMMA_RF`` on the first bin, half on the second.
    """
    support = [i for i in bins]
    if len(support) < 2:
        raise ValueError('neighbor-bins demo needs at least two bin indices')
    burn_idx = support[0]
    gamma_rates = [GAMMA_RF] + [0.5 * GAMMA_RF] * (len(support) - 1)
    (model, f, ip0, im0) = load_equilibrium_model()
    mirror_idx = len(f) - 1 - burn_idx
    configure_discrete_bins_ssrf(model, support, gamma_rates)
    traces = {'iplus_burn': [], 'iminus_burn': [], 'iplus_neighbor': [], 'iminus_neighbor': [], 'iplus_mirror': [], 'iminus_mirror': []}
    (ip_ref, im_ref, _) = model.physical_intensities()
    traces['iplus_burn'].append(ip_ref[burn_idx])
    traces['iminus_burn'].append(im_ref[burn_idx])
    traces['iplus_neighbor'].append(ip_ref[support[1]])
    traces['iminus_neighbor'].append(im_ref[support[1]])
    traces['iplus_mirror'].append(ip_ref[mirror_idx])
    traces['iminus_mirror'].append(im_ref[mirror_idx])
    steps_applied = 0
    (ip, im) = (ip_ref, im_ref)
    for step in range(MAX_SSRF_STEPS):
        model.step_once(dt=DT, rf_on=True, dnp_on=False)
        (ip, im, _) = model.physical_intensities()
        traces['iplus_burn'].append(ip[burn_idx])
        traces['iminus_burn'].append(im[burn_idx])
        traces['iplus_neighbor'].append(ip[support[1]])
        traces['iminus_neighbor'].append(im[support[1]])
        traces['iplus_mirror'].append(ip[mirror_idx])
        traces['iminus_mirror'].append(im[mirror_idx])
        steps_applied = step + 1
        if step > 5:
            prev = traces['iplus_burn'][-2]
            cur = traces['iplus_burn'][-1]
            if abs(cur - prev) < 1e-08 * max(abs(prev), 1e-12):
                break
    branch_ok = burn_preserves_branch_order(ip0, im0, ip, im, burn_idx)
    rf_profile = np.asarray(model.params.rf_profile)
    print(f'\nssRF burn checks (single-bin style on neighboring bins {support}):')
    print(f'  RF bins={support}  R={[f[i] for i in support]}')
    print(f'  gamma_rf per bin={gamma_rates}  sum_gamma={rf_profile.sum():.4g}')
    print(f'  steps applied: {steps_applied}')
    print(f'  branch order preserved at {burn_idx}: {branch_ok}')
    for (i, g) in zip(support, gamma_rates):
        check = verify_burn_response(ip0, im0, ip, im, i, rtol=0.12)
        d_ip_burn = check['d_iplus_burn']
        d_im_burn = check['d_iminus_burn']
        d_ip_mirror = check['d_iplus_mirror']
        d_im_mirror = check['d_iminus_mirror']
        ratio_ip_mirror_over_im_burn = abs(d_ip_mirror) / max(abs(d_im_burn), 1e-30)
        ratio_im_mirror_over_ip_burn = abs(d_im_mirror) / max(abs(d_ip_burn), 1e-30)
        mirror_i = check['mirror_idx']
        print(f'  bin {i} (gamma={g:.4g}, mirror={mirror_i}): I+={ip[i]:.6e}  I-={im[i]:.6e}  dI+={d_ip_burn:.6e}  dI-={d_im_burn:.6e}')
        print(f'    dI+_mirror={d_ip_mirror:.6e}  dI-_mirror={d_im_mirror:.6e}')
        print(f'    |dI+_mirror|/|dI-_burn| = {ratio_ip_mirror_over_im_burn:.4f} (target ~0.5)')
        print(f'    |dI-_mirror|/|dI+_burn| = {ratio_im_mirror_over_ip_burn:.4f} (target ~0.5)')
        print(f"    amp_burn/amp_mirror = {check['ratios']['amp_burn_over_amp_mirror']:.4f} (target ~2.0)")
    out_lineshape = out_lineshape or OUTPUT_DIR / 'v2_neighbor_bins_lineshape.png'
    out_trajectory = out_trajectory or OUTPUT_DIR / 'v2_neighbor_bins_trajectory.png'
    out_profile = out_profile or OUTPUT_DIR / 'v2_neighbor_bins_rf_profile.png'
    (fig, ax) = plt.subplots(figsize=(8.0, 3.2), layout='constrained')
    ax.plot(f, rf_profile, color='green', linewidth=1.4, label='$\\Gamma_{\\mathrm{RF}}(R)$')
    for i in support:
        ax.axvline(f[i], color='black', linestyle=':', alpha=0.45)
    ax.axvspan(f[support[0]], f[support[-1]], color='green', alpha=0.12, label='RF support')
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('RF rate $\\Gamma_{\\mathrm{RF}}$')
    ax.set_title(f'Flat single-bin RF on bins {support}  |  gamma={gamma_rates}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    out_profile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_profile, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_profile}')
    (fig, ax) = plt.subplots(figsize=(8.0, 4.0), layout='constrained')
    ax.plot(f, ip0, color='tab:red', alpha=0.35, linestyle='--', linewidth=1.0, label='$I_+$ before')
    ax.plot(f, im0, color='tab:blue', alpha=0.35, linestyle='--', linewidth=1.0, label='$I_-$ before')
    ax.plot(f, ip, color='tab:red', linewidth=1.2, label='$I_+$ after ssRF')
    ax.plot(f, im, color='tab:blue', linewidth=1.2, label='$I_-$ after ssRF')
    for i in support:
        ax.axvline(f[i], color='green', linestyle=':', alpha=0.55)
    ax.axvspan(f[support[0]], f[support[-1]], color='green', alpha=0.08)
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('intensity')
    ax.set_title(f'ssRF burn (bins {support})  |  {steps_applied} steps  |  gamma={gamma_rates}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    out_lineshape.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_lineshape, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_lineshape}')
    steps = np.arange(len(traces['iplus_burn']))
    (fig, ax) = plt.subplots(figsize=(8.0, 4.2), layout='constrained')
    ax.plot(steps, traces['iplus_burn'], color='tab:red', label=f'$I_+$ bin {burn_idx}')
    ax.plot(steps, traces['iminus_burn'], color='tab:blue', label=f'$I_-$ bin {burn_idx}')
    ax.plot(steps, traces['iplus_neighbor'], color='tab:orange', label=f'$I_+$ bin {support[1]}')
    ax.plot(steps, traces['iminus_neighbor'], color='tab:cyan', label=f'$I_-$ bin {support[1]}')
    ax.plot(steps, traces['iplus_mirror'], color='tab:red', linestyle='--', alpha=0.7, label=f'$I_+$ mirror({burn_idx})')
    ax.plot(steps, traces['iminus_mirror'], color='tab:blue', linestyle='--', alpha=0.7, label=f'$I_-$ mirror({burn_idx})')
    ax.set_xlabel('RF integration step')
    ax.set_ylabel('intensity')
    ax.set_title(f'ssRF intensities at neighboring bins {support}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    fig.savefig(out_trajectory, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_trajectory}')

def compare_voigt_capacity_weights():
    """Side-by-side Voigt burns: center-shared vs separate w[kp]/w[km] per pair."""
    shared = _run_voigt_burn(capacity_weighting='center_shared')
    separate = _run_voigt_burn(capacity_weighting='separate_branch')
    f = shared['f']
    burn_idx = shared['burn_idx']
    print('\nVoigt capacity-weighting comparison (full-spectrum recovery):')
    for (label, run) in (('center_shared (current)', shared), ('separate_branch (w[kp], w[km])', separate)):
        ip = run['ip']
        im = run['im']
        print(f"  {label}:  I+={ip[burn_idx]:.4e}  I-={im[burn_idx]:.4e}  I->=I+={run['im_ge_ip']}  branch_order={run['branch_ok']}")
    steps = np.arange(len(shared['traces']['iplus_burn']))
    (fig, axes) = plt.subplots(1, 2, figsize=(12.0, 4.2), layout='constrained')
    ax = axes[0]
    ax.plot(steps, shared['traces']['iplus_burn'], color='tab:red', label='$I_+$ burn (shared)')
    ax.plot(steps, shared['traces']['iminus_burn'], color='tab:blue', label='$I_-$ burn (shared)')
    ax.plot(steps, separate['traces']['iplus_burn'], color='tab:red', linestyle='--', label='$I_+$ burn (separate)')
    ax.plot(steps, separate['traces']['iminus_burn'], color='tab:blue', linestyle='--', label='$I_-$ burn (separate)')
    ax.set_xlabel('RF step')
    ax.set_ylabel('intensity at burn bin')
    ax.set_title('Burn-bin trajectory')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(f, shared['ip0'], color='tab:red', alpha=0.25, linestyle=':', label='$I_+$ before')
    ax.plot(f, shared['im0'], color='tab:blue', alpha=0.25, linestyle=':', label='$I_-$ before')
    ax.plot(f, shared['ip'], color='tab:red', linewidth=1.3, label='$I_+$ after (shared)')
    ax.plot(f, shared['im'], color='tab:blue', linewidth=1.3, label='$I_-$ after (shared)')
    ax.plot(f, separate['ip'], color='tab:red', linewidth=1.0, linestyle='--', label='$I_+$ after (separate)')
    ax.plot(f, separate['im'], color='tab:blue', linewidth=1.0, linestyle='--', label='$I_-$ after (separate)')
    ax.axvline(BURN_R, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('intensity')
    ax.set_title('Lineshape after Voigt burn')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')
    out = OUTPUT_DIR / 'v2_voigt_capacity_compare.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def _print_ssrf_run_summary(label, run, burn_idx):
    ip = run['ip']
    im = run['im']
    support = run['support']
    print(f"  {label}:  support={support[0]}..{support[-1]} ({len(support)} bins)  peak_gamma={run['gamma_rf']:.4g}  sum_gamma={run['profile_sum']:.4g}  steps={run['steps_applied']}/{run['max_steps']}  I+={ip[burn_idx]:.4e}  I-={im[burn_idx]:.4e}  Ps={ip[burn_idx] + im[burn_idx]:.4e}  I->=I+={run['im_ge_ip']}  branch_order={run['branch_ok']}")

def _plot_ssrf_mode_comparison(runs, *, out, title_suffix, trajectory_title, lineshape_title):
    f = runs[0][1]['f']
    step_lens = {len(run['traces']['iplus_burn']) for (_, run) in runs}
    if len(step_lens) != 1:
        raise ValueError(f'runs have mismatched trace lengths: {step_lens}')
    steps = np.arange(next(iter(step_lens)))
    (fig, axes) = plt.subplots(1, 2, figsize=(12.0, 4.2), layout='constrained')
    linestyles = ('-', '--', '-.')
    colors_ip = ('tab:red', 'tab:red', 'tab:red')
    colors_im = ('tab:blue', 'tab:blue', 'tab:blue')
    ax = axes[0]
    for (idx, (label, run)) in enumerate(runs):
        ls = linestyles[idx % len(linestyles)]
        traces = run['traces']
        ax.plot(steps, traces['iplus_burn'], color=colors_ip[idx % len(colors_ip)], linestyle=ls, label=f'$I_+$ burn ({label})')
        ax.plot(steps, traces['iminus_burn'], color=colors_im[idx % len(colors_im)], linestyle=ls, label=f'$I_-$ burn ({label})')
    ax.set_xlabel('RF step')
    ax.set_ylabel('intensity at burn bin')
    ax.set_title(trajectory_title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    ip0 = runs[0][1]['ip0']
    im0 = runs[0][1]['im0']
    ax.plot(f, ip0, color='tab:red', alpha=0.2, linestyle=':', label='$I_+$ before')
    ax.plot(f, im0, color='tab:blue', alpha=0.2, linestyle=':', label='$I_-$ before')
    for (idx, (label, run)) in enumerate(runs):
        ls = linestyles[idx % len(linestyles)]
        ax.plot(f, run['ip'], color='tab:red', linewidth=1.2, linestyle=ls, label=f'$I_+$ after ({label})')
        ax.plot(f, run['im'], color='tab:blue', linewidth=1.2, linestyle=ls, label=f'$I_-$ after ({label})')
    ax.axvline(BURN_R, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('intensity')
    ax.set_title(lineshape_title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')
    fig.suptitle(title_suffix, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def compare_ssrf_modes():
    """Side-by-side single-bin vs Voigt multi-bin ssRF burns at matched power and steps."""
    compare_gamma = GAMMA_RF
    compare_steps = MAX_SSRF_STEPS
    burn_kwargs = {'max_steps': compare_steps, 'early_stop': False, 'gamma_rf': compare_gamma}
    single = _run_ssrf_burn('single', **burn_kwargs)
    voigt = _run_ssrf_burn('voigt', **burn_kwargs)
    burn_idx = single['burn_idx']
    print('\nssRF mode comparison (single-bin vs Voigt multi-bin):')
    print(f'  matched peak: gamma_rf={compare_gamma}  steps={compare_steps}  dt={DT}')
    for (label, run) in (('single-bin', single), ('Voigt peak-matched', voigt)):
        _print_ssrf_run_summary(label, run, burn_idx)
    _plot_ssrf_mode_comparison([('single', single), ('Voigt', voigt)], out=OUTPUT_DIR / 'v2_ssrf_mode_compare.png', title_suffix=f'ssRF comparison with recovery  |  {compare_steps} steps  |  peak gamma_rf={compare_gamma}', trajectory_title=f'Burn-bin trajectory  |  peak gamma_rf={compare_gamma}', lineshape_title=f'Lineshape after ssRF burn  |  peak gamma_rf={compare_gamma}')

def compare_ssrf_modes_rf_only():
    """Single-bin vs Voigt with recovery/diffusion disabled (RF-only dynamics)."""
    compare_steps = MAX_SSRF_STEPS
    burn_kwargs = {'max_steps': compare_steps, 'early_stop': False, 'gamma_rf': GAMMA_RF, 'rf_only': True}
    single = _run_ssrf_burn('single', **burn_kwargs)
    voigt = _run_ssrf_burn('voigt', **burn_kwargs)
    burn_idx = single['burn_idx']
    print('\nssRF mode comparison (RF only, recovery/diffusion off):')
    print(f'  matched peak: gamma_rf={GAMMA_RF}  steps={compare_steps}  dt={DT}')
    for (label, run) in (('single-bin', single), ('Voigt peak-matched', voigt)):
        _print_ssrf_run_summary(label, run, burn_idx)
    _plot_ssrf_mode_comparison([('single', single), ('Voigt', voigt)], out=OUTPUT_DIR / 'v2_ssrf_mode_compare_rf_only.png', title_suffix=f'ssRF comparison, RF only (no recovery/diffusion)  |  {compare_steps} steps  |  peak gamma_rf={GAMMA_RF}', trajectory_title='Burn-bin trajectory  |  center RF rate matched', lineshape_title='Lineshape after ssRF burn  |  center RF rate matched')

def compare_voigt_renorm():
    """Compare peak-matched vs sum-normalized Voigt against single-bin (recovery on)."""
    compare_steps = MAX_SSRF_STEPS
    burn_kwargs = {'max_steps': compare_steps, 'early_stop': False, 'gamma_rf': GAMMA_RF}
    single = _run_ssrf_burn('single', **burn_kwargs)
    voigt_peak = _run_ssrf_burn('voigt', **burn_kwargs)
    voigt_renorm = _run_ssrf_burn('voigt', **burn_kwargs, voigt_sum_normalize=True)
    burn_idx = single['burn_idx']
    renorm_peak = voigt_renorm['gamma_rf']
    print('\nVoigt renormalization comparison (recovery on):')
    print(f'  steps={compare_steps}  dt={DT}')
    print(f"  single-bin sum(profile)={single['profile_sum']:.4g}")
    print(f"  Voigt peak-matched: peak={voigt_peak['gamma_rf']:.4g}  sum={voigt_peak['profile_sum']:.4g}")
    print(f"  Voigt sum-normalized: peak={renorm_peak:.4g}  sum={voigt_renorm['profile_sum']:.4g}  (target sum={GAMMA_RF:.4g})")
    for (label, run) in (('single-bin', single), ('Voigt peak-matched', voigt_peak), ('Voigt sum-normalized', voigt_renorm)):
        _print_ssrf_run_summary(label, run, burn_idx)
    _plot_ssrf_mode_comparison([('single', single), ('Voigt peak', voigt_peak), ('Voigt renorm', voigt_renorm)], out=OUTPUT_DIR / 'v2_ssrf_mode_compare_voigt_renorm.png', title_suffix=f'Voigt renormalization  |  {compare_steps} steps  |  renorm peak={renorm_peak:.3f}  (sum profile = {GAMMA_RF:.1f})', trajectory_title=f"Burn-bin trajectory  |  single sum={single['profile_sum']:.2f}, Voigt renorm peak={renorm_peak:.3f}", lineshape_title='Lineshape after ssRF burn  |  recovery on')
    (fig, ax) = plt.subplots(figsize=(8.0, 3.2), layout='constrained')
    f = single['f']
    ax.plot(f, single['rf_profile'], color='black', linewidth=1.4, label='single-bin')
    ax.plot(f, voigt_peak['rf_profile'], color='tab:green', linewidth=1.2, linestyle='--', label=f"Voigt peak-matched (sum={voigt_peak['profile_sum']:.2f})")
    ax.plot(f, voigt_renorm['rf_profile'], color='tab:orange', linewidth=1.2, linestyle='-.', label=f'Voigt sum-normalized (peak={renorm_peak:.3f})')
    ax.axvline(BURN_R, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('frequency (MHz)')
    ax.set_ylabel('RF rate $\\Gamma_{\\mathrm{RF}}$')
    ax.set_title('RF profiles: peak-matched vs sum-normalized Voigt')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    out_profile = OUTPUT_DIR / 'v2_voigt_renorm_rf_profile.png'
    out_profile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_profile, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_profile}')

def _parse_args():
    p = argparse.ArgumentParser(description='ssrf_realtime manipulation demo plots')
    p.add_argument('-m', '--ssrf-mode', choices=('single', 'voigt', 'both'), default='both', help='ssRF profile: single-bin, Voigt multi-bin, or both (default: both)')
    p.add_argument('--neighbor-bins', action='store_true', help='Flat single-bin-style RF on neighboring bins 173 and 174')
    p.add_argument('--compare-ssrf-modes', action='store_true', help='Overlay single-bin vs Voigt multi-bin ssRF burn comparison')
    p.add_argument('--compare-ssrf-rf-only', action='store_true', help='Compare single-bin vs Voigt with recovery/diffusion disabled')
    p.add_argument('--compare-voigt-renorm', action='store_true', help='Compare peak-matched vs sum-normalized Voigt against single-bin')
    p.add_argument('--compare-voigt-weights', action='store_true', help='Compare Voigt center-shared vs separate w[kp]/w[km] capacity weighting')
    p.add_argument('--skip-afp', action='store_true', help='Skip AFP demo plot')
    return p.parse_args()

def main():
    args = _parse_args()
    if args.neighbor_bins:
        run_neighbor_bins_demo()
        return
    if args.compare_voigt_weights:
        compare_voigt_capacity_weights()
        return
    if args.compare_ssrf_rf_only:
        compare_ssrf_modes_rf_only()
        return
    if args.compare_voigt_renorm:
        compare_voigt_renorm()
        return
    if args.compare_ssrf_modes:
        compare_ssrf_modes()
        return
    (model, f, ip0, im0) = load_equilibrium_model()
    plot_baseline(f, ip0, im0, OUTPUT_DIR / 'v2_baseline_lineshape.png')
    if args.ssrf_mode == 'both':
        modes = ('single', 'voigt')
    else:
        modes = (args.ssrf_mode,)
    for mode in modes:
        prefix = 'v2_ssrf' if mode == 'single' else 'v2_voigt_ssrf'
        (_, f2, ip2, im2) = load_equilibrium_model()
        run_ssrf_demo(f2, ip2, im2, OUTPUT_DIR / f'{prefix}_lineshape.png', OUTPUT_DIR / f'{prefix}_trajectory.png', mode=mode, out_profile=OUTPUT_DIR / f'{prefix}_rf_profile.png' if mode == 'voigt' else None)
    if not args.skip_afp:
        (afp_model, f3, _, _) = load_equilibrium_model()
        run_afp_demo(afp_model, f3, OUTPUT_DIR / 'v2_afp_before_after.png')
if __name__ == '__main__':
    main()
