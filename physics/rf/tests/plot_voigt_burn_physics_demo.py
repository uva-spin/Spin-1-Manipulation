"""
Example burn plots using the same setup as spin1_ssrf_realtime_voigt_burn.

Uses analytic Boltzmann initialization (not GenerateVectorLineshape), default
701-bin grid, physical Voigt RF, and population-dependent spin diffusion during
both RF and recovery.  Burn durations are chosen to preserve I-(R) >= I+(R) at
the burn center in the moderate-power regime.

Run from repo root:
  python physics/ssrf_realtime/tests/plot_voigt_burn_physics_demo.py
  python physics/ssrf_realtime/tests/plot_voigt_burn_physics_demo.py --all
  python physics/ssrf_realtime/tests/plot_voigt_burn_physics_demo.py --single-bin
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent.parent
VOIGT_BURN_ROOT = REPO_ROOT / 'physics' / 'spin1_ssrf_realtime_voigt_burn'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VOIGT_BURN_ROOT) not in sys.path:
    sys.path.insert(0, str(VOIGT_BURN_ROOT))
from physics.ssrf_realtime import Spin1Model
from physics.ssrf_realtime.rate_equations_realtime import configure_single_bin_ssrf, configure_voigt_burn_spectral_recovery, configure_voigt_ssrf, create_voigt_burn_model, voigt_burn_params
from physics.ssrf_realtime.voigt_physical import approximate_voigt_fwhm
OUTPUT_DIR = TESTS_DIR / 'output'
GAUSS_FWHM_R = 0.03
LORENTZ_FWHM_R = 0.015
BROAD_GAUSS_FWHM_R = 0.12
BROAD_LORENTZ_FWHM_R = 0.04
BURN_RF_STEPS = 100
BURN_GAMMA_RF = 2.0
WIDTH_COMPARE_STEPS = 50
_ACTIVE_GAUSS_FWHM_R = GAUSS_FWHM_R
_ACTIVE_LORENTZ_FWHM_R = LORENTZ_FWHM_R
_SINGLE_BIN_MODE = False

def _rf_mode_label():
    return 'single-bin' if _SINGLE_BIN_MODE else 'physical Voigt'

def _out_name(stem):
    if _SINGLE_BIN_MODE:
        return f'{stem}_single_bin.png'
    return f'{stem}.png'

def _branch_order_ok(model, R=None):
    loc = model.local_intensities(R)
    return (loc['Iplus'], loc['Iminus'], loc['Iminus'] >= loc['Iplus'])

def _spectrum(model):
    (R, Ip, Im, total) = model.spectrum()
    return (R, Ip, Im, total)

def _run_voigt_burn_model(*, gamma_rf=BURN_GAMMA_RF, gaussian_fwhm_R=None, lorentzian_fwhm_R=None, diffusion_scale=None, rf_burn_R=None, dt=None, max_steps=BURN_RF_STEPS, recovery_steps=0, legacy_discrete=False, single_bin=None):
    if single_bin is None:
        single_bin = _SINGLE_BIN_MODE
    if single_bin and legacy_discrete:
        raise ValueError('single_bin and legacy_discrete are mutually exclusive')
    if gaussian_fwhm_R is None:
        gaussian_fwhm_R = _ACTIVE_GAUSS_FWHM_R
    if lorentzian_fwhm_R is None:
        lorentzian_fwhm_R = _ACTIVE_LORENTZ_FWHM_R
    overrides = {'gamma_rf': gamma_rf, 'rf_gaussian_fwhm_R': gaussian_fwhm_R, 'rf_lorentzian_fwhm_R': lorentzian_fwhm_R}
    if diffusion_scale is not None:
        overrides['diffusion_scale'] = diffusion_scale
    if rf_burn_R is not None:
        overrides['rf_burn_R'] = rf_burn_R
    if dt is not None:
        overrides['dt'] = dt
    model = create_voigt_burn_model(**overrides)
    burn_R = model.params.rf_burn_R
    burn_idx = model.burn_index(burn_R)
    mirror_idx = len(model.Rplus) - 1 - burn_idx
    if legacy_discrete:
        model.params.use_physical_voigt_rf = False
        configure_voigt_burn_spectral_recovery(model)
        configure_voigt_ssrf(model, burn_idx, gamma_rf, half_width=15, full_spectrum_recovery=True)
    else:
        configure_voigt_burn_spectral_recovery(model)
        if single_bin:
            configure_single_bin_ssrf(model, burn_idx, gamma_rf, apply_demo_recovery=False)
    (R0, ip0, im0, _) = _spectrum(model)
    if single_bin or legacy_discrete:
        Rprof = model.Rplus
        rf_physical = np.asarray(model.params.rf_profile)
        summary = {'approx_fwhm_R': model.dR, 'peak_rate': np.max(rf_physical), 'normalization': 'single_bin' if single_bin else 'legacy_discrete'}
    else:
        (Rprof, rf_physical) = model.rf_profile_physical()
        summary = model.rf_profile_summary()
    traces = {k: [] for k in ('iplus_burn', 'iminus_burn', 'iplus_mirror', 'iminus_mirror', 'P')}
    (ip0_b, im0_b, ok0) = _branch_order_ok(model, burn_R)
    traces['iplus_burn'].append(ip0_b)
    traces['iminus_burn'].append(im0_b)
    traces['iplus_mirror'].append(model.local_intensities(-burn_R)['Iplus'])
    traces['iminus_mirror'].append(model.local_intensities(-burn_R)['Iminus'])
    traces['P'].append(model.polarizations()['P'])
    for _ in range(max_steps):
        model.step(1, rf_on=True, dnp_on=False)
        loc = model.local_intensities(burn_R)
        traces['iplus_burn'].append(loc['Iplus'])
        traces['iminus_burn'].append(loc['Iminus'])
        mirror = model.local_intensities(-burn_R)
        traces['iplus_mirror'].append(mirror['Iplus'])
        traces['iminus_mirror'].append(mirror['Iminus'])
        traces['P'].append(model.polarizations()['P'])
    (ip_b, im_b, branch_ok) = _branch_order_ok(model, burn_R)
    (R, ip, im, _) = _spectrum(model)
    recovery_traces = None
    if recovery_steps > 0:
        recovery_traces = {k: [traces[k][-1]] for k in traces}
        for _ in range(recovery_steps):
            model.step(1, rf_on=False, dnp_on=False)
            loc = model.local_intensities(burn_R)
            recovery_traces['iplus_burn'].append(loc['Iplus'])
            recovery_traces['iminus_burn'].append(loc['Iminus'])
            mirror = model.local_intensities(-burn_R)
            recovery_traces['iplus_mirror'].append(mirror['Iplus'])
            recovery_traces['iminus_mirror'].append(mirror['Iminus'])
            recovery_traces['P'].append(model.polarizations()['P'])
        (R, ip, im, _) = _spectrum(model)
    return {'model': model, 'R': R, 'R0': R0, 'Rprof': Rprof, 'rf_physical': rf_physical, 'rf_summary': summary, 'ip0': ip0, 'im0': im0, 'ip': ip, 'im': im, 'ip_burn_end': ip.copy(), 'im_burn_end': im.copy(), 'burn_R': burn_R, 'burn_idx': burn_idx, 'mirror_idx': mirror_idx, 'branch_ok': branch_ok, 'gamma_rf': gamma_rf, 'gaussian_fwhm_R': gaussian_fwhm_R, 'lorentzian_fwhm_R': lorentzian_fwhm_R, 'max_steps': max_steps, 'single_bin': single_bin, 'legacy_discrete': legacy_discrete, 'traces': {k: np.asarray(v) for (k, v) in traces.items()}, 'recovery_traces': None if recovery_traces is None else {k: np.asarray(v) for (k, v) in recovery_traces.items()}}

def _print_burn_bin_intensities(run):
    """Print initial and final I+, I- at the burn center."""
    burn_R = run['burn_R']
    traces = run['traces']
    ip0 = traces['iplus_burn'][0]
    im0 = traces['iminus_burn'][0]
    ipf = traces['iplus_burn'][-1]
    imf = traces['iminus_burn'][-1]
    ratio = imf / max(ipf, 1e-30)
    print(f"burn bin R={burn_R:.4f} ({_rf_mode_label()}): I+ {ip0:.6f} -> {ipf:.6f}, I- {im0:.6f} -> {imf:.6f}, I-/I+={ratio:.4f}, branch_ok={run['branch_ok']}")

def plot_physical_rf_profile(out):
    run = _run_voigt_burn_model(max_steps=0)
    model = run['model']
    Rprof = run['Rprof']
    profile = run['rf_physical']
    summary = run['rf_summary']
    burn_R = run['burn_R']
    (_, Ip, _, total) = _spectrum(model)
    (gp, _) = model.rf_rate_fields()
    (fig, axes) = plt.subplots(2, 1, figsize=(8.8, 6.0), layout='constrained', sharex=True)
    (ax0, ax1) = axes
    ax0.plot(Rprof, profile, color='green', linewidth=1.5, label=f'{_rf_mode_label()} profile')
    ax0.axvline(burn_R, color='black', linestyle=':', alpha=0.5, label='burn center')
    ax0.set_ylabel('profile weight')
    ax0.set_title(f"{_rf_mode_label()} RF rate profile  |  Gauss={model.params.rf_gaussian_fwhm_R:.3f}  Lorentz={model.params.rf_lorentzian_fwhm_R:.3f}  approx FWHM={summary['approx_fwhm_R']:.3f}")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8, loc='upper right')
    scale = 0.3 * np.max(total) if total.size else 1.0
    ax1.plot(Rprof, scale * profile, color='green', linestyle='--', linewidth=1.2, label='RF profile (scaled)')
    ax1.plot(model.Rplus, gp, color='darkgreen', linewidth=1.0, label='$\\Gamma_+(R)$ rate field')
    ax1.axvline(burn_R, color='black', linestyle=':', alpha=0.5)
    ax1.set_xlabel('physical R')
    ax1.set_ylabel('RF rate / scaled profile')
    ax1.set_title(f'Capacity-weighted + branch rates  |  gamma_rf={model.params.gamma_rf}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='upper right')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def plot_physical_burn_lineshape(out, *, broad=False):
    if broad:
        run = _run_voigt_burn_model(gaussian_fwhm_R=BROAD_GAUSS_FWHM_R, lorentzian_fwhm_R=BROAD_LORENTZ_FWHM_R)
        tag = 'broad Voigt'
    else:
        run = _run_voigt_burn_model()
        tag = f'{_rf_mode_label()} burn'
    R = run['R']
    burn_R = run['burn_R']
    (fig, ax) = plt.subplots(figsize=(8.8, 4.2), layout='constrained')
    ax.plot(R, run['ip0'], drawstyle='steps-mid', color='tab:red', alpha=0.35, linestyle='--', label='$I_+$ before')
    ax.plot(R, run['im0'], drawstyle='steps-mid', color='tab:blue', alpha=0.35, linestyle='--', label='$I_-$ before')
    ax.plot(R, run['ip'], drawstyle='steps-mid', color='tab:red', linewidth=1.2, label='$I_+$ after RF')
    ax.plot(R, run['im'], drawstyle='steps-mid', color='tab:blue', linewidth=1.2, label='$I_-$ after RF')
    ax.axvline(burn_R, color='green', linestyle=':', alpha=0.6)
    ax.set_xlabel('physical R')
    ax.set_ylabel('intensity')
    ax.set_title(f"{tag} burn (voigt_burn setup)  |  {run['max_steps']} steps  |  gamma={run['gamma_rf']:.1f}  Gauss={run['gaussian_fwhm_R']:.4f}  Lorentz={run['lorentzian_fwhm_R']:.4f}  I->=I+={run['branch_ok']}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}  branch order preserved={run['branch_ok']}")

def plot_physical_burn_trajectory(out):
    run = _run_voigt_burn_model()
    traces = run['traces']
    steps = np.arange(len(traces['iplus_burn']))
    (fig, axes) = plt.subplots(2, 1, figsize=(8.8, 6.0), layout='constrained', sharex=True)
    (ax0, ax1) = axes
    ax0.plot(steps, traces['iplus_burn'], color='tab:red', label=f"$I_+(R={run['burn_R']:.2f})$")
    ax0.plot(steps, traces['iminus_burn'], color='tab:blue', label=f"$I_-(R={run['burn_R']:.2f})$")
    ax0.set_ylabel('burn-center intensity')
    ax0.set_title(f"Burn-center intensities  |  gamma={run['gamma_rf']:.1f}  branch order preserved={run['branch_ok']}")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8, loc='upper right')
    ax1.plot(steps, traces['P'], color='black', linewidth=1.2, label='$P(t)$')
    ax1.set_xlabel('integration step')
    ax1.set_ylabel('vector polarization')
    ax1.set_title('Vector polarization during RF (spin diffusion active)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='upper right')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def plot_discrete_vs_physical_compare(out):
    if _SINGLE_BIN_MODE:
        single = _run_voigt_burn_model(single_bin=True)
        physical = _run_voigt_burn_model(single_bin=False)
        burn_R = physical['burn_R']
        (fig, axes) = plt.subplots(1, 2, figsize=(11.0, 4.2), layout='constrained')
        for (ax, run, title) in ((axes[0], single, 'single-bin RF (frozen profile)'), (axes[1], physical, 'physical-R Voigt (voigt_burn)')):
            R = run['R']
            ax.plot(R, run['ip0'], drawstyle='steps-mid', color='tab:red', alpha=0.25, linestyle='--')
            ax.plot(R, run['im0'], drawstyle='steps-mid', color='tab:blue', alpha=0.25, linestyle='--')
            ax.plot(R, run['ip'], drawstyle='steps-mid', color='tab:red', linewidth=1.1, label='$I_+$ after')
            ax.plot(R, run['im'], drawstyle='steps-mid', color='tab:blue', linewidth=1.1, label='$I_-$ after')
            ax.axvline(burn_R, color='green', linestyle=':', alpha=0.5)
            ax.set_xlabel('physical R')
            ax.set_ylabel('intensity')
            ax.set_title(f"{title}  |  I->=I+={run['branch_ok']}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
        fig.suptitle(f'Same voigt_burn init/recovery  |  gamma={BURN_GAMMA_RF}  steps={BURN_RF_STEPS}', fontsize=11)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
        return
    physical = _run_voigt_burn_model()
    discrete = _run_voigt_burn_model(legacy_discrete=True)
    burn_R = physical['burn_R']
    (fig, axes) = plt.subplots(1, 2, figsize=(11.0, 4.2), layout='constrained')
    for (ax, run, title) in ((axes[0], discrete, 'legacy v2 discrete Voigt'), (axes[1], physical, 'physical-R Voigt (voigt_burn)')):
        R = run['R']
        ax.plot(R, run['ip0'], drawstyle='steps-mid', color='tab:red', alpha=0.25, linestyle='--')
        ax.plot(R, run['im0'], drawstyle='steps-mid', color='tab:blue', alpha=0.25, linestyle='--')
        ax.plot(R, run['ip'], drawstyle='steps-mid', color='tab:red', linewidth=1.1, label='$I_+$ after')
        ax.plot(R, run['im'], drawstyle='steps-mid', color='tab:blue', linewidth=1.1, label='$I_-$ after')
        ax.axvline(burn_R, color='green', linestyle=':', alpha=0.5)
        ax.set_xlabel('physical R')
        ax.set_ylabel('intensity')
        ax.set_title(f"{title}  |  I->=I+={run['branch_ok']}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Same voigt_burn setup  |  gamma={BURN_GAMMA_RF}  steps={BURN_RF_STEPS}  Gauss={voigt_burn_params().rf_gaussian_fwhm_R:.3f}  Lorentz={voigt_burn_params().rf_lorentzian_fwhm_R:.3f}', fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def plot_v2_vs_voigt_burn_package(out):
    """Overlay v2 and spin1_ssrf_realtime_voigt_burn trajectories (must match)."""
    from ssrf_realtime.model import Spin1Model as RefModel, Spin1Params as RefParams
    ref = RefModel(RefParams(gamma_rf=BURN_GAMMA_RF, diffusion_scale=5.0, dnp_enabled=False, t1_rate=0.0))
    v2 = create_voigt_burn_model(gamma_rf=BURN_GAMMA_RF)
    burn_R = v2.params.rf_burn_R
    steps = np.arange(BURN_RF_STEPS + 1)
    (ref_ip, ref_im, v2_ip, v2_im) = ([], [], [], [])
    ref_ip.append(ref.local_intensities(burn_R)['Iplus'])
    ref_im.append(ref.local_intensities(burn_R)['Iminus'])
    v2_ip.append(v2.local_intensities(burn_R)['Iplus'])
    v2_im.append(v2.local_intensities(burn_R)['Iminus'])
    for _ in range(BURN_RF_STEPS):
        ref.step(1, rf_on=True, dnp_on=False)
        v2.step(1, rf_on=True, dnp_on=False)
        ref_ip.append(ref.local_intensities(burn_R)['Iplus'])
        ref_im.append(ref.local_intensities(burn_R)['Iminus'])
        v2_ip.append(v2.local_intensities(burn_R)['Iplus'])
        v2_im.append(v2.local_intensities(burn_R)['Iminus'])
    ref_ip = np.asarray(ref_ip)
    ref_im = np.asarray(ref_im)
    v2_ip = np.asarray(v2_ip)
    v2_im = np.asarray(v2_im)
    (fig, ax) = plt.subplots(figsize=(8.8, 4.2), layout='constrained')
    ax.plot(steps, ref_im, color='tab:blue', linewidth=1.3, label='voigt_burn package $I_-$')
    ax.plot(steps, v2_im, color='tab:cyan', linestyle='--', linewidth=1.1, label='ssrf_realtime $I_-$')
    ax.plot(steps, ref_ip, color='tab:red', linewidth=1.3, label='voigt_burn package $I_+$')
    ax.plot(steps, v2_ip, color='tab:orange', linestyle='--', linewidth=1.1, label='ssrf_realtime $I_+$')
    ax.set_xlabel('integration step')
    ax.set_ylabel(f'intensity at R={burn_R:.2f}')
    ax.set_title(f'Package parity at burn center  |  max |ΔI+|={np.max(np.abs(v2_ip - ref_ip)):.2e}  max |ΔI-|={np.max(np.abs(v2_im - ref_im)):.2e}')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def plot_post_burn_diffusion_recovery(out):
    """Match spin1_ssrf_realtime_voigt_burn/examples/headless_demo.py no_dnp_burn_recovery."""
    dt = 0.0005
    burn_end = 0.125
    total_time = 1.75
    run = _run_voigt_burn_model(rf_burn_R=0.4, gamma_rf=6.0, gaussian_fwhm_R=0.055, lorentzian_fwhm_R=0.02, diffusion_scale=50.0, dt=dt, max_steps=burn_end / dt, recovery_steps=(total_time - burn_end) / dt)
    model = run['model']
    burn_R = run['burn_R']
    (R, ip, im, total) = _spectrum(model)
    (Rprof, profile) = model.rf_profile_physical()
    (fig, (ax0, ax1)) = plt.subplots(2, 1, figsize=(9.0, 5.8), layout='constrained')
    ax0.plot(R, ip, drawstyle='steps-mid', label='$I_+(R)$')
    ax0.plot(R, im, drawstyle='steps-mid', label='$I_-(R)$')
    ax0.plot(R, total, label='total')
    ax0.plot(Rprof, 0.3 * np.max(total) * profile, linestyle='--', label='RF rate profile (scaled)')
    ax0.axvline(burn_R, linestyle=':', linewidth=1.0)
    ax0.set_xlabel('physical R')
    ax0.set_ylabel('intensity')
    ax0.set_title('Voigt burn followed by population-dependent recovery (DNP off)')
    ax0.legend(fontsize=8, ncols=3)
    ax0.grid(True, alpha=0.3)
    traces = run['traces']
    rec = run['recovery_traces']
    assert rec is not None
    t_rf = np.arange(len(traces['iplus_burn'])) * model.params.dt
    t_rec = np.arange(len(rec['iplus_burn'])) * model.params.dt + t_rf[-1]
    ax1.plot(t_rf, traces['iplus_burn'], label=f'$I_+(R={burn_R:.2f})$ RF on')
    ax1.plot(t_rf, traces['iminus_burn'], label=f'$I_-(R={burn_R:.2f})$ RF on')
    ax1.plot(t_rec, rec['iplus_burn'], linestyle='--', label='RF off, diffusion')
    ax1.plot(t_rec, rec['iminus_burn'], linestyle='--')
    ax1.axvspan(0.0, t_rf[-1], alpha=0.12, label='RF on')
    ax1.set_xlabel('time [arb.]')
    ax1.set_ylabel('burn-center intensity')
    ax1.set_title('RF lowers P; spin diffusion redistributes the reduced state')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def plot_narrow_vs_broad_physical(out):
    narrow = _run_voigt_burn_model(max_steps=WIDTH_COMPARE_STEPS)
    broad = _run_voigt_burn_model(gaussian_fwhm_R=BROAD_GAUSS_FWHM_R, lorentzian_fwhm_R=BROAD_LORENTZ_FWHM_R, max_steps=WIDTH_COMPARE_STEPS)
    R = narrow['R']
    burn_R = narrow['burn_R']
    (fig, axes) = plt.subplots(1, 2, figsize=(11.0, 4.2), layout='constrained')
    for (ax, run, label) in ((axes[0], narrow, f"active Gauss={narrow['gaussian_fwhm_R']:.4f} Lorentz={narrow['lorentzian_fwhm_R']:.4f}"), (axes[1], broad, f'broad Gauss={BROAD_GAUSS_FWHM_R:.4f} Lorentz={BROAD_LORENTZ_FWHM_R:.4f}')):
        ax.plot(R, run['ip0'], drawstyle='steps-mid', color='tab:red', alpha=0.25, linestyle='--')
        ax.plot(R, run['im0'], drawstyle='steps-mid', color='tab:blue', alpha=0.25, linestyle='--')
        ax.plot(R, run['ip'], drawstyle='steps-mid', color='tab:red', linewidth=1.1, label='$I_+$ after')
        ax.plot(R, run['im'], drawstyle='steps-mid', color='tab:blue', linewidth=1.1, label='$I_-$ after')
        ax.axvline(burn_R, color='green', linestyle=':', alpha=0.5)
        ax.set_xlabel('physical R')
        ax.set_ylabel('intensity')
        ax.set_title(f'{label}  |  {WIDTH_COMPARE_STEPS} steps')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(f'Voigt width comparison (shorter burn to show wings)  |  gamma={BURN_GAMMA_RF}', fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

def main():
    global _ACTIVE_GAUSS_FWHM_R, _ACTIVE_LORENTZ_FWHM_R, _SINGLE_BIN_MODE
    parser = argparse.ArgumentParser(description='voigt_burn-aligned burn demo plots')
    parser.add_argument('--all', action='store_true', help='Generate extended plot set')
    parser.add_argument('--single-bin', action='store_true', help='Use v15-style single-bin RF at burn center with full-spectrum spectral-neighbor recovery')
    parser.add_argument('--gauss-fwhm', type=float, default=None, help=f'Gaussian RF FWHM in R (default: {GAUSS_FWHM_R})')
    parser.add_argument('--lorentz-fwhm', type=float, default=None, help=f'Lorentzian RF FWHM in R (default: {LORENTZ_FWHM_R})')
    args = parser.parse_args()
    _ACTIVE_GAUSS_FWHM_R = GAUSS_FWHM_R if args.gauss_fwhm is None else args.gauss_fwhm
    _ACTIVE_LORENTZ_FWHM_R = LORENTZ_FWHM_R if args.lorentz_fwhm is None else args.lorentz_fwhm
    _SINGLE_BIN_MODE = args.single_bin
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = voigt_burn_params()
    print(f'voigt_burn setup: n_bins={base.n_bins}, p0={base.p0}, rf_burn_R={base.rf_burn_R}, diffusion_scale={base.diffusion_scale}, gamma={BURN_GAMMA_RF}, steps={BURN_RF_STEPS}, RF mode={_rf_mode_label()}')
    if not _SINGLE_BIN_MODE:
        print(f'active RF widths: Gauss={_ACTIVE_GAUSS_FWHM_R:.4f}  Lorentz={_ACTIVE_LORENTZ_FWHM_R:.4f}  approx FWHM={approximate_voigt_fwhm(_ACTIVE_GAUSS_FWHM_R, _ACTIVE_LORENTZ_FWHM_R):.4f}  (center_bin norm: peak rate at burn center is unchanged by width)')
    print(f'spectral-neighbor recovery on (d_spec) + spin diffusion (scale={base.diffusion_scale})')
    if _SINGLE_BIN_MODE:
        print(f'single-bin RF: gamma_rf={BURN_GAMMA_RF} at burn_idx for R={base.rf_burn_R}')
    _print_burn_bin_intensities(_run_voigt_burn_model())
    plot_physical_rf_profile(OUTPUT_DIR / _out_name('v2_voigt_burn_rf_profile'))
    plot_physical_burn_lineshape(OUTPUT_DIR / _out_name('v2_voigt_burn_lineshape'))
    plot_physical_burn_trajectory(OUTPUT_DIR / _out_name('v2_voigt_burn_trajectory'))
    plot_discrete_vs_physical_compare(OUTPUT_DIR / _out_name('v2_voigt_burn_discrete_vs_physical'))
    if not _SINGLE_BIN_MODE:
        plot_v2_vs_voigt_burn_package(OUTPUT_DIR / 'v2_voigt_burn_package_parity.png')
    if args.all:
        if _SINGLE_BIN_MODE:
            print('Skipping --all Voigt width / package plots in single-bin mode.')
        else:
            plot_physical_burn_lineshape(OUTPUT_DIR / 'v2_voigt_burn_broad_lineshape.png', broad=True)
            plot_narrow_vs_broad_physical(OUTPUT_DIR / 'v2_voigt_burn_narrow_vs_broad.png')
            plot_post_burn_diffusion_recovery(OUTPUT_DIR / 'v2_voigt_burn_diffusion_recovery.png')
if __name__ == '__main__':
    main()
