"""Example Dulya-fit lineshape with one selective ssRF burn.

Run from the repo root:
  python Data_Creation/plot_dulya_fit_ssrf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data_Creation.spectrum_gen.constants import (
    DT,
    F_MAX,
    F_MIN,
    NUM_BINS,
)
from Data_Creation.spectrum_gen.dulya import apply_active_fit, use_fit_params
from Data_Creation.spectrum_gen.ssrf_traj import run_one_polarization
from physics.rf.optimal_profile import DATA_GEN_SETTINGS, run_optimal_profile_polarization

P0 = 0.45
GAMMA_RF = 2.0
N_STEPS = 320
BURN_R = -0.94
RF_GAUSSIAN_FWHM_R = 0.15
RF_LORENTZIAN_FWHM_R = 0.15
FIT_PARAMS = SCRIPT_DIR / 'fit_params.json'
OUT_SSRF = SCRIPT_DIR / 'dulya_fit_ssrf_example.png'
OUT_PROFILE = SCRIPT_DIR / 'dulya_fit_ssrf_profile_example.png'


def _style(ax):
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', color='0.78', linewidth=0.8)
    ax.tick_params(labelsize=11, width=1.0, length=4)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _pq_box(ax, note):
    ax.text(
        0.02,
        0.97,
        note,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=12,
        family='monospace',
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.92, 'edgecolor': '0.7'},
    )


def _save(fig, path):
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(path)


def plot_selective(frequency):
    center = int(np.argmin(np.abs(frequency - BURN_R)))
    center_r = float(frequency[center])
    traj = run_one_polarization(
        center,
        P0,
        gamma_rf=GAMMA_RF,
        n_steps=N_STEPS,
        dt=DT,
        gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R,
        lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R,
        capture_spectrum=True,
        legacy_spectral_recovery=False,
    )
    if traj['skipped']:
        raise SystemExit(f'ssRF skipped at R={center_r:.4f} (bin {center}): intensity too small')

    ip0 = np.asarray(traj['iplus_full'][0])
    im0 = np.asarray(traj['iminus_full'][0])
    ip1 = np.asarray(traj['iplus_full'][-1])
    im1 = np.asarray(traj['iminus_full'][-1])
    p_true = float(traj['p_initial'])
    q_true = float(traj['q_initial'])
    p_mod = float(traj['p_final'])
    q_mod = float(traj['q_final'])

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(frequency, -(ip0 + im0), color='0.45', lw=1.6, ls=(0, (3, 2)), label='equilibrium')
    ax.plot(frequency, -(ip1 + im1), color='#1a5276', lw=1.8, label='after ssRF')
    ax.axvline(center_r, color='0.2', ls=':', lw=1.15, label=f'burn R={center_r:.3f}')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'signal  $I_+ + I_-$')
    ax.set_title(
        rf'Dulya fit   $P_0={P0:.2f}$   ssRF at $R={center_r:.3f}$'
        rf'   $\gamma={GAMMA_RF:g}$   $n={N_STEPS}$'
    )
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    _style(ax)
    _save(fig, OUT_SSRF)
    print(f'ssRF true        P = {p_true:+.6f}   Q = {q_true:+.6f}')
    print(f'ssRF after       P = {p_mod:+.6f}   Q = {q_mod:+.6f}')
    print(f'burn bin {center}  R = {center_r:.6f}')


def plot_optimal_profile():
    traj = run_optimal_profile_polarization(
        P0,
        n_steps=0,
        n_bins=NUM_BINS,
        r_min=F_MIN,
        r_max=F_MAX,
        dt=DT,
        settings=DATA_GEN_SETTINGS,
        equilibrium_loader=apply_active_fit,
    )
    frequency = np.asarray(traj['frequency'], dtype=float)
    end = int(traj['program_end_step'])
    ip0 = np.asarray(traj['iplus_full'][0])
    im0 = np.asarray(traj['iminus_full'][0])
    ip1 = np.asarray(traj['iplus_full'][end])
    im1 = np.asarray(traj['iminus_full'][end])
    p_true = float(traj['p_initial'])
    q_true = float(traj['q_initial'])
    p_mod = float(traj['p_full'][end])
    q_mod = float(traj['q_full'][end])
    peak_u = float(traj['gamma_rf'])
    duration = float(traj['duration'])

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(frequency, -(ip0 + im0), color='0.45', lw=1.6, ls=(0, (3, 2)), label='equilibrium')
    ax.plot(frequency, -(ip1 + im1), color='#1a5276', lw=1.8, label='after optimal profile')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'signal  $I_+ + I_-$')
    ax.set_title(
        rf'Dulya fit   $P_0={P0:.2f}$   optimal ssRF profile'
        rf'   $U_\mathrm{{max}}={peak_u:.2f}$   $T={duration:.3g}$'
    )
    _pq_box(
        ax,
        f'true           P = {p_true:+.4f}    Q = {q_true:+.4f}\n'
        f'after profile  P = {p_mod:+.4f}    Q = {q_mod:+.4f}',
    )
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    _style(ax)
    _save(fig, OUT_PROFILE)
    print(f'profile true     P = {p_true:+.6f}   Q = {q_true:+.6f}')
    print(f'profile after    P = {p_mod:+.6f}   Q = {q_mod:+.6f}')
    print(f'U_max = {peak_u:.6f}  T = {duration:.6g}  status = {traj["status"]}')


def main():
    use_fit_params(FIT_PARAMS)
    frequency = np.linspace(F_MIN, F_MAX, NUM_BINS)
    plot_selective(frequency)
    # plot_optimal_profile()


if __name__ == '__main__':
    main()
