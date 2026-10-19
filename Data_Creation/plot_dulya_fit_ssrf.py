"""Example Dulya-fit lineshape with one ssRF burn at R = -1.

Equilibrium P and Q are the unmanipulated polarizations. The same spectrum
is then burned with the create_data ssRF Voigt profile centered on the bin
nearest R = -1, and the post-burn P and Q are written on the figure.

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

from Data_Creation.spectrum_gen.constants import (  # noqa: E402
    DT,
    F_MAX,
    F_MIN,
    NUM_BINS,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
)
from Data_Creation.spectrum_gen.dulya import use_fit_params  # noqa: E402
from Data_Creation.spectrum_gen.ssrf_traj import run_one_polarization  # noqa: E402

P0 = 0.45
GAMMA_RF = 5.0
N_STEPS = 200
BURN_R = -0.92
FIT_PARAMS = SCRIPT_DIR / 'fit_params.json'
OUT = SCRIPT_DIR / 'dulya_fit_ssrf_example.png'


def main():
    use_fit_params(FIT_PARAMS)
    frequency = np.linspace(F_MIN, F_MAX, NUM_BINS)
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
    ax.plot(frequency, ip0 + im0, color='0.45', lw=1.6, ls=(0, (3, 2)), label='equilibrium')
    ax.plot(frequency, ip1 + im1, color='#1a5276', lw=1.8, label='after ssRF')
    ax.axvline(center_r, color='0.2', ls=':', lw=1.15, label=f'burn R={center_r:.3f}')
    ax.set_xlim(-3.5, 3.5)
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'signal  $I_+ + I_-$')
    ax.set_title(
        rf'Dulya fit   $P_0={P0:.2f}$   ssRF at $R={center_r:.3f}$'
        rf'   $\gamma={GAMMA_RF:g}$   $n={N_STEPS}$'
    )
    note = (
        f'true        P = {p_true:+.4f}    Q = {q_true:+.4f}\n'
        f'after ssRF  P = {p_mod:+.4f}    Q = {q_mod:+.4f}'
    )
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
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', color='0.78', linewidth=0.8)
    ax.tick_params(labelsize=11, width=1.0, length=4)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'true        P = {p_true:+.6f}   Q = {q_true:+.6f}')
    print(f'after ssRF  P = {p_mod:+.6f}   Q = {q_mod:+.6f}')
    print(f'burn bin {center}  R = {center_r:.6f}')
    print(OUT)


if __name__ == '__main__':
    main()
