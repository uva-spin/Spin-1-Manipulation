"""Example Dulya-fit lineshape: selective AFP window and AFP Q<0 profile.

Writes two figures:
  dulya_fit_afp_example.png          — one windowed AFP near R = -0.92
  dulya_fit_afp_profile_example.png  — full Q<0 AFP Profile sweep

Run from the repo root:
  python Data_Creation/plot_dulya_fit_afp.py
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

from Data_Creation.spectrum_gen.afp_traj import run_one_polarization
from Data_Creation.spectrum_gen.bridge import (
    build_equilibrium_spin1_model,
    full_spectrum_intensities,
)
from Data_Creation.spectrum_gen.constants import (
    AFP_WINDOW,
    DT,
    F_MAX,
    F_MIN,
    NUM_BINS,
)
from Data_Creation.spectrum_gen.dulya import use_fit_params
from Data_Creation.spectrum_gen.events import _burn_window_bins

P0 = 0.45
N_RELAX = 0
BURN_R = -0.92
FIT_PARAMS = SCRIPT_DIR / 'fit_params.json'
OUT_AFP = SCRIPT_DIR / 'dulya_fit_afp_example.png'
OUT_PROFILE = SCRIPT_DIR / 'dulya_fit_afp_profile_example.png'


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


def _equilibrium():
    eq = build_equilibrium_spin1_model(
        polarization=P0,
        num_bins=NUM_BINS,
        dt=DT,
        rf_enabled=False,
        relax_enabled=True,
        r_min=F_MIN,
        r_max=F_MAX,
        legacy_spectral_recovery=False,
    )
    return full_spectrum_intensities(eq)


def plot_selective(frequency):
    center = int(np.argmin(np.abs(frequency - BURN_R)))
    center_r = float(frequency[center])
    ip0, im0, _ = _equilibrium()
    traj = run_one_polarization(
        center,
        P0,
        afp_window=AFP_WINDOW,
        n_relax=N_RELAX,
        dt=DT,
        capture_spectrum=True,
        legacy_spectral_recovery=False,
    )
    if traj['skipped']:
        raise SystemExit(f'AFP skipped at R={center_r:.4f} (bin {center}): empty subset')

    ip1 = np.asarray(traj['iplus_full'][-1])
    im1 = np.asarray(traj['iminus_full'][-1])
    p_true = float(traj['p_initial'])
    q_true = float(traj['q_initial'])
    p_mod = float(traj['p_final'])
    q_mod = float(traj['q_final'])
    subset = traj['afp_subset']
    r_lo = float(frequency[min(subset)])
    r_hi = float(frequency[max(subset)])

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(frequency, np.asarray(ip0) + np.asarray(im0), color='0.45', lw=1.6, ls=(0, (3, 2)), label='equilibrium')
    ax.plot(frequency, ip1 + im1, color='#1a5276', lw=1.8, label='after AFP')
    ax.axvline(center_r, color='0.2', ls=':', lw=1.15, label=f'center R={center_r:.3f}')
    ax.axvspan(r_lo, r_hi, color='0.75', alpha=0.25, label=f'window={AFP_WINDOW}')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'signal  $I_+ + I_-$')
    ax.set_title(
        rf'Dulya fit   $P_0={P0:.2f}$   AFP at $R={center_r:.3f}$'
        rf'   window$={AFP_WINDOW}$'
    )
    _pq_box(
        ax,
        f'true       P = {p_true:+.4f}    Q = {q_true:+.4f}\n'
        f'after AFP  P = {p_mod:+.4f}    Q = {q_mod:+.4f}',
    )
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    _style(ax)
    _save(fig, OUT_AFP)
    print(f'AFP true       P = {p_true:+.6f}   Q = {q_true:+.6f}')
    print(f'AFP after      P = {p_mod:+.6f}   Q = {q_mod:+.6f}')
    print(f'center bin {center}  R = {center_r:.6f}  window bins {min(subset)}..{max(subset)}')


def plot_afp_profile(frequency):
    ip0, im0, _ = _equilibrium()
    burn_window = _burn_window_bins(num_bins=NUM_BINS, r_min=F_MIN, r_max=F_MAX)
    traj = run_one_polarization(
        0,
        P0,
        n_relax=N_RELAX,
        qneg_bins=burn_window,
        dt=DT,
        capture_spectrum=True,
        legacy_spectral_recovery=False,
    )
    if traj['skipped']:
        raise SystemExit('AFP profile skipped: no Q<0 bins in burn window')

    ip1 = np.asarray(traj['iplus_full'][-1])
    im1 = np.asarray(traj['iminus_full'][-1])
    p_true = float(traj['p_initial'])
    q_true = float(traj['q_initial'])
    p_mod = float(traj['p_final'])
    q_mod = float(traj['q_final'])
    subset = np.asarray(traj['afp_subset'], dtype=int)
    n_swept = int(subset.size)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(frequency, np.asarray(ip0) + np.asarray(im0), color='0.45', lw=1.6, ls=(0, (3, 2)), label='equilibrium')
    ax.plot(frequency, ip1 + im1, color='#1a5276', lw=1.8, label='after AFP profile')
    if n_swept:
        ax.axvspan(
            float(frequency[subset.min()]),
            float(frequency[subset.max()]),
            color='0.75',
            alpha=0.18,
            label=f'Q<0 sweep ({n_swept} bins)',
        )
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'signal  $I_+ + I_-$')
    ax.set_title(rf'Dulya fit   $P_0={P0:.2f}$   AFP profile (Q$<0$ sweep)')
    _pq_box(
        ax,
        f'true           P = {p_true:+.4f}    Q = {q_true:+.4f}\n'
        f'after profile  P = {p_mod:+.4f}    Q = {q_mod:+.4f}',
    )
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    _style(ax)
    _save(fig, OUT_PROFILE)
    print(f'profile true   P = {p_true:+.6f}   Q = {q_true:+.6f}')
    print(f'profile after  P = {p_mod:+.6f}   Q = {q_mod:+.6f}')
    print(f'swept {n_swept} Q<0 bins')


def main():
    use_fit_params(FIT_PARAMS)
    frequency = np.linspace(F_MIN, F_MAX, NUM_BINS)
    plot_selective(frequency)
    plot_afp_profile(frequency)


if __name__ == '__main__':
    main()
