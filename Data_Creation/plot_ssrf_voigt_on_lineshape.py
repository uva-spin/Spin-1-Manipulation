"""
Wide and narrow ssRF Voigt burns on one lineshape.

Top row: manipulation with no noise (shared lineshape before the burn).
Middle row: those same results with one shared noise draw.
Bottom row: the power profiles that were applied.

Run from the repo root:
  python Data_Creation/plot_ssrf_voigt_on_lineshape.py
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

from Data_Creation.spectrum_gen.events import ssrf_voigt_power_profile
from Data_Creation.spectrum_gen.ssrf_traj import run_one_polarization
from physics import rf
from physics.rf.voigt_physical import approximate_voigt_fwhm

P0 = 0.45
N_BINS = 500
R_MIN = -6.0
R_MAX = 6.0
N_STEPS = 40
NOISE_SIGMA = 0.1
SEED = 42
# (name, gamma_rf, gaussian FWHM, lorentzian FWHM, color) — the demonstrated profiles
PROFILES = (
    ('wide', 5.47, 0.089, 0.038, '#1b7f4e'),
    ('narrow', 8.89, 0.042, 0.029, '#1a5276'),
)
OUT = SCRIPT_DIR / 'spectra_data' / 'plots' / 'ssrf_voigt_power_on_lineshape_noise0.1.png'


def left_horn_center(frequency):
    """Bin of the most negative Q on the left horn."""
    params = rf.Spin1Params(
        n_bins=N_BINS,
        r_min=R_MIN,
        r_max=R_MAX,
        p0=P0,
        q0=None,
        line_gamma=0.05,
        line_asym=0.04,
        dt=0.0015,
        rf_enabled=False,
        rf_burn_R=-1.0,
    )
    iplus, iminus, _ = rf.Spin1Model(params).physical_intensities()
    q = iplus - iminus
    left = np.flatnonzero(frequency < 0.0)
    return int(left[np.argmin(q[left])])


def main():
    frequency = np.linspace(R_MIN, R_MAX, N_BINS)
    center = left_horn_center(frequency)
    center_r = float(frequency[center])

    spectra = []
    for name, gamma, gauss, lorentz, color in PROFILES:
        traj = run_one_polarization(
            center,
            P0,
            gamma_rf=gamma,
            n_steps=N_STEPS,
            gaussian_fwhm_R=gauss,
            lorentzian_fwhm_R=lorentz,
            capture_spectrum=True,
            legacy_spectral_recovery=False,
        )
        before = traj['iplus_full'][0] + traj['iminus_full'][0]
        after = traj['iplus_full'][N_STEPS] + traj['iminus_full'][N_STEPS]
        spectra.append((name, gamma, gauss, lorentz, color, before, after))

    before = spectra[0][5]
    noise = np.random.default_rng(SEED).normal(0.0, NOISE_SIGMA, size=before.shape)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.8, 10.2),
        sharex=True,
        sharey='row',
        gridspec_kw={'height_ratios': [1.15, 1.15, 0.95], 'hspace': 0.22, 'wspace': 0.08},
    )
    for col, (name, gamma, gauss, lorentz, color, _, after) in enumerate(spectra):
        ax_clean, ax_noisy, ax_u = axes[0, col], axes[1, col], axes[2, col]
        fwhm = approximate_voigt_fwhm(gauss, lorentz)
        tag = rf'$G={gauss:.3f}$  $L={lorentz:.3f}$  $\gamma={gamma:.2f}$  FWHM$={fwhm:.3f}$'

        ax_clean.plot(frequency, before, color='0.45', lw=1.35, ls=(0, (3, 2)), label='before', zorder=2)
        ax_clean.plot(frequency, after, color=color, lw=1.6, label='after', zorder=3)
        ax_clean.axvline(center_r, color='0.25', ls=':', lw=1.15, zorder=1)
        ax_clean.set_title(f'{name}   no noise\n{tag}', fontsize=12)
        ax_clean.legend(fontsize=11, loc='upper right', frameon=False)

        ax_noisy.plot(frequency, after + noise, color=color, lw=1.15, zorder=3)
        ax_noisy.axvline(center_r, color='0.25', ls=':', lw=1.15, zorder=1)
        ax_noisy.set_title(rf'{name}   $\sigma={NOISE_SIGMA}$', fontsize=12)

        profile = np.asarray(
            ssrf_voigt_power_profile(
                center,
                gamma,
                num_bins=N_BINS,
                r_min=R_MIN,
                r_max=R_MAX,
                gaussian_fwhm_R=gauss,
                lorentzian_fwhm_R=lorentz,
            ),
            dtype=float,
        )
        ax_u.step(frequency, profile, where='mid', color=color, lw=1.8, zorder=3)
        ax_u.fill_between(frequency, profile, step='mid', color=color, alpha=0.22, zorder=2)
        ax_u.axvline(center_r, color='0.25', ls=':', lw=1.15, zorder=1)
        ax_u.set_title(f'{name} power profile', fontsize=12)
        ax_u.set_xlabel(r'$R$')
        ax_u.set_xlim(R_MIN, R_MAX)
        ax_u.set_ylim(-0.35, 10.4)

    axes[0, 0].set_ylabel('signal (a.u.)')
    axes[1, 0].set_ylabel('signal (a.u.)')
    axes[2, 0].set_ylabel(r'$U(R)$')
    axes[0, 0].set_ylim(-0.15, 1.05)
    axes[1, 0].set_ylim(-0.45, 1.15)

    for ax in axes.ravel():
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', color='0.78', linewidth=0.8)
        ax.tick_params(labelsize=11, width=1.0, length=4)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    fig.suptitle(
        rf'Same lineshape  $P_0={P0:.2f}$,  ssRF Voigt burn  $n={N_STEPS}$,  $R={center_r:.2f}$',
        fontsize=14,
        y=0.995,
    )
    fig.patch.set_facecolor('white')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
