"""Diagnostic plots for full-spectrum create_data output."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import (
    AFP_WINDOW,
    COMBO_LAYOUT_PROFILE_AFP_SELECTIVE,
    COMBO_LAYOUT_PROFILE_PROFILE,
    COMBO_LAYOUT_SELECTIVE_SELECTIVE,
    COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE,
    COMBO_SCENARIO_AFP_FIRST_REGION,
    COMBO_SCENARIO_BOTH_SELECTIVE,
    COMBO_SCENARIO_SSRF_FIRST_REGION,
    F_MAX,
    F_MIN,
    NUM_BINS,
    SOURCE_AFP,
    SOURCE_AFP_PROFILE,
    SOURCE_PROFILE,
    SOURCE_SSRF,
    SOURCE_SSRF_AFP,
    SOURCE_UNMANIP,
)

FREQUENCY = np.linspace(F_MIN, F_MAX, NUM_BINS)


def _event_mask(data, *, source=None, p0=None, center_bin=None, n_steps=None, combo_scenario=None, combo_layout=None):
    n = data['spectra'].shape[0]
    mask = np.ones(n, dtype=bool)
    if source is not None:
        mask &= data['source'] == source
    if p0 is not None:
        mask &= np.isclose(data['p0'], p0, atol=1e-05, rtol=0.0)
    if center_bin is not None:
        mask &= data['center_bin'] == center_bin
    if n_steps is not None:
        mask &= data['n_steps'] == n_steps
    if combo_scenario is not None and 'combo_scenario' in data:
        mask &= data['combo_scenario'] == int(combo_scenario)
    if combo_layout is not None and 'combo_layout' in data:
        mask &= data['combo_layout'] == int(combo_layout)
    return mask


def _combo_scenario_slug(scenario):
    return {
        COMBO_SCENARIO_SSRF_FIRST_REGION: 'sc1_ssrf_r0_afp_r1',
        COMBO_SCENARIO_AFP_FIRST_REGION: 'sc2_ssrf_r1_afp_r0',
        COMBO_SCENARIO_BOTH_SELECTIVE: 'sc3_both_selective',
    }.get(int(scenario), f'sc{int(scenario)}')


def _combo_layout_slug(layout):
    return {
        COMBO_LAYOUT_PROFILE_PROFILE: 'lay0_profile_profile',
        COMBO_LAYOUT_PROFILE_AFP_SELECTIVE: 'lay1_profile_afp_sel',
        COMBO_LAYOUT_SSRF_SELECTIVE_PROFILE: 'lay2_ssrf_sel_profile',
        COMBO_LAYOUT_SELECTIVE_SELECTIVE: 'lay3_selective_selective',
    }.get(int(layout), f'lay{int(layout)}')


def _pick_combined_example_groups(data):
    mask = data['source'] == SOURCE_SSRF_AFP
    if not np.any(mask):
        return []
    groups = {}
    for i in np.flatnonzero(mask):
        key = (
            int(data['combo_scenario'][i]),
            int(data['combo_layout'][i]),
            round(float(data['p0'][i]), 5),
            int(data['center_bin'][i]),
        )
        groups.setdefault(key, set()).add(int(data['n_steps'][i]))
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1], kv[0][2], kv[0][3]))
    seen, picks = set(), []
    for (scen, lay, p0, center), _ in ranked:
        slot = (scen, lay)
        if slot in seen:
            continue
        seen.add(slot)
        picks.append((float(p0), scen, lay, center))
    return picks


def _pick_keys(data, source, *, max_examples=2, with_center=True):
    mask = data['source'] == source
    if not np.any(mask):
        return []
    p0s = data['p0'][mask]
    steps = data['n_steps'][mask]
    if with_center:
        bins = data['center_bin'][mask]
        pairs = {}
        for p0, b, s in zip(p0s.tolist(), bins.tolist(), steps.tolist()):
            key = (round(p0, 5), int(b))
            pairs.setdefault(key, set()).add(s)
        ranked = sorted(pairs.items(), key=lambda kv: (-len(kv[1]), kv[0][0], kv[0][1]))
        return [k for k, _ in ranked[: max(0, max_examples)]]
    groups = {}
    for p0, s in zip(p0s.tolist(), steps.tolist()):
        key = round(p0, 5)
        groups.setdefault(key, set()).add(s)
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [k for k, _ in ranked[: max(0, max_examples)]]


def _style_axes(ax, *, label_size=16, tick_size=14, title_size=15):
    ax.tick_params(axis='both', which='major', labelsize=tick_size, width=1.1, length=5)
    ax.tick_params(axis='both', which='minor', width=0.8, length=3)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_size(title_size)
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
    ax.grid(True, alpha=0.28, linewidth=0.7)


def save_example_plots(
    data,
    plot_dir,
    *,
    frequency=None,
    max_ssrf_examples=2,
    max_afp_examples=2,
    max_afp_profile_examples=2,
    max_unmanip_examples=2,
    profile_rates=None,
):
    """Write diagnostic PNGs for enabled event sources."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    f = np.asarray(FREQUENCY if frequency is None else frequency).reshape(-1)
    profile_rates = {} if profile_rates is None else profile_rates
    saved = []

    for p0, center in _pick_keys(data, SOURCE_SSRF, max_examples=max_ssrf_examples):
        mask = _event_mask(data, source=SOURCE_SSRF, p0=p0, center_bin=center)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        idx = idx[np.argsort(data['n_steps'][idx])]
        steps, spectra = data['n_steps'][idx], data['spectra'][idx]
        p_tot, q_tot = data['P_total'][idx], data['Q_total'][idx]
        power = data['applied_power'][idx[0]]
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
        ax_ps, ax_q = axes
        show_i = np.unique(np.round(np.linspace(0, idx.size - 1, min(6, idx.size))).astype(int))
        cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(show_i)))
        for color, j in zip(cmap, show_i):
            ip, im = spectra[j, 0], spectra[j, 1]
            ax_ps.plot(f, ip + im, color=color, lw=1.4, label=f'n={steps[j]}')
            ax_q.plot(f, ip - im, color=color, lw=1.2, label=f'n={steps[j]}')
        for ax in axes:
            ax.axvline(f[center], color='0.35', ls=':', lw=1.0, label='center')
            ax.grid(True, alpha=0.3)
        ax_ps.set_ylabel('$P_s = I_+ + I_-$')
        ax_ps.set_title(
            f'ssRF Voigt burn  P0={p0:.3f}  center={center}  γ_rf={power:.1f}  '
            f'P_tot∈[{p_tot.min():.3f},{p_tot.max():.3f}]'
        )
        ax_ps.legend(fontsize=8, ncols=3, loc='upper right')
        ax_q.set_xlabel('R')
        ax_q.set_ylabel('$Q = I_+ - I_-$')
        ax_q.legend(fontsize=8, ncols=3, loc='upper right')
        fig.tight_layout()
        path = plot_dir / f'ssrf_ps_q_steps_P{p0:.2f}_bin{center:04d}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

    for source, n_ex, label, prefix, cmap_fn in (
        (SOURCE_AFP, max_afp_examples, 'AFP', 'afp', plt.cm.plasma),
        (SOURCE_AFP_PROFILE, max_afp_profile_examples, 'AFP Profile', 'afp_profile', plt.cm.cividis),
    ):
        for p0, center in _pick_keys(data, source, max_examples=n_ex):
            mask = _event_mask(data, source=source, p0=p0, center_bin=center)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            idx = idx[np.argsort(data['n_steps'][idx])]
            steps, spectra = data['n_steps'][idx], data['spectra'][idx]
            p_tot, q_tot = data['P_total'][idx], data['Q_total'][idx]
            whole = source == SOURCE_AFP_PROFILE or int(center) < 0
            where = 'Q<0 sweep' if whole else f'center={center}  window={AFP_WINDOW}'
            tag = 'profile' if whole else f'bin{center:04d}'
            fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
            ax_ps, ax_q = axes
            show_i = np.unique(np.round(np.linspace(0, idx.size - 1, min(6, idx.size))).astype(int))
            cmap = cmap_fn(np.linspace(0.15, 0.9, len(show_i)))
            for color, j in zip(cmap, show_i):
                ip, im = spectra[j, 0], spectra[j, 1]
                ax_ps.plot(f, ip + im, color=color, lw=1.4, label=f'n={steps[j]}')
                ax_q.plot(f, ip - im, color=color, lw=1.2, label=f'n={steps[j]}')
            for ax in axes:
                if not whole:
                    ax.axvline(f[center], color='green', ls=':', lw=1.1, label='center')
                ax.grid(True, alpha=0.3)
            ax_ps.set_ylabel('$P_s = I_+ + I_-$')
            ax_ps.set_title(
                f'{label}  P0={p0:.3f}  {where}  P_tot∈[{p_tot.min():.3f},{p_tot.max():.3f}]  '
                f'Q_tot∈[{q_tot.min():.3f},{q_tot.max():.3f}]'
            )
            ax_ps.legend(fontsize=8, ncols=3, loc='upper right')
            ax_q.set_xlabel('R')
            ax_q.set_ylabel('$Q = I_+ - I_-$')
            fig.tight_layout()
            path = plot_dir / f'{prefix}_ps_q_P{p0:.2f}_{tag}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path)

    unmanip = data['source'] == SOURCE_UNMANIP
    if np.any(unmanip):
        for p0 in np.unique(np.round(data['p0'][unmanip], 5))[:max(0, max_unmanip_examples)]:
            mask = _event_mask(data, source=SOURCE_UNMANIP, p0=float(p0))
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            j = int(idx[0])
            ip, im = data['spectra'][j]
            fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
            axes[0].plot(f, ip + im, color='tab:green', lw=1.6, label='$P_s$')
            axes[0].plot(f, ip - im, color='tab:purple', lw=1.2, label='$Q$')
            axes[1].plot(f, ip, color='tab:red', lw=1.5, label='$I_+$')
            axes[1].plot(f, im, color='tab:blue', lw=1.5, label='$I_-$')
            for ax in axes:
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc='upper right')
            axes[0].set_title(
                f'Unmanipulated  P0={p0:.3f}  P_tot={float(data["P_total"][j]):.4f}  '
                f'Q_tot={float(data["Q_total"][j]):.4f}'
            )
            axes[1].set_xlabel('R')
            fig.tight_layout()
            path = plot_dir / f'unmanipulated_P{p0:.2f}.png'
            fig.savefig(path, dpi=140)
            plt.close(fig)
            saved.append(path)

    for p0, scen, lay, center in _pick_combined_example_groups(data):
        mask = _event_mask(
            data, source=SOURCE_SSRF_AFP, p0=p0, center_bin=center, combo_scenario=scen, combo_layout=lay
        )
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        idx = idx[np.argsort(data['n_steps'][idx])]
        steps, spectra = data['n_steps'][idx], data['spectra'][idx]
        p_tot = data['P_total'][idx]
        scen_tag, lay_tag = _combo_scenario_slug(scen), _combo_layout_slug(lay)
        center_tag = 'whole' if int(center) < 0 else f'bin{int(center):04d}'
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
        show_i = np.unique(np.round(np.linspace(0, idx.size - 1, min(6, idx.size))).astype(int))
        cmap = plt.cm.magma(np.linspace(0.15, 0.9, len(show_i)))
        for color, j in zip(cmap, show_i):
            ip, im = spectra[j, 0], spectra[j, 1]
            axes[0].plot(f, ip + im, color=color, lw=1.4, label=f'n={steps[j]}')
            axes[1].plot(f, ip - im, color=color, lw=1.2, label=f'n={steps[j]}')
        axes[0].set_title(f'Combined ssRF+AFP  P0={p0:.3f}  scen={scen} lay={lay}  P∈[{p_tot.min():.3f},{p_tot.max():.3f}]')
        axes[1].set_xlabel('R')
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncols=3, loc='upper right')
        fig.tight_layout()
        path = plot_dir / f'combined_ps_q_{scen_tag}_{lay_tag}_P{p0:.2f}_{center_tag}.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

    for p0 in _pick_keys(data, SOURCE_PROFILE, max_examples=max_ssrf_examples, with_center=False):
        mask = _event_mask(data, source=SOURCE_PROFILE, p0=p0)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        idx = idx[np.argsort(data['n_steps'][idx])]
        steps, spectra = data['n_steps'][idx], data['spectra'][idx]
        p_tot, power = data['P_total'][idx], data['applied_power'][idx[0]]
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        show_i = np.unique(np.round(np.linspace(0, idx.size - 1, min(6, idx.size))).astype(int))
        cmap = plt.cm.cividis(np.linspace(0.15, 0.9, len(show_i)))
        for color, j in zip(cmap, show_i):
            ax.plot(f, spectra[j, 0] + spectra[j, 1], color=color, lw=2.0, label=f'$n={steps[j]}$')
        ax.set_xlabel('$R$')
        ax.set_xlim(-6, 6)
        ax.set_ylabel('Intensity (arb. units)')
        ax.set_title(f'Optimal RF profile  $P_0={p0:.3f}$  $U_\\mathrm{{max}}={power:.2f}$  $P\\in[{p_tot.min():.3f},{p_tot.max():.3f}]$')
        ax.legend(fontsize=12, ncols=2, loc='upper right', frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        path = plot_dir / f'profile_ps_q_steps_P{p0:.2f}.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)
        rates_info = profile_rates.get(round(float(p0), 5))
        rates = None
        f_u, peak = f, float(power)
        if rates_info is not None:
            rates = np.asarray(rates_info['rates'], dtype=float).reshape(-1)
            f_u = np.asarray(rates_info.get('frequency', f), dtype=float).reshape(-1)
            peak = float(rates_info.get('peak_u', power))
        else:
            rates = np.asarray(data['power_profile'][idx[0]], dtype=float).reshape(-1)
        if rates is not None and rates.size and f_u.size == rates.size:
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            ax.step(f_u, rates, where='mid', color='#1b7f4e', lw=2.0)
            ax.fill_between(f_u, rates, step='mid', color='#1b7f4e', alpha=0.22)
            ax.set_xlabel('$R$')
            ax.set_ylabel('$U(R)$')
            ax.set_title(f'Applied RF power profile  $P_0={p0:.3f}$  $U_\\mathrm{{max}}={peak:.3f}$')
            _style_axes(ax)
            fig.tight_layout()
            path = plot_dir / f'profile_applied_power_P{p0:.2f}.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)
    return saved
