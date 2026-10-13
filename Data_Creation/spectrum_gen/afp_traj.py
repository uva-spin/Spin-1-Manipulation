"""Full-spectrum AFP trajectories (no per-bin shard I/O)."""
from __future__ import annotations

import numpy as np

from physics.rf import mirror_bin_idx

from .bridge import (
    afp_touched_bins,
    afp_window_indices,
    build_equilibrium_spin1_model,
    commit_touched_bins_only,
    configure_afp_recovery,
    full_spectrum_intensities,
    level_pq,
)
from .constants import (
    AFP_CENTER_MARGIN,
    AFP_EFFICIENCY,
    AFP_WINDOW,
    DIFFUSION_SCALE,
    DT,
    F_MAX,
    F_MIN,
    NUM_BINS,
)


def qneg_sweep_indices(q, allowed=None):
    """Bins with Q < 0, keeping at most one index from each mirror pair."""
    q = np.asarray(q)
    neg = np.flatnonzero(q < 0.0)
    if allowed is not None:
        allow = np.zeros(q.size, dtype=bool)
        idx = np.asarray(list(allowed), dtype=int)
        idx = idx[(idx >= 0) & (idx < q.size)]
        allow[idx] = True
        neg = neg[allow[neg]]
    if neg.size == 0:
        return []
    n = q.size
    chosen, blocked = [], set()
    for i in neg[np.argsort(q[neg])]:
        i = int(i)
        if i in blocked:
            continue
        chosen.append(i)
        blocked.add(i)
        blocked.add(n - 1 - i)
    return sorted(chosen)


def run_one_polarization(
    bin_idx,
    polarization,
    *,
    num_bins=NUM_BINS,
    dt=DT,
    n_relax=0,
    afp_window=AFP_WINDOW,
    afp_efficiency=AFP_EFFICIENCY,
    diffusion_scale=DIFFUSION_SCALE,
    capture_spectrum=True,
    r_min=None,
    r_max=None,
    legacy_spectral_recovery=False,
    subset_indices=None,
    qneg_bins=None,
):
    r_lo = F_MIN if r_min is None else r_min
    r_hi = F_MAX if r_max is None else r_max
    model = build_equilibrium_spin1_model(
        polarization=polarization,
        num_bins=num_bins,
        dt=dt,
        rf_enabled=False,
        relax_enabled=True,
        diffusion_scale=diffusion_scale,
        r_min=r_lo,
        r_max=r_hi,
        legacy_spectral_recovery=legacy_spectral_recovery,
    )
    ip0, im0, _ = full_spectrum_intensities(model)
    q_eq = np.asarray(ip0) - np.asarray(im0)
    if qneg_bins is not None:
        subset = qneg_sweep_indices(q_eq, qneg_bins)
    elif subset_indices is None:
        subset = afp_window_indices(bin_idx, num_bins, window=afp_window)
    else:
        subset = [int(i) for i in subset_indices if 0 <= int(i) < num_bins]
    if subset and (qneg_bins is not None or subset_indices is not None):
        bin_idx = int(subset[int(np.argmin(q_eq[subset]))])
    if not subset:
        return {
            'skipped': True,
            'afp_subset': [],
            'p_full': None,
            'q_full': None,
            'iplus_full': None,
            'iminus_full': None,
        }

    touched = afp_touched_bins(num_bins, subset)
    model.params.afp_enabled = True
    model.params.afp_efficiency = afp_efficiency
    model.params.afp_center_margin = AFP_CENTER_MARGIN
    model.params.afp_preserve_intensity_area = False
    model.params.afp_subset_indices = list(subset)
    model._afp_pending = True
    p_initial, q_initial = level_pq(model)
    model.afp_sweep()
    model.params.afp_enabled = False
    model._afp_pending = False
    ip_sim, im_sim, _ = model.physical_intensities()
    base_ip, base_im = commit_touched_bins_only(ip0, im0, ip_sim, im_sim, touched)
    model.load_from_physical_intensities(base_ip, base_im)
    configure_afp_recovery(model)

    t_len = int(n_relax) + 1
    p_full = np.empty(t_len)
    q_full = np.empty(t_len)
    iplus_full = iminus_full = None
    if capture_spectrum:
        iplus_full = np.empty((t_len, num_bins))
        iminus_full = np.empty((t_len, num_bins))

    def _record(k):
        pol = model.polarizations()
        p_full[k] = pol['P']
        q_full[k] = pol['Q']
        if capture_spectrum:
            ip, im, _ = full_spectrum_intensities(model)
            iplus_full[k] = ip
            iminus_full[k] = im

    _record(0)
    for k in range(1, t_len):
        model.step_once(dt=dt, rf_on=False, dnp_on=False, copy=False)
        _record(k)

    return {
        'skipped': False,
        'afp_subset': subset,
        'iplus_full': iplus_full,
        'iminus_full': iminus_full,
        'p_full': p_full,
        'q_full': q_full,
        'p_initial': float(p_initial),
        'q_initial': float(q_initial),
        'p_final': float(p_full[-1]),
        'q_final': float(q_full[-1]),
        'center_bin': int(bin_idx),
        'frequency': np.asarray(model.Rplus),
    }
