import numpy as np
from bin_setup import equilibrium_lineshape, polarization_grid
from common import BURN_BIN_CHOICES, F_MAX, F_MIN, NUM_BINS, P_ABS_MIN, EXCLUDED_MANIPULATION_BURN_BINS, is_burn_bin

def positive_polarization_grid(p_min, p_max, p_step, *, p_abs_min=P_ABS_MIN):
    """Polarization grid with only strictly positive P values."""
    g = polarization_grid(p_min, p_max, p_step)
    return g[g > p_abs_min]

def equilibrium_q_profile(polarization, *, num_bins=NUM_BINS, shape_params=None, r_min=None, r_max=None):
    """Equilibrium Q = I+ - I- at each spectral bin."""
    shape = shape_params
    r_lo = F_MIN if r_min is None else r_min
    r_hi = F_MAX if r_max is None else r_max
    f = np.linspace(r_lo, r_hi, num_bins)
    (_, ip, im) = equilibrium_lineshape(polarization, f, shape)
    return np.asarray(ip) - np.asarray(im)

def q_negative_burn_mask(polarization, *, num_bins=NUM_BINS, shape_params=None):
    """True at burn-window bins where equilibrium Q < 0."""
    q = equilibrium_q_profile(polarization, num_bins=num_bins, shape_params=shape_params)
    mask = np.zeros(num_bins, dtype=bool)
    for b in BURN_BIN_CHOICES:
        if q[b] < 0.0:
            mask[b] = True
    return mask

def is_q_negative_burn_center(polarization, bin_idx, *, num_bins=NUM_BINS, shape_params=None):
    """True when ``bin_idx`` is in the burn window and equilibrium Q < 0."""
    if not is_burn_bin(bin_idx):
        return False
    q = equilibrium_q_profile(polarization, num_bins=num_bins, shape_params=shape_params)
    return q[bin_idx] < 0.0

def border_neighbor_mask(polarization, *, num_bins=NUM_BINS, shape_params=None):
    """Burn-window bins with Q >= 0 adjacent to a Q < 0 bin."""
    q = equilibrium_q_profile(polarization, num_bins=num_bins, shape_params=shape_params)
    qneg = q_negative_burn_mask(polarization, num_bins=num_bins, shape_params=shape_params)
    border = np.zeros(num_bins, dtype=bool)
    for b in BURN_BIN_CHOICES:
        bi = b
        if q[bi] >= 0.0:
            for nb in (bi - 1, bi + 1):
                if 0 <= nb < num_bins and qneg[nb]:
                    border[bi] = True
                    break
    return border

def neighbor_border_offsets(q_eq, burn_bin, *, num_bins=NUM_BINS):
    """Offsets (-1, +1) of border neighbors to record when burning at ``burn_bin``."""
    if q_eq[burn_bin] >= 0.0:
        return ()
    offsets = []
    for d in (-1, 1):
        nb = burn_bin + d
        if 0 <= nb < num_bins and is_burn_bin(nb) and (q_eq[nb] >= 0.0):
            offsets.append(d)
    return tuple(offsets)

def union_q_negative_burn_centers(p_values, *, num_bins=NUM_BINS, shape_params=None):
    """Sorted burn-window bins that are Q < 0 for at least one P in ``p_values``."""
    union = np.zeros(num_bins, dtype=bool)
    for p0 in np.asarray(p_values):
        union |= q_negative_burn_mask(p0, num_bins=num_bins, shape_params=shape_params)
    return np.flatnonzero(union).astype(int)

def manipulation_shard_bins(*, num_bins=NUM_BINS):
    """Burn-window bins that receive ssRF/AFP shard generation (minus exclusions)."""
    choices = np.asarray(BURN_BIN_CHOICES, dtype=int)
    valid = choices[(choices >= 0) & (choices < num_bins)]
    excluded = {b for b in EXCLUDED_MANIPULATION_BURN_BINS if 0 <= b < num_bins}
    return frozenset((b for b in valid if b not in excluded))

def is_manipulation_shard_bin(bin_idx):
    """True when ``bin_idx`` may be an ssRF/AFP manipulation center."""
    bi = bin_idx
    return is_burn_bin(bi) and bi not in EXCLUDED_MANIPULATION_BURN_BINS
