import numpy as np

from bin_setup import equilibrium_lineshape, get_shape_params, polarization_grid
from common import BURN_BIN_CHOICES, F_MAX, F_MIN, NUM_BINS, P_ABS_MIN, EXCLUDED_MANIPULATION_BURN_BINS, is_burn_bin


def positive_polarization_grid(
    p_min: float,
    p_max: float,
    p_step: float,
    *,
    p_abs_min: float = P_ABS_MIN,
) -> np.ndarray:
    """Polarization grid with only strictly positive P values."""
    g = polarization_grid(float(p_min), float(p_max), float(p_step))
    return g[g > float(p_abs_min)]


def equilibrium_q_profile(
    polarization: float,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> np.ndarray:
    """Equilibrium Q = I+ - I- at each spectral bin."""
    shape = shape_params if shape_params is not None else get_shape_params()
    f = np.linspace(float(F_MIN), float(F_MAX), int(num_bins))
    _, ip, im = equilibrium_lineshape(float(polarization), f, shape)
    return np.asarray(ip, dtype=float) - np.asarray(im, dtype=float)


def q_negative_burn_mask(
    polarization: float,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> np.ndarray:
    """True at burn-window bins where equilibrium Q < 0 (diagnostic helper)."""
    q = equilibrium_q_profile(
        polarization, num_bins=int(num_bins), shape_params=shape_params
    )
    mask = np.zeros(int(num_bins), dtype=bool)
    for b in BURN_BIN_CHOICES:
        if q[int(b)] < 0.0:
            mask[int(b)] = True
    return mask


def is_q_negative_burn_center(
    polarization: float,
    bin_idx: int,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> bool:
    """True when ``bin_idx`` is in the burn window and equilibrium Q < 0."""
    if not is_burn_bin(int(bin_idx)):
        return False
    q = equilibrium_q_profile(
        polarization, num_bins=int(num_bins), shape_params=shape_params
    )
    return float(q[int(bin_idx)]) < 0.0


def border_neighbor_mask(
    polarization: float,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> np.ndarray:
    """Burn-window bins with Q >= 0 adjacent to a Q < 0 bin (spillover targets)."""
    q = equilibrium_q_profile(
        polarization, num_bins=int(num_bins), shape_params=shape_params
    )
    qneg = q_negative_burn_mask(
        polarization, num_bins=int(num_bins), shape_params=shape_params
    )
    border = np.zeros(int(num_bins), dtype=bool)
    for b in BURN_BIN_CHOICES:
        bi = int(b)
        if q[bi] >= 0.0:
            for nb in (bi - 1, bi + 1):
                if 0 <= nb < int(num_bins) and qneg[nb]:
                    border[bi] = True
                    break
    return border


def neighbor_border_offsets(
    q_eq: np.ndarray,
    burn_bin: int,
    *,
    num_bins: int = NUM_BINS,
) -> tuple[int, ...]:
    """Offsets (-1, +1) of border neighbors to record when burning at ``burn_bin``."""
    if float(q_eq[int(burn_bin)]) >= 0.0:
        return ()
    offsets: list[int] = []
    for d in (-1, 1):
        nb = int(burn_bin) + d
        if 0 <= nb < int(num_bins) and is_burn_bin(nb) and float(q_eq[nb]) >= 0.0:
            offsets.append(d)
    return tuple(offsets)


def union_q_negative_burn_centers(
    p_values: np.ndarray,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> np.ndarray:
    """Sorted burn-window bin indices that are Q < 0 for at least one P in ``p_values``."""
    shape = shape_params if shape_params is not None else get_shape_params()
    union = np.zeros(int(num_bins), dtype=bool)
    for p0 in np.asarray(p_values, dtype=float):
        union |= q_negative_burn_mask(
            float(p0), num_bins=int(num_bins), shape_params=shape
        )
    return np.flatnonzero(union).astype(int)


def manipulation_shard_bins(
    p_values: np.ndarray | None = None,
    *,
    num_bins: int = NUM_BINS,
) -> frozenset[int]:
    """Burn-window bins that receive ssRF/AFP shard generation (any Q sign, minus exclusions)."""
    del p_values  # kept for API compatibility with older call sites
    choices = np.asarray(BURN_BIN_CHOICES, dtype=int)
    valid = choices[(choices >= 0) & (choices < int(num_bins))]
    excluded = {int(b) for b in EXCLUDED_MANIPULATION_BURN_BINS if 0 <= int(b) < int(num_bins)}
    return frozenset(int(b) for b in valid if int(b) not in excluded)


def is_manipulation_shard_bin(
    bin_idx: int,
    p_values: np.ndarray | None = None,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> bool:
    """True when ``bin_idx`` may be an ssRF/AFP manipulation center."""
    del p_values, num_bins, shape_params
    bi = int(bin_idx)
    return is_burn_bin(bi) and bi not in EXCLUDED_MANIPULATION_BURN_BINS


# Backward-compatible aliases
is_q_negative_burn_shard_bin = is_manipulation_shard_bin
is_ssrf_burn_shard_bin = is_manipulation_shard_bin


def required_manipulation_shard_bins(
    p_values: np.ndarray,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> frozenset[int]:
    """Burn bins that must have their own ssRF/AFP shard (full burn window)."""
    del shape_params
    return manipulation_shard_bins(p_values, num_bins=int(num_bins))
