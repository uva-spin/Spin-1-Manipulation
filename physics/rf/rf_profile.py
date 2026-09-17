"""
Multi-bin ssRF burn profiles for ssrf_realtime.

Each bin in the local support receives RF power from a discretized Voigt
envelope peaked at the burn center. ssRF is applied independently at every
active bin via ``ssrf_subset_indices`` (see ``Spin1Model.ssrf_burn``).
"""
import numpy as np
from scipy.special import wofz
from .model import Spin1Model
HALF_WIDTH = 5
PROFILE_REL_THRESHOLD = 0.05
SIGMA_BINS = 2.0
VOIGT_GAMMA_BINS = 1.0

def mirror_bin_idx(n_bins, bin_idx):
    return n_bins - 1 - bin_idx

def ssrf_touched_bins(n_bins, subset):
    """Packet/intensity bins ssRF changes: each burn index i also updates mirror(i)."""
    touched = set()
    for i in subset:
        touched.add(i)
        touched.add(mirror_bin_idx(n_bins, i))
    return sorted(touched)

def _voigt_kernel(x, x0, sigma, lorentz_gamma):
    """Discretized Voigt (Faddeeva), same form as ``ssRFMapper._voigt_profile``."""
    sigma = max(sigma, 1e-12)
    x_norm = (np.asarray(x) - x0) / (sigma * np.sqrt(2.0))
    z = x_norm + 1j * (lorentz_gamma / (sigma * np.sqrt(2.0)))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))

def make_voigt_rf_profile(n_bins, center, gamma_rf, *, sigma=SIGMA_BINS, lorentz_gamma=VOIGT_GAMMA_BINS, half_width=None, rel_threshold=PROFILE_REL_THRESHOLD):
    """
    Rounded Voigt RF envelope on discrete bins, peaked at ``center``.

    The Voigt is sampled on bin indices within ``center ± half_width``,
    normalized so ``profile[center] == gamma_rf``, then bins below
    ``rel_threshold * |gamma_rf|`` are dropped from support.
    """
    n_bins = n_bins
    c = np.clip(center, 0, n_bins - 1)
    hw = HALF_WIDTH if half_width is None else max(0, half_width)
    g_peak = gamma_rf
    floor = rel_threshold * abs(g_peak)
    lo = max(0, c - hw)
    hi = min(n_bins - 1, c + hw)
    xs = np.arange(lo, hi + 1)
    kernel = _voigt_kernel(xs, c, sigma, lorentz_gamma)
    peak = kernel[c - lo] if kernel.size else 0.0
    profile = np.zeros(n_bins)
    if peak <= 0.0:
        profile[c] = g_peak
        return (profile, [c])
    weights = kernel / peak
    profile[lo:hi + 1] = g_peak * weights
    profile[c] = g_peak
    support_idx = np.flatnonzero(np.abs(profile) >= floor)
    if support_idx.size == 0:
        profile[c] = g_peak
        return (profile, [c])
    compact = np.zeros(n_bins)
    compact[support_idx] = profile[support_idx]
    support = [i for i in support_idx]
    return (compact, support)

def make_multi_bin_rf_profile(n_bins, center, gamma_rf, *, half_width=HALF_WIDTH, rel_threshold=PROFILE_REL_THRESHOLD, sigma=SIGMA_BINS, lorentz_gamma=VOIGT_GAMMA_BINS):
    """Alias for the discrete Voigt multi-bin profile."""
    return make_voigt_rf_profile(n_bins, center, gamma_rf, sigma=sigma, lorentz_gamma=lorentz_gamma, half_width=half_width, rel_threshold=rel_threshold)

def freeze_rf_profile(model, profile):
    """Keep ``params.rf_profile`` fixed; ``ssrf_burn`` always calls ``set_rf_profile``."""
    frozen = np.asarray(profile).copy()
    model.params.rf_profile = frozen.copy()
    model._rf_profile_frozen = True

    def _frozen_set_rf_profile():
        model.params.rf_profile = frozen.copy()
    model.set_rf_profile = _frozen_set_rf_profile
    return _frozen_set_rf_profile

def unfreeze_rf_profile(model):
    """Restore dynamic Q-shaped ``set_rf_profile`` and clear frozen discrete state."""
    model._rf_profile_frozen = False
    model.params.rf_profile = None
    model.params.ssrf_subset_indices = None
    if hasattr(model, 'invalidate_rf_profile'):
        model.invalidate_rf_profile()
    model.set_rf_profile = Spin1Model.set_rf_profile.__get__(model, type(model))
    model.set_rf_profile()
