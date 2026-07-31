"""
Multi-bin ssRF burn profiles for ssrf_realtime_v2.

Each bin in the local support receives RF power from a discretized Voigt
envelope peaked at the burn center. ssRF is applied independently at every
active bin via ``ssrf_subset_indices`` (see ``Spin1Model.ssrf_burn``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import wofz

from .model import Spin1Model

HALF_WIDTH = 5
PROFILE_REL_THRESHOLD = 0.05
SIGMA_BINS = 2.0
VOIGT_GAMMA_BINS = 1.0


def mirror_bin_idx(n_bins: int, bin_idx: int) -> int:
    return int(n_bins) - 1 - int(bin_idx)


def ssrf_touched_bins(n_bins: int, subset: list[int] | np.ndarray) -> list[int]:
    """Packet/intensity bins ssRF changes: each burn index i also updates mirror(i)."""
    touched: set[int] = set()
    for i in subset:
        touched.add(int(i))
        touched.add(mirror_bin_idx(n_bins, int(i)))
    return sorted(touched)


def _voigt_kernel(x: np.ndarray, x0: float, sigma: float, lorentz_gamma: float) -> np.ndarray:
    """Discretized Voigt (Faddeeva), same form as ``ssRFMapper._voigt_profile``."""
    sigma = max(float(sigma), 1e-12)
    x_norm = (np.asarray(x, dtype=float) - float(x0)) / (sigma * np.sqrt(2.0))
    z = x_norm + 1j * (float(lorentz_gamma) / (sigma * np.sqrt(2.0)))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))


def make_voigt_rf_profile(
    n_bins: int,
    center: int,
    gamma_rf: float,
    *,
    sigma: float = SIGMA_BINS,
    lorentz_gamma: float = VOIGT_GAMMA_BINS,
    half_width: int | None = None,
    rel_threshold: float = PROFILE_REL_THRESHOLD,
) -> tuple[np.ndarray, list[int]]:
    """
    Rounded Voigt RF envelope on discrete bins, peaked at ``center``.

    The Voigt is sampled on bin indices within ``center ± half_width``,
    normalized so ``profile[center] == gamma_rf``, then bins below
    ``rel_threshold * |gamma_rf|`` are dropped from support.
    """
    n_bins = int(n_bins)
    c = int(np.clip(int(center), 0, n_bins - 1))
    hw = HALF_WIDTH if half_width is None else max(0, int(half_width))
    g_peak = float(gamma_rf)
    floor = float(rel_threshold) * abs(g_peak)

    lo = max(0, c - hw)
    hi = min(n_bins - 1, c + hw)
    xs = np.arange(lo, hi + 1, dtype=float)
    kernel = _voigt_kernel(xs, float(c), float(sigma), float(lorentz_gamma))
    peak = float(kernel[c - lo]) if kernel.size else 0.0

    profile = np.zeros(n_bins, dtype=float)
    if peak <= 0.0:
        profile[c] = g_peak
        return profile, [c]

    weights = kernel / peak
    profile[lo : hi + 1] = g_peak * weights
    profile[c] = g_peak

    support_idx = np.flatnonzero(np.abs(profile) >= floor)
    if support_idx.size == 0:
        profile[c] = g_peak
        return profile, [c]

    compact = np.zeros(n_bins, dtype=float)
    compact[support_idx] = profile[support_idx]
    support = [int(i) for i in support_idx]
    return compact, support


def make_multi_bin_rf_profile(
    n_bins: int,
    center: int,
    gamma_rf: float,
    *,
    half_width: int = HALF_WIDTH,
    rel_threshold: float = PROFILE_REL_THRESHOLD,
    sigma: float = SIGMA_BINS,
    lorentz_gamma: float = VOIGT_GAMMA_BINS,
) -> tuple[np.ndarray, list[int]]:
    """Alias for the discrete Voigt multi-bin profile."""
    return make_voigt_rf_profile(
        n_bins,
        center,
        gamma_rf,
        sigma=sigma,
        lorentz_gamma=lorentz_gamma,
        half_width=half_width,
        rel_threshold=rel_threshold,
    )


def freeze_rf_profile(model: Spin1Model, profile: np.ndarray) -> Callable[[], None]:
    """Keep ``params.rf_profile`` fixed; ``ssrf_burn`` always calls ``set_rf_profile``."""
    frozen = np.asarray(profile, dtype=float).copy()
    model.params.rf_profile = frozen.copy()
    model._rf_profile_frozen = True

    def _frozen_set_rf_profile() -> None:
        model.params.rf_profile = frozen.copy()

    model.set_rf_profile = _frozen_set_rf_profile  # type: ignore[method-assign]
    return _frozen_set_rf_profile


def unfreeze_rf_profile(model: Spin1Model) -> None:
    """Restore dynamic Q-shaped ``set_rf_profile`` and clear frozen discrete state."""
    model._rf_profile_frozen = False
    model.params.rf_profile = None
    model.params.ssrf_subset_indices = None
    if hasattr(model, "invalidate_rf_profile"):
        model.invalidate_rf_profile()
    model.set_rf_profile = Spin1Model.set_rf_profile.__get__(model, type(model))  # type: ignore[method-assign]
    model.set_rf_profile()
