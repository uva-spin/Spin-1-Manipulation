"""Intensity/population conversions matching the spin-1 ss-RF realtime model."""

from __future__ import annotations

import numpy as np

PLUS, ZERO, MINUS = 0, 1, 2


def transition_differences(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return packet-space I+ and I- transition differences from populations."""
    return n[:, PLUS] - n[:, ZERO], n[:, ZERO] - n[:, MINUS]


def packet_differences_to_physical_intensities(
    diff_plus: np.ndarray,
    diff_minus: np.ndarray,
    Rplus: np.ndarray,
    *,
    display_cal: float,
    dR: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project packet transition differences to physical R-bin intensities.

    I+(R_k) uses packet k.  I-(R_k) uses the packet at -R_k (mirror index).
    """
    n_bins = len(Rplus)
    scale = float(display_cal) / float(dR)
    Iplus = scale * np.asarray(diff_plus, dtype=float)
    Iminus = np.zeros(n_bins, dtype=float)
    for k in range(n_bins):
        mirror = n_bins - 1 - k
        Iminus[k] = scale * diff_minus[mirror]
    return Iplus, Iminus


def physical_intensities_to_packet_n(
    Iplus: np.ndarray,
    Iminus: np.ndarray,
    mu: np.ndarray,
    *,
    display_cal: float,
    dR: float,
    min_population: float = 1e-30,
) -> np.ndarray:
    """
    Recover packet populations from physical R-grid intensities.

    Each packet k carries fixed weight mu[k] and obeys

        n[k,+] - n[k,0] = I+(R_k) * dR / display_cal
        n[k,0] - n[k,-] = I-(R_{-k}) * dR / display_cal

    with n[k,+] + n[k,0] + n[k,-] = mu[k].
    """
    Iplus = np.asarray(Iplus, dtype=float)
    Iminus = np.asarray(Iminus, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n_bins = len(Iplus)

    inv_scale = float(dR) / float(display_cal)
    n = np.zeros((n_bins, 3), dtype=float)
    for k in range(n_bins):
        mirror = n_bins - 1 - k
        a = Iplus[k] * inv_scale
        b = Iminus[mirror] * inv_scale
        n_zero = (mu[k] - a + b) / 3.0
        n[k, ZERO] = n_zero
        n[k, PLUS] = n_zero + a
        n[k, MINUS] = n_zero - b

    n = np.maximum(n, min_population)
    row_sums = n.sum(axis=1, keepdims=True)
    n *= mu[:, None] / np.maximum(row_sums, min_population)
    return n


def packet_n_to_physical_intensities(
    n: np.ndarray,
    Rplus: np.ndarray,
    *,
    display_cal: float,
    dR: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert packet populations to physical R-axis intensities and total signal."""
    diff_plus, diff_minus = transition_differences(n)
    Iplus, Iminus = packet_differences_to_physical_intensities(
        diff_plus, diff_minus, Rplus, display_cal=display_cal, dR=dR
    )
    return Iplus, Iminus, Iplus + Iminus
