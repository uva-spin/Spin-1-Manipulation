"""Intensity/population conversions matching the spin-1 ss-RF realtime model."""
import numpy as np
(PLUS, ZERO, MINUS) = (0, 1, 2)

def transition_differences(n):
    """Return packet-space I+ and I- transition differences from populations."""
    return (n[:, PLUS] - n[:, ZERO], n[:, ZERO] - n[:, MINUS])

def packet_differences_to_physical_intensities(diff_plus, diff_minus, Rplus, *, display_cal, dR):
    """
    Project packet transition differences to physical R-bin intensities.

    I+(R_k) uses packet k.  I-(R_k) uses the packet at -R_k (mirror index).
    """
    n_bins = len(Rplus)
    scale = display_cal / dR
    Iplus = scale * np.asarray(diff_plus)
    Iminus = np.zeros(n_bins)
    for k in range(n_bins):
        mirror = n_bins - 1 - k
        Iminus[k] = scale * diff_minus[mirror]
    return (Iplus, Iminus)

def physical_intensities_to_packet_n(Iplus, Iminus, mu, *, display_cal, dR, min_population=1e-30):
    """
    Recover packet populations from physical R-grid intensities.

    Each packet k carries fixed weight mu[k] and obeys

        n[k,+] - n[k,0] = I+(R_k) * dR / display_cal
        n[k,0] - n[k,-] = I-(R_{-k}) * dR / display_cal

    with n[k,+] + n[k,0] + n[k,-] = mu[k].
    """
    Iplus = np.asarray(Iplus)
    Iminus = np.asarray(Iminus)
    mu = np.asarray(mu)
    n_bins = len(Iplus)
    inv_scale = dR / display_cal
    n = np.zeros((n_bins, 3))
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

def packet_n_to_physical_intensities(n, Rplus, *, display_cal, dR):
    """Convert packet populations to physical R-axis intensities and total signal."""
    (diff_plus, diff_minus) = transition_differences(n)
    (Iplus, Iminus) = packet_differences_to_physical_intensities(diff_plus, diff_minus, Rplus, display_cal=display_cal, dR=dR)
    return (Iplus, Iminus, Iplus + Iminus)
