"""Physical-R bin-integrated Voigt RF rate profiles (from spin1_ssrf_realtime_voigt_burn)."""
from functools import lru_cache
import numpy as np
try:
    from scipy.special import voigt_profile as _scipy_voigt_profile
except Exception:
    _scipy_voigt_profile = None
_SQRT_2LN2 = np.sqrt(2.0 * np.log(2.0))

def gaussian_sigma_from_fwhm(fwhm):
    return max(0.0, fwhm) / (2.0 * _SQRT_2LN2)

def lorentzian_hwhm_from_fwhm(fwhm):
    return 0.5 * max(0.0, fwhm)

def approximate_voigt_fwhm(gaussian_fwhm, lorentzian_fwhm):
    g = max(0.0, gaussian_fwhm)
    l = max(0.0, lorentzian_fwhm)
    if g == 0.0:
        return l
    if l == 0.0:
        return g
    return 0.5346 * l + np.sqrt(0.2166 * l * l + g * g)

def _pseudo_voigt_peak_normalized(x, gaussian_fwhm, lorentzian_fwhm):
    x = np.asarray(x)
    g = max(0.0, gaussian_fwhm)
    l = max(0.0, lorentzian_fwhm)
    tiny = 1e-30
    if g <= tiny and l <= tiny:
        return np.zeros_like(x)
    if l <= tiny:
        return np.exp(-4.0 * np.log(2.0) * (x / max(g, tiny)) ** 2)
    if g <= tiny:
        return 1.0 / (1.0 + 4.0 * (x / max(l, tiny)) ** 2)
    f = (g ** 5 + 2.69269 * g ** 4 * l + 2.42843 * g ** 3 * l ** 2 + 4.47163 * g ** 2 * l ** 3 + 0.07842 * g * l ** 4 + l ** 5) ** 0.2
    ratio = np.clip(l / max(f, tiny), 0.0, 1.0)
    eta = np.clip(1.36603 * ratio - 0.47719 * ratio ** 2 + 0.11116 * ratio ** 3, 0.0, 1.0)
    gaussian = np.exp(-4.0 * np.log(2.0) * (x / max(f, tiny)) ** 2)
    lorentzian = 1.0 / (1.0 + 4.0 * (x / max(f, tiny)) ** 2)
    return eta * lorentzian + (1.0 - eta) * gaussian

def voigt_peak_normalized(x, gaussian_fwhm, lorentzian_fwhm):
    arr = np.asarray(x)
    g = max(0.0, gaussian_fwhm)
    l = max(0.0, lorentzian_fwhm)
    tiny = 1e-30
    if g <= tiny and l <= tiny:
        return np.zeros_like(arr)
    if _scipy_voigt_profile is None:
        return _pseudo_voigt_peak_normalized(arr, g, l)
    sigma = gaussian_sigma_from_fwhm(g)
    gamma = lorentzian_hwhm_from_fwhm(l)
    values = _scipy_voigt_profile(arr, sigma, gamma)
    peak = _scipy_voigt_profile(0.0, sigma, gamma)
    if not np.isfinite(peak) or peak <= 0.0:
        return _pseudo_voigt_peak_normalized(arr, g, l)
    return np.asarray(values / peak)

@lru_cache(maxsize=32)
def _legendre_rule(order):
    (nodes, weights) = np.polynomial.legendre.leggauss(max(2, order))
    return (nodes.astype(float), weights.astype(float))

def recommended_quadrature_order(bin_width, gaussian_fwhm, lorentzian_fwhm, minimum=16, maximum=256):
    width = approximate_voigt_fwhm(gaussian_fwhm, lorentzian_fwhm)
    if width <= 0.0:
        return minimum
    raw = np.ceil(12.0 * abs(bin_width) / max(width, 1e-15))
    return int(np.clip(max(minimum, raw), minimum, maximum))

def bin_averaged_voigt(bin_centers, center_R, bin_width_R, gaussian_fwhm_R, lorentzian_fwhm_R, normalization='center_bin', quadrature_order=0):
    centers = np.asarray(bin_centers)
    if centers.ndim != 1:
        raise ValueError('bin_centers must be one-dimensional')
    dR = abs(bin_width_R)
    if dR <= 0.0:
        raise ValueError('bin_width_R must be positive')
    g = max(0.0, gaussian_fwhm_R)
    l = max(0.0, lorentzian_fwhm_R)
    if g == 0.0 and l == 0.0:
        out = np.zeros_like(centers)
        if out.size:
            out[np.argmin(np.abs(centers - center_R))] = 1.0
        return out
    order = quadrature_order
    if order <= 0:
        order = recommended_quadrature_order(dR, g, l)
    order = int(order)
    (nodes, weights) = _legendre_rule(order)
    sample_R = centers[:, None] + 0.5 * dR * nodes[None, :]
    values = voigt_peak_normalized(sample_R - center_R, g, l)
    averaged = 0.5 * np.sum(values * weights[None, :], axis=1)
    averaged = np.maximum(np.nan_to_num(averaged, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    mode = str(normalization).lower()
    if mode == 'center_bin':
        peak = np.max(averaged) if averaged.size else 0.0
        if peak > 0.0 and np.isfinite(peak):
            averaged = averaged / peak
    elif mode != 'continuous_peak':
        raise ValueError("normalization must be 'center_bin' or 'continuous_peak'")
    return averaged

def implementation_name():
    return 'SciPy exact Voigt' if _scipy_voigt_profile is not None else 'pseudo-Voigt fallback'

def discrete_bins_to_physical_fwhm(sigma_bins, lorentz_gamma_bins, bin_width_R):
    """Map legacy discrete-bin Voigt widths to physical-R FWHM values."""
    dR = abs(bin_width_R)
    return (sigma_bins * dR, lorentz_gamma_bins * dR)
