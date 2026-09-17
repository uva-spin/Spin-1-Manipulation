"""Dulya/Hamada lineshape from frozen single-site fit parameters."""
import json
from pathlib import Path
import numpy as np
from dulya_kernel import polynomial_background, qmeter_gain, site_transition_components
_HERE = Path(__file__).resolve().parent
DEFAULT_FIT_PARAMS_PATH = _HERE / 'fit_params.json'
SHAPE_KEYS = ('amp', 'center', 'cc', 'split', 'sigma', 'eta', 'xi', 'b0', 'b1', 'b2', 'b3')

def load_fit_params(path=None):
    path = Path(path) if path is not None else DEFAULT_FIT_PARAMS_PATH
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def shape_params_from_fit(fit_blob=None):
    """Frozen fit params used for all polarizations."""
    p = fit_blob if fit_blob is not None else load_fit_params()
    missing = [k for k in SHAPE_KEYS if k not in p]
    if missing:
        raise KeyError(f'fit_params missing required keys {missing}; got {sorted(p)}')
    return {k: p[k] for k in SHAPE_KEYS}

def _build_dulya_intensities(P, x, shape_params, *, wd=16.35, exact_intensity=True, nphi=64):
    """Fit-scale Ps, I+, I- before optional P normalization."""
    P_clip = np.clip(P, -0.999999, 0.999999)
    amp = shape_params['amp']
    split = shape_params['split']
    sigma = shape_params['sigma']
    eta = shape_params['eta']
    xi = shape_params['xi']
    x_eff = shape_params['cc'] * (x - shape_params['center'])
    (plus, minus) = site_transition_components(x_eff, P_clip, split, sigma, eta, wd=wd, exact_intensity=exact_intensity, nphi=nphi)
    gain = qmeter_gain(x_eff, split, xi)
    background = polynomial_background(x, shape_params['b0'], shape_params['b1'], shape_params['b2'], shape_params['b3'])
    iplus = amp * np.asarray(plus) * gain + 0.5 * background
    iminus = amp * np.asarray(minus) * gain + 0.5 * background
    ps = iplus + iminus
    return (ps, iplus, iminus)

def GenerateDulyaLineshape(P, x, shape_params=None, *, wd=16.35, exact_intensity=True, nphi=64):
    """I+/I- at fit signal scale (amp, Q-meter gain, baseline)."""
    P = P
    x = np.asarray(x).reshape(-1)
    if shape_params is None:
        shape_params = shape_params_from_fit()
    if abs(P) < 1e-15:
        z = np.zeros_like(x)
        return (z, z.copy(), z.copy())
    return _build_dulya_intensities(P, x, shape_params, wd=wd, exact_intensity=exact_intensity, nphi=nphi)
