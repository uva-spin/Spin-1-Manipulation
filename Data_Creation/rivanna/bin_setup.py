import os
import numpy as np
from common import D_SAME_0MINUS, D_SAME_PLUS0, D_SPEC_0MINUS, D_SPEC_PLUS0, DIFFUSION_SCALE, F_MAX, F_MIN, FIT_PARAMS_PATH, NUM_BINS, PHYSICS_MODEL, RF_GAUSSIAN_FWHM_R, RF_LORENTZIAN_FWHM_R, RF_MODE, ZQ_WIDTH_R
from lineshape import GenerateDulyaLineshape, shape_params_from_fit
_SHAPE_PARAMS = None
LINESHape_MODEL = 'dulya_fit'

def get_shape_params():
    """Frozen fit params from ``fit_params.json``."""
    global _SHAPE_PARAMS
    if _SHAPE_PARAMS is None:
        _SHAPE_PARAMS = shape_params_from_fit()
    return dict(_SHAPE_PARAMS)

def equilibrium_lineshape(P, f, shape_params=None):
    """Equilibrium I± from the fitted Dulya/Hamada formulation."""
    return GenerateDulyaLineshape(P, np.asarray(f), shape_params if shape_params is not None else get_shape_params())

def spin1_scale_factors(P, iplus, iminus):
    """Map fit-scale intensities to Spin1 units (sum ~ P) and back."""
    area_fit = np.sum(np.asarray(iplus) + np.asarray(iminus))
    P = P
    if abs(area_fit) > 1e-30 and abs(P) > 1e-15:
        return (P / area_fit, area_fit / P)
    return (1.0, 1.0)

def generate_unmanipulated_cube(*, num_bins, p_min, p_max, p_step, shape_params=None):
    """(n_p, n_bins) equilibrium cubes via ``GenerateDulyaLineshape``."""
    shape = shape_params if shape_params is not None else get_shape_params()
    p_values = polarization_grid(p_min, p_max, p_step)
    f = np.linspace(F_MIN, F_MAX, num_bins)
    n_p = p_values.size
    n_bins_i = num_bins
    ps = np.zeros((n_p, n_bins_i))
    iplus = np.zeros((n_p, n_bins_i))
    iminus = np.zeros((n_p, n_bins_i))
    for (j, p0) in enumerate(p_values):
        if (j + 1) % 50 == 0 or j == 0 or j == n_p - 1:
            print(f'  GenerateDulyaLineshape P={p0:+.3f} ({j + 1}/{n_p})', flush=True)
        (signal, ip, im) = equilibrium_lineshape(p0, f, shape)
        ps[j] = np.asarray(signal)
        iplus[j] = np.asarray(ip)
        iminus[j] = np.asarray(im)
    return {'p_values': p_values, 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'amp': np.abs(ps), 'R': f, 'shape_params': shape}

def shape_meta(shape_params=None, *, rf_mode=None, gaussian_fwhm_R=None, lorentzian_fwhm_R=None, diffusion_scale=None):
    """Provenance blob stored in NPZ ``meta_json`` fields."""
    shape = shape_params if shape_params is not None else get_shape_params()
    return {'lineshape_model': LINESHape_MODEL, 'equilibrium_kernel': 'GenerateDulyaLineshape', 'fit_params_path': str(FIT_PARAMS_PATH), 'physics_model': PHYSICS_MODEL, 'rf_mode': str(RF_MODE if rf_mode is None else rf_mode), 'rf_gaussian_fwhm_R': RF_GAUSSIAN_FWHM_R if gaussian_fwhm_R is None else gaussian_fwhm_R, 'rf_lorentzian_fwhm_R': RF_LORENTZIAN_FWHM_R if lorentzian_fwhm_R is None else lorentzian_fwhm_R, 'diffusion_scale': DIFFUSION_SCALE if diffusion_scale is None else diffusion_scale, 'zq_width_R': ZQ_WIDTH_R, 'd_same_plus0': D_SAME_PLUS0, 'd_same_0minus': D_SAME_0MINUS, 'd_spec_plus0': D_SPEC_PLUS0, 'd_spec_0minus': D_SPEC_0MINUS, 'shape_params': {k: v for (k, v) in shape.items()}}

def resolve_bin_idx(cli_bin_idx, *, num_bins=NUM_BINS):
    """Resolve a zero-indexed spectral bin (0 .. num_bins-1) from CLI or SLURM."""
    if cli_bin_idx is not None:
        bin_idx = cli_bin_idx
    else:
        env_idx = os.environ.get('SLURM_ARRAY_TASK_ID')
        if env_idx is None or str(env_idx).strip() == '':
            return None
        bin_idx = env_idx
    nb = num_bins
    if bin_idx < 0 or bin_idx >= nb:
        raise ValueError(f'bin_idx={bin_idx} out of range for num_bins={nb} (zero-indexed valid range 0..{nb - 1})')
    return bin_idx

def print_shape_banner(shape, *, num_bins):
    print('Dulya fitted lineshape:', ', '.join((f'{k}={v:.6g}' for (k, v) in shape.items())), flush=True)
    print(f'R grid: [{F_MIN}, {F_MAX}]  n={num_bins}', flush=True)

def polarization_grid(p_min, p_max, p_step):
    g = np.arange(p_min, p_max + 1e-12, p_step)
    return g[np.abs(g) >= 1e-12]
