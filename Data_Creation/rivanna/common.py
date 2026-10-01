"""Shared constants for data generation."""
import sys
from pathlib import Path
import numpy as np
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
PHYSICS_MODEL = 'physics.rf'
RF_MODE = 'physical_voigt'
RF_MODE_SINGLE_BIN = 'single_bin'
RF_MODE_PHYSICAL_VOIGT = 'physical_voigt'
NUM_BINS = 500
F_MIN = -6.0
F_MAX = 6.0
FREQUENCY = np.linspace(F_MIN, F_MAX, NUM_BINS)
P_MIN = -0.9
P_MAX = 0.9
P_STEP = 0.05
P_ABS_MIN = 0.02
BURN_R_MIN = -3.0
BURN_R_MAX = 3.0
GAMMA_RF_MIN = 10.0
GAMMA_RF_MAX = 10.0
GAMMA_RF_STEP = 10.0
MIN_BURN_STEPS = 1
MAX_BURN_STEPS = 500
BURN_STEPS_STEP = 50
DT = 0.0015
SSRF_GAMMA_RF = 10.0
SSRF_MAX_STEPS = 50
AFP_N_RELAX = 0
AFP_WINDOW = 8
AFP_EFFICIENCY = 1.0
AFP_CENTER_MARGIN = 0

def gamma_rf_grid(g_min=GAMMA_RF_MIN, g_max=GAMMA_RF_MAX, g_step=GAMMA_RF_STEP):
    """Inclusive gamma_rf sample grid."""
    g_min = g_min
    g_max = g_max
    g_step = g_step
    n = np.floor((g_max - g_min) / g_step + 1e-12) + 1
    return g_min + g_step * np.arange(n)

def burn_steps_grid(n_min=MIN_BURN_STEPS, n_max=MAX_BURN_STEPS, n_step=BURN_STEPS_STEP):
    """Inclusive burn-length (macro-step) sample grid."""
    return np.arange(n_min, n_max + 1, n_step, dtype=np.int32)
DIFFUSION_SCALE = 5.0
ZQ_WIDTH_R = 0.05
RF_GAUSSIAN_FWHM_R = 0.03
RF_LORENTZIAN_FWHM_R = 0.015
D_SAME_PLUS0 = 0.18
D_SAME_0MINUS = 0.1
D_SPEC_PLUS0 = 2.0
D_SPEC_0MINUS = 1.0
MIRROR_AMP_EPS = 1e-15
MIRROR_AMP_RTOL = 1e-06
MAX_GDT = 0.05
MAX_NSUB = 20
PS_ABS_MIN = 1e-12
MIRROR_OVER_BURN_AREA_TARGET = 0.5
MIRROR_OVER_BURN_AREA_RTOL = 0.1
IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET = 0.5
IPLUS_MIRROR_OVER_IMINUS_BURN_TARGET = 0.5
SSRF_INTENSITY_RATIO_RTOL = 0.1
MAX_SSRF_AREA_RATIO_RETRIES = 32
STORE_DTYPE = np.float32
SEED = 42
DEFAULT_SAMPLE_COUNT = 100
DEMO_P = 0.48
DEMO_BURN_BIN = 210
DATA_DIR = _HERE / 'data'
FIT_PARAMS_PATH = _HERE / 'fit_params.json'
SSRF_SHARD_DIR = DATA_DIR / 'ssrf_shards'
SSRF_TRAIN_DIR = DATA_DIR / 'ssrf_train'
AFP_SHARD_DIR = DATA_DIR / 'afp_shards'
AFP_TRAIN_DIR = DATA_DIR / 'afp_train'
UNMANIP_TRAIN_DIR = DATA_DIR / 'unmanip_train'
COMBINED_TRAIN_ALL_DIR = DATA_DIR / 'combined_train_all'
AFP_STEP_SUBSAMPLE = 50

def effective_afp_step_subsample(n_relax, step_subsample=AFP_STEP_SUBSAMPLE):
    """Keep every relax step when there is no relaxation trajectory to thin."""
    if n_relax <= 0:
        return 1
    return max(1, step_subsample)
SOURCE_SSRF = 0
SOURCE_AFP = 1
SOURCE_UNMANIP = 2
SOURCE_PROFILE = 3
SOURCE_AFP_PROFILE = 4
PLOT_DIR = DATA_DIR / 'plots'
SLURM_LOG_DIR = _HERE / 'slurm_logs'
BURN_BIN_CHOICES = np.flatnonzero((FREQUENCY > BURN_R_MIN) & (FREQUENCY < BURN_R_MAX)).astype(int)
EXCLUDED_MANIPULATION_BURN_BINS = frozenset({250})
BURN_BIN_ARRAY_START = BURN_BIN_CHOICES[0] if BURN_BIN_CHOICES.size else 0
BURN_BIN_ARRAY_END = BURN_BIN_CHOICES[-1] if BURN_BIN_CHOICES.size else -1

def burn_bin_position(bin_idx):
    """Index of ``bin_idx`` within ``BURN_BIN_CHOICES``, or None if outside the burn window."""
    choices = np.asarray(BURN_BIN_CHOICES, dtype=int)
    if choices.size == 0:
        return None
    pos = np.searchsorted(choices, bin_idx)
    if pos >= choices.size or choices[pos] != bin_idx:
        return None
    return pos

def is_burn_bin(bin_idx):
    return burn_bin_position(bin_idx) is not None

def intensity_pq(iplus, iminus):
    """Per-bin intensity targets: P = I+ + I-, Q = I+ - I-."""
    ip = np.asarray(iplus)
    im = np.asarray(iminus)
    return (ip + im, ip - im)
