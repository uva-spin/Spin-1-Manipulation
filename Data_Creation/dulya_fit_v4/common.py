"""Shared constants for Dulya-fit v2 data generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

PHYSICS_MODEL = "ssrf_realtime_v2"
# Default RF: physical-R Voigt + shared spectral recovery (ssrf_realtime_v2 voigt_burn path).
RF_MODE = "physical_voigt"
RF_MODE_SINGLE_BIN = "single_bin"
RF_MODE_PHYSICAL_VOIGT = "physical_voigt"

NUM_BINS = 500
F_MIN = -6.0
F_MAX = 6.0
FREQUENCY = np.linspace(F_MIN, F_MAX, NUM_BINS, dtype=np.float32)

P_MIN = -0.9
P_MAX = 0.9
P_STEP = 0.05
P_ABS_MIN = 0.02

BURN_R_MIN = -3.0
BURN_R_MAX = 3.0
# Discrete ssRF burn parameter grids (no continuous burn-to-turnover).
GAMMA_RF_MIN = 5.0
GAMMA_RF_MAX = 10.0
GAMMA_RF_STEP = 2.5
MIN_BURN_STEPS = 20
MAX_BURN_STEPS = 100
BURN_STEPS_STEP = 20
DT = 0.0015

SSRF_GAMMA_RF = 5.0  # demo / single-shot default
SSRF_MAX_STEPS = 100  # alias of MAX_BURN_STEPS for CLI compatibility
# Instantaneous AFP flip by default; raise for post-flip relaxation trajectories.
AFP_N_RELAX = 0
AFP_WINDOW = 8
AFP_EFFICIENCY = 1.0
AFP_CENTER_MARGIN = 0


def gamma_rf_grid(
    g_min: float = GAMMA_RF_MIN,
    g_max: float = GAMMA_RF_MAX,
    g_step: float = GAMMA_RF_STEP,
) -> np.ndarray:
    """Inclusive gamma_rf sample grid."""
    g_min = float(g_min)
    g_max = float(g_max)
    g_step = float(g_step)
    if g_step <= 0.0:
        raise ValueError(f"g_step must be > 0, got {g_step}")
    if g_max < g_min:
        raise ValueError(f"g_max ({g_max}) must be >= g_min ({g_min})")
    n = int(np.floor((g_max - g_min) / g_step + 1e-12)) + 1
    return g_min + g_step * np.arange(n, dtype=float)


def burn_steps_grid(
    n_min: int = MIN_BURN_STEPS,
    n_max: int = MAX_BURN_STEPS,
    n_step: int = BURN_STEPS_STEP,
) -> np.ndarray:
    """Inclusive burn-length (macro-step) sample grid."""
    n_min = int(n_min)
    n_max = int(n_max)
    n_step = int(n_step)
    if n_step < 1:
        raise ValueError(f"n_step must be >= 1, got {n_step}")
    if n_max < n_min:
        raise ValueError(f"n_max ({n_max}) must be >= n_min ({n_min})")
    return np.arange(n_min, n_max + 1, n_step, dtype=np.int32)

# Shared recovery rates (AFP and ssRF) + voigt_burn RF widths (physical R).
DIFFUSION_SCALE = 5.0
ZQ_WIDTH_R = 0.05
RF_GAUSSIAN_FWHM_R = 0.030
RF_LORENTZIAN_FWHM_R = 0.015
D_SAME_PLUS0 = 0.18
D_SAME_0MINUS = 0.10
D_SPEC_PLUS0 = 2.0
D_SPEC_0MINUS = 1.0

MIRROR_AMP_EPS = 1e-15
MIRROR_AMP_RTOL = 1e-6
MAX_GDT = 0.05
MAX_NSUB = 20
PS_ABS_MIN = 1e-12

MIRROR_OVER_BURN_AREA_TARGET = 0.5
MIRROR_OVER_BURN_AREA_RTOL = 0.10
IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET = 0.5
IPLUS_MIRROR_OVER_IMINUS_BURN_TARGET = 0.5
SSRF_INTENSITY_RATIO_RTOL = 0.10
MAX_SSRF_AREA_RATIO_RETRIES = 32

STORE_DTYPE = np.float32
SEED = 42
DEFAULT_SAMPLE_COUNT = 100

# Demo / plot defaults (realistic training example).
DEMO_P = 0.48
DEMO_BURN_BIN = 228

DATA_DIR = _HERE / "data"
FIT_PARAMS_PATH = _HERE / "fit_params.json"

SSRF_SHARD_DIR = DATA_DIR / "ssrf_shards"
SSRF_TRAIN_DIR = DATA_DIR / "ssrf_train"
AFP_SHARD_DIR = DATA_DIR / "afp_shards"
AFP_TRAIN_DIR = DATA_DIR / "afp_train"
UNMANIP_TRAIN_DIR = DATA_DIR / "unmanip_train"
# Full-spectrum shards (ssrf_spectrum_bin_*.npz) live alongside trajectory shards.
SPECTRUM_SSRF_SHARD_DIR = SSRF_SHARD_DIR
SPECTRUM_AFP_SHARD_DIR = AFP_SHARD_DIR
SPECTRUM_TRAIN_DIR = DATA_DIR / "spectrum_train"
SPECTRUM_TRAIN_NPZ = SPECTRUM_TRAIN_DIR / "spectrum_train.npz"
SSRF_SPECTRUM_ROWS_DIR = DATA_DIR / "ssrf_spectrum_rows"
AFP_SPECTRUM_ROWS_DIR = DATA_DIR / "afp_spectrum_rows"
UNMANIP_SPECTRUM_ROWS_DIR = DATA_DIR / "unmanip_spectrum_rows"
# Unified per-bin train files (source 0=ssrf, 1=afp, 2=unmanipulated).
COMBINED_TRAIN_ALL_DIR = DATA_DIR / "combined_train_all"
# Full-spectrum training coverage defaults.
MULTI_BURN_MIN = 2
MULTI_BURN_MAX = 5
AFP_STEP_SUBSAMPLE = 50
# Authoritative unmanip rows come from unmanip_bin_XXXX.npz at combine time.
UNMANIP_TRAIN_FRACTION = 0.0
DEFAULT_RANDOM_SSRF_SAMPLES = 0
# Hybrid spectrum sampling: dense Cartesian on every Nth burn bin; MC elsewhere.
SPECTRUM_DENSE_BIN_STRIDE = 5
SPECTRUM_MC_DRAWS_PER_BIN = 2
SPECTRUM_MAX_TRAIN_ROWS = 5_000_000
SPECTRUM_MIN_BURN_BIN_COVERAGE = 0.95


def effective_afp_step_subsample(
    n_relax: int,
    step_subsample: int = AFP_STEP_SUBSAMPLE,
) -> int:
    """Keep every relax step when there is no relaxation trajectory to thin."""
    if int(n_relax) <= 0:
        return 1
    return max(1, int(step_subsample))
# Process P×gamma×steps combos in batches during spectrum generation (peak RAM control).
# Spectrum mode allocates a full (t_max, num_bins) cube per combo, so it needs a small
# batch. Plain trajectory mode only allocates 6 center-bin (t_max,) arrays per combo
# (no num_bins factor), so it tolerates a much larger batch for the same RAM budget.
DEFAULT_SSRF_COMBO_BATCH_SIZE = 64
DEFAULT_SSRF_TRAJ_COMBO_BATCH_SIZE = 2048
SOURCE_SSRF = 0
SOURCE_AFP = 1
SOURCE_UNMANIP = 2
PLOT_DIR = DATA_DIR / "plots"
SLURM_LOG_DIR = _HERE / "slurm_logs"

BURN_BIN_CHOICES = np.flatnonzero(
    (FREQUENCY > BURN_R_MIN) & (FREQUENCY < BURN_R_MAX)
).astype(int)
# Inclusive SLURM array bounds for burn-window spectrum jobs (R in (BURN_R_MIN, BURN_R_MAX)).
BURN_BIN_ARRAY_START = int(BURN_BIN_CHOICES[0]) if BURN_BIN_CHOICES.size else 0
BURN_BIN_ARRAY_END = int(BURN_BIN_CHOICES[-1]) if BURN_BIN_CHOICES.size else -1


def burn_bin_position(bin_idx: int) -> int | None:
    """Index of ``bin_idx`` within ``BURN_BIN_CHOICES``, or None if outside the burn window."""
    choices = np.asarray(BURN_BIN_CHOICES, dtype=int)
    if choices.size == 0:
        return None
    pos = int(np.searchsorted(choices, int(bin_idx)))
    if pos >= int(choices.size) or int(choices[pos]) != int(bin_idx):
        return None
    return pos


def is_burn_bin(bin_idx: int) -> bool:
    return burn_bin_position(bin_idx) is not None


def is_dense_spectrum_bin(
    bin_idx: int,
    *,
    stride: int = SPECTRUM_DENSE_BIN_STRIDE,
) -> bool:
    """True for every ``stride``-th burn-window bin (Cartesian γ×steps coverage)."""
    pos = burn_bin_position(bin_idx)
    if pos is None:
        return False
    return (int(pos) % max(1, int(stride))) == 0
