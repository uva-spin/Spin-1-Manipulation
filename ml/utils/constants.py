"""Shared constants for the P/Q training scripts.

Single source of truth — imported by ``utils.helpers`` and re-exported
through ``lstm`` (sibling scripts do ``import lstm as lstm_pq``).
"""
from pathlib import Path

import torch

ML_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SPECTRA_PATH = ML_DIR / 'data' / 'spectra_combined.npz'
DEFAULT_OUTPUT_DIR = ML_DIR / 'results' / 'lstm' / 'lstm_result_ssrf_afp_combined_v3'
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 500
BATCH_SIZE = 1024
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001
MIN_DELTA = 1e-06
T_0 = 10
T_MULT = 2
LR_MIN = 1e-07
RESTART_LR_DECAY = 0.5
RESTART_WARMUP_EPOCHS = 5
MAX_GRAD_NORM = 1.0
HIDDEN_SIZE = 32
NUM_LAYERS = 2
DROPOUT = 0.1
VAL_FRAC = 0.15
TEST_FRAC = 0.15
REL_LOSS_EPS = 0.0001
RPE_ABS_EPS = 1e-10
NOISE_STD = 0.01
N_EXAMPLE_PLOTS = 36
POL_ABS_BANDS = tuple(((lo / 100.0, (lo + 5) / 100.0) for lo in range(5, 95, 5)))
SOURCE_SSRF = 0
SOURCE_AFP = 1
SOURCE_UNMANIP = 2
SOURCE_PROFILE = 3
SOURCE_AFP_PROFILE = 4
SOURCE_NAME = {SOURCE_SSRF: 'ssRF', SOURCE_AFP: 'AFP', SOURCE_UNMANIP: 'unmanipulated', SOURCE_PROFILE: 'optimal profile', SOURCE_AFP_PROFILE: 'AFP Profile'}
SPECTRUM_R_MIN = -6.0
SPECTRUM_R_MAX = 6.0
STATS_KEYS = (
    'ps_mean', 'ps_std', 'pwr_mean', 'pwr_std', 'steps_mean', 'steps_std',
    'P_mean', 'P_std', 'Q_mean', 'Q_std', 'noise_std', 'test_idx', 'n_test',
)
