"""Shared constants for Dulya-fit data generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

NUM_BINS = 500
# Match PolySignal / Dulya-fit R-grid (horns near ±split ≈ ±1).
F_MIN = -6.0
F_MAX = 6.0
FREQUENCY = np.linspace(F_MIN, F_MAX, NUM_BINS, dtype=np.float32)

P_MIN = -0.9
P_MAX = 0.9
P_STEP = 0.05
# Avoid near-zero P (vanishing intensity) when sampling uniformly.
P_ABS_MIN = 0.02

BURN_R_MIN = -2.0
BURN_R_MAX = 2.0
GAMMA_RF_MIN = 1.0
GAMMA_RF_MAX = 2.0
MIN_BURN_STEPS = 20
MAX_BURN_STEPS = 100
DT = 0.0005

# Match rate_eqs_test traj defaults (overridable via CLI).
SSRF_GAMMA_RF = 100.0
SSRF_MAX_STEPS = 5000
AFP_N_RELAX = 5000
AFP_WINDOW = 8
AFP_EFFICIENCY = 1.0
AFP_CENTER_MARGIN = 0

MIRROR_OVER_BURN_AREA_TARGET = 0.5
MIRROR_OVER_BURN_AREA_RTOL = 0.10
IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET = 0.5
IPLUS_MIRROR_OVER_IMINUS_BURN_TARGET = 0.5
SSRF_INTENSITY_RATIO_RTOL = 0.10
MAX_SSRF_AREA_RATIO_RETRIES = 32

STORE_DTYPE = np.float32
SEED = 42

# Modest defaults for transportable sample sets; override via CLI.
DEFAULT_SAMPLE_COUNT = 100

DATA_DIR = _HERE / "data"
FIT_PARAMS_PATH = _HERE / "fit_params.json"

SSRF_SHARD_DIR = DATA_DIR / "ssrf_shards"
SSRF_TRAIN_DIR = DATA_DIR / "ssrf_train"
AFP_SHARD_DIR = DATA_DIR / "afp_shards"
AFP_TRAIN_DIR = DATA_DIR / "afp_train"
UNMANIP_TRAIN_DIR = DATA_DIR / "unmanip_train"
SLURM_LOG_DIR = _HERE / "slurm_logs"

BURN_BIN_CHOICES = np.flatnonzero(
    (FREQUENCY > BURN_R_MIN) & (FREQUENCY < BURN_R_MAX)
).astype(int)
