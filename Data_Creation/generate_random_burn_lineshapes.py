"""
Generate burned lineshape samples using rate_equations_realtime.solve_rate_equations.

Each sample starts from an equilibrium lineshape, applies repeated single-bin RF
burns until the requested step count (or early stop), and stores one data point:
the final manipulated 500-bin spectrum only (no intermediate burn-step rows).

True vector/tensor polarization are integrated over the final Ps / Qs lineshape.

Run:
  python Data_Creation/generate_random_burn_lineshapes.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.burn_lookup_realtime import BurnTrajectoryConfig, initial_lineshape  # noqa: E402
from physics.ssrf_realtime.rate_equations_realtime import (  # noqa: E402
    burn_preserves_ps_sign,
    solve_rate_equations,
)


# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------

NUM_SAMPLES = 1000
SEED = 42
OUTPUT_PATH = SCRIPT_DIR / "random_burn_lineshapes_1000.pkl"

NUM_BINS = 500
F_MIN = -3.0
F_MAX = 3.0

P_MIN = -0.6
P_MAX = 0.6

# Only sample burn bins whose center frequency lies in this R range.
BURN_R_MIN = -2.0
BURN_R_MAX = 2.0

# RF burn intensity and duration are randomized per sample.
GAMMA_RF_MIN = 0.5
GAMMA_RF_MAX = 4.0
MIN_BURN_STEPS = 1
MAX_BURN_STEPS = 100

DT = 0.015
INTEGRATION_STEPS = 50

STORE_DTYPE = np.float32


def _build_config() -> BurnTrajectoryConfig:
    return BurnTrajectoryConfig(
        n_bins=NUM_BINS,
        f_min=F_MIN,
        f_max=F_MAX,
        gamma_rf=GAMMA_RF_MIN,
        dt=DT,
        steps=INTEGRATION_STEPS,
        burn_r_min=BURN_R_MIN,
        burn_r_max=BURN_R_MAX,
        store_dtype=STORE_DTYPE,
    )


def integrated_polarizations(
    ps: np.ndarray, qs: np.ndarray
) -> tuple[float, float]:
    """Integrate the final manipulated lineshape for true P and Q."""
    return float(np.sum(ps)), float(np.sum(qs))


def apply_burn_sequence(
    iplus0: np.ndarray,
    iminus0: np.ndarray,
    burn_bin_idx: int,
    polarization: float,
    gamma_rf: float,
    num_burn_steps: int,
    config: BurnTrajectoryConfig,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float, float]:
    """
    Apply repeated single-bin RF burns and return the final spectrum.

    Returns
    -------
    iplus, iminus, steps_applied, ps_at_burn_bin, ps_ratio, burn_step_norm, burn_progress
    """
    params = config.spin1_params(polarization)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    burn_bin_idx = int(burn_bin_idx)

    ps0_at_burn = float((iplus + iminus)[burn_bin_idx])
    if ps0_at_burn == 0.0:
        return (
            iplus.astype(STORE_DTYPE),
            iminus.astype(STORE_DTYPE),
            0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    steps_applied = 0
    for _ in range(int(num_burn_steps)):
        iplus_new, iminus_new, _, _, _ = solve_rate_equations(
            iplus,
            iminus,
            config.dt,
            float(gamma_rf),
            burn_bin_idx,
            params=params,
            rf_only=True,
        )
        iplus_new = np.asarray(iplus_new, dtype=float)
        iminus_new = np.asarray(iminus_new, dtype=float)

        if not burn_preserves_ps_sign(iplus, iminus, iplus_new, iminus_new, burn_bin_idx):
            break

        ps_before = float(iplus[burn_bin_idx] + iminus[burn_bin_idx])
        ps_after = float(iplus_new[burn_bin_idx] + iminus_new[burn_bin_idx])
        if ps_after == ps_before:
            break

        iplus = iplus_new
        iminus = iminus_new
        steps_applied += 1

    ps_at_burn = float(iplus[burn_bin_idx] + iminus[burn_bin_idx])
    if abs(ps0_at_burn) > 1e-12:
        ps_ratio = float(ps_at_burn / ps0_at_burn)
    else:
        ps_ratio = 0.0
    burn_progress = float(1.0 - abs(ps_ratio))
    burn_step_norm = float(steps_applied / max(MAX_BURN_STEPS, 1))

    return (
        iplus.astype(STORE_DTYPE),
        iminus.astype(STORE_DTYPE),
        steps_applied,
        ps_at_burn,
        ps_ratio,
        burn_step_norm,
        burn_progress,
    )


def generate_sample(
    sample_id: int,
    rng: np.random.Generator,
    config: BurnTrajectoryConfig,
    burn_bin_choices: np.ndarray,
) -> Dict[str, Any]:
    polarization = float(rng.uniform(P_MIN, P_MAX))
    burn_bin_idx = int(rng.choice(burn_bin_choices))
    gamma_rf = float(rng.uniform(GAMMA_RF_MIN, GAMMA_RF_MAX))
    requested_steps = int(rng.integers(MIN_BURN_STEPS, MAX_BURN_STEPS + 1))

    frequency, iplus0, iminus0, ps0 = initial_lineshape(polarization, config)
    ps0_at_burn = float(ps0[burn_bin_idx])
    true_p_initial = float(np.sum(ps0))

    iplus, iminus, steps_applied, ps_at_burn, ps_ratio, burn_step_norm, burn_progress = (
        apply_burn_sequence(
            iplus0=iplus0,
            iminus0=iminus0,
            burn_bin_idx=burn_bin_idx,
            polarization=polarization,
            gamma_rf=gamma_rf,
            num_burn_steps=requested_steps,
            config=config,
        )
    )

    ps = (iplus + iminus).astype(STORE_DTYPE)
    qs = (iplus - iminus).astype(STORE_DTYPE)
    true_p, true_q = integrated_polarizations(ps, qs)

    return {
        "sample_id": int(sample_id),
        "P_initial": polarization,
        "true_P_initial": STORE_DTYPE(true_p_initial),
        "true_P": STORE_DTYPE(true_p),
        "true_Q": STORE_DTYPE(true_q),
        "burn_bin_idx": burn_bin_idx,
        "burn_freq": float(frequency[burn_bin_idx]),
        "gamma_rf": gamma_rf,
        "burn_step": int(steps_applied),
        "burn_step_norm": burn_step_norm,
        "ps0_at_burn_bin": STORE_DTYPE(ps0_at_burn),
        "ps_at_burn_bin": STORE_DTYPE(ps_at_burn),
        "ps_ratio": STORE_DTYPE(ps_ratio),
        "burn_progress": STORE_DTYPE(burn_progress),
        "frequency": frequency.astype(STORE_DTYPE),
        "Ps": ps,
        "Qs": qs,
        "Iplus": iplus,
        "Iminus": iminus,
    }


def generate_dataset(
    num_samples: int,
    seed: int,
    config: BurnTrajectoryConfig,
) -> pd.DataFrame:
    burn_bin_choices = config.burn_bin_indices()
    if burn_bin_choices.size == 0:
        raise ValueError(
            f"No burn bins in R range ({BURN_R_MIN}, {BURN_R_MAX}) for {NUM_BINS} bins."
        )

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for sample_id in tqdm.tqdm(range(num_samples), desc="Generating burned lineshapes"):
        rows.append(generate_sample(sample_id, rng, config, burn_bin_choices))
    return pd.DataFrame(rows)


def main() -> None:
    config = _build_config()
    print(
        f"Generating {NUM_SAMPLES} burned lineshapes "
        f"(P in [{P_MIN}, {P_MAX}], burn bins in ({BURN_R_MIN}, {BURN_R_MAX}))"
    )
    print(
        f"Random gamma_rf in [{GAMMA_RF_MIN}, {GAMMA_RF_MAX}], "
        f"burn steps in [{MIN_BURN_STEPS}, {MAX_BURN_STEPS}]"
    )

    df = generate_dataset(NUM_SAMPLES, SEED, config)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUTPUT_PATH)

    print(f"Saved {len(df)} samples to {OUTPUT_PATH}")
    print(
        "Burn bin idx range:",
        int(df["burn_bin_idx"].min()),
        "..",
        int(df["burn_bin_idx"].max()),
    )
    print(
        "Applied burn steps:",
        f"mean={df['burn_step'].mean():.1f}, "
        f"median={df['burn_step'].median():.0f}, "
        f"max={df['burn_step'].max()}",
    )
    print(
        "True P (integrated final Ps):",
        f"mean={df['true_P'].mean():.3f}, "
        f"median={df['true_P'].median():.3f}, "
        f"min={df['true_P'].min():.3f}, "
        f"max={df['true_P'].max():.3f}",
    )
    print(
        "True P shift from burn (integrated sum Ps):",
        f"mean true_P - true_P_initial = {(df['true_P'] - df['true_P_initial']).mean():.3f}",
    )


if __name__ == "__main__":
    main()
