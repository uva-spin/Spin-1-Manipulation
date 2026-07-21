"""
Generate unmanipulated test lineshapes via GenerateVectorLineshape.

Each row is one equilibrium 500-bin spectrum at a random P in [P_MIN, P_MAX].

Run:
  python Data_Creation/generate_vector_lineshape_test_data.py
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

from physics.lineshape.Lineshape import GenerateVectorLineshape  # noqa: E402

NUM_SAMPLES = 1000
SEED = 42
OUTPUT_PATH = REPO_ROOT / "data" / "vector_lineshape_test_1000.pkl"

NUM_BINS = 500
F_MIN = -3.0
F_MAX = 3.0

P_MIN = 0.25
P_MAX = 0.50

STORE_DTYPE = np.float32


def generate_sample(
    sample_id: int,
    polarization: float,
    frequency: np.ndarray,
) -> Dict[str, Any]:
    ps, iplus, iminus = GenerateVectorLineshape(float(polarization), frequency)
    ps = np.asarray(ps, dtype=STORE_DTYPE)
    iplus = np.asarray(iplus, dtype=STORE_DTYPE)
    iminus = np.asarray(iminus, dtype=STORE_DTYPE)
    qs = (iplus - iminus).astype(STORE_DTYPE)

    return {
        "sample_id": int(sample_id),
        "P_initial": float(polarization),
        "true_P_initial": STORE_DTYPE(np.sum(ps)),
        "true_P": STORE_DTYPE(np.sum(ps)),
        "true_Q": STORE_DTYPE(np.sum(qs)),
        "burn_bin_idx": None,
        "frequency": frequency.astype(STORE_DTYPE),
        "Ps": ps,
        "Qs": qs,
        "Iplus": iplus,
        "Iminus": iminus,
    }


def generate_dataset(num_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frequency = np.linspace(F_MIN, F_MAX, NUM_BINS, dtype=STORE_DTYPE)
    polarizations = rng.uniform(P_MIN, P_MAX, size=num_samples)

    rows: List[Dict[str, Any]] = []
    for sample_id, p in tqdm.tqdm(
        enumerate(polarizations),
        total=num_samples,
        desc="Generating vector lineshapes",
    ):
        rows.append(generate_sample(sample_id, float(p), frequency))
    return pd.DataFrame(rows)


def main() -> None:
    print(
        f"Generating {NUM_SAMPLES} unmanipulated lineshapes "
        f"(P in [{P_MIN}, {P_MAX}])"
    )
    df = generate_dataset(NUM_SAMPLES, SEED)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(OUTPUT_PATH)

    print(f"Saved {len(df)} samples to {OUTPUT_PATH}")
    print(
        "P range:",
        f"min={df['P_initial'].min():.3f}, "
        f"max={df['P_initial'].max():.3f}, "
        f"mean={df['P_initial'].mean():.3f}",
    )
    print(
        "Integrated P (sum Ps):",
        f"mean={df['true_P'].mean():.3f}, "
        f"min={df['true_P'].min():.3f}, "
        f"max={df['true_P'].max():.3f}",
    )


if __name__ == "__main__":
    main()
