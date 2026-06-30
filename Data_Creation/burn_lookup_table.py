"""
Build a burn-augmented lookup table using realtime ss-RF dynamics.

For each polarization and frequency bin, applies repeated single-bin RF burns
from an unburned lineshape. Each row stores only the burn-bin Ps/Qs/Iplus/Iminus
values (not the full lineshape). Every trajectory has exactly ``max_steps + 1``
rows (burn_step 0 .. max_steps).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.burn_lookup_realtime import (  
    BurnTrajectoryConfig,
    burn_trajectory_realtime,
    initial_lineshape,
)
from physics.ssrf_realtime.rate_equations_realtime import verify_rates_response  

NUM_BINS = 500


@dataclass
class BurnLookupConfig:
    """Configuration for full burn lookup table generation."""

    p_min: float = -0.7
    p_max: float = 0.70
    p_step: float = 0.05
    num_bins: int = NUM_BINS
    output_path: Path = SCRIPT_DIR / "burn_lookup_table.pkl"
    trajectory: BurnTrajectoryConfig = field(default_factory=BurnTrajectoryConfig)

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.trajectory.n_bins = self.num_bins

    @property
    def polarizations(self) -> np.ndarray:
        return np.arange(self.p_min, self.p_max, self.p_step, dtype=float)

    @property
    def burn_bin_indices(self) -> np.ndarray:
        return np.arange(self.num_bins, dtype=int)



def per_bin_output_path(output_dir: Path, burn_bin_idx: int) -> Path:
    """Path for a single burn-bin shard."""
    return output_dir / f"burn_lookup_bin_{burn_bin_idx}.pkl"


def generate_burn_lookup_for_bin(
    config: BurnLookupConfig,
    burn_bin_idx: int,
    *,
    polarizations: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """Generate burn trajectories for one burn bin across all polarizations."""
    cfg = config.trajectory
    p_values = np.asarray(list(polarizations) if polarizations is not None else config.polarizations)
    burn_freq = float(cfg.f[int(burn_bin_idx)])

    rows: List[dict] = []
    progress = tqdm.tqdm(p_values, desc=f"Burn bin {burn_bin_idx}")

    for polarization in p_values:
        _f, iplus0, iminus0, _ps0 = initial_lineshape(float(polarization), cfg)
        traj_rows = burn_trajectory_realtime(
            iplus0,
            iminus0,
            int(burn_bin_idx),
            float(polarization),
            burn_freq,
            config=cfg,
        )
        rows.extend(traj_rows)
        progress.update(1)
    progress.close()

    return pd.DataFrame(rows).dropna()


def generate_burn_lookup_table(
    config: BurnLookupConfig,
    *,
    polarizations: Optional[Iterable[float]] = None,
    burn_bin_indices: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Generate all burn trajectories and return a DataFrame."""
    burn_bins = np.asarray(
        list(burn_bin_indices) if burn_bin_indices is not None else config.burn_bin_indices,
        dtype=int,
    )

    frames: List[pd.DataFrame] = []
    for burn_idx in burn_bins:
        frames.append(
            generate_burn_lookup_for_bin(
                config,
                int(burn_idx),
                polarizations=polarizations,
            )
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate realtime burn lookup table.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small subset (2 P values, first 5 burn bins, max 20 steps).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle path (default: burn_lookup_table.pkl or smoke variant).",
    )
    parser.add_argument("--p-step", type=float, default=0.005, help="Polarization grid step.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max burns per trajectory.")
    parser.add_argument(
        "--num-bins",
        type=int,
        default=NUM_BINS,
        help=f"Number of frequency bins (default: {NUM_BINS}).",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip post-generation checks.")
    parser.add_argument(
        "--burn-bin-idx",
        type=int,
        default=None,
        help="Generate data for a single burn bin only (for Slurm array tasks).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-bin shard pickles when using --burn-bin-idx.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Exit successfully if the target output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()


    trajectory = BurnTrajectoryConfig(max_steps=args.max_steps, n_bins=args.num_bins)
    config = BurnLookupConfig(
        p_step=args.p_step,
        num_bins=args.num_bins,
        trajectory=trajectory,
    )
    polarizations = None
    burn_bins = None

    config.output_path = Path(args.output)

    traj_cfg = config.trajectory
    f = traj_cfg.f

    if args.burn_bin_idx is not None:
        burn_bin_idx = int(args.burn_bin_idx)
        if burn_bin_idx < 0 or burn_bin_idx >= config.num_bins:
            raise ValueError(
                f"burn_bin_idx={burn_bin_idx} is outside valid range "
                f"[0, {config.num_bins - 1}]"
            )

        output_dir = Path(args.output_dir) if args.output_dir is not None else config.output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = output_dir / f"burn_lookup_bin_{burn_bin_idx}.pkl"
        print(f"Single-bin mode: burn_bin_idx={burn_bin_idx}")
        print(f"Output shard: {shard_path}")

        if args.skip_if_exists and shard_path.exists():
            print(f"Shard already exists, skipping: {shard_path}")
            return

        df = generate_burn_lookup_for_bin(
            config,
            burn_bin_idx,
            polarizations=polarizations,
        )
        df.to_pickle(shard_path)
        print(f"Saved {len(df)} rows to {shard_path}")
        if not args.no_validate:
            run_validation(df, config.trajectory)
        return

    df = generate_burn_lookup_table(
        config,
        polarizations=polarizations,
        burn_bin_indices=burn_bins,
    )
    df.to_pickle(config.output_path)
    print(f"Saved {len(df)} rows to {config.output_path}")

    if not args.no_validate:
        run_validation(df, config.trajectory)


if __name__ == "__main__":
    main()
