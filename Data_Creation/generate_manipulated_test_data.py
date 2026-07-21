"""
Generate manipulated test lineshapes using Spin1Model (same physics as ssrf_afp.py).

Each sample starts from GenerateVectorLineshape at P in [P_MIN, P_MAX], then applies
one of:
  - ssRF Voigt burn only (Spin1Model relaxation on touched bins)
  - AFP sweep only (3-bin window, matches afp_bin_traj.py training physics)
  - both ssRF burn then AFP (same AFP commit path as training)

AFP-only samples are post-AFP snapshots with no relaxation (matches
afp_bin_traj.py training step index 0).

Run:
  python Data_Creation/generate_manipulated_test_data.py
  python Data_Creation/generate_manipulated_test_data.py --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape  # noqa: E402
from physics.lineshape.rate_eqs_test.afp_bin_traj import (  # noqa: E402
    afp_touched_bins,
    afp_window_indices,
    commit_touched_bins_only,
    restore_touched_intensity_area,
)
from physics.lineshape.rate_eqs_test.ssrf_afp import (  # noqa: E402
    commit_ssrf_bins_only,
    mirror_bin_idx,
    resolve_burn_bin,
    ssrf_touched_bins,
)
from physics.lineshape.rate_eqs_test.ssrf_bin_traj import (  # noqa: E402
    SIGMA_BINS,
    VOIGT_GAMMA_BINS,
    freeze_rf_profile,
    make_voigt_rf_profile,
)
from physics.ssrf_realtime.model import Spin1Model, Spin1Params  # noqa: E402

ManipulationMode = Literal["ssrf", "afp", "both"]

NUM_SAMPLES = 10_000
SEED = 42
OUTPUT_PATH = REPO_ROOT / "data" / "manipulated_test_10000.pkl"

NUM_BINS = 500
F_MIN = -3.0
F_MAX = 3.0
FREQUENCY = np.linspace(F_MIN, F_MAX, NUM_BINS, dtype=np.float32)

P_MIN = 0.25
P_MAX = 0.50

BURN_R_MIN = -2.0
BURN_R_MAX = 2.0
GAMMA_RF_MIN = 0.5
GAMMA_RF_MAX = 4.0
MIN_BURN_STEPS = 20
MAX_BURN_STEPS = 100
DT = 0.005

# Match afp_bin_traj.py / combined_train AFP shards.
AFP_WINDOW = 8
AFP_EFFICIENCY = 1.0
AFP_CENTER_MARGIN = 0

RF_SIGMA_BINS = SIGMA_BINS
RF_VOIGT_GAMMA_BINS = VOIGT_GAMMA_BINS

MANIPULATION_MODE_WEIGHTS = (0.40, 0.40, 0.20)
STORE_DTYPE = np.float32

_BURN_BIN_CHOICES = np.flatnonzero(
    (FREQUENCY > BURN_R_MIN) & (FREQUENCY < BURN_R_MAX)
).astype(int)


def _sample_afp_center_bin(n_bins: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, int(n_bins)))


def apply_afp_to_lineshape(
    iplus: np.ndarray,
    iminus: np.ndarray,
    afp_subset: List[int],
    *,
    polarization: float,
    num_bins: int,
    afp_efficiency: float,
    afp_center_margin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous AFP on ``afp_subset`` with training commit/restore (no relaxation)."""
    iplus0 = np.asarray(iplus, dtype=float).copy()
    iminus0 = np.asarray(iminus, dtype=float).copy()
    subset = [int(i) for i in afp_subset]
    touched = afp_touched_bins(int(num_bins), subset)
    area0 = float(np.sum(iplus0 + iminus0))

    params = Spin1Params(
        n_bins=int(num_bins),
        r_min=F_MIN,
        r_max=F_MAX,
        p0=float(polarization),
        initial_polarization=float(polarization),
        q0=0.0,
        p_dnp_sat=float(polarization),
        dnp_enabled=False,
        rf_enabled=False,
        relax_enabled=False,
        afp_enabled=True,
        afp_efficiency=float(afp_efficiency),
        afp_center_margin=int(afp_center_margin),
        afp_preserve_intensity_area=True,
        afp_subset_indices=subset,
        gamma_rf=0.0,
    )
    model = Spin1Model(params, initial_polarization=float(polarization))
    model.load_from_physical_intensities(iplus0, iminus0)
    model.params.afp_enabled = True
    model.params.afp_preserve_intensity_area = True
    model.params.afp_subset_indices = subset
    model._afp_pending = True

    model.afp_sweep()
    ip_sim, im_sim, _ = model.physical_intensities()
    out_ip, out_im = commit_touched_bins_only(
        iplus0, iminus0, ip_sim, im_sim, touched
    )
    return restore_touched_intensity_area(out_ip, out_im, touched, area0)


def _run_ssrf_burn(
    *,
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    gamma_rf: float,
    n_steps: int,
    num_bins: int,
    dt: float,
    polarization: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """ssRF Voigt burn + relaxation on touched bins; returns committed spectrum."""
    f = np.asarray(FREQUENCY[:num_bins], dtype=float)
    profile, ssrf_subset = make_voigt_rf_profile(
        num_bins,
        int(burn_idx),
        float(gamma_rf),
        sigma=float(RF_SIGMA_BINS),
        lorentz_gamma=float(RF_VOIGT_GAMMA_BINS),
    )
    touched = ssrf_touched_bins(num_bins, ssrf_subset)
    baseline_ip = np.asarray(iplus, dtype=float).copy()
    baseline_im = np.asarray(iminus, dtype=float).copy()

    params = Spin1Params(
        n_bins=num_bins,
        r_min=F_MIN,
        r_max=F_MAX,
        p0=float(polarization),
        initial_polarization=float(polarization),
        q0=0.0,
        p_dnp_sat=float(polarization),
        dnp_enabled=False,
        rf_enabled=True,
        relax_enabled=True,
        afp_enabled=False,
        dt=float(dt),
    )
    model = Spin1Model(params)
    model.load_from_physical_intensities(baseline_ip, baseline_im)
    model.params.gamma_rf = float(gamma_rf)
    model.params.ssrf_subset_indices = [int(i) for i in ssrf_subset]
    model.params.rf_burn_R = float(f[int(burn_idx)])
    freeze_rf_profile(model, profile)
    model._active_idx = np.asarray(touched, dtype=int) if touched else None

    step_count = max(1, int(n_steps))
    model.step(n_steps=step_count)
    ip_sim, im_sim, _ = model.physical_intensities()
    out_ip, out_im = commit_ssrf_bins_only(
        baseline_ip, baseline_im, ip_sim, im_sim, touched
    )
    return out_ip, out_im, step_count


def _sample_manipulation_mode(rng: np.random.Generator) -> ManipulationMode:
    modes: Tuple[ManipulationMode, ...] = ("ssrf", "afp", "both")
    idx = int(rng.choice(len(modes), p=MANIPULATION_MODE_WEIGHTS))
    return modes[idx]


def run_manipulation_event(
    *,
    polarization: float,
    mode: ManipulationMode,
    burn_bin: int | None,
    afp_center_bin: int | None,
    afp_bins: List[int] | None,
    gamma_rf: float,
    n_steps: int,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    afp_efficiency: float = AFP_EFFICIENCY,
    afp_center_margin: int = AFP_CENTER_MARGIN,
) -> Dict[str, Any]:
    """Build one manipulated lineshape; AFP path matches afp_bin_traj.py."""
    apply_ssrf = mode in ("ssrf", "both")
    apply_afp = mode in ("afp", "both")

    f = np.asarray(FREQUENCY[:num_bins], dtype=float)
    _, iplus0, iminus0 = GenerateVectorLineshape(float(polarization), f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    iplus_unburned = iplus.copy()
    iminus_unburned = iminus.copy()

    q_signal = iplus_unburned - iminus_unburned
    burn_idx = resolve_burn_bin(q_signal, burn_bin if apply_ssrf else None)
    ps0_at_burn = (
        float(iplus_unburned[burn_idx] + iminus_unburned[burn_idx])
        if apply_ssrf
        else np.nan
    )

    step_count = 0
    if apply_ssrf:
        iplus, iminus, step_count = _run_ssrf_burn(
            iplus=iplus,
            iminus=iminus,
            burn_idx=int(burn_idx),
            gamma_rf=float(gamma_rf),
            n_steps=int(n_steps),
            num_bins=num_bins,
            dt=float(dt),
            polarization=float(polarization),
        )

    if apply_afp:
        if not afp_bins:
            raise ValueError("AFP mode requires afp_bins (3-bin window indices).")
        iplus, iminus = apply_afp_to_lineshape(
            iplus,
            iminus,
            afp_bins,
            polarization=float(polarization),
            num_bins=num_bins,
            afp_efficiency=float(afp_efficiency),
            afp_center_margin=int(afp_center_margin),
        )

    ps = (iplus + iminus).astype(STORE_DTYPE)
    qs = (iplus - iminus).astype(STORE_DTYPE)
    ps_at_burn = float(iplus[burn_idx] + iminus[burn_idx]) if apply_ssrf else np.nan
    if apply_ssrf and abs(ps0_at_burn) > 1e-12:
        ps_ratio = float(ps_at_burn / ps0_at_burn)
    else:
        ps_ratio = 1.0

    afp_lo = int(afp_bins[0]) if afp_bins else -1
    afp_hi = int(afp_bins[-1]) + 1 if afp_bins else -1
    afp_center = int(afp_center_bin) if afp_center_bin is not None else -1

    return {
        "P_initial": float(polarization),
        "true_P_initial": STORE_DTYPE(float(np.sum(iplus_unburned + iminus_unburned))),
        "true_P": STORE_DTYPE(float(np.sum(ps))),
        "true_Q": STORE_DTYPE(float(np.sum(qs))),
        "manipulation_mode": mode,
        "ssrf_applied": apply_ssrf,
        "afp_applied": apply_afp,
        "burn_bin_idx": int(burn_idx) if apply_ssrf else None,
        "burn_freq": float(f[burn_idx]) if apply_ssrf else np.nan,
        "mirror_bin_idx": int(mirror_bin_idx(num_bins, burn_idx)) if apply_ssrf else None,
        "gamma_rf": float(gamma_rf) if apply_ssrf else np.nan,
        "burn_step_requested": int(n_steps) if apply_ssrf else 0,
        "burn_step": int(step_count) if apply_ssrf else 0,
        "burn_step_norm": float(step_count / max(MAX_BURN_STEPS, 1)) if apply_ssrf else 0.0,
        "ps0_at_burn_bin": STORE_DTYPE(ps0_at_burn) if apply_ssrf else np.nan,
        "ps_at_burn_bin": STORE_DTYPE(ps_at_burn) if apply_ssrf else np.nan,
        "ps_ratio": STORE_DTYPE(ps_ratio),
        "burn_progress": STORE_DTYPE(float(1.0 - abs(ps_ratio))),
        "afp_center_bin_idx": afp_center if apply_afp else None,
        "afp_mirror_bin_idx": (
            int(mirror_bin_idx(num_bins, afp_center)) if apply_afp and afp_center >= 0 else None
        ),
        "afp_bin_start": afp_lo,
        "afp_bin_stop": afp_hi,
        "afp_sweep_width": afp_hi - afp_lo if apply_afp and afp_bins else 0,
        "afp_freq_start": float(f[afp_lo]) if apply_afp and afp_bins else np.nan,
        "afp_freq_stop": float(f[afp_hi - 1]) if apply_afp and afp_bins else np.nan,
        "afp_efficiency": AFP_EFFICIENCY if apply_afp else np.nan,
        "frequency": FREQUENCY.astype(STORE_DTYPE),
        "Ps": ps,
        "Qs": qs,
        "Iplus": iplus.astype(STORE_DTYPE),
        "Iminus": iminus.astype(STORE_DTYPE),
    }


def generate_sample(sample_id: int, rng: np.random.Generator) -> Dict[str, Any]:
    polarization = float(rng.uniform(P_MIN, P_MAX))
    mode = _sample_manipulation_mode(rng)

    burn_bin: int | None = None
    gamma_rf = np.nan
    n_steps = 0
    if mode in ("ssrf", "both"):
        burn_bin = int(rng.choice(_BURN_BIN_CHOICES))
        gamma_rf = float(rng.uniform(GAMMA_RF_MIN, GAMMA_RF_MAX))
        n_steps = int(rng.integers(MIN_BURN_STEPS, MAX_BURN_STEPS + 1))

    afp_center_bin: int | None = None
    afp_bins: List[int] | None = None
    if mode in ("afp", "both"):
        afp_center_bin = _sample_afp_center_bin(NUM_BINS, rng)
        afp_bins = afp_window_indices(int(afp_center_bin), NUM_BINS, window=AFP_WINDOW)

    row = run_manipulation_event(
        polarization=polarization,
        mode=mode,
        burn_bin=burn_bin,
        afp_center_bin=afp_center_bin,
        afp_bins=afp_bins,
        gamma_rf=gamma_rf,
        n_steps=n_steps,
    )
    row["sample_id"] = int(sample_id)
    return row


def _generate_sample_task(sample_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED + sample_id)
    return generate_sample(sample_id, rng)


def generate_dataset(num_samples: int, workers: int) -> pd.DataFrame:
    if _BURN_BIN_CHOICES.size == 0:
        raise ValueError(
            f"No burn bins in R range ({BURN_R_MIN}, {BURN_R_MAX}) for {NUM_BINS} bins."
        )

    if workers <= 1:
        rng = np.random.default_rng(SEED)
        rows = [
            generate_sample(sample_id, rng)
            for sample_id in tqdm.tqdm(range(num_samples), desc="Generating manipulated lineshapes")
        ]
        return pd.DataFrame(rows)

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in tqdm.tqdm(
            pool.map(_generate_sample_task, range(num_samples), chunksize=32),
            total=num_samples,
            desc=f"Generating manipulated lineshapes ({workers} workers)",
        ):
            rows.append(row)
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Spin1Model manipulated test data.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel worker processes (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Number of samples to generate (default: {NUM_SAMPLES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output pickle path (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        f"Generating {args.num_samples} manipulated lineshapes "
        f"(P in [{P_MIN}, {P_MAX}]) via Spin1Model / ssrf_afp physics"
    )
    print(
        "Manipulation mix (ssRF / AFP / both): "
        f"{MANIPULATION_MODE_WEIGHTS[0]:.0%} / "
        f"{MANIPULATION_MODE_WEIGHTS[1]:.0%} / "
        f"{MANIPULATION_MODE_WEIGHTS[2]:.0%}"
    )
    print(f"AFP window: {AFP_WINDOW} bins (matches afp_bin_traj.py; no post-AFP relaxation)")
    print(f"Workers: {args.workers}")

    df = generate_dataset(args.num_samples, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.output)

    print(f"Saved {len(df)} samples to {args.output}")
    print("Mode counts:")
    print(df["manipulation_mode"].value_counts().to_string())
    print(
        "P_initial:",
        f"min={df['P_initial'].min():.3f}, max={df['P_initial'].max():.3f}, "
        f"mean={df['P_initial'].mean():.3f}",
    )
    print(
        "true_P (final integrated):",
        f"min={df['true_P'].min():.3f}, max={df['true_P'].max():.3f}, "
        f"mean={df['true_P'].mean():.3f}",
    )
    print(
        "true_P shift:",
        f"mean={(df['true_P'] - df['true_P_initial']).mean():.4f}",
    )


if __name__ == "__main__":
    main()
