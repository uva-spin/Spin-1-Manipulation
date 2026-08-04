"""
Sample random physical-Voigt ssRF burns for DAE lineshape denoising.

Each sample:
  - starts from a Dulya equilibrium lineshape at random vector polarization
  - applies one Voigt burn only at a burn-window bin where initial Q < 0
  - wider-than-default Voigt RF profile
  - fixed RF power gamma_rf=10.0, random burn length in [10, 100] macro-steps
  - dt=0.0015

Only the final full spectra are saved (I+ / I-), with no burn metadata.

Examples (from repo root):
  python Data_Creation/create_dae_voigt_burn_spectra.py --quick
  python Data_Creation/create_dae_voigt_burn_spectra.py --num-samples 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DULYA_V5 = SCRIPT_DIR / "dulya_fit_v5"

if str(DULYA_V5) not in sys.path:
    sys.path.insert(0, str(DULYA_V5))

from bin_setup import get_shape_params  # noqa: E402
from burn_selection import equilibrium_q_profile  # noqa: E402
from common import (  # noqa: E402
    BURN_BIN_CHOICES,
    BURN_R_MAX,
    BURN_R_MIN,
    NUM_BINS,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE_PHYSICAL_VOIGT,
)
from ssrf_bin_traj import run_one_polarization  # noqa: E402

DEFAULT_OUTPUT = SCRIPT_DIR / "dae_voigt_burn_spectra" / "spectra.npz"

P_MIN = 0.2
P_MAX = 0.6
GAMMA_RF = 10.0
MIN_BURN_STEPS = 10
MAX_BURN_STEPS = 100
DT = 0.0015
# Wider Voigt than the dulya_fit_v5 defaults (0.030 / 0.015).
GAUSSIAN_FWHM_R = 3*RF_GAUSSIAN_FWHM_R  # 0.090
LORENTZIAN_FWHM_R = 3*RF_LORENTZIAN_FWHM_R  # 0.045
STORE_DTYPE = np.float32
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_SEED = 42
MAX_ATTEMPTS_FACTOR = 20


def _burn_window_bins() -> np.ndarray:
    choices = np.asarray(BURN_BIN_CHOICES, dtype=np.int32)
    if choices.size == 0:
        raise RuntimeError(
            f"No burn bins in R window ({BURN_R_MIN}, {BURN_R_MAX})"
        )
    return choices


def q_negative_bins_for_p0(
    p0: float,
    burn_window: np.ndarray,
    *,
    shape_params: dict[str, float],
) -> np.ndarray:
    """Burn-window bin indices where equilibrium Q = I+ - I- is negative."""
    q = equilibrium_q_profile(float(p0), shape_params=shape_params)
    mask = q[burn_window] < 0.0
    return burn_window[mask]


def sample_one_spectrum(
    rng: np.random.Generator,
    burn_window: np.ndarray,
    *,
    shape_params: dict[str, float],
) -> np.ndarray | None:
    """Return final spectrum as float32 array shaped (2, num_bins), or None if skipped."""
    p0 = float(rng.uniform(P_MIN, P_MAX))
    qneg_bins = q_negative_bins_for_p0(p0, burn_window, shape_params=shape_params)
    if qneg_bins.size == 0:
        return None

    bin_idx = int(rng.choice(qneg_bins))
    n_steps = int(rng.integers(MIN_BURN_STEPS, MAX_BURN_STEPS + 1))

    traj = run_one_polarization(
        bin_idx,
        p0,
        dt=float(DT),
        gamma_rf=float(GAMMA_RF),
        n_steps=n_steps,
        rf_mode=RF_MODE_PHYSICAL_VOIGT,
        gaussian_fwhm_R=float(GAUSSIAN_FWHM_R),
        lorentzian_fwhm_R=float(LORENTZIAN_FWHM_R),
        shape_params=shape_params,
        capture_spectrum=True,
    )
    if bool(traj.get("skipped", False)):
        return None

    iplus = np.asarray(traj["ip_spectrum"], dtype=STORE_DTYPE).reshape(-1)
    iminus = np.asarray(traj["im_spectrum"], dtype=STORE_DTYPE).reshape(-1)
    if iplus.size != NUM_BINS or iminus.size != NUM_BINS:
        raise ValueError(
            f"Unexpected spectrum length: I+={iplus.size}, I-={iminus.size}, "
            f"expected {NUM_BINS}"
        )
    return np.stack([iplus, iminus], axis=0)


def generate_spectra(
    num_samples: int,
    *,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Generate ``num_samples`` spectra with shape (N, 2, NUM_BINS)."""
    n = int(num_samples)
    if n < 1:
        raise ValueError(f"num_samples must be >= 1, got {n}")

    burn_window = _burn_window_bins()
    shape_params = get_shape_params()
    rng = np.random.default_rng(int(seed))
    out = np.empty((n, 2, NUM_BINS), dtype=STORE_DTYPE)

    max_attempts = max(n * MAX_ATTEMPTS_FACTOR, n + 10)
    filled = 0
    attempts = 0
    while filled < n and attempts < max_attempts:
        attempts += 1
        spectrum = sample_one_spectrum(
            rng,
            burn_window,
            shape_params=shape_params,
        )
        if spectrum is None:
            continue
        out[filled] = spectrum
        filled += 1
        if filled == 1 or filled % 50 == 0 or filled == n:
            print(f"  sampled {filled}/{n} spectra ({attempts} attempts)", flush=True)

    if filled < n:
        raise RuntimeError(
            f"Only collected {filled}/{n} spectra after {attempts} attempts "
            "(too many skipped burns)"
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random Voigt-burned full spectra (I+/I- only) for DAE training. "
            "Burns only where initial Q < 0, with a widened Voigt RF profile."
        )
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of burned spectra to save (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output .npz path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate 8 spectra for a local smoke run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_samples = 8 if args.quick else int(args.num_samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Sampling {num_samples} Voigt-burn spectra | "
        f"P in [{P_MIN}, {P_MAX}] | gamma_rf={GAMMA_RF} | "
        f"steps in [{MIN_BURN_STEPS}, {MAX_BURN_STEPS}] | dt={DT} | "
        f"burn R in ({BURN_R_MIN}, {BURN_R_MAX}) | Q<0 centers only | "
        f"Voigt FWHM G={GAUSSIAN_FWHM_R:.3f} L={LORENTZIAN_FWHM_R:.3f}",
        flush=True,
    )
    spectra = generate_spectra(num_samples, seed=int(args.seed))

    # Channel 0 = I+, channel 1 = I-. No metadata by design.
    np.savez_compressed(output, spectra=spectra)
    print(
        f"Saved spectra array shape={spectra.shape} dtype={spectra.dtype} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
