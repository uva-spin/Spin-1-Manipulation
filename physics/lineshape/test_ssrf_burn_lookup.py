"""
Mirror of test_ssrf.py using burn_lookup_table.pkl trajectories.

Uses ``GenerateVectorLineshape`` (``Lineshape.py`` normalization: scale so
``sum(Iplus + Iminus) == P``) for starting spectra, matching
``burn_lookup_table.pkl`` and ``burn_lookup_realtime.initial_lineshape``.
Loads ``burn_lookup_table.pkl`` directly (no on-the-fly trajectory generation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.burn_lookup_realtime import BurnTrajectoryConfig, build_burn_bin_index
from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.lineshape.burnLookupMapper import BurnLookupMapper


class PolarizationBurnLookupMapper(BurnLookupMapper):
    """Map burned Ps using the lookup trajectory for this initial polarization."""

    def compute_lookup_index(self) -> None:
        self.burn_index = build_burn_bin_index(
            self.burn_lookup_df,
            self.burn_bin_idx,
            polarization=self.polarization,
        )

from rate_equations_realtime import verify_burn_response, verify_rates_response  

P = 0.60
P_REF = 0.45
P_REF2 = 0.30
NUM_BINS = 500
x0 = 0.92

trajectory_cfg = BurnTrajectoryConfig(n_bins=NUM_BINS, max_steps=500)
f = trajectory_cfg.f
signal, iplus, iminus = GenerateVectorLineshape(P, f)
ref_signal, ref_iplus, ref_iminus = GenerateVectorLineshape(P_REF, f)
ref2_signal, _ref2_iplus, _ref2_iminus = GenerateVectorLineshape(P_REF2, f)

print(f"bin size: {f[1] - f[0]:.6f} MHz")

test_bin_idx = int(np.argmin(np.abs(f - x0)))
mirror_bin_idx = len(f) - 1 - test_bin_idx
target_signal_amp = ref2_signal[test_bin_idx]

print(f"Burn bin: idx={test_bin_idx}, R={f[test_bin_idx]:.4f} MHz")
print(f"Mirror bin: idx={mirror_bin_idx}, R={f[mirror_bin_idx]:.4f} MHz")
print(f"Signal Amplitude at burned bin: {signal[test_bin_idx]:.6e}")
print(f"Iplus Amplitude at burned bin: {iplus[test_bin_idx]:.6e}")
print(f"Iminus Amplitude at burned bin: {iminus[test_bin_idx]:.6e}")


def _load_burn_lookup(
    burn_bin_idx: int,
    *,
    lookup_path: Path,
) -> pd.DataFrame:
    df = pd.read_pickle(lookup_path)
    return df[df["burn_bin_idx"] == burn_bin_idx]


lookup_path = _DATA_CREATION_DIR / "burn_lookup_table.pkl"


def _find_brentq_bracket(func, amps):
    """Return (lo, hi) where func changes sign, or raise ValueError."""
    values = [(a, func(a)) for a in amps]
    for (a0, f0), (a1, f1) in zip(values, values[1:]):
        if f0 == 0.0:
            return a0, a0
        if f0 * f1 < 0.0:
            return a0, a1
    raise ValueError(
        "No sign change in residual; cannot bracket root. "
        f"Sampled amps={amps}, residuals={[v for _, v in values]}"
    )


def _find_burn_amp(
    polarization: float,
    burn_signal: np.ndarray,
    burn_iplus: np.ndarray,
    burn_iminus: np.ndarray,
    *,
    target_signal: float,
) -> float:
    burn_lookup_df = _load_burn_lookup(test_bin_idx, lookup_path=lookup_path)
    mapper = PolarizationBurnLookupMapper(f, polarization, test_bin_idx, burn_lookup_df)
    mapper.compute_lookup_index()

    def _burned_signal_at_bin(burn_amp: float) -> float:
        burned_signal, _, _ = mapper.apply_bin_burn(
            burn_signal.copy(),
            burn_iplus.copy(),
            burn_iminus.copy(),
            bin_idx=test_bin_idx,
            amp=burn_amp,
        )
        return burned_signal[test_bin_idx]

    residual = lambda a: _burned_signal_at_bin(a) - target_signal
    bracket_lo, bracket_hi = _find_brentq_bracket(
        residual, np.linspace(0.0, 1.0, 41),
    )
    return (
        bracket_lo
        if bracket_lo == bracket_hi
        else brentq(residual, bracket_lo, bracket_hi)
    )


def _apply_burn(
    polarization: float,
    burn_signal: np.ndarray,
    burn_iplus: np.ndarray,
    burn_iminus: np.ndarray,
    burn_amp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    burn_lookup_df = _load_burn_lookup(test_bin_idx, lookup_path=lookup_path)
    mapper = PolarizationBurnLookupMapper(f, polarization, test_bin_idx, burn_lookup_df)
    mapper.compute_lookup_index()
    return mapper.apply_bin_burn(
        burn_signal,
        burn_iplus,
        burn_iminus,
        bin_idx=test_bin_idx,
        amp=burn_amp,
    )


matched_amp_p = _find_burn_amp(
    P, signal, iplus, iminus, target_signal=target_signal_amp,
)
matched_amp_ref = _find_burn_amp(
    P_REF, ref_signal, ref_iplus, ref_iminus, target_signal=target_signal_amp,
)

print(f"Burn amplitude (P={P*100:.0f}% -> P_REF2={P_REF2*100:.0f}%): {matched_amp_p:.6e}")
print(f"Burn amplitude (P_REF={P_REF*100:.0f}% -> P_REF2={P_REF2*100:.0f}%): {matched_amp_ref:.6e}")

new_signal, new_iplus, new_iminus = _apply_burn(
    P, signal, iplus, iminus, matched_amp_p,
)
ref_burn_signal, ref_burn_iplus, ref_burn_iminus = _apply_burn(
    P_REF, ref_signal, ref_iplus, ref_iminus, matched_amp_ref,
)

rates_check = verify_burn_response(iplus, iminus, new_iplus, new_iminus, test_bin_idx)
reference_rates_check = verify_rates_response(
    iplus,
    iminus,
    test_bin_idx,
    trajectory_cfg.gamma_rf,
    dt=trajectory_cfg.dt,
)

print(f"Signal Amplitude at burned bin (after): {new_signal[test_bin_idx]:.6e}")
print(f"Iplus Amplitude at burned bin (after): {new_iplus[test_bin_idx]:.6e}")
print(f"Iminus Amplitude at burned bin (after): {new_iminus[test_bin_idx]:.6e}")

print(f"Change in Iplus: {new_iplus[test_bin_idx] - iplus[test_bin_idx]:.6e}")
print(f"Change in Iminus: {new_iminus[test_bin_idx] - iminus[test_bin_idx]:.6e}")

print(f"Burned signal amplitude at burn bin (P):     {new_signal[test_bin_idx]:.6e}")
print(f"Burned signal amplitude at burn bin (P_REF):  {ref_burn_signal[test_bin_idx]:.6e}")

print(f"Burned Iplus ampltiude at burn bin (P):     {new_iplus[test_bin_idx]:.6e}")
print(f"Burned Iplus ampltiude at burn bin (P_REF):  {ref_burn_iplus[test_bin_idx]:.6e}")
print(f"Burned Iminus ampltiude at burn bin (P):     {new_iminus[test_bin_idx]:.6e}")
print(f"Total burned area in Iplus + Iminus: {np.sum(new_iplus[test_bin_idx] + new_iminus[test_bin_idx]):.6e}")
print(f"Burned Iminus ampltiude at burn bin (P_REF):  {ref_burn_iminus[test_bin_idx]:.6e}")

iplus_diff = abs(new_iplus - iplus[test_bin_idx])
iminus_diff = abs(new_iminus - iminus[test_bin_idx])

print("iplus difference:")
print(f"  at burn bin ({f[test_bin_idx]:.4f} MHz): {iplus_diff[test_bin_idx]:.6e}")
print()
print("iminus difference:")
print(f"  at burn bin ({f[test_bin_idx]:.4f} MHz): {iminus_diff[test_bin_idx]:.6e}")

print(f"Iplus < Iminus: {iplus[test_bin_idx] < iminus[test_bin_idx]}")
print(f"iplus_diff < iminus_diff: {iplus_diff[test_bin_idx] < iminus_diff[test_bin_idx]}")
print()
print("Rates response check (burn lookup mapper):")
print(f"  passed: {rates_check['passed']}")
print(f"  |Ps| decreased at burn bin: {rates_check['magnitude_decreased']}")
print(f"  amp_burn / amp_mirror: {rates_check['ratios']['amp_burn_over_amp_mirror']:.4f}")
print(f"  dI+_burn / dI-_mirror: {rates_check['ratios']['iplus_burn_over_iminus_mirror']:.4f}")
print(f"  dI-_burn / dI+_mirror: {rates_check['ratios']['iminus_burn_over_iplus_mirror']:.4f}")
print()
print("Rates response check (single rate-equation step reference):")
print(f"  passed: {reference_rates_check['passed']}")
print(f"  |Ps| decreased at burn bin: {reference_rates_check['magnitude_decreased']}")
print(f"  amp_burn / amp_mirror: {reference_rates_check['ratios']['amp_burn_over_amp_mirror']:.4f}")
print(f"  dI+_burn / dI-_mirror: {reference_rates_check['ratios']['iplus_burn_over_iminus_mirror']:.4f}")
print(f"  dI-_burn / dI+_mirror: {reference_rates_check['ratios']['iminus_burn_over_iplus_mirror']:.4f}")

fig, axes = plt.subplots(1, 1, figsize=(12, 8))
axes.plot(f, new_iplus + new_iminus, label=f"signal (after, P={P*100:.0f}%)")
axes.plot(f, new_iplus, label=f"iplus (after, P={P*100:.0f}%)", linestyle="--")
axes.plot(f, new_iminus, label=f"iminus (after, P={P*100:.0f}%)", linestyle="--")
axes.plot(
    f,
    ref_burn_iplus + ref_burn_iminus,
    label=f"signal (after, P_REF={P_REF*100:.0f}%)",
    alpha=0.8,
)
axes.plot(
    f,
    ref_burn_iplus,
    label=f"iplus (after, P_REF={P_REF*100:.0f}%)",
    linestyle="--",
    alpha=0.8,
)
axes.plot(
    f,
    ref_burn_iminus,
    label=f"iminus (after, P_REF={P_REF*100:.0f}%)",
    linestyle="--",
    alpha=0.8,
)

axes.set_xlabel("Frequency")
axes.set_ylabel("Amplitude")
axes.legend()
plt.savefig("test_ssrf_burn_lookup.png", dpi=300)
plt.close()
