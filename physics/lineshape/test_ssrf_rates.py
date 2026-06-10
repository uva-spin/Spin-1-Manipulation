import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rate_equations import solve_rate_equations, verify_rates_response
from Lineshape import GenerateVectorLineshape

f = np.linspace(-3, 3, 249)
burn_R = -0.88
dt = 1.0

print(f"bin size: {f[1] - f[0]:.6f} MHz")

P = 0.65
# P_REF = 0.35

signal, iplus, iminus = GenerateVectorLineshape(P, f)
ref_signal, ref_iplus, ref_iminus = GenerateVectorLineshape(P_REF, f)

test_bin_idx = int(np.argmin(np.abs(f - burn_R)))
mirror_bin_idx = len(f) - 1 - test_bin_idx
target_signal_amp = ref_signal[test_bin_idx]

print(f"Burn bin: idx={test_bin_idx}, R={f[test_bin_idx]:.4f} MHz")
print(f"Mirror bin: idx={mirror_bin_idx}, R={f[mirror_bin_idx]:.4f} MHz")
print(f"Signal amplitude at burned bin: {signal[test_bin_idx]:.6e}")
print(f"Iplus amplitude at burned bin:  {iplus[test_bin_idx]:.6e}")
print(f"Iminus amplitude at burned bin: {iminus[test_bin_idx]:.6e}")
print(f"Tensor polarization (sum I+ - I-) before: {np.sum(iplus - iminus):.6e}")
print(f"Tensor polarization (sum I+ - I-) target: {np.sum(ref_iplus - ref_iminus):.6e}")


def _burned_signal_at_bin(xi: float) -> float:
    iplus_new, iminus_new, _, _, _ = solve_rate_equations(
        iplus, iminus, dt, xi, test_bin_idx
    )
    return (iplus_new + iminus_new)[test_bin_idx]


def _find_brentq_bracket(func, xis):
    """Return (lo, hi) where func changes sign, or raise ValueError."""
    values = [(x, func(x)) for x in xis]
    for (x0, f0), (x1, f1) in zip(values, values[1:]):
        if f0 == 0.0:
            return x0, x0
        if f0 * f1 < 0.0:
            return x0, x1
    raise ValueError(
        "No sign change in residual; cannot bracket root. "
        f"Sampled xi={xis}, residuals={[v for _, v in values]}"
    )


residual = lambda xi: _burned_signal_at_bin(xi) - target_signal_amp
bracket_lo, bracket_hi = _find_brentq_bracket(
    residual, np.linspace(0.0, 1.0, 41),
)
matched_xi = (
    bracket_lo
    if bracket_lo == bracket_hi
    else brentq(residual, bracket_lo, bracket_hi)
)

new_iplus, new_iminus, _, _, _ = solve_rate_equations(
    iplus, iminus, dt, matched_xi, test_bin_idx
)
new_signal = new_iplus + new_iminus

iplus_diff = new_iplus - ref_iplus
iminus_diff = new_iminus - ref_iminus

tensor_after = np.sum(new_iplus - new_iminus)
tensor_ref = np.sum(ref_iplus - ref_iminus)

rates_check = verify_rates_response(iplus, iminus, test_bin_idx, matched_xi, dt)

print(f"Signal amplitude at burned bin (after): {new_signal[test_bin_idx]:.6e}")
print(f"Iplus amplitude at burned bin (after):  {new_iplus[test_bin_idx]:.6e}")
print(f"Iminus amplitude at burned bin (after): {new_iminus[test_bin_idx]:.6e}")
print(f"Change in Iplus:  {new_iplus[test_bin_idx] - iplus[test_bin_idx]:.6e}")
print(f"Change in Iminus: {new_iminus[test_bin_idx] - iminus[test_bin_idx]:.6e}")
print()
print(f"Matched xi: {matched_xi:.6e}")
print(f"Burned signal amplitude at burn bin: {new_signal[test_bin_idx]:.6e}")
print(f"35% signal amplitude at burn bin:    {target_signal_amp:.6e}")
print()
print(f"Tensor polarization (sum I+ - I-) after:  {tensor_after:.6e}")
print(f"Tensor polarization (sum I+ - I-) target: {tensor_ref:.6e}")
print()
print("iplus difference:")
print(f"  at burn bin ({f[test_bin_idx]:.4f} MHz): {iplus_diff[test_bin_idx]:.6e}")
print()
print("iminus difference:")
print(f"  at burn bin ({f[test_bin_idx]:.4f} MHz): {iminus_diff[test_bin_idx]:.6e}")
print()
print(f"iplus_diff < iminus_diff: {iplus_diff[test_bin_idx] < iminus_diff[test_bin_idx]}")
print()
print("Rates response check:")
print(f"  passed: {rates_check['passed']}")
print(f"  |Ps| decreased at burn bin: {rates_check['magnitude_decreased']}")
print(f"  amp_burn / amp_mirror: {rates_check['ratios']['amp_burn_over_amp_mirror']:.4f}")
print(f"  dI+_burn / dI-_mirror: {rates_check['ratios']['iplus_burn_over_iminus_mirror']:.4f}")
print(f"  dI-_burn / dI+_mirror: {rates_check['ratios']['iminus_burn_over_iplus_mirror']:.4f}")

fig, axes = plt.subplots(1, 1, figsize=(12, 8))
axes.plot(f, signal, label="signal (before)", linestyle=":", alpha=0.8)
axes.plot(f, iplus, label="iplus (before)", linestyle=":", alpha=0.8)
axes.plot(f, iminus, label="iminus (before)", linestyle=":", alpha=0.8)
axes.plot(f, new_signal, label="signal (after)")
axes.plot(f, new_iplus, label="iplus (after)", linestyle="--")
axes.plot(f, new_iminus, label="iminus (after)", linestyle="--")
axes.plot(f, ref_signal, label=f"signal ({P_REF*100:.2f}%)", alpha=0.8)
axes.plot(f, ref_iplus, label=f"iplus ({P_REF*100:.2f}%)", alpha=0.8)
axes.plot(f, ref_iminus, label=f"iminus ({P_REF*100:.2f}%)", alpha=0.8)
axes.axvline(f[test_bin_idx], color="k", linestyle=":", alpha=0.4, label=f"burn R={burn_R}")
axes.set_xlabel("Frequency (MHz)")
axes.set_ylabel("Amplitude")
axes.legend()
plt.show()
plt.close()
