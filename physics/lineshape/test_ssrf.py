import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.optimize import brentq

from ssRFMapper import ssRFMapper
from Lineshape import GenerateVectorLineshape

f = np.linspace(-3, 3, 249)
sigma = 0.04
gamma = 0.05
x0 = 0.9
amp = 0.4

print(f"bin size: {f[1] - f[0]:.6f} MHz")

lookup_path = Path(__file__).resolve().parent.parent.parent / "Data_Creation" / "lookup_table.pkl"
mapping_data = pd.read_pickle(lookup_path)

P = -0.65
P_REF = -0.35

signal, iplus, iminus = GenerateVectorLineshape(P, f)
ref_signal, ref_iplus, ref_iminus = GenerateVectorLineshape(P_REF, f)


# plt.figure(figsize=(12, 8))
# # plt.plot(f, signal, label="signal", linestyle=":", alpha=0.8)
# plt.plot(f, iplus, label=f"iplus ({P*100:.2f}%)", linestyle=":")
# plt.plot(f, iminus, label=f"iminus ({P*100:.2f}%)", linestyle=":")
# plt.plot(f, ref_iplus, label=f"ref_iplus ({P_REF*100:.2f}%)", linestyle=":")
# plt.plot(f, ref_iminus, label=f"ref_iminus ({P_REF*100:.2f}%)", linestyle=":")
# plt.legend()
# plt.show()
# plt.close()


test_bin_idx = int(np.argmin(np.abs(f - x0)))
# test_bin_idx = 180
target_signal_amp = ref_signal[test_bin_idx]
print(f"Signal Amplitude at burned bin: {signal[test_bin_idx]:.6e}")
print(f"Iplus Amplitude at burned bin: {iplus[test_bin_idx]:.6e}")
print(f"Iminus Amplitude at burned bin: {iminus[test_bin_idx]:.6e}")


mapper = ssRFMapper(f, sigma, gamma, x0, amp)
mapper.compute_lookup_tables(mapping_data)


def _burned_signal_at_bin(burn_amp: float) -> float:
    burned_signal, _, _ = mapper.apply_bin_burn(
        signal.copy(), iplus.copy(), iminus.copy(),
        bin_idx=test_bin_idx, amp=burn_amp,
    )
    return burned_signal[test_bin_idx]


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


residual = lambda a: _burned_signal_at_bin(a) - target_signal_amp
bracket_lo, bracket_hi = _find_brentq_bracket(
    residual, np.linspace(0.0, 10.0, 41),
)
matched_amp = (
    bracket_lo
    if bracket_lo == bracket_hi
    else brentq(residual, bracket_lo, bracket_hi)
)

mapper.amp = matched_amp
new_signal, new_iplus, new_iminus = mapper.apply_bin_burn(
    signal,
    iplus,
    iminus,
    bin_idx=test_bin_idx,
    amp=matched_amp,
)

iplus_diff = new_iplus - ref_iplus
# iplus_diff = new_iplus - ref_iminus

iminus_diff = new_iminus - ref_iminus
# iminus_diff = new_iminus - ref_iplus
mirror_bin_idx = int(np.argmin(np.abs(f - (2 * x0 - f[test_bin_idx]))))

print(f"Signal Amplitude at burned bin (after): {new_signal[test_bin_idx]:.6e}")
print(f"Iplus Amplitude at burned bin (after): {new_iplus[test_bin_idx]:.6e}")
print(f"Iminus Amplitude at burned bin (after): {new_iminus[test_bin_idx]:.6e}")

print(f"Change in Iplus: {new_iplus[test_bin_idx] - iplus[test_bin_idx]:.6e}")
print(f"Change in Iminus: {new_iminus[test_bin_idx] - iminus[test_bin_idx]:.6e}")


print(f"Burn amplitude: {matched_amp:.6e}")
print(f"Burned signal amplitude at burn bin: {new_signal[test_bin_idx]:.6e}")
print(f"35% signal amplitude at burn bin:    {target_signal_amp:.6e}")


# iplus_diff = abs(new_iplus[test_bin_idx] - iplus[test_bin_idx])
# iminus_diff = abs(new_iminus[test_bin_idx] - iminus[test_bin_idx])

print("iplus difference:")
print(f"  at burn bin ({f[test_bin_idx]:.4f} MHz): {iplus_diff[test_bin_idx]:.6e}")
print()
print("iminus difference:")
print(f"  at burn bin ({f[test_bin_idx]:.4f} MHz): {iminus_diff[test_bin_idx]:.6e}")


print(f"iplus_diff < iminus_diff: {iplus_diff[test_bin_idx] < iminus_diff[test_bin_idx]}")

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

axes.set_xlabel("Frequency")
axes.set_ylabel("Amplitude")
axes.legend()
plt.show()
plt.close()

