import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.optimize import brentq

from ssRFMapper_sim import ssRFMapper
from Lineshape import GenerateVectorLineshape

f = np.linspace(-3, 3, 249)
sigma = 0.04
gamma = 0.05
x0 = -0.9
amp = 0.5

print(f"bin size: {f[1] - f[0]:.6f} MHz")

lookup_path = Path(__file__).resolve().parent.parent.parent / "Data_Creation" / "lookup_table.pkl"
mapping_data = pd.read_pickle(lookup_path)

P = 0.45
P_REF = 0.25

signal, iplus, iminus = GenerateVectorLineshape(P, f)
ref_signal, ref_iplus, ref_iminus = GenerateVectorLineshape(P_REF, f)


plt.figure(figsize=(12, 8))
# plt.plot(f, signal, label="signal", linestyle=":", alpha=0.8)
plt.plot(f, iplus, label=f"iplus ({P*100:.2f}%)", linestyle=":")
plt.plot(f, iminus, label=f"iminus ({P*100:.2f}%)", linestyle=":")
plt.plot(f, ref_iplus, label=f"ref_iplus ({P_REF*100:.2f}%)", linestyle=":")
plt.plot(f, ref_iminus, label=f"ref_iminus ({P_REF*100:.2f}%)", linestyle=":")
plt.legend()
plt.show()
plt.close()


test_bin_idx = int(np.argmin(np.abs(f - x0)))
target_signal_amp = ref_signal[test_bin_idx]

mapper = ssRFMapper(f, sigma, gamma, x0, amp)


def _burned_signal_at_bin(burn_amp: float) -> float:
    burned_signal, _, _, _, _, _ = mapper.apply_bin_burn(
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
    residual, np.linspace(0.0, 2.0, 41),
)
matched_amp = (
    bracket_lo
    if bracket_lo == bracket_hi
    else brentq(residual, bracket_lo, bracket_hi)
)


new_signal, new_iplus, new_iminus, rho_plus, rho_zero, rho_minus = mapper.apply_bin_burn(
    signal,
    iplus,
    iminus,
    bin_idx=test_bin_idx,
    amp=matched_amp,
)


fig, axes = plt.subplots(1, 1, figsize=(12, 8))
axes.plot(f, signal, label="signal (before)", linestyle=":", alpha=0.8)
axes.plot(f, iplus, label="iplus (before)", linestyle=":", alpha=0.8)
axes.plot(f, iminus, label="iminus (before)", linestyle=":", alpha=0.8)
axes.plot(f, new_signal, label=f"signal ({P*100:.2f}%)")
axes.plot(f, new_iplus, label=f"iplus ({P*100:.2f}%)", linestyle="--")
axes.plot(f, new_iminus, label=f"iminus ({P*100:.2f}%)", linestyle="--")
axes.plot(f, ref_signal, label=f"signal ({P_REF*100:.2f}%)", alpha=0.8)
axes.plot(f, ref_iplus, label=f"iplus ({P_REF*100:.2f}%)", alpha=0.8)
axes.plot(f, ref_iminus, label=f"iminus ({P_REF*100:.2f}%)", alpha=0.8)

axes.set_xlabel("Frequency")
axes.set_ylabel("Amplitude")
axes.legend()
plt.show()
plt.close()

# fig, axes = plt.subplots(1, 1, figsize=(12, 8))
# axes.plot(f, rho_plus, label="rho_plus")
# axes.plot(f, rho_zero, label="rho_zero")
# axes.plot(f, rho_minus, label="rho_minus")
# axes.legend()
# plt.show()
# plt.close()

