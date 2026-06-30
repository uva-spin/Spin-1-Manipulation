from pathlib import Path

from afp import AFP
from lineshape.Lineshape import GenerateVectorLineshape
import numpy as np
import matplotlib.pyplot as plt

R = np.linspace(-3, 3, 500)
signal, Iplus, Iminus = GenerateVectorLineshape(0.45, R)

subset_indices = np.arange(250, 500)

# --- before (unaltered lineshape → populations) ---
afp_before = AFP.from_intensities(Iplus, Iminus)
Iplus_b, Iminus_b = afp_before.to_intensities()

# --- half swap: same subset, partial mixing ---
afp_half = AFP.from_intensities(Iplus, Iminus)
afp_half.perform_afp(subset_indices=subset_indices, efficiency=1.0)
Iplus_h, Iminus_h = afp_half.to_intensities()
n_plus_h = afp_half.n_plus.copy()
n_naught_h = afp_half.n_naught.copy()
n_minus_h = afp_half.n_minus.copy()

# --- full AFP on same subset, starting from half-swap populations ---
afp_full = AFP(n_plus_h.copy(), n_naught_h.copy(), n_minus_h.copy())
afp_full.perform_afp(subset_indices=np.arange(0, 500), efficiency=1.0)
Iplus_f, Iminus_f = afp_full.to_intensities()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
panels = (
    ("before AFP", Iplus_b, Iminus_b),
    ("after half swap (η=0.5)", Iplus_h, Iminus_h),
    ("after full sweep on same subset (η=1.0)", Iplus_f, Iminus_f),
)
for ax, (title, Ip, Im) in zip(axes, panels):
    ax.step(R, Ip + Im, label="$I_+ + I_-$", color="black")
    ax.step(R, Ip, label="$I_+$", color="red")
    ax.step(R, Im, label="$I_-$", color="blue")
    ax.set_ylabel("intensity")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.set_title(title)
axes[-1].set_xlim(-3, 3)
axes[-1].set_xlabel("$R$")
plt.tight_layout()
plt.show()
