import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_LINESHAPE_DIR = Path(__file__).resolve().parent.parent
if str(_LINESHAPE_DIR) not in sys.path:
    sys.path.insert(0, str(_LINESHAPE_DIR))

from Lineshape import GenerateVectorLineshape

P = 0.60
NUM_BINS = 500
burn_idx = 172
mirror_idx = NUM_BINS - burn_idx - 1 # 500 - 172 - 1 = 327

f = np.linspace(-3, 3, NUM_BINS)
signal, Iplus_ls, Iminus_ls = GenerateVectorLineshape(P, f)

Iplus_i = float(Iplus_ls[burn_idx])
Iminus_i = float(Iminus_ls[burn_idx])
Iplus_f = float(Iplus_ls[mirror_idx])
Iminus_f = float(Iminus_ls[mirror_idx])

print(f"P={P}")
print(f"Burn bin:   idx={burn_idx}, R={f[burn_idx]:.4f}, I+={Iplus_i:.6e}, I-={Iminus_i:.6e}")
print(f"Mirror bin: idx={mirror_idx}, R={f[mirror_idx]:.4f}, I+={Iplus_f:.6e}, I-={Iminus_f:.6e}")

xi = 1.3
t = np.linspace(0.0, 5.0, 100)


def Iplus(t, prefactor):
    return prefactor * np.exp(t * (1 - 2 * xi))


def Iminus(t, prefactor):
    return prefactor * np.exp(t * (1 - 2 * xi))


# Mirror gains the depleted burn amount (crossed, 2:1), not the remaining intensity.
Iplus_new = Iplus_f + (Iminus_i - Iminus(t, Iminus_i)) / 2
Iminus_new = Iminus_f + (Iplus_i - Iplus(t, Iplus_i)) / 2

Q_theta1 = Iplus(t, Iplus_i) - Iminus(t, Iminus_i)
Q_theta2 = Iplus_new - Iminus_new
P_theta1 = Iplus(t, Iplus_i) + Iminus(t, Iminus_i)
P_theta2 = Iplus_new + Iminus_new

Q_total = Q_theta1 + Q_theta2
P_total = P_theta1 + P_theta2

# Final burn / mirror bin heights on the lineshape.
Iplus_ls_final = np.array(Iplus_ls, dtype=float, copy=True)
Iminus_ls_final = np.array(Iminus_ls, dtype=float, copy=True)
Iplus_ls_final[burn_idx] = float(Iplus(t, Iplus_i)[-1])
Iminus_ls_final[burn_idx] = float(Iminus(t, Iminus_i)[-1])
Iplus_ls_final[mirror_idx] = float(Iplus_new[-1])
Iminus_ls_final[mirror_idx] = float(Iminus_new[-1])
signal_final = Iplus_ls_final + Iminus_ls_final

plt.figure()
plt.plot(f, signal_final, label=r'$P_s$')
plt.plot(f, Iplus_ls_final, label=r'$I_+$')
plt.plot(f, Iminus_ls_final, label=r'$I_-$')
# plt.vlines(f[burn_idx], ymin=0, ymax=signal_final[burn_idx], color='red', linestyle='--', label='burn')
# plt.vlines(f[mirror_idx], ymin=0, ymax=signal_final[mirror_idx], color='blue', linestyle='--', label='mirror')
plt.legend()
plt.savefig('rate_eqs_test_lineshape.png')



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax1.plot(t, Iplus(t, Iplus_i), label=r'$I_+$')
ax1.plot(t, Iminus(t, Iminus_i), label=r'$I_-$')
ax1.plot(t, Iplus(t, Iplus_i) + Iminus(t, Iminus_i), label=r'$I_+$ + $I_-$')
ax1.plot(t, Iplus(t, Iplus_i) - Iminus(t, Iminus_i), label=r'$Q$')
ax1.axhline(0.0, color='black', linestyle='--')
ax1.set_xlim(0.0, 5.0)
ax1.set_title(rf'$R$ (burn idx={burn_idx})')
ax1.grid(True)

ax2.plot(t, Iplus_new)
ax2.plot(t, Iminus_new)
ax2.plot(t, Iplus_new + Iminus_new)
ax2.plot(t, Iplus_new - Iminus_new)
ax2.axhline(0.0, color='black', linestyle='--')
ax2.set_xlim(0.0, 5.0)
ax2.set_title(rf'$-R$ (mirror idx={mirror_idx})')
ax2.grid(True)
fig.legend(loc='outside right upper')
fig.tight_layout()
fig.savefig('rate_eqs_test.png')



### for how Iplus and Iminus change with time
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(t, Iminus(t, Iminus_i), label=r'$I_-$')
axes[0].plot(t, Iplus(t, Iplus_i), label=r'$I_+$')
axes[0].axhline(0.0, color='black', linestyle='--')
axes[0].set_xlim(0.0, 5.0)
axes[0].set_ylabel(r'$I_-, I_+$')
axes[0].set_title(r'$I_+$ (burn idx={burn_idx})')
axes[1].set_title(r'$I_-$ (mirror idx={mirror_idx})')
axes[1].plot(t, Iplus_new)
axes[1].plot(t, Iminus_new)
axes[1].axhline(0.0, color='black', linestyle='--')
axes[1].set_xlim(0.0, 5.0)
axes[1].set_ylabel(r'$I_-, I_+$')
axes[0].grid(True)
axes[1].grid(True)
fig.tight_layout()
fig.savefig('rate_eqs_test_Iplus_Iminus.png')

fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True, sharey=True)
axes[0].plot(t, Q_theta1, label=r'$Q$')
axes[0].plot(t, P_theta1, label=r'$P$')
axes[0].axhline(0.0, color='black', linestyle='--')
axes[0].set_xlim(0.0, 5.0)
axes[0].grid(True)
axes[0].legend()
axes[0].set_ylabel(r'$Q, P$')
axes[0].set_title(r'$R$')
axes[1].plot(t, Q_theta2, label=r'$Q$')
axes[1].plot(t, P_theta2, label=r'$P$')
axes[1].axhline(0.0, color='black', linestyle='--')
axes[1].set_xlim(0.0, 5.0)
axes[1].grid(True)
axes[1].legend()
axes[1].set_ylabel(r'$Q, P$')
axes[1].set_title(r'$-R$')
axes[2].plot(t, Q_total, label=r'$Q$')
axes[2].plot(t, P_total, label=r'$P$')
axes[2].axhline(0.0, color='black', linestyle='--')
axes[2].set_xlim(0.0, 5.0)
axes[2].grid(True)
axes[2].legend()
axes[2].set_ylabel(r'$Q, P$')
axes[2].set_title(r'$Total$')
fig.supxlabel(r'$t$')
fig.savefig('rate_eqs_test_Q.png')
