import numpy as np
import matplotlib.pyplot as plt

Iplus_i = 2.6
Iminus_i = 5.3

Iplus_f = 7.0
Iminus_f = 2.0

xi = 1.3

x = np.linspace(0.,5,100)

def Iplus(t, prefactor):
    return prefactor * np.exp(t*(1-2*xi))

def Iminus(t, prefactor):
    return prefactor * np.exp(t*(1-2*xi))

Iplus_new = Iplus_f - Iminus(x, Iminus_i) / 2
Iminus_new = Iminus_f - Iplus(x, Iplus_i) / 2

Q_theta1 = Iplus(x, Iplus_i) - Iminus(x, Iminus_i)
Q_theta2 = Iplus_f - Iminus(x, Iminus_i) / 2 - Iminus_f - Iplus(x, Iplus_i) / 2
P_theta1 = Iplus(x, Iplus_i) + Iminus(x, Iminus_i)
P_theta2 = Iplus_new + Iminus_new

fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax1.plot(x, Iplus(x, Iplus_i), label=r'$I_+$')
ax1.plot(x, Iminus(x, Iminus_i), label=r'$I_-$')
ax1.plot(x, Iplus(x, Iplus_i) + Iminus(x, Iminus_i), label=r'$I_+$ + $I_-$')
ax1.plot(x, Iplus(x, Iplus_i) - Iminus(x, Iminus_i), label=r'$Q$')
ax1.axhline(0., color='black', linestyle='--')
ax1.set_xlim(0., 5.)
ax1.grid(True)

ax2.plot(x, Iplus_new)
ax2.plot(x, Iminus_new)
ax2.plot(x, Iplus_new + Iminus_new)
ax2.plot(x, Iplus_new - Iminus_new)
ax2.axhline(0., color='black', linestyle='--')
ax2.set_xlim(0., 5.)
ax2.grid(True)
fig.legend(loc='outside right upper')
fig.tight_layout()
fig.savefig('rate_eqs_test.png')

fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True, sharey=True)
axes[0].plot(x, Q_theta1, label=r'$Q$')
axes[0].plot(x, P_theta1, label=r'$P$')
axes[0].axhline(0., color='black', linestyle='--')
axes[0].set_xlim(0., 5.)
axes[0].grid(True)
axes[0].legend()
axes[0].set_ylabel(r'$Q, P$')
axes[0].set_title(r'$R$')
axes[1].plot(x, Q_theta2, label=r'$Q$')
axes[1].plot(x, P_theta2, label=r'$P$')
axes[1].axhline(0., color='black', linestyle='--')
axes[1].set_xlim(0., 5.)
axes[1].grid(True)
axes[1].legend()
axes[1].set_ylabel(r'$Q, P$')
axes[1].set_title(r'$-R$')
axes[2].plot(x, Q_theta1 + Q_theta2, label=r'$Q$')
axes[2].plot(x, P_theta1 + P_theta2, label=r'$P$')
axes[2].axhline(0., color='black', linestyle='--')
axes[2].set_xlim(0., 5.)
axes[2].grid(True)
axes[2].legend()
axes[2].set_ylabel(r'$Q, P$')
axes[2].set_title(r'$Total$')
# fig.legend(loc='outside right upper')
# fig.tight_layout()
fig.supxlabel(r'$t$')
fig.savefig('rate_eqs_test_Q.png')
