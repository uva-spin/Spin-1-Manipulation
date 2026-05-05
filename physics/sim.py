from afp import AFP
from lineshape.Lineshape import GenerateVectorLineshape
import numpy as np
import matplotlib.pyplot as plt

R = np.linspace(-6, 6, 500)
signal, Iplus, Iminus = GenerateVectorLineshape(0.25, R)


afp = AFP(0.0, 0.0, 0.0)
n_plus = afp.calculate_n_plus(Iplus, Iminus)
n_minus = afp.calculate_n_minus(Iplus, Iminus)
n_naught = afp.calculate_n_naught(Iplus, Iminus)

afp.norm_mu(Iplus, Iminus)
n_plus = afp.calculate_n_plus(Iplus, Iminus)
n_minus = afp.calculate_n_minus(Iplus, Iminus)
n_naught = afp.calculate_n_naught(Iplus, Iminus)

fig, ax = plt.subplots(3, 1, figsize=(12,8))
ax[0].step(R, n_plus, label='$n_+$')
ax[0].step(R, n_minus, label='$n_-$')
ax[0].step(R, n_naught, label='$n_0$')
ax[0].grid(True)
ax[0].set_xlim(-3,3)
ax[0].legend(loc='upper right')
ax[1].step(R, signal, label='signal', color='black')
ax[1].grid(True)
ax[1].set_xlim(-3,3)
ax[1].legend(loc='upper right')
ax[2].step(R,(n_plus - n_minus) / (n_plus + n_minus + n_naught), label='$P$', color='black')
ax[2].step(R, ((n_plus + n_minus + n_naught) - 3*n_naught)/(n_plus + n_minus + n_naught), label='$Q$', color='red')
ax[2].grid(True)
ax[2].set_xlim(-3,3)
ax[2].legend(loc='upper right')
plt.show()

print(f"Sum of lineshape: {np.sum(signal)}")
print(f" Sum of population: {np.sum(n_plus) + np.sum(n_minus) + np.sum(n_naught)}")
# print(f"$\mu$: {afp.mu}")

