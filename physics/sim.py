from afp import AFP
from lineshape.Lineshape import GenerateVectorLineshape
import numpy as np
import matplotlib.pyplot as plt

R = np.linspace(-6, 6, 500)
signal, Iplus, Iminus = GenerateVectorLineshape(0.45, R)

def theta_space(Iplus, Iminus):
    Iplus_theta = np.zeros(len(Iplus))
    Iminus_theta = np.zeros(len(Iminus))
    for i in range(len(Iplus)):
        Iplus_theta[i] = Iplus[i]
        Iminus_theta[i] = Iminus[len(Iplus) - i - 1]
    return Iplus_theta, Iminus_theta

Iplus_theta, Iminus_theta = theta_space(Iplus, Iminus)


afp = AFP(0.0, 0.0, 0.0)
n_plus = afp.calculate_n_plus(Iplus, Iminus)
n_minus = afp.calculate_n_minus(Iplus, Iminus)
n_naught = afp.calculate_n_naught(Iplus, Iminus)

P_theta = afp.calc_P_theta(Iplus, Iminus)
Q_theta = afp.calc_Q_theta(Iplus, Iminus)

plt.figure(figsize=(12,8))
plt.step(R, Iplus_theta, label='$\\rho_+ - \\rho_0$', color='red')
plt.step(R, Iminus_theta, label='$\\rho_0 - \\rho_-$', color='blue')
plt.grid(True)
plt.xlim(0,np.pi/2)
plt.xlabel('$\\Theta$', fontsize=18, fontfamily='Times New Roman')
plt.legend(loc='upper right')
plt.show()

# afp.norm_mu(Iplus, Iminus)
# n_plus_norm = afp.calculate_n_plus(Iplus, Iminus).copy()
# n_minus_norm = afp.calculate_n_minus(Iplus, Iminus).copy()
# n_naught_norm = afp.calculate_n_naught(Iplus, Iminus).copy()

# plt.figure(figsize=(12,8))
# plt.step(R, P_theta, label='$P_\\theta$', color='red')
# plt.step(R, Q_theta, label='$Q_\\theta$', color='blue')
# plt.grid(True)
# plt.xlim(-3,3)
# plt.xlabel('$\\Theta$', fontsize=18, fontfamily='Times New Roman')
# plt.legend(loc='upper right')
# plt.show()


# fig, ax = plt.subplots(3, 1, figsize=(12,8))
# ax[0].step(R, n_plus_norm, label='$n_+$')
# ax[0].step(R, n_minus_norm, label='$n_-$')
# ax[0].step(R, n_naught_norm, label='$n_0$')
# ax[0].grid(True)
# ax[0].set_xlim(-3,3)
# ax[0].legend(loc='upper right')
# ax[1].step(R, signal, label='signal', color='black')
# ax[1].grid(True)
# ax[1].set_xlim(-3,3)
# ax[1].legend(loc='upper right')
# ax[2].step(R,(n_plus_norm - n_minus_norm) / (n_plus_norm + n_minus_norm + n_naught_norm), label='$P$', color='black')
# ax[2].step(R, ((n_plus_norm + n_minus_norm + n_naught_norm) - 3*n_naught_norm)/(n_plus_norm + n_minus_norm + n_naught_norm), label='$Q$', color='red')
# ax[2].grid(True)
# ax[2].set_xlim(-3,3)
# ax[2].legend(loc='upper right')
# plt.show()

# subset_indices = np.arange(150, 151)
# n_plus_afp, n_minus_afp, n_naught_afp = afp.perform_afp(
#     len(subset_indices), subset_indices=subset_indices
# )

# fig, ax = plt.subplots(2, 1, figsize=(12,8))
# ax[0].step(R, n_plus, label='$n_+$', color='red')
# ax[0].step(R, n_minus, label='$n_-$', color='blue')
# ax[0].step(R, n_naught, label='$n_0$', color='green')
# ax[0].grid(True)
# ax[0].set_xlim(-3,3)
# ax[0].legend(loc='upper right')
# ax[1].step(R, n_plus_afp, label='$\Delta n_+$', color='red')
# ax[1].step(R, n_minus_afp, label='$\Delta n_-$', color='blue')
# ax[1].step(R, n_naught_afp, label='$\Delta n_0$', color='green')
# ax[1].grid(True)
# ax[1].set_xlim(-3,3)
# ax[1].legend(loc='upper right', fontsize=14)
# plt.show()



# fig, ax = plt.subplots(3, 1, figsize=(12,8))
# # ax[0].step(R[subset_indices], signal[subset_indices], label='signal', color='black')
# ax[0].step(R, n_plus_norm - n_naught_norm, label='$I_+$', color='red')
# # ax[0].step(R, n_naught_norm - n_minus_norm, label='$I_-$', color='blue')
# ax[0].grid(True)
# ax[0].legend(loc='upper right')
# # ax[1].step(R, (n_plus_afp - n_minus_afp), label='P', color='black')
# ax[1].step(R, (n_plus_afp - n_naught_afp), label=f'$I_+$', color='red')
# # ax[1].step(R, n_plus_afp + n_minus_afp - 2*n_naught_afp, label=f'$Q$', color='purple')
# ax[1].grid(True)
# ax[1].legend(loc='upper right', fontsize=14)
# ax[2].step(R, n_naught_afp - n_minus_afp, label=f'$I_-$', color='blue')
# ax[2].grid(True)
# ax[2].legend(loc='upper right')
# plt.show()

# fig, ax = plt.subplots(2, 1, figsize=(12,8))
# ax[0].step(R, n_plus_norm, label='$n_+$', color='red')
# ax[0].step(R, n_naught_norm, label='$n_0$', color='blue', linestyle='--')
# ax[0].step(R, n_minus_norm, label='$n_-$', color='green', linestyle='--')
# ax[0].grid(True)
# ax[0].legend(loc='upper right')
# ax[1].step(R, n_plus_afp, label=f'$n_+$', color='red', linestyle='--')
# ax[1].step(R, n_naught_afp, label=f'$n_0$', color='blue', linestyle='--')
# ax[1].step(R, n_minus_afp, label=f'$n_-$', color='green', linestyle='--')
# ax[1].grid(True)
# ax[1].legend(loc='upper right')
# plt.show()