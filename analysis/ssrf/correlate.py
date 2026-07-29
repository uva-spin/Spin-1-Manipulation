import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
data = pd.read_csv('Combined_burn_shards.csv')


# BIN_IDX = 0
# DT = 0.00015
# P0 = 0.55
GAMMA_A = 0.8
GAMMA_B = 0.7
data = data[((data['gamma_a'] == GAMMA_A) & (data['gamma_b'] == GAMMA_B)) & (data['dt'] !=  0.15 )]


data = data.sort_values(by='bin_idx')

data = data.dropna()

### Plot difference of iplus_a and iplus_b and iminus_a and iminus_b as a function of gamma, dt, and initial polarization

iplus_diff = data['iplus_a'] - data['iplus_b']
iminus_diff = data['iminus_a'] - data['iminus_b']

# # Plot difference of iplus_a and iplus_b as a function of gamma
# fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# ax[0].scatter(data['gamma_a'], iplus_diff)
# ax[0].scatter(data['gamma_b'], iplus_diff)
# ax[0].set_xlabel('gamma')
# ax[0].set_ylabel('difference of iplus_a and iplus_b')

# ax[1].scatter(data['gamma_a'], iminus_diff)
# ax[1].scatter(data['gamma_b'], iminus_diff)
# ax[1].set_xlabel('gamma')
# ax[1].set_ylabel('difference of iminus_a and iminus_b')
# plt.tight_layout()
# plt.savefig('difference_of_iplus_and_iminus.png')

# ### Now create 3D scatter plot of difference of iplus_a and iplus_b and iminus_a and iminus_b as a function of gamma, dt, and initial polarization

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax.scatter(data['gamma_a'], data['p0']*100, iplus_diff, color='tab:blue', label=r'$\Delta I_+$')
# ax.scatter(data['gamma_b'], data['p0']*100, iminus_diff, color='tab:red', label=r'$\Delta I_-$')
# ax.legend()
# ax.set_xlabel('gamma')
# ax.set_ylabel('initial polarization (%)')
# ax.set_zlabel(r'$\Delta I_{\pm}$')
# plt.savefig('difference_of_iminus_and_iminus_3d.png')

# print("="*100)
# print(f"Mean of iplus_diff: {iplus_diff.mean()}")
# print(f"Mean of iminus_diff: {iminus_diff.mean()}")
# print("="*100)



### Plot across frequency spectrum of iplus_diff and iminus_diff
# fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# ax[0].scatter(np.arange(len(iplus_diff)), iplus_diff, s=4)
# ax[0].set_xlabel('bin index', fontsize=14)
# ax[0].set_ylabel(r'$\Delta I_+$', fontsize=14)
# ax[0].grid(True)
# ax[1].scatter(np.arange(len(iminus_diff)), iminus_diff, s=4)
# ax[1].set_xlabel('bin index', fontsize=14)
# ax[1].set_ylabel(r'$\Delta I_-$', fontsize=14)
# ax[1].grid(True)
# plt.tight_layout()
# plt.savefig('frequency_spectrum_of_iplus_and_iminus.png')



fig = plt.figure(figsize=(16, 10))
ax0 = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(1, 2, 2, projection='3d')

bin_idx = data['rf_burn_R'].to_numpy()

ax0.scatter(bin_idx, data['dt'], iplus_diff, s=4, color='tab:blue')
ax0.set_xlabel(r'$R$', fontsize=14)
ax0.set_ylabel(r'$\Delta t$', fontsize=14)
ax0.set_zlabel(r'$\Delta I_+$', fontsize=14)

ax1.scatter(bin_idx, data['dt'], iminus_diff, s=4, color='tab:red')
ax1.set_xlabel(r'$R$', fontsize=14)
ax1.set_ylabel(r'$\Delta t$', fontsize=14)
ax1.set_zlabel(r'$\Delta I_-$', fontsize=14)

# plt.tight_layout()
plt.savefig('dt_vs_diff_3D.png')

fig = plt.figure(figsize=(16, 10))
ax0 = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(1, 2, 2, projection='3d')

bin_idx = data['rf_burn_R'].to_numpy()

ax0.scatter(bin_idx, data['p0'], iplus_diff, s=4, color='tab:blue')
ax0.set_xlabel(r'$R$', fontsize=14)
ax0.set_ylabel(r'initial polarization (%)', fontsize=14)
ax0.set_zlabel(r'$\Delta I_+$', fontsize=14)

ax1.scatter(bin_idx, data['p0'], iminus_diff, s=4, color='tab:red')
ax1.set_xlabel(r'$R$', fontsize=14)
ax1.set_ylabel(r'initial polarization (%)', fontsize=14)
ax1.set_zlabel(r'$\Delta I_-$', fontsize=14)

# plt.tight_layout()
plt.savefig('p0_vs_diff_3D.png')

fig = plt.figure(figsize=(16, 10))
ax0 = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(1, 2, 2, projection='3d')

bin_idx = data['rf_burn_R'].to_numpy()

ax0.scatter(data['dt'], data['p0'], iplus_diff, s=4, color='tab:blue')
ax0.set_xlabel(r'$\Delta t$', fontsize=14)
ax0.set_ylabel(r'initial polarization (%)', fontsize=14)
ax0.set_zlabel(r'$\Delta I_+$', fontsize=14)

ax1.scatter(data['dt'], data['p0'], iminus_diff, s=4, color='tab:red')
ax1.set_xlabel(r'$\Delta t$', fontsize=14)
ax1.set_ylabel(r'initial polarization (%)', fontsize=14)
ax1.set_zlabel(r'$\Delta I_-$', fontsize=14)

# plt.tight_layout()
plt.savefig('dt_vs_p0_vs_diff_3D.png')


print("="*100)
print(f"Mean of iplus_diff: {iplus_diff.mean()}")
print(f"Mean of iminus_diff: {iminus_diff.mean()}")
print(f"Max of iplus_diff: {iplus_diff.max()}")
print(f"Max of iminus_diff: {iminus_diff.max()}")
print(f"Min of iplus_diff: {iplus_diff.min()}")
print(f"Min of iminus_diff: {iminus_diff.min()}")
print("="*100)
