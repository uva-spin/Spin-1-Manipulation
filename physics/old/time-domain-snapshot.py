import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
import matplotlib.font_manager as font_manager
from scipy.signal import hilbert
from Lineshape import *
from mpl_toolkits.mplot3d import Axes3D
import tqdm as tqdm

# --- System and lineshape setup ---
g = 0.05
s = 0.04
bigy = np.sqrt(3 - s)
labelfontsize = 30
P = 0.50  # Input polarization
r = (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)
phi_rad = 2*np.pi

# --- Generate frequency axis and base lineshape ---
steps = 500
n_timesteps = 500
noise_std = 0.0  # Gaussian noise standard deviation (relative to signal scale)

xvals = np.linspace(-6, 6, steps)  # frequency (R)
yvals_absorp1 = Lineshape(xvals, 1)        # χ''₊
yvals_absorp2 = Lineshape(-xvals, 1)       # χ''₋
yvals_disp1 = np.imag(hilbert(yvals_absorp1))  # χ'₊
yvals_disp2 = np.imag(hilbert(yvals_absorp2))  # χ'₋

signal1 = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
signal2 = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)

# --- Loop over 500 timesteps: add Gaussian noise to each event ---
# Amplitude = magnitude of (absorptive + i*dispersive) for the combined signal
signal_scale = np.max(np.abs(signal1 + signal2))
amplitude_grid = np.zeros((n_timesteps, steps))

for t in tqdm.tqdm(range(n_timesteps), desc="Timesteps"):
    # Noisy absorptive and dispersive at this timestep
    noise = np.random.normal(0, noise_std , steps)
    # Combined complex signal; amplitude = magnitude
    signal = signal1 + signal2 + noise
    amplitude_grid[t, :] = np.sqrt(signal**2)

# --- Heatmap: frequency (R) vs timestep, amplitude as z (color) ---
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.pcolormesh(xvals, np.arange(n_timesteps), amplitude_grid, shading="auto", cmap="viridis")
ax.set_xlabel("Frequency (R)", fontsize=labelfontsize)
ax.set_ylabel("Timestep", fontsize=labelfontsize)
ax.tick_params(axis='both', which='major', labelsize=labelfontsize-8)
ax.tick_params(axis='both', which='minor', labelsize=labelfontsize-8)
ax.tick_params(axis='x', labelsize=labelfontsize-8)
ax.tick_params(axis='y', labelsize=labelfontsize-8)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Amplitude", fontsize=labelfontsize)
# ax.set_title("Amplitude vs frequency and timestep (500 steps, Gaussian noise)", fontsize=14)
plt.tight_layout()
plt.savefig("plots/time_domain_amplitude_heatmap.png", dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(xvals, signal1 + signal2 + np.random.normal(0, noise_std, steps))
ax.set_xlabel("Frequency (R)", fontsize=labelfontsize)
ax.set_ylabel("Amplitude", fontsize=labelfontsize)
ax.tick_params(axis='both', which='major', labelsize=labelfontsize-8)
ax.tick_params(axis='both', which='minor', labelsize=labelfontsize-8)
ax.tick_params(axis='x', labelsize=labelfontsize-8)
ax.tick_params(axis='y', labelsize=labelfontsize-8)
plt.tight_layout()
plt.savefig("plots/time_domain_signal_1_2.png", dpi=150, bbox_inches="tight")
# plt.show()
plt.close()


