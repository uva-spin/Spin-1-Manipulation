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


def cosal(x, eps):
    return (1 - eps * x - s) / bigxsquare(x, eps)

def c(x):
    return np.sqrt(np.sqrt(g**2 + (1 - x - s)**2))

def bigxsquare(x, eps):
    return np.sqrt(g**2 + (1 - eps * x - s)**2)

def mult_term(x, eps):
    return 1 / (2 * np.pi * np.sqrt(bigxsquare(x, eps)))

def cosaltwo(x, eps):
    return np.sqrt((1 + cosal(x, eps)) / 2)

def sinaltwo(x, eps):
    return np.sqrt((1 - cosal(x, eps)) / 2)

def termone(x, eps):
    return np.pi / 2 + np.arctan((bigy**2 - bigxsquare(x, eps)) / (2 * bigy * np.sqrt(bigxsquare(x, eps)) * sinaltwo(x, eps)))

def termtwo(x, eps):
    return np.log((bigy**2 + bigxsquare(x, eps) + 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)) /
                  (bigy**2 + bigxsquare(x, eps) - 2 * bigy * np.sqrt(bigxsquare(x, eps)) * cosaltwo(x, eps)))

def icurve(x, eps):
    return mult_term(x, eps) * (2 * cosaltwo(x, eps) * termone(x, eps) + sinaltwo(x, eps) * termtwo(x, eps))

# --- Generate x and absorptive signal ---

steps = 1000

xvals = np.linspace(-6, 6, steps)
yvals_absorp1 = icurve(xvals, 1) / 10        # χ''₊
yvals_absorp2 = icurve(-xvals, 1) / 10       # χ''₋

# --- Get dispersive part via Hilbert transform (numerical Kramers–Kronig) ---
yvals_disp1 = np.imag(hilbert(yvals_absorp1))  # χ'₊
yvals_disp2 = np.imag(hilbert(yvals_absorp2))  # χ'₋

# --- Plotting setup ---
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(1, 1, 1)

# --- Generate and plot signals for different phase values ---
phi_values = np.linspace(-90, 90, steps)  # steps values from 0 to 360 degrees
# P_values = np.linspace(0.0005, .60, 36)
P_values = np.full(steps, 0.25)
r_values = (np.sqrt(4 - 3 * P_values**2) + P_values) / (2 - 2 * P_values)

U = 2.4283
eta = 1.04e-2
# self.phi = 6.1319
Cstray = 10**(-20)
shift = 0
Cknob = 0.220
cable = 3
center_freq = 32.68

noise = np.random.normal(0, 0.1, len(xvals))

for i, (phi_deg, r) in tqdm.tqdm(enumerate(zip(phi_values, r_values)), desc="Creating Signal Phase Variation"):
    phi_rad = np.deg2rad(phi_deg)
    
    # Phase-sensitive linear combination
    signal1 = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    signal2 = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)

    # baseline = Baseline(xvals, U, Cknob, eta, shift, Cstray, phi_deg, 0)
    baseline = 0

    total_signal = signal1 + signal2 + baseline
    
    # Use a colormap to create a gradient of colors
    color = plt.cm.plasma(i / len(phi_values))
    
    # Plot with decreasing opacity for better visualization
    alpha = 0.7 - (i / len(phi_values)) * 0.5  # Fade out as phi increases
    
    plt.plot(xvals, total_signal, color=color, alpha=alpha, linewidth=2)
    # plt.plot(xvals, signal1, color='red', alpha=alpha, linewidth=2)
    # plt.plot(xvals, signal2, color='green', alpha=alpha, linewidth=2)


norm = plt.Normalize(min(phi_values), max(phi_values))
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
cbar.set_label('φ', fontsize=labelfontsize, fontweight='bold')
cbar.ax.tick_params(labelsize=labelfontsize-8)

# --- Plot formatting ---
axisFontSize = 38
legendFontSize = 38

ax.set_xlabel('R', fontsize=24, fontfamily='Times New Roman')
ax.set_ylabel('Signal [$C_E$ mV]', fontsize=24, fontfamily='Times New Roman')

# Set up major and minor ticks
major_ticks_x = np.arange(-6, 6, 1)  # Major ticks every 1 unit
minor_ticks_x = np.arange(-6, 6, 1)  # Minor ticks every 0.5 units
major_ticks_y = np.arange(-1.2, 1.2, 0.2)  # Major ticks every 0.2 units
minor_ticks_y = np.arange(-1.2, 1.2, 0.1)  # Minor ticks every 0.1 units



# Set the ticks
ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)
ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

# Set the grid
ax.grid(True, which='major', linestyle='-', linewidth=1.5, alpha=0.3, color='gray')
ax.grid(True, which='minor', linestyle='--', linewidth=1, alpha=0.2, color='lightgray')

ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Add black outline
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Apply scalar formatter
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.xaxis.set_minor_formatter(ScalarFormatter(useMathText=True))

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=18, width=2)
ax.tick_params(axis='both', which='minor', labelsize=18, width=2)
ax.tick_params(axis='x', labelsize=18, width=2)

# Format tick labels - show only every other major tick
for label in ax.xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
for label in ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)

ax.set_axisbelow(True)

plt.xlim(-6, 6)

# Add title
# plt.title('Phase-Sensitive Signal Variation (φ = 0° to 360°)', fontsize=28, pad=20)


# Save figure
fig.set_size_inches(24, 16)
fig.savefig('plots/signal_phase_variation.jpeg', dpi=600, bbox_inches='tight', facecolor='white')
# plt.show()

def create_signal_heatmap(signal1, signal2, xvals, phi_values):
    """
    Create a heatmap visualization of real vs imaginary signal variations.
    
    Args:
        signal1: Array of signal1 values
        signal2: Array of signal2 values
        xvals: x-axis values
        phi_values: Array of phase values
    """
    # Create figure for real vs imaginary plot only
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    # Calculate signal values for each combination
    Z1 = np.zeros((len(phi_values), len(xvals)))
    real_part = np.zeros_like(Z1)
    imag_part = np.zeros_like(Z1)
    for i, phi_deg in enumerate(phi_values):
        phi_rad = np.deg2rad(phi_deg)
        # baseline = Baseline(xvals, U, Cknob, eta, shift, Cstray, phi_deg, 0)
        baseline = 0
        real_part[i, :] = (r *yvals_disp1 + yvals_disp2 ) * np.cos(phi_rad) + baseline 
        imag_part[i, :] = (r * yvals_absorp1 + yvals_absorp2) * np.sin(phi_rad) + baseline
    
    # --- Real vs Imag, colored by phase ---
    # For each R (column), plot a line/points in (imag, real) space, colored by phase
    for j in range(len(xvals)):
        ax2.plot(imag_part[:, j], real_part[:, j], color='gray', alpha=0.2, linewidth=0.5)
    # Now scatter all points, colored by phase
    scatter = ax2.scatter(imag_part.flatten(), real_part.flatten(), c=np.repeat(phi_values, len(xvals)), 
                          cmap='hsv', s=8, alpha=0.8)
    cbar2 = plt.colorbar(scatter, ax=ax2, orientation='horizontal', pad=0.1)
    cbar2.set_label('φ', fontsize=labelfontsize, fontfamily='Times New Roman')
    cbar2.ax.tick_params(labelsize=labelfontsize-8)
    # ax2.set_title('Real vs Imaginary Signal', fontsize=32, pad=20, fontweight='bold')
    ax2.set_xlabel('Imaginary Signal', fontsize=36, fontfamily='Times New Roman')
    ax2.set_ylabel('Real Signal', fontsize=36, fontfamily='Times New Roman')
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
    ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.xaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
    ax2.tick_params(axis='both', which='major', labelsize=22, width=2)
    ax2.tick_params(axis='both', which='minor', labelsize=22, width=2)
    ax2.tick_params(axis='x', labelsize=22, width=2)
    ax2.set_xlim(imag_part.min(), imag_part.max())
    ax2.set_ylim(real_part.min(), real_part.max())
    ax2.grid(True, which='major', linestyle='-', linewidth=1.5, alpha=0.3, color='gray')
    ax2.grid(True, which='minor', linestyle='--', linewidth=1, alpha=0.4, color='lightgray')
    
    ax2.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Add black outline
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.set_size_inches(24, 16)
    fig.savefig('plots/signal_heatmap_real_imag_only.jpeg', dpi=600, bbox_inches='tight', facecolor='white')
    # plt.show()

# Add this after your main plot code:
create_signal_heatmap(yvals_absorp1, yvals_disp1, xvals, phi_values)

# Create array to store total signal for heatmap
signal_heatmap = np.zeros((len(phi_values), len(xvals)))

# Fill the array with total signal values for each phase and R value
for i, (phi_deg, r) in tqdm.tqdm(enumerate(zip(phi_values, r_values)), desc="Creating Heatmap of Signal"):
    phi_rad = np.deg2rad(phi_deg)
    
    # Phase-sensitive linear combination
    signal1 = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    signal2 = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)
    # baseline = Baseline(xvals, U, Cknob, eta, shift, Cstray, phi_deg, 0)
    baseline = 0

    signal_heatmap[i, :] = signal1 + signal2 + baseline 

# # Create heatmap
# fig_heatmap = plt.figure(figsize=(16, 10))
# ax_heatmap = fig_heatmap.add_subplot(111)
# im = ax_heatmap.imshow(
#     signal_heatmap,
#     aspect='auto',
#     extent=[xvals[0], xvals[-1], phi_values[0], phi_values[-1]],
#     origin='lower',
#     cmap='RdBu_r'
# )

# # Add colorbar
# cbar = plt.colorbar(im, ax=ax_heatmap, orientation='vertical', pad=0.05)
# cbar.set_label('Signal [$C_E$ mV]', fontsize=labelfontsize, fontfamily='Times New Roman')
# cbar.ax.tick_params(labelsize=labelfontsize-8)

# # Add labels and title
# ax_heatmap.set_xlabel('R', fontsize=36, fontfamily='Times New Roman')
# ax_heatmap.set_ylabel('φ', fontsize=36, fontfamily='Times New Roman')
# # ax_heatmap.set_title('Phase-Sensitive Signal Variation', fontsize=32, pad=20, fontweight='bold')

# # Format ticks
# ax_heatmap.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax_heatmap.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
# ax_heatmap.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax_heatmap.xaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
# ax_heatmap.tick_params(axis='both', which='major', labelsize=22, width=2)
# ax_heatmap.tick_params(axis='both', which='minor', labelsize=22, width=2)
# ax_heatmap.tick_params(axis='x', labelsize=22, width=2)

# # Set white background
# ax_heatmap.set_facecolor('white')
# fig_heatmap.patch.set_facecolor('white')

# # Add black outline
# for spine in ax_heatmap.spines.values():
#     spine.set_edgecolor('black')
#     spine.set_linewidth(2)

# plt.tight_layout()
# fig_heatmap.savefig('plots/signal_phase_heatmap.jpeg', dpi=600, bbox_inches='tight', facecolor='white')



real_signal = np.zeros((len(phi_values), len(xvals)))
imag_signal = np.zeros((len(phi_values), len(xvals)))

for i, (phi_deg, r) in tqdm.tqdm(enumerate(zip(phi_values, r_values)), desc="Creating Heatmap of Real Vs. Imag"):
    phi_rad = np.deg2rad(phi_deg)
    # baseline = Baseline(xvals, U, Cknob, eta, shift, Cstray, phi_deg, 0)
    baseline = 0
    # Real component (cosine terms)
    real_signal[i, :] = r * yvals_disp1 * np.cos(phi_rad) + yvals_disp2 * np.cos(phi_rad) + baseline 
    # Imaginary component (sine terms)
    imag_signal[i, :] = r * yvals_absorp1 * np.sin(phi_rad) + yvals_absorp2 * np.sin(phi_rad) + baseline 




# Create combined heatmap with total signal, real, and imaginary components
fig = plt.figure(figsize=(36, 10))
fig.patch.set_facecolor('white')

# Create a GridSpec to handle the subplots and colorbar
gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1], height_ratios=[1], wspace=0.05)

# Total signal heatmap
ax0 = plt.subplot(gs[0])
ax0.set_facecolor('white')
im0 = ax0.imshow(
    real_signal,
    aspect='auto',
    extent=[xvals[0], xvals[-1], phi_values[0], phi_values[-1]],
    origin='lower',
    cmap='RdBu_r'
)

# Real component heatmap
ax1 = plt.subplot(gs[1])
ax1.set_facecolor('white')
im1 = ax1.imshow(
    signal_heatmap,
    aspect='auto',
    extent=[xvals[0], xvals[-1], phi_values[0], phi_values[-1]],
    origin='lower',
    cmap='RdBu_r'
)
ax1.set_xlabel('R', fontsize=36, fontfamily='Times New Roman')
ax1.set_title('Total Signal', fontsize=36, pad=20, fontfamily='Times New Roman')
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.xaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax1.tick_params(axis='both', which='major', labelsize=22, width=2)
ax1.tick_params(axis='both', which='minor', labelsize=22, width=2)
ax1.tick_params(axis='x', labelsize=22, width=2)
ax1.tick_params(axis='y', labelleft=False)  # Remove y-axis tick labels from middle subplot

# Add black outline
for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)


ax0.set_xlabel('R', fontsize=36, fontfamily='Times New Roman')
ax0.set_ylabel('φ', fontsize=36, fontfamily='Times New Roman')
ax0.set_title('Real Component', fontsize=36, pad=20, fontfamily='Times New Roman')
ax0.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax0.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax0.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax0.xaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax0.tick_params(axis='both', which='major', labelsize=22, width=2)
ax0.tick_params(axis='both', which='minor', labelsize=22, width=2)
ax0.tick_params(axis='x', labelsize=22, width=2)

# Add black outline
for spine in ax0.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Imaginary component heatmap
ax2 = plt.subplot(gs[2])
ax2.set_facecolor('white')
im2 = ax2.imshow(
    imag_signal,
    aspect='auto',
    extent=[xvals[0], xvals[-1], phi_values[0], phi_values[-1]],
    origin='lower',
    cmap='RdBu_r'
)
ax2.set_xlabel('R', fontsize=36, fontfamily='Times New Roman')
ax2.set_title('Imaginary Component', fontsize=36, pad=20, fontfamily='Times New Roman')
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.xaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax2.tick_params(axis='both', which='major', labelsize=22, width=2)
ax2.tick_params(axis='both', which='minor', labelsize=22, width=2)
ax2.tick_params(axis='x', labelsize=22, width=2)
ax2.tick_params(axis='y', labelleft=False)  # Remove y-axis tick labels from right subplot

# Add black outline
for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Add a shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
cbar_ax.set_facecolor('white')
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label('Signal [$C_E$ mV]', fontsize=labelfontsize, fontfamily='Times New Roman')
cbar.ax.tick_params(labelsize=labelfontsize-8)

plt.tight_layout()  # Adjust layout to make room for colorbar
fig.savefig('plots/signal_phase_heatmap_real_imag.jpeg', dpi=600, bbox_inches='tight', facecolor='white')
# Also save the combined plot as the signal_phase_heatmap
fig.savefig('plots/signal_phase_heatmap.jpeg', dpi=600, bbox_inches='tight', facecolor='white')
# plt.show()

# Create side-by-side 2D histograms for real and imaginary components
fig_hist = plt.figure(figsize=(24, 10))
fig_hist.patch.set_facecolor('white')

# Real component histogram
ax_hist1 = plt.subplot(1, 2, 1)
ax_hist1.set_facecolor('white')
plt.hist2d(imag_signal.flatten(), real_signal.flatten(), 
          bins=125, cmap='plasma', density=True)
cbar1 = plt.colorbar(label='Density', ax=ax_hist1)
cbar1.ax.tick_params(labelsize=22)
ax_hist1.set_xlabel('Imaginary Signal', fontsize=36, fontfamily='Times New Roman')
ax_hist1.set_ylabel('Real Signal', fontsize=36, fontfamily='Times New Roman')
# ax_hist1.set_title('Real Component Phase Space', fontsize=32, pad=20, fontweight='bold')
ax_hist1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_hist1.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax_hist1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_hist1.xaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax_hist1.tick_params(axis='both', which='major', labelsize=22, width=2)
ax_hist1.tick_params(axis='both', which='minor', labelsize=22, width=2)
ax_hist1.tick_params(axis='x', labelsize=22, width=2)
plt.grid(True, alpha=0.3, color='gray', linewidth=1)

# Add black outline
for spine in ax_hist1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

# Imaginary component histogram
ax_hist2 = plt.subplot(1, 2, 2)
ax_hist2.set_facecolor('white')
plt.hist2d(imag_signal.flatten(), real_signal.flatten(), 
          bins=125, cmap='plasma', density=True)
cbar2 = plt.colorbar(label='Density', ax=ax_hist2)
cbar2.ax.tick_params(labelsize=22)
ax_hist2.set_xlabel('Imaginary Signal', fontsize=36, fontfamily='Times New Roman')
ax_hist2.set_ylabel('Real Signal', fontsize=36, fontfamily='Times New Roman')
# ax_hist2.set_title('Imaginary Component Phase Space', fontsize=32, pad=20, fontweight='bold')
ax_hist2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_hist2.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax_hist2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_hist2.xaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
ax_hist2.tick_params(axis='both', which='major', labelsize=22, width=2)
ax_hist2.tick_params(axis='both', which='minor', labelsize=22, width=2)
ax_hist2.tick_params(axis='x', labelsize=22, width=2)
plt.grid(True, alpha=0.3, color='gray', linewidth=1)

# Add black outline
for spine in ax_hist2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(2)

plt.tight_layout()
fig_hist.savefig('plots/phase_space_components_side_by_side.jpeg', dpi=300, bbox_inches='tight', facecolor='white')
# plt.show()

def calculate_total_signal(x, P, phi_deg):
    """
    Calculate the total signal for given x, polarization P, and phase angle phi.
    
    Parameters:
    -----------
    x : float or array-like
        The x-coordinate value(s)
    P : float
        Input polarization (between 0 and 1)
    phi_deg : float
        Phase angle in degrees
        
    Returns:
    --------
    float or array-like
        The total signal value(s)
    """
    # System parameters
    g = 0.05
    s = 0.04
    bigy = np.sqrt(3 - s)
    
    # Calculate r from P
    r = (np.sqrt(4 - 3 * P**2) + P) / (2 - 2 * P)
    
    # Convert phase to radians
    phi_rad = np.deg2rad(phi_deg)
    
    # Calculate absorptive signals
    yvals_absorp1 = icurve(x, 1) / 10        # χ''₊
    yvals_absorp2 = icurve(-x, 1) / 10       # χ''₋
    
    # Calculate dispersive signals using Hilbert transform
    yvals_disp1 = np.imag(hilbert(yvals_absorp1))  # χ'₊
    yvals_disp2 = np.imag(hilbert(yvals_absorp2))  # χ'₋
    
    # Calculate phase-sensitive linear combination
    signal1 = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    signal2 = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)
    
    # Return total signal
    return signal1 + signal2

# --- 3D Plotting of Signal Snapshots over R, phi, and P ---
fig_3d = plt.figure(figsize=(20, 20))
fig_3d.patch.set_facecolor('white')
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_facecolor('white')
ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False

# Set box aspect ratio to make the plot more square
ax_3d.set_box_aspect([1, 1, 0.8])  # Adjust these ratios to make it more square

# Create a colormap for the polarization values
norm = plt.Normalize(vmin=P_values.min(), vmax=P_values.max())
cmap = plt.cm.viridis

# Iterate over each phase angle to plot snapshots
for i, (phi_deg, r) in tqdm.tqdm(enumerate(zip(phi_values, r_values)), desc="Creating 3D Signal Snapshots"):
    phi_rad = np.deg2rad(phi_deg)
    
    # Calculate signal1 and signal2 for each phase angle
    signal1 = r * (yvals_absorp1 * np.sin(phi_rad) + yvals_disp1 * np.cos(phi_rad))
    signal2 = yvals_absorp2 * np.sin(phi_rad) + yvals_disp2 * np.cos(phi_rad)
    
    # Calculate baseline for the current phase angle
    # baseline = Baseline(xvals, U, Cknob, eta, shift, Cstray, phi_deg, 0)
    baseline = 0

    # Total signal for the current phase angle
    total_signal = signal1 + signal2 + baseline

    # Plot the snapshot for the current phase angle
    # Use color to represent polarization
    # color = cmap(norm(P_values[i]))
    color = 'red'
    ax_3d.plot(xvals, np.full_like(xvals, phi_deg), total_signal, color=color, alpha=0.7, linewidth=1.5)

# Create a ScalarMappable for the color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You need to set an array for the ScalarMappable

# Add color bar
# cbar = fig_3d.colorbar(sm, ax=ax_3d, shrink=0.6, aspect=10, pad=0.1)
# cbar.set_label('Polarization (P)', fontsize=26, fontweight='bold')
# cbar.ax.tick_params(labelsize=20)

# Set labels with better spacing
ax_3d.set_xlabel('R', fontsize=36, fontfamily='Times New Roman', labelpad=28)
ax_3d.set_ylabel('φ', fontsize=36, fontfamily='Times New Roman', labelpad=28)
ax_3d.set_zlabel('Signal Amplitude', fontsize=36, fontfamily='Times New Roman', labelpad=28)

# Adjust tick parameters
ax_3d.tick_params(axis='x', labelsize=22, width=2, pad=8)
ax_3d.tick_params(axis='y', labelsize=22, width=2, pad=8)
ax_3d.tick_params(axis='z', labelsize=22, width=2, pad=12)

# Make grid lines with adjustable opacity
ax_3d.xaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)
ax_3d.yaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)
ax_3d.zaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.5)

# Add black outline to 3D plot panes
ax_3d.xaxis.pane.set_edgecolor('black')
ax_3d.yaxis.pane.set_edgecolor('black')
ax_3d.zaxis.pane.set_edgecolor('black')
ax_3d.xaxis.pane.set_linewidth(2)
ax_3d.yaxis.pane.set_linewidth(2)
ax_3d.zaxis.pane.set_linewidth(2)

# Set better viewing angle - elevated view from the side
# ax_3d.view_init(elev=25, azim=135)
ax_3d.view_init(elev=25, azim=135)

# Adjust the plot limits for better spacing
ax_3d.set_xlim(xvals[0], xvals[-1])
ax_3d.set_ylim(phi_values[0], phi_values[-1])

# Add some distance between the plot and the edges - increase right margin for z-axis label
fig_3d.subplots_adjust(left=0.05, right=0.7, top=0.95, bottom=0.05)

fig_3d.savefig('plots/3d_signal_snapshots_with_polarization.jpeg', dpi=600, bbox_inches='tight', facecolor='white', pad_inches=0.8)
# plt.show()
