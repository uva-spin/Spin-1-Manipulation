import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 0.05
s = 0.04
bigy = np.sqrt(3 - s)

def Lineshape(x,eps):
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
    
    return icurve(x,eps)/10



def GenerateVectorLineshape(P,x):

    r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
    
    if P > 0:
        Iplus = r*Lineshape(x,1)
        Iminus = Lineshape(x,-1)
        r = r
    else:
        r = 1/r
        Iplus = -Lineshape(x,1)
        Iminus = -r*Lineshape(x,-1)
    
    ### Scaling
    pSummed = np.sum(Iplus + Iminus)
    deltaP = P/pSummed
    Iplus = Iplus*deltaP
    Iminus = Iminus*deltaP
    signal = Iplus + Iminus

    return signal,Iplus,Iminus

x_values = np.linspace(-3, 3, 500)
signal, Iplus, Iminus = GenerateVectorLineshape(0.5, x_values)

# Number of bins for the vertical lines
num_bins = 50
bin_indices = np.linspace(0, len(signal)-1, num_bins, dtype=int)
bin_x_positions = x_values[bin_indices]  # Actual x positions

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the signal lines (these remain static) - use x_values for proper scaling
line_signal, = ax.plot(x_values, signal, color='black', linewidth=2, label='Signal', zorder=3)
line_iplus, = ax.plot(x_values, Iplus, color='blue', linewidth=2, alpha=0.5, label='Iplus', zorder=2)
line_iminus, = ax.plot(x_values, Iminus, color='green', linewidth=2, alpha=0.5, label='Iminus', zorder=2)

# Draw all vertical lines from 0 upward to the signal value (already visible)
for idx in bin_indices:
    x_pos = x_values[idx]
    ax.plot([x_pos, x_pos], [0, signal[idx]], color='black', linewidth=1, alpha=0.3, zorder=1)

# Create filled regions between vertical lines (will be animated)
fills = []
for i in range(len(bin_indices) - 1):
    # Get x indices for this bin
    start_idx = bin_indices[i]
    end_idx = bin_indices[i + 1]
    x_bin = x_values[start_idx:end_idx + 1]
    y_bin = signal[start_idx:end_idx + 1]
    
    # Create a polygon fill from 0 to signal, initially invisible
    fill = ax.fill_between(x_bin, 0, y_bin, color='red', alpha=0, zorder=2)
    fills.append(fill)

ax.set_xlabel('R', fontsize=12)
ax.set_ylabel('arb units', fontsize=12)
ax.set_title('Binning of r', fontsize=14)
ax.legend(fontsize=12)
ax.tick_params(labelsize=12)
ax.set_xticks(np.arange(-3, 3.1, 0.5))
plt.tight_layout()

# Animation function
def animate(frame):
    # frame ranges from 0 to num_bins-1
    for i, fill in enumerate(fills):
        # Remove the old fill and create a new one with updated alpha
        fill.remove()
        
        start_idx = bin_indices[i]
        end_idx = bin_indices[i + 1]
        x_bin = x_values[start_idx:end_idx + 1]
        y_bin = signal[start_idx:end_idx + 1]
        
        # Only fill the current bin (frame), unfill others
        if i == frame:
            fills[i] = ax.fill_between(x_bin, 0, y_bin, color='red', alpha=0.6, zorder=2)
        else:
            fills[i] = ax.fill_between(x_bin, 0, y_bin, color='red', alpha=0, zorder=2)
    
    return fills

# Create animation
# Each frame will show one bin filled
anim = FuncAnimation(fig, animate, frames=len(fills), interval=100, blit=False, repeat=True)

plt.show()
# To save the animation, uncomment:
anim.save('binning_animation.gif', writer='pillow', fps=10)