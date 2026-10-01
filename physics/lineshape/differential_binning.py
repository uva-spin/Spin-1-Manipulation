import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from Lineshape import GenerateVectorLineshape

P = 0.5
NUM_BINS = 60
R_MIN, R_MAX = -3.0, 3.0

# Evaluate on a fine grid, then downsample to discrete bin centers.
fine_x = np.linspace(R_MIN, R_MAX, 5000)
signal_fine, iplus_fine, iminus_fine = GenerateVectorLineshape(P, fine_x)

bin_edges = np.linspace(R_MIN, R_MAX, NUM_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_indices = np.searchsorted(fine_x, bin_centers, side="left")
bin_indices = np.clip(bin_indices, 0, len(fine_x) - 1)

signal = signal_fine[bin_indices]
iplus = iplus_fine[bin_indices]
iminus = iminus_fine[bin_indices]
idiff = iplus - iminus

# Extend one bin past the last center so plt.step(where='post') draws the final edge.
x_step = np.append(bin_edges[:-1], bin_edges[-1])
signal_step = np.append(signal, signal[-1])
iplus_step = np.append(iplus, iplus[-1])
iminus_step = np.append(iminus, iminus[-1])
idiff_step = np.append(idiff, idiff[-1])

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 2.0,
})

COLOR_SIGNAL = "#000000"
COLOR_IPLUS = "#d62728"
COLOR_IMINUS = "#1f77b4"
COLOR_DIFF = "#9467bd"  # purple — stands out from I+/I-

step_kw = dict(where="post", linewidth=2.8, solid_joinstyle="miter")


def draw_bin_borders(ax, series_list):
    """Vertical bin edges only under the plotted curves (envelope span)."""
    stacked = np.vstack(series_list)
    for i, edge in enumerate(bin_edges):
        if i == 0:
            col = stacked[:, 0]
        elif i == len(bin_edges) - 1:
            col = stacked[:, -1]
        else:
            col = np.concatenate([stacked[:, i - 1], stacked[:, i]])
        y_bot = min(0.0, float(col.min()))
        y_top = max(0.0, float(col.max()))
        if y_top <= y_bot:
            continue
        ax.plot(
            [edge, edge], [y_bot, y_top],
            color="0.45", linewidth=1.0, solid_capstyle="butt", zorder=1,
        )


def style_axes(ax, y_values, *, ylabel=False):
    y_lo = min(0.0, float(np.min(y_values)))
    y_hi = max(0.0, float(np.max(y_values)))
    pad = 0.05 * (y_hi - y_lo if y_hi > y_lo else 1.0)

    ax.set_xlabel(r"$R$", fontsize=22)
    if ylabel:
        ax.set_ylabel("Intensity (arb. units)", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=20, width=2.0, length=7)
    ax.tick_params(axis="both", which="minor", width=1.5, length=4)
    ax.set_xlim(R_MIN, R_MAX)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xticks(np.arange(R_MIN, R_MAX + 0.1, 1.0))
    ax.grid(True, linestyle="--", color="0.8", linewidth=1.0)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-2, 2))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.get_offset_text().set_fontsize(18)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(2.0)
    ax.set_facecolor("white")


fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7), sharey=False)

# Left: intensity curves
ax_left.step(x_step, signal_step, color=COLOR_SIGNAL, zorder=3, **step_kw)
# ax_left.step(x_step, iplus_step, color=COLOR_IPLUS, zorder=2, **step_kw)
# ax_left.step(x_step, iminus_step, color=COLOR_IMINUS, zorder=2, **step_kw)
draw_bin_borders(ax_left, [signal_step])
style_axes(ax_left, np.concatenate([signal]), ylabel=True)

# Right: intensity curves + I+ − I−
# ax_right.step(x_step, signal_step, color=COLOR_SIGNAL, zorder=3, **step_kw)
# ax_right.step(x_step, iplus_step, color=COLOR_IPLUS, zorder=2, **step_kw)
# ax_right.step(x_step, iminus_step, color=COLOR_IMINUS, zorder=2, **step_kw)
ax_right.step(x_step, idiff_step, color=COLOR_DIFF, zorder=4, **step_kw)
ax_right.axhline(0.0, color="0.35", linewidth=1.2, linestyle=":", zorder=0)
draw_bin_borders(ax_right, [idiff_step])
style_axes(ax_right, np.concatenate([idiff]), ylabel=False)

# Match left panel lower y-limit to the right panel.
(_, left_top) = ax_left.get_ylim()
(right_bot, _) = ax_right.get_ylim()
ax_left.set_ylim(right_bot, left_top)

fig.patch.set_facecolor("white")
fig.tight_layout()

out_path = "differential_binning.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved {out_path}")
plt.show()
