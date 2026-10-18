import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics import rf

P = 0.45
N_BINS = 500
R = np.linspace(-6.0, 6.0, N_BINS)
BURN_R = -1.0
GAMMA_RF = 10.0
DT = 0.0015
MAX_STEPS = 40
GAUSS_FWHM_R = 0.03*3
LORENTZ_FWHM_R = 0.015*3
AFP_HALF = 0.25
OUT = Path(__file__).resolve().parent


def _q_negative_intervals(R, Q):
    """Index ranges (inclusive) where Q < 0."""
    mask = Q < 0.0
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) != 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def _zero_crossing_r(R, Q, left_index, right_index):
    """Linear R where Q crosses zero between ``left_index`` and ``right_index``."""
    y0, y1 = float(Q[left_index]), float(Q[right_index])
    r0, r1 = float(R[left_index]), float(R[right_index])
    if y0 == y1:
        return r0
    t = y0 / (y0 - y1)
    return r0 + t * (r1 - r0)


def q_negative_bounds(R, Q):
    """Left/right R edges of each shaded Q < 0 region (at zero crossings)."""
    bounds = []
    for i0, i1 in _q_negative_intervals(R, Q):
        r_left = float(R[0]) if i0 == 0 else _zero_crossing_r(R, Q, i0 - 1, i0)
        r_right = float(R[-1]) if i1 >= len(Q) - 1 else _zero_crossing_r(R, Q, i1, i1 + 1)
        bounds.append((r_left, r_right))
    return bounds


def base_model():
    """Analytic Pake equilibrium, same construction as create_data."""
    params = rf.Spin1Params(
        n_bins=N_BINS,
        r_min=float(R[0]),
        r_max=float(R[-1]),
        p0=P,
        q0=None,
        line_gamma=0.05,
        line_asym=0.04,
        dt=DT,
        rf_enabled=False,
        rf_burn_R=BURN_R,
    )
    return rf.Spin1Model(params)


model = base_model()
iplus, iminus, signal = model.physical_intensities()


Q = iplus - iminus  # local tensor polarization at each bin (physics/README)
Ps = iplus + iminus

fig, ax = plt.subplots(figsize=(9.2, 5.1))
ax.fill_between(
    R,
    Q,
    0.0,
    where=Q < 0.0,
    interpolate=True,
    color="#E74C3C",
    alpha=0.55,
    zorder=1,
)
for r_left, r_right in q_negative_bounds(R, Q):
    ax.axvline(r_left, color="#C0392B", ls=":", lw=1.4, alpha=0.9, zorder=2)
    ax.axvline(r_right, color="#C0392B", ls=":", lw=1.4, alpha=0.9, zorder=2)
ax.plot(R, Ps, color="#2980B9", lw=1.8, alpha=0.38, zorder=2)
ax.plot(R, Q, color="black", lw=2.15, zorder=3)
ax.set_xlabel(r"$R$", fontsize=16)
ax.set_ylabel("amplitude (a.u.)", fontsize=16)
ax.set_xlim(float(R[0]), float(R[-1]))
ax.margins(x=0.0, y=0.08)
ax.grid(True, linestyle="--", color="0.75", linewidth=1.0)
ax.axhline(0.0, color="0.55", linewidth=1.0)
ax.set_facecolor("white")
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_linewidth(1.0)
ax.tick_params(labelsize=13, width=1.0, length=4)
fig.savefig(OUT / "tensor_pol.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# def style(ax):
#     ax.set_xlabel(r"$R$", fontsize=16)
#     ax.set_ylabel("intensity (a.u.)", fontsize=16)
#     ax.set_xlim(float(R[0]), float(R[-1]))
#     ax.margins(x=0.0, y=0.08)
#     ax.tick_params(labelsize=13, width=0.8, length=4)
#     ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
#     ax.grid(True, linestyle="--", color="0.86", linewidth=0.8, zorder=0)
#     ax.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
#     ax.set_facecolor("white")
#     for side in ("top", "right", "bottom", "left"):
#         ax.spines[side].set_visible(True)
#         ax.spines[side].set_color("0.2")
#         ax.spines[side].set_linewidth(0.9)
#     # ax.legend(
#     #     fontsize=11,
#     #     frameon=True,
#     #     fancybox=False,
#     #     edgecolor="0.75",
#     #     framealpha=0.95,
#     #     loc="center left",
#     #     bbox_to_anchor=(1.02, 0.5),
#     # )


# def save(fig, name):
#     path = OUT / name
#     fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     print(path)


# def draw_before(ax):
#     ax.plot(R, iplus, color="#E74C3C", lw=1.15, ls=(0, (4, 2)), alpha=0.5, label=r"$I_+$ before", zorder=2)
#     ax.plot(R, iminus, color="#2980B9", lw=1.15, ls=(0, (4, 2)), alpha=0.5, label=r"$I_-$ before", zorder=2)
#     ax.plot(R, signal, color="0.15", lw=1.25, ls=(0, (4, 2)), alpha=0.45, label="signal before", zorder=2)


# model = base_model()
# (ip, im, _, support) = model.apply_voigt_burn(BURN_R, GAUSS_FWHM_R, LORENTZ_FWHM_R, GAMMA_RF, MAX_STEPS, dt=DT)

# fig, ax = plt.subplots(figsize=(9.2, 5.1))
# ax.plot(R, iplus, color="#E74C3C", lw=2.0, label=r"$I_+$ before", zorder=2)
# ax.plot(R, iminus, color="#2980B9", lw=2.0, label=r"$I_-$ before", zorder=2)
# ax.plot(R, signal, color="black", lw=2.15, label="signal before", zorder=2)
# style(ax)
# fig.tight_layout()
# save(fig, "signal.png")

# fig, ax = plt.subplots(figsize=(9.2, 5.1))
# ax.axvspan(R[support[0]], R[support[-1]], color="#1E8449", alpha=0.16, label="ssRF window", zorder=0)
# ax.axvline(BURN_R, color="#1E8449", ls=":", lw=1.6, label=r"$R = -1$", zorder=1)
# draw_before(ax)
# ax.plot(R, ip, color="#E74C3C", lw=2.0, label=r"$I_+$ after ssRF", zorder=3)
# ax.plot(R, im, color="#2980B9", lw=2.0, label=r"$I_-$ after ssRF", zorder=3)
# ax.plot(R, ip + im, color="black", lw=2.15, label="signal after ssRF", zorder=3)
# style(ax)
# fig.tight_layout()
# print(f"ssRF steps: {MAX_STEPS}  window bins: {len(support)}")
# save(fig, "ssrf.png")

# model = base_model()
# window = np.flatnonzero((R >= BURN_R - AFP_HALF) & (R <= BURN_R + AFP_HALF))
# model.params.afp_subset_indices = [int(i) for i in window]
# model.afp_sweep()
# ip, im, _ = model.physical_intensities()

# fig, ax = plt.subplots(figsize=(9.2, 5.1))
# ax.axvspan(R[window[0]], R[window[-1]], color="#7D3C98", alpha=0.16, label="AFP sweep", zorder=0)
# ax.axvline(BURN_R, color="#7D3C98", ls=":", lw=1.6, label=r"$R = -1$", zorder=1)
# draw_before(ax)
# ax.plot(R, ip, color="#E74C3C", lw=2.0, label=r"$I_+$ after AFP", zorder=3)
# ax.plot(R, im, color="#2980B9", lw=2.0, label=r"$I_-$ after AFP", zorder=3)
# ax.plot(R, ip + im, color="black", lw=2.15, label="signal after AFP", zorder=3)
# style(ax)
# fig.tight_layout()
# save(fig, "afp.png")



# ### afp + ssrf
# model = base_model()
# window = np.flatnonzero((R >= BURN_R - AFP_HALF) & (R <= BURN_R + AFP_HALF))
# model.params.afp_subset_indices = [int(i) for i in window]
# model.afp_sweep()
# burn_idx = int(model.burn_index(BURN_R))
# support = rf.configure_voigt_ssrf(model, burn_idx, GAMMA_RF, half_width=15, full_spectrum_recovery=True)
# (ip, im, _, support) = model.apply_voigt_burn(BURN_R, GAUSS_FWHM_R, LORENTZ_FWHM_R, GAMMA_RF, MAX_STEPS, dt=DT)

# fig, ax = plt.subplots(figsize=(9.2, 5.1))
# ax.axvspan(R[window[0]], R[window[-1]], color="#7D3C98", alpha=0.16, label="AFP sweep", zorder=0)
# ax.axvspan(R[support[0]], R[support[-1]], color="#1E8449", alpha=0.16, label="ssRF window", zorder=0)
# ax.axvline(BURN_R, color="#7D3C98", ls=":", lw=1.6, label=r"$R = -1$", zorder=1)
# draw_before(ax)
# ax.plot(R, ip, color="#E74C3C", lw=2.0, label=r"$I_+$ after AFP", zorder=3)
# ax.plot(R, im, color="#2980B9", lw=2.0, label=r"$I_-$ after AFP", zorder=3)
# ax.plot(R, ip + im, color="black", lw=2.15, label="signal after AFP", zorder=3)
# style(ax)
# fig.tight_layout()
# save(fig, "afp_ssrf.png")