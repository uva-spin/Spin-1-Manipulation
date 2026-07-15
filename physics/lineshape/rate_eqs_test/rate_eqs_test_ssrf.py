"""
rate_eqs_test variant using spin-1 ss-RF population transfer (RF burns only).

Replaces the analytic Iplus_new / Iminus_new mirror-gain formulas with the
packet-level RF transfers from ``physics/ssrf_realtime`` (same as
``spin1_ssrf_realtime``):

    at burn packet kp:   n+ -> n0   (rate gamma_rf * (n+ - n0))
    at mirror packet km: n0 -> n-   (rate gamma_rf * (n0 - n-))

RF burn strength is profiled by local Q = I+ - I-:
  - Q >= 0  -> gamma = 0
  - Q < 0   -> gamma approaches GAMMA_RF as Q approaches the most-negative bin

Recovery, spectral diffusion, DNP, and T1 are disabled so only SSRF burns act.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime import Spin1Params
from physics.ssrf_realtime.rate_equations_realtime import (
    build_model_for_intensities,
    burn_preserves_ps_sign,
)

P = 0.60
NUM_BINS = 500
burn_idx = 172
mirror_idx = NUM_BINS - burn_idx - 1  # 500 - 172 - 1 = 327

GAMMA_RF = 2.0
DT = 0.015
N_STEPS = 100

OUT_DIR = Path(__file__).resolve().parent


def rf_burn_profile(q: np.ndarray, gamma_max: float) -> np.ndarray:
    """Per-bin RF burn rate from Q: zero for Q>=0, up to gamma_max at deepest Q<0."""
    q = np.asarray(q, dtype=float)
    q_min = float(np.min(q))
    if q_min >= 0.0:
        return np.zeros_like(q)
    # q/q_min is in (0, 1] for q in [q_min, 0); clip kills positive-Q bins.
    scale = np.clip(q / q_min, 0.0, 1.0)
    return float(gamma_max) * scale


f = np.linspace(-3, 3, NUM_BINS)
signal, Iplus_ls, Iminus_ls = GenerateVectorLineshape(P, f)

Q_ls = np.asarray(Iplus_ls, dtype=float) - np.asarray(Iminus_ls, dtype=float)
gamma_profile = rf_burn_profile(Q_ls, GAMMA_RF)
gamma_rf_eff = float(gamma_profile[burn_idx])

Iplus_i = float(Iplus_ls[burn_idx])
Iminus_i = float(Iminus_ls[burn_idx])
Iplus_f = float(Iplus_ls[mirror_idx])
Iminus_f = float(Iminus_ls[mirror_idx])

print(f"P={P}")
print(f"Burn bin:   idx={burn_idx}, R={f[burn_idx]:.4f}, I+={Iplus_i:.6e}, I-={Iminus_i:.6e}")
print(f"Mirror bin: idx={mirror_idx}, R={f[mirror_idx]:.4f}, I+={Iplus_f:.6e}, I-={Iminus_f:.6e}")
print(
    f"Q profile:  Q_burn={Q_ls[burn_idx]:.6e}, Q_min={float(np.min(Q_ls)):.6e}, "
    f"gamma_max={GAMMA_RF}, gamma_eff={gamma_rf_eff:.6e}"
)
print(f"SSRF RF-only: gamma_rf={gamma_rf_eff}, dt={DT}, n_steps={N_STEPS}")

# ---------------------------------------------------------------------------
# Old analytic rate-eq transfer (commented out; kept for comparison).
# ---------------------------------------------------------------------------
# xi = 1.3
# t = np.linspace(0.0, 5.0, 100)
#
# def Iplus(t, prefactor):
#     return prefactor * np.exp(t * (1 - 2 * xi))
#
# def Iminus(t, prefactor):
#     return prefactor * np.exp(t * (1 - 2 * xi))
#
# # Mirror gains the depleted burn amount (crossed, 2:1), not the remaining intensity.
# Iplus_new = Iplus_f + (Iminus_i - Iminus(t, Iminus_i)) / 2
# Iminus_new = Iminus_f + (Iplus_i - Iplus(t, Iplus_i)) / 2
#
# Q_theta1 = Iplus(t, Iplus_i) - Iminus(t, Iminus_i)
# Q_theta2 = Iplus_new - Iminus_new
# P_theta1 = Iplus(t, Iplus_i) + Iminus(t, Iminus_i)
# P_theta2 = Iplus_new + Iminus_new
# Q_total = Q_theta1 + Q_theta2
# P_total = P_theta1 + P_theta2

# ---------------------------------------------------------------------------
# SSRF population transfer (RF burns only).
# ---------------------------------------------------------------------------
params = Spin1Params(
    n_bins=NUM_BINS,
    r_min=float(f[0]),
    r_max=float(f[-1]),
    p0=P,
    initial_polarization=P,
    rf_burn_R=float(f[burn_idx]),
    gamma_rf=gamma_rf_eff,
    # RF burns only: disable everything else from the realtime model.
    d_same_plus0=0.0,
    d_same_0minus=0.0,
    d_spec_plus0=0.0,
    d_spec_0minus=0.0,
    dnp_enabled=False,
    t1_rate=0.0,
    dt=DT,
    steps=1,
)

model = build_model_for_intensities(
    Iplus_ls,
    Iminus_ls,
    params=params,
    rf_burn_R=float(f[burn_idx]),
    initial_polarization=P,
)
model.params.gamma_rf = float(gamma_rf_eff)

t_hist = [0.0]
Iplus_burn = [float(Iplus_ls[burn_idx])]
Iminus_burn = [float(Iminus_ls[burn_idx])]
Iplus_mirror = [float(Iplus_ls[mirror_idx])]
Iminus_mirror = [float(Iminus_ls[mirror_idx])]

Iplus_cur = np.asarray(Iplus_ls, dtype=float).copy()
Iminus_cur = np.asarray(Iminus_ls, dtype=float).copy()
steps_applied = 0

for _ in range(N_STEPS):
    state_before = model.n.copy()
    model.step_once(dt=DT, rf_on=True, dnp_on=False)
    Iplus_new, Iminus_new, _ = model.physical_intensities()
    if not burn_preserves_ps_sign(Iplus_cur, Iminus_cur, Iplus_new, Iminus_new, burn_idx):
        model.n = state_before
        break
    Iplus_cur = np.asarray(Iplus_new, dtype=float).copy()
    Iminus_cur = np.asarray(Iminus_new, dtype=float).copy()
    steps_applied += 1
    t_hist.append(steps_applied * DT)
    Iplus_burn.append(float(Iplus_cur[burn_idx]))
    Iminus_burn.append(float(Iminus_cur[burn_idx]))
    Iplus_mirror.append(float(Iplus_cur[mirror_idx]))
    Iminus_mirror.append(float(Iminus_cur[mirror_idx]))

t = np.asarray(t_hist, dtype=float)
Iplus_burn = np.asarray(Iplus_burn, dtype=float)
Iminus_burn = np.asarray(Iminus_burn, dtype=float)
Iplus_mirror = np.asarray(Iplus_mirror, dtype=float)
Iminus_mirror = np.asarray(Iminus_mirror, dtype=float)

print(f"Steps applied: {steps_applied} (t_final={t[-1]:.4f})")
print(
    f"Burn final:   I+={Iplus_burn[-1]:.6e}, I-={Iminus_burn[-1]:.6e}, "
    f"Ps={Iplus_burn[-1] + Iminus_burn[-1]:.6e}"
)
print(
    f"Mirror final: I+={Iplus_mirror[-1]:.6e}, I-={Iminus_mirror[-1]:.6e}, "
    f"Ps={Iplus_mirror[-1] + Iminus_mirror[-1]:.6e}"
)

Q_theta1 = Iplus_burn - Iminus_burn
Q_theta2 = Iplus_mirror - Iminus_mirror
P_theta1 = Iplus_burn + Iminus_burn
P_theta2 = Iplus_mirror + Iminus_mirror
Q_total = Q_theta1 + Q_theta2
P_total = P_theta1 + P_theta2

# Final lineshape from RF-only population transfer (only burn/mirror bins change).
Iplus_ls_final = Iplus_cur
Iminus_ls_final = Iminus_cur
signal_final = Iplus_ls_final + Iminus_ls_final

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(f, Q_ls, label=r"$Q = I_+ - I_-$")
axes[0].axhline(0.0, color="black", linestyle="--")
axes[0].axvline(f[burn_idx], color="red", linestyle="--", label="burn")
axes[0].set_ylabel(r"$Q$")
axes[0].legend()
axes[0].grid(True)
axes[1].plot(f, gamma_profile, label=r"$\gamma_{\mathrm{RF}}(Q)$")
axes[1].axhline(GAMMA_RF, color="gray", linestyle=":", label=r"$\gamma_{\max}$")
axes[1].axvline(f[burn_idx], color="red", linestyle="--", label="burn")
axes[1].plot(f[burn_idx], gamma_rf_eff, "ro", label=rf"$\gamma_{{\mathrm{{eff}}}}={gamma_rf_eff:.3g}$")
axes[1].set_xlabel(r"$R$")
axes[1].set_ylabel(r"$\gamma_{\mathrm{RF}}$")
axes[1].legend()
axes[1].grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "rate_eqs_test_ssrf_burn_profile.png")

plt.figure()
plt.plot(f, signal_final, label=r"$P_s$")
plt.plot(f, Iplus_ls_final, label=r"$I_+$")
plt.plot(f, Iminus_ls_final, label=r"$I_-$")
plt.legend()
plt.savefig(OUT_DIR / "rate_eqs_test_ssrf_lineshape.png")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax1.plot(t, Iplus_burn, label=r"$I_+$")
ax1.plot(t, Iminus_burn, label=r"$I_-$")
ax1.plot(t, Iplus_burn + Iminus_burn, label=r"$I_+$ + $I_-$")
ax1.plot(t, Iplus_burn - Iminus_burn, label=r"$Q$")
ax1.axhline(0.0, color="black", linestyle="--")
ax1.set_xlim(0.0, float(t[-1]))
ax1.set_title(rf"$R$ (burn idx={burn_idx})")
ax1.grid(True)

ax2.plot(t, Iplus_mirror, label=r"$I_+$")
ax2.plot(t, Iminus_mirror, label=r"$I_-$")
ax2.plot(t, Iplus_mirror + Iminus_mirror, label=r"$I_+$ + $I_-$")
ax2.plot(t, Iplus_mirror - Iminus_mirror, label=r"$Q$")
ax2.axhline(0.0, color="black", linestyle="--")
ax2.set_xlim(0.0, float(t[-1]))
ax2.set_title(rf"$-R$ (mirror idx={mirror_idx})")
ax2.grid(True)
fig.legend(loc="outside right upper")
fig.tight_layout()
fig.savefig(OUT_DIR / "rate_eqs_test_ssrf.png")

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(t, Iminus_burn, label=r"$I_-$")
axes[0].plot(t, Iplus_burn, label=r"$I_+$")
axes[0].axhline(0.0, color="black", linestyle="--")
axes[0].set_xlim(0.0, float(t[-1]))
axes[0].set_ylabel(r"$I_-, I_+$")
axes[0].set_title(rf"$I_\pm$ burn (idx={burn_idx})")
axes[0].legend()
axes[0].grid(True)
axes[1].plot(t, Iplus_mirror, label=r"$I_+$")
axes[1].plot(t, Iminus_mirror, label=r"$I_-$")
axes[1].axhline(0.0, color="black", linestyle="--")
axes[1].set_xlim(0.0, float(t[-1]))
axes[1].set_ylabel(r"$I_-, I_+$")
axes[1].set_title(rf"$I_\pm$ mirror (idx={mirror_idx})")
axes[1].legend()
axes[1].grid(True)
fig.tight_layout()
fig.savefig(OUT_DIR / "rate_eqs_test_ssrf_Iplus_Iminus.png")

fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True, sharey=True)
axes[0].plot(t, Q_theta1, label=r"$Q$")
axes[0].plot(t, P_theta1, label=r"$P$")
axes[0].axhline(0.0, color="black", linestyle="--")
axes[0].set_xlim(0.0, float(t[-1]))
axes[0].grid(True)
axes[0].legend()
axes[0].set_ylabel(r"$Q, P$")
axes[0].set_title(r"$R$")
axes[1].plot(t, Q_theta2, label=r"$Q$")
axes[1].plot(t, P_theta2, label=r"$P$")
axes[1].axhline(0.0, color="black", linestyle="--")
axes[1].set_xlim(0.0, float(t[-1]))
axes[1].grid(True)
axes[1].legend()
axes[1].set_ylabel(r"$Q, P$")
axes[1].set_title(r"$-R$")
axes[2].plot(t, Q_total, label=r"$Q$")
axes[2].plot(t, P_total, label=r"$P$")
axes[2].axhline(0.0, color="black", linestyle="--")
axes[2].set_xlim(0.0, float(t[-1]))
axes[2].grid(True)
axes[2].legend()
axes[2].set_ylabel(r"$Q, P$")
axes[2].set_title(r"$Total$")
axes[3].plot(t, Q_total + P_total, label=r"$Q + P$")
axes[3].axhline(0.0, color="black", linestyle="--")
axes[3].set_xlim(0.0, float(t[-1]))
axes[3].grid(True)
axes[3].legend()
axes[3].set_ylabel(r"$Q + P$")
axes[3].set_title(r"$Total$")
fig.supxlabel(r"$t$")
fig.savefig(OUT_DIR / "rate_eqs_test_ssrf_Q.png")

print(f"Wrote plots to {OUT_DIR}")
