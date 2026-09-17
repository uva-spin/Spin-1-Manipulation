"""
Mirror of test_ssrf_burn_lookup.py using direct rate-equation burns.

Burns P and P_REF down to the P_REF2 signal level at one bin via a single
``solve_rate_equations`` step each (``dt`` chosen by root-finding), then
prints the resulting Iplus/Iminus at the burn bin.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from physics.burn_lookup_realtime import BurnTrajectoryConfig
from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime.rate_equations_realtime import solve_rate_equations, verify_burn_response, verify_rates_response
P = 0.6
P_REF = 0.45
P_REF2 = 0.44
NUM_BINS = 500
x0 = -0.92
gamma_rf = 2.0
trajectory_cfg = BurnTrajectoryConfig(n_bins=NUM_BINS, max_steps=10000, dt=0.0015, gamma_rf=gamma_rf)
f = trajectory_cfg.f
(signal, iplus, iminus) = GenerateVectorLineshape(P, f)
(ref_signal, ref_iplus, ref_iminus) = GenerateVectorLineshape(P_REF, f)
(ref2_signal, _ref2_iplus, _ref2_iminus) = GenerateVectorLineshape(P_REF2, f)
print(f'bin size: {f[1] - f[0]:.6f} MHz')
test_bin_idx = np.argmin(np.abs(f - x0))
mirror_bin_idx = len(f) - 1 - test_bin_idx
target_signal_amp = ref2_signal[test_bin_idx]
print(f'Burn bin: idx={test_bin_idx}, R={f[test_bin_idx]:.4f} MHz')
print(f'Mirror bin: idx={mirror_bin_idx}, R={f[mirror_bin_idx]:.4f} MHz')
print(f'Signal Amplitude at burned bin: {signal[test_bin_idx]:.6e}')
print(f'Iplus Amplitude at burned bin: {iplus[test_bin_idx]:.6e}')
print(f'Iminus Amplitude at burned bin: {iminus[test_bin_idx]:.6e}')

def _find_brentq_bracket(func, values):
    """Return (lo, hi) where func changes sign, or raise ValueError."""
    samples = [(value, func(value)) for value in values]
    for ((v0, f0), (v1, f1)) in zip(samples, samples[1:]):
        if f0 == 0.0:
            return (v0, v0)
        if f0 * f1 < 0.0:
            return (v0, v1)
    raise ValueError(f'No sign change in residual; cannot bracket root. Sampled values={list(values)}, residuals={[v for (_, v) in samples]}')

def _burned_ps_single_step(polarization, burn_iplus, burn_iminus, dt):
    """Return burn-bin Ps after one ``solve_rate_equations`` step at ``dt``."""
    params = trajectory_cfg.spin1_params(polarization)
    (iplus_new, iminus_new, _, _, _) = solve_rate_equations(burn_iplus, burn_iminus, dt, trajectory_cfg.gamma_rf, test_bin_idx, params=params, rf_only=True)
    return iplus_new[test_bin_idx] + iminus_new[test_bin_idx]

def _find_burn_dt(polarization, burn_iplus, burn_iminus, *, target_ps):
    """Find ``dt`` for a single RF step that lands exactly on ``target_ps``."""
    burn_iplus = np.asarray(burn_iplus)
    burn_iminus = np.asarray(burn_iminus)

    def residual(dt):
        return _burned_ps_single_step(polarization, burn_iplus, burn_iminus, dt) - target_ps
    max_dt = trajectory_cfg.dt * trajectory_cfg.max_steps
    dts = np.linspace(trajectory_cfg.dt, max_dt, 200)
    (bracket_lo, bracket_hi) = _find_brentq_bracket(residual, dts)
    return bracket_lo if bracket_lo == bracket_hi else brentq(residual, bracket_lo, bracket_hi)

def _apply_single_step_burn(polarization, burn_iplus, burn_iminus, dt):
    """Apply one ``solve_rate_equations`` step at ``dt``."""
    params = trajectory_cfg.spin1_params(polarization)
    (iplus_new, iminus_new, _, _, _) = solve_rate_equations(burn_iplus, burn_iminus, dt, trajectory_cfg.gamma_rf, test_bin_idx, params=params, rf_only=True)
    iplus_new = np.asarray(iplus_new)
    iminus_new = np.asarray(iminus_new)
    burned_signal = iplus_new + iminus_new
    ps_at_burn = burned_signal[test_bin_idx]
    return (burned_signal, iplus_new, iminus_new, ps_at_burn)
matched_dt_p = _find_burn_dt(P, iplus, iminus, target_ps=target_signal_amp)
matched_dt_ref = _find_burn_dt(P_REF, ref_iplus, ref_iminus, target_ps=target_signal_amp)
(new_signal, new_iplus, new_iminus, ps_at_burn_p) = _apply_single_step_burn(P, iplus, iminus, matched_dt_p)
(ref_burn_signal, ref_burn_iplus, ref_burn_iminus, ps_at_burn_ref) = _apply_single_step_burn(P_REF, ref_iplus, ref_iminus, matched_dt_ref)
print(f'Burn dt (P={P * 100:.0f}% -> P_REF2={P_REF2 * 100:.0f}%): {matched_dt_p:.6e}, Ps at burn bin={ps_at_burn_p:.6e}')
print(f'Burn dt (P_REF={P_REF * 100:.0f}% -> P_REF2={P_REF2 * 100:.0f}%): {matched_dt_ref:.6e}, Ps at burn bin={ps_at_burn_ref:.6e}')
print(f'Target signal amplitude (P_REF2): {target_signal_amp:.6e}')
rates_check = verify_burn_response(iplus, iminus, new_iplus, new_iminus, test_bin_idx)
reference_rates_check = verify_rates_response(iplus, iminus, test_bin_idx, trajectory_cfg.gamma_rf, dt=trajectory_cfg.dt)
print(f'Signal Amplitude at burned bin (after): {new_signal[test_bin_idx]:.6e}')
print(f'Iplus Amplitude at burned bin (after): {new_iplus[test_bin_idx]:.6e}')
print(f'Iminus Amplitude at burned bin (after): {new_iminus[test_bin_idx]:.6e}')
print(f'Change in Iplus: {new_iplus[test_bin_idx] - iplus[test_bin_idx]:.6e}')
print(f'Change in Iminus: {new_iminus[test_bin_idx] - iminus[test_bin_idx]:.6e}')
print(f'Burned signal amplitude at burn bin (P):     {new_signal[test_bin_idx]:.6e}')
print(f'Burned signal amplitude at burn bin (P_REF): {ref_burn_signal[test_bin_idx]:.6e}')
print(f'Burned Iplus amplitude at burn bin (P):     {new_iplus[test_bin_idx]:.10e}')
print(f'Burned Iplus amplitude at burn bin (P_REF): {ref_burn_iplus[test_bin_idx]:.10e}')
print(f'Burned Iminus amplitude at burn bin (P):     {new_iminus[test_bin_idx]:.10e}')
print(f'Burned Iminus amplitude at burn bin (P_REF): {ref_burn_iminus[test_bin_idx]:.10e}')
iplus_diff = abs(new_iplus - iplus[test_bin_idx])
iminus_diff = abs(new_iminus - iminus[test_bin_idx])
print('iplus difference:')
print(f'  at burn bin ({f[test_bin_idx]:.4f} MHz): {iplus_diff[test_bin_idx]:.6e}')
print()
print('iminus difference:')
print(f'  at burn bin ({f[test_bin_idx]:.4f} MHz): {iminus_diff[test_bin_idx]:.6e}')
print(f'Iplus < Iminus: {iplus[test_bin_idx] < iminus[test_bin_idx]}')
print(f'iplus_diff < iminus_diff: {iplus_diff[test_bin_idx] < iminus_diff[test_bin_idx]}')
print()
print('Rates response check (direct burn trajectory):')
print(f"  passed: {rates_check['passed']}")
print(f"  |Ps| decreased at burn bin: {rates_check['magnitude_decreased']}")
print(f"  amp_burn / amp_mirror: {rates_check['ratios']['amp_burn_over_amp_mirror']:.4f}")
print(f"  dI+_burn / dI-_mirror: {rates_check['ratios']['iplus_burn_over_iminus_mirror']:.4f}")
print(f"  dI-_burn / dI+_mirror: {rates_check['ratios']['iminus_burn_over_iplus_mirror']:.4f}")
print()
print('Rates response check (single rate-equation step reference):')
print(f"  passed: {reference_rates_check['passed']}")
print(f"  |Ps| decreased at burn bin: {reference_rates_check['magnitude_decreased']}")
print(f"  amp_burn / amp_mirror: {reference_rates_check['ratios']['amp_burn_over_amp_mirror']:.4f}")
print(f"  dI+_burn / dI-_mirror: {reference_rates_check['ratios']['iplus_burn_over_iminus_mirror']:.4f}")
print(f"  dI-_burn / dI+_mirror: {reference_rates_check['ratios']['iminus_burn_over_iplus_mirror']:.4f}")
(fig, axes) = plt.subplots(1, 1, figsize=(12, 8))
axes.plot(f, new_iplus + new_iminus, label=f'signal (after, P={P * 100:.0f}%)')
axes.plot(f, new_iplus, label=f'iplus (after, P={P * 100:.0f}%)', linestyle='--')
axes.plot(f, new_iminus, label=f'iminus (after, P={P * 100:.0f}%)', linestyle='--')
axes.plot(f, ref_burn_iplus + ref_burn_iminus, label=f'signal (after, P_REF={P_REF * 100:.0f}%)', alpha=0.8)
axes.plot(f, ref_burn_iplus, label=f'iplus (after, P_REF={P_REF * 100:.0f}%)', linestyle='--', alpha=0.8)
axes.plot(f, ref_burn_iminus, label=f'iminus (after, P_REF={P_REF * 100:.0f}%)', linestyle='--', alpha=0.8)
axes.set_xlabel('Frequency')
axes.set_ylabel('Amplitude')
axes.legend()
plt.savefig('test_ssrf.png', dpi=300)
plt.close()
