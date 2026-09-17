"""
SSRF all-bins burn: fixed DT, ≤N_STEPS; bisect min gamma_rf to null local Q(R).

Burn-down uses RF + sameθ + neighbor diffusion (DNP/T1 off). Neighbor spillover
is discarded on commit (burn + RF-mirror bins only). Optional AFP sweep (physics.afp)
runs after all burns, matching Data_Creation/ssRFData_mc intensity rescaling.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tqdm
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from physics.afp import AFP
from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime import Spin1Params
from physics.ssrf_realtime.rate_equations_realtime import build_model_for_intensities
P = 0.4
NUM_BINS = 500
DT = 0.05
N_STEPS = 500
Q_ABS_TOL = 1e-10
Q_FRAC_TOL = 1e-07
GAMMA_HI_INIT = 5.0
GAMMA_MAX = GAMMA_HI_INIT
N_BISECT = 12
MAX_GDT = 0.05
MAX_NSUB = 20
AFP_ENABLED = True
AFP_EFFICIENCY = 1.0
AFP_CENTER_EXCLUSION_BINS = 5
AFP_BIN_RANGE = None
OUT_DIR = Path(__file__).resolve().parent
D_SAME_PLUS0 = 0.25
D_SAME_0MINUS = 0.15
D_SPEC_PLUS0 = 1.5
D_SPEC_0MINUS = 0.8

def max_euler_gamma(dt):
    return MAX_NSUB * MAX_GDT / max(dt, 1e-30)

def mirror_bin_idx(n_bins, bin_idx):
    return n_bins - 1 - bin_idx

def q_at_r_bin(iplus, iminus, bin_idx):
    return iplus[bin_idx] - iminus[bin_idx]

def lineshape_area(iplus, iminus, f):
    return np.trapezoid(np.asarray(iplus) + np.asarray(iminus), f)

def q_total(iplus, iminus):
    return np.sum(iplus - iminus)

def p_total(iplus, iminus):
    return np.sum(iplus + iminus)

def q_target_tol(q_before):
    return max(Q_ABS_TOL, Q_FRAC_TOL * abs(q_before))

def _crossed_zero(before, after):
    if before > 0.0:
        return after <= 0.0
    if before < 0.0:
        return after >= 0.0
    return after != 0.0

def physical_ip_im_at_bin(model, bin_idx):
    """Physical I±(R) at one bin from packet state."""
    bin_idx = bin_idx
    m = mirror_bin_idx(len(model.n), bin_idx)
    scale = model.display_cal / model.dR
    n = model.n
    return (scale * (n[bin_idx, 0] - n[bin_idx, 1]), scale * (n[m, 1] - n[m, 2]))

def q_from_packet(model, burn_idx):
    (ip, im) = physical_ip_im_at_bin(model, burn_idx)
    return ip - im

def enrich_trial_totals(trial, iplus_before, iminus_before, f):
    if 'area_loss' in trial:
        return trial
    (iplus_cur, iminus_cur) = (trial['iplus'], trial['iminus'])
    (qt0, qt1) = (q_total(iplus_before, iminus_before), q_total(iplus_cur, iminus_cur))
    (a0, a1) = (lineshape_area(iplus_before, iminus_before, f), lineshape_area(iplus_cur, iminus_cur, f))
    out = dict(trial)
    out.update(q_total_before=qt0, q_total_after=qt1, q_total_gain=qt1 - qt0, area_before=a0, area_after=a1, area_loss=a0 - a1)
    return out

def afp_touched_bins(n_bins, subset):
    """Intensity bins AFP changes: each sweep index i also updates mirror(i)."""
    touched = set()
    for i in subset:
        touched.add(i)
        touched.add(mirror_bin_idx(n_bins, i))
    return sorted(touched)

def apply_afp_sweep(iplus, iminus, *, bin_range=None, subset_indices=None, efficiency=AFP_EFFICIENCY, center_margin=AFP_CENTER_EXCLUSION_BINS):
    """
    AFP on I± via physics.afp.AFP.

    AFP.intensities_to_populations re-normalizes Σρ, so raw to_intensities()×Σ(I++I−)
    globally rescales the spectrum. Match ssRF commit_burn_bins_only: recover intensity
    units from a pre-AFP round-trip, then write only sweep bins and their mirrors.

    ``subset_indices`` / ``bin_range`` are sweep frequencies only. Do not also pass
    RF-mirror indices — each sweep at i already swaps the mirror packet. Untouched
    bins keep their pre-AFP intensities exactly.

    Returns (Iplus, Iminus, subset_used).
    """
    iplus = np.asarray(iplus)
    iminus = np.asarray(iminus)
    n = len(iplus)
    total_area = np.sum(iplus + iminus)
    if total_area <= 0.0:
        return (iplus.copy(), iminus.copy(), [])
    if subset_indices is not None:
        subset = [i for i in subset_indices]
    elif bin_range is not None:
        (start, stop) = (bin_range[0], bin_range[1])
        subset = list(range(start, stop))
    else:
        subset = list(range(n))
    if center_margin > 0:
        c = n // 2
        forbidden = set(range(max(0, c - center_margin), min(n, c + center_margin + 1)))
        subset = [i for i in subset if i not in forbidden]
    afp = AFP.from_intensities(iplus, iminus)
    (ip_rt, im_rt) = afp.to_intensities()
    rt_sum = np.sum(np.asarray(ip_rt) + np.asarray(im_rt))
    scale = total_area / rt_sum if abs(rt_sum) > 1e-30 else total_area
    if subset:
        afp.perform_afp(subset_indices=subset, efficiency=efficiency, show_progress=False)
    (ip_new, im_new) = afp.to_intensities()
    ip_new = np.asarray(ip_new) * scale
    im_new = np.asarray(im_new) * scale
    out_ip = iplus.copy()
    out_im = iminus.copy()
    touched = afp_touched_bins(n, subset)
    touched_set = set(touched)
    for k in touched:
        out_ip[k] = ip_new[k]
        out_im[k] = im_new[k]
    if touched:
        unt_idx = [i for i in range(n) if i not in touched_set]
        area_unt = np.sum(out_ip[unt_idx] + out_im[unt_idx]) if unt_idx else 0.0
        ps_touch = out_ip[touched] + out_im[touched]
        area_touch = np.sum(ps_touch)
        missing = total_area - area_unt - area_touch
        if abs(missing) > 1e-15:
            weights = np.maximum(ps_touch, 0.0)
            wsum = np.sum(weights)
            if wsum > 1e-30:
                adds = missing * (weights / wsum)
            else:
                adds = np.full(len(touched), missing / len(touched))
            for (k, add) in zip(touched, adds):
                out_ip[k] += 0.5 * add
                out_im[k] += 0.5 * add
    return (out_ip, out_im, subset)

def _burn_params(n_bins, r_min, r_max, polarization):
    return Spin1Params(n_bins=n_bins, r_min=r_min, r_max=r_max, p0=polarization, initial_polarization=polarization, gamma_rf=0.0, d_same_plus0=D_SAME_PLUS0, d_same_0minus=D_SAME_0MINUS, d_spec_plus0=D_SPEC_PLUS0, d_spec_0minus=D_SPEC_0MINUS, dnp_enabled=False, t1_rate=0.0, dt=DT, steps=1)

def commit_burn_bins_only(iplus, iminus, iplus_sim, iminus_sim, burn_idx):
    burn_idx = burn_idx
    mirror_idx = mirror_bin_idx(len(iplus), burn_idx)
    iplus_out = np.asarray(iplus).copy()
    iminus_out = np.asarray(iminus).copy()
    for idx in (burn_idx, mirror_idx):
        iplus_out[idx] = iplus_sim[idx]
        iminus_out[idx] = iminus_sim[idx]
    return (iplus_out, iminus_out)

def apply_rf_burn(iplus, iminus, burn_idx, gamma_rf, *, f, polarization, dt=DT, n_steps=N_STEPS, q_tol=None, model=None, n0=None, light=False, gamma_off_after_steps=10):
    if gamma_rf <= 0.0 or n_steps <= 0:
        return None
    burn_idx = burn_idx
    q_before = q_at_r_bin(iplus, iminus, burn_idx)
    tol = q_tol if q_tol is not None else q_target_tol(q_before)
    if model is None or n0 is None:
        params = _burn_params(len(f), f[0], f[-1], polarization)
        model = build_model_for_intensities(iplus, iminus, params=params, rf_burn_R=f[burn_idx], initial_polarization=polarization)
        n0 = model.n.copy()
    else:
        model.params.rf_burn_R = f[burn_idx]
    model.params.gamma_rf = gamma_rf
    model.n = n0.copy()
    model.t = 0.0
    mirror_idx = mirror_bin_idx(len(iplus), burn_idx)
    ip_prev = {burn_idx: iplus[burn_idx], mirror_idx: iplus[mirror_idx]}
    im_prev = {burn_idx: iminus[burn_idx], mirror_idx: iminus[mirror_idx]}
    g = gamma_rf
    dt_f = dt
    n_sub = min(max(1, np.ceil(abs(g) * dt_f / MAX_GDT)), MAX_NSUB) if g else 1
    dt_sub = dt_f / n_sub
    steps_done = 0
    off_after = None if gamma_off_after_steps is None else gamma_off_after_steps
    for _step in range(n_steps):
        if off_after is not None and steps_done >= off_after:
            model.params.gamma_rf = 0.0
        if abs(q_from_packet(model, burn_idx)) <= tol:
            break
        state_before = model.n.copy()
        rf_on = model.params.gamma_rf != 0.0
        for _ in range(n_sub if rf_on else 1):
            model.step_once(dt=dt_sub if rf_on else dt_f, rf_on=rf_on, dnp_on=False, copy=False)
        (ip_new, im_new) = ({}, {})
        sign_ok = True
        for idx in (burn_idx, mirror_idx):
            (ip_new[idx], im_new[idx]) = physical_ip_im_at_bin(model, idx)
            for (b, a) in ((ip_prev[idx], ip_new[idx]), (im_prev[idx], im_new[idx]), (ip_prev[idx] + im_prev[idx], ip_new[idx] + im_new[idx])):
                if _crossed_zero(b, a):
                    sign_ok = False
                    break
            if not sign_ok:
                break
        if not sign_ok:
            model.n = state_before
            break
        (ip_prev, im_prev) = (ip_new, im_new)
        steps_done += 1
    if steps_done == 0:
        return None
    (iplus_sim, iminus_sim, _) = model.physical_intensities()
    (iplus_cur, iminus_cur) = commit_burn_bins_only(iplus, iminus, np.asarray(iplus_sim), np.asarray(iminus_sim), burn_idx)
    q_after = iplus_cur[burn_idx] - iminus_cur[burn_idx]
    out = {'burn_idx': burn_idx, 'gamma_rf': g, 'n_steps': steps_done, 't_burn': steps_done * dt_f, 'q_before': q_before, 'q_after': q_after, 'q_gain': q_after - q_before, 'iplus': iplus_cur, 'iminus': iminus_cur}
    return out if light else enrich_trial_totals(out, iplus, iminus, f)

def find_gamma_to_null_q_r(iplus, iminus, burn_idx, *, f, polarization, dt=DT, n_steps=N_STEPS, gamma_hi=GAMMA_HI_INIT, n_bisect=N_BISECT, gamma_guess=None):
    q_before = q_at_r_bin(iplus, iminus, burn_idx)
    if q_before >= 0.0:
        return None
    tol = q_target_tol(q_before)
    params = _burn_params(len(f), f[0], f[-1], polarization)
    model = build_model_for_intensities(iplus, iminus, params=params, rf_burn_R=f[burn_idx], initial_polarization=polarization)
    n0 = model.n.copy()

    def trial_at(gamma):
        return apply_rf_burn(iplus, iminus, burn_idx, gamma, f=f, polarization=polarization, dt=dt, n_steps=n_steps, q_tol=tol, model=model, n0=n0, light=True)

    def meets(trial):
        return trial is not None and abs(trial['q_after']) <= tol
    expand_cap = max(gamma_hi, 2.0 * max_euler_gamma(dt))
    hi = min(gamma_hi, expand_cap)
    hi_trial = None
    if gamma_guess is not None and gamma_guess > 0.0:
        warm = trial_at(min(gamma_guess, expand_cap))
        if meets(warm):
            (hi_trial, hi) = (warm, warm['gamma_rf'])
        else:
            hi = min(max(hi, gamma_guess), expand_cap)
    if hi_trial is None:
        hi_trial = trial_at(hi)
        last_ok = hi_trial
        for _ in range(10):
            if hi_trial is None:
                hi *= 0.5
                if hi < 1e-12:
                    return None
                hi_trial = trial_at(hi)
                if hi_trial is not None:
                    last_ok = hi_trial
                continue
            if meets(hi_trial) or hi >= expand_cap * (1.0 - 1e-12):
                break
            last_ok = hi_trial
            hi = min(hi * 2.0, expand_cap)
            nxt = trial_at(hi)
            if nxt is None:
                hi_trial = last_ok
                break
            hi_trial = nxt
    if hi_trial is None:
        return None
    if not meets(hi_trial):
        return enrich_trial_totals(hi_trial, iplus, iminus, f)
    (lo_ok, hi_ok, best) = (0.0, hi_trial['gamma_rf'], hi_trial)
    for _ in range(n_bisect):
        mid = 0.5 * (lo_ok + hi_ok)
        trial = trial_at(mid)
        if meets(trial):
            (hi_ok, best) = (mid, trial)
        else:
            lo_ok = mid
    return enrich_trial_totals(best, iplus, iminus, f)

def _skip_trace(burn_idx, mirror_idx, f, q_before, iplus, iminus):
    return {'bin_idx': burn_idx, 'mirror_idx': mirror_idx, 'f': f[burn_idx], 'gamma_rf': 0.0, 'n_steps': 0, 't_burn': 0.0, 'q_before': q_before, 'q_after': q_before, 'q_gain': 0.0, 'q_total': q_total(iplus, iminus), 'p_total': p_total(iplus, iminus), 'area_loss': 0.0, 'skipped': True}

def optimize_all_bins(polarization=P, num_bins=NUM_BINS, dt=DT, n_steps=N_STEPS, afp=AFP_ENABLED, afp_efficiency=AFP_EFFICIENCY, afp_bin_range=AFP_BIN_RANGE, afp_center_margin=AFP_CENTER_EXCLUSION_BINS):
    f = np.linspace(-3.0, 3.0, num_bins)
    (_, iplus0, iminus0) = GenerateVectorLineshape(polarization, f)
    iplus = np.asarray(iplus0).copy()
    iminus = np.asarray(iminus0).copy()
    (iplus_unburned, iminus_unburned) = (iplus.copy(), iminus.copy())
    q0 = iplus_unburned - iminus_unburned
    candidates = [i for i in range(num_bins) if q0[i] < 0.0]
    (initial_q, initial_p) = (q_total(iplus, iminus), p_total(iplus, iminus))
    area0 = lineshape_area(iplus, iminus, f)
    gamma_profile = np.zeros(num_bins)
    steps_profile = np.zeros(num_bins, dtype=int)
    trace = []
    applied = skipped = 0
    gamma_guess = None
    pbar = tqdm.tqdm(candidates, desc='gamma-opt bins', unit='bin')
    for burn_idx in pbar:
        mirror_idx = mirror_bin_idx(num_bins, burn_idx)
        q_before = q_at_r_bin(iplus, iminus, burn_idx)
        if q_before >= 0.0:
            skipped += 1
            trace.append(_skip_trace(burn_idx, mirror_idx, f, q_before, iplus, iminus))
            continue
        trial = find_gamma_to_null_q_r(iplus, iminus, burn_idx, f=f, polarization=polarization, dt=dt, n_steps=n_steps, gamma_guess=gamma_guess)
        if trial is None:
            skipped += 1
            trace.append(_skip_trace(burn_idx, mirror_idx, f, q_before, iplus, iminus))
            continue
        (iplus, iminus) = (trial['iplus'], trial['iminus'])
        gamma_profile[burn_idx] = trial['gamma_rf']
        steps_profile[burn_idx] = trial['n_steps']
        gamma_guess = trial['gamma_rf']
        applied += 1
        pbar.set_postfix(idx=burn_idx, R=f'{f[burn_idx]:+.3f}', gamma=f"{trial['gamma_rf']:.4g}", steps=f"{trial['n_steps']}/{n_steps}", Q=f"{trial['q_after']:.2e}", applied=applied, refresh=False)
        trace.append({'bin_idx': burn_idx, 'mirror_idx': mirror_idx, 'f': f[burn_idx], 'gamma_rf': trial['gamma_rf'], 'n_steps': trial['n_steps'], 't_burn': trial['t_burn'], 'q_before': trial['q_before'], 'q_after': trial['q_after'], 'q_gain': trial['q_gain'], 'q_total_gain': trial['q_total_gain'], 'q_total': trial['q_total_after'], 'p_total': p_total(iplus, iminus), 'area_loss': trial['area_loss'], 'skipped': False})
    (iplus_pre_afp, iminus_pre_afp) = (iplus.copy(), iminus.copy())
    afp_subset = []
    if afp:
        subset = None
        if afp_bin_range is None:
            burned = [i for i in range(num_bins) if gamma_profile[i] > 0.0]
            if burned:
                subset = [i for i in burned]
        (iplus, iminus, afp_subset) = apply_afp_sweep(iplus, iminus, bin_range=afp_bin_range, subset_indices=subset, efficiency=afp_efficiency, center_margin=afp_center_margin)
    return {'polarization': polarization, 'dt': dt, 'n_steps': n_steps, 'f': f, 'iplus_unburned': iplus_unburned, 'iminus_unburned': iminus_unburned, 'iplus_pre_afp': iplus_pre_afp, 'iminus_pre_afp': iminus_pre_afp, 'iplus': iplus, 'iminus': iminus, 'gamma_profile': gamma_profile, 'steps_profile': steps_profile, 'q0': q0, 'area0': area0, 'area_final': lineshape_area(iplus, iminus, f), 'area_loss_total': area0 - lineshape_area(iplus, iminus, f), 'initial_q': initial_q, 'final_q': q_total(iplus, iminus), 'initial_p': initial_p, 'final_p': p_total(iplus, iminus), 'q_pre_afp': q_total(iplus_pre_afp, iminus_pre_afp), 'p_pre_afp': p_total(iplus_pre_afp, iminus_pre_afp), 'n_applied': applied, 'n_skipped': skipped, 'n_candidates': len(candidates), 'trace': trace, 'afp_enabled': afp, 'afp_efficiency': afp_efficiency, 'afp_subset': afp_subset, 'afp_center_margin': afp_center_margin}

def save_gamma_profile(result, output_path):
    q_final = result['iplus'] - result['iminus']
    data = np.column_stack([result['f'], result['gamma_profile'], result['steps_profile'], result['q0'], q_final])
    np.savetxt(output_path, data, delimiter=',', header='R,gamma_rf,n_steps,Q_unburned,Q_final', comments='')

def _afp_spans(ax, result):
    subset = result.get('afp_subset') or []
    if not subset:
        return
    f = result['f']
    n = len(f)
    (lo, hi) = (min(subset), max(subset))
    ax.axvspan(f[lo], f[hi], color='gold', alpha=0.18, label='AFP sweep')
    (m_lo, m_hi) = (n - 1 - hi, n - 1 - lo)
    if m_lo != lo or m_hi != hi:
        ax.axvspan(f[m_lo], f[m_hi], color='gold', alpha=0.08, label='AFP mirrors')

def plot_result(result, output_path):
    f = result['f']
    (iplus, iminus) = (result['iplus'], result['iminus'])
    (iplus0, iminus0) = (result['iplus_unburned'], result['iminus_unburned'])
    (fig, axes) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for (ax_data, color, label) in ((iplus0 + iminus0, 'black', '$P_s$'), (iplus0, 'tab:red', '$I_+$'), (iminus0, 'tab:blue', '$I_-$')):
        axes[0].step(f, ax_data, color=color, linestyle='--', alpha=0.55, linewidth=1.0, label=f'{label} (unburned)')
    axes[0].step(f, iplus + iminus, color='black', label='$P_s$')
    axes[0].step(f, iplus, color='tab:red', label='$I_+$')
    axes[0].step(f, iminus, color='tab:blue', label='$I_-$')
    if result.get('afp_enabled') and 'iplus_pre_afp' in result:
        axes[0].step(f, result['iplus_pre_afp'] + result['iminus_pre_afp'], color='gray', alpha=0.5, linewidth=0.9, label='$P_s$ (pre-AFP)')
    _afp_spans(axes[0], result)
    for row in result['trace']:
        if not row['skipped']:
            axes[0].axvline(row['f'], color='green', alpha=0.15, linestyle=':')
    axes[0].set_ylabel('intensity')
    axes[0].legend(loc='upper right', fontsize=7)
    axes[0].grid(True, alpha=0.3)
    axes[1].step(f, iplus0 - iminus0, color='tab:purple', linestyle='--', alpha=0.55, label='$Q$ (unburned)')
    axes[1].step(f, iplus - iminus, color='tab:purple', label='$Q = I_+ - I_-$')
    axes[1].axhline(0.0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('$R$')
    axes[1].set_ylabel('Q profile')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    afp_tag = ' + AFP' if result.get('afp_enabled') else ''
    delta_q = result['final_q'] - result['initial_q']
    fig.suptitle(f"SSRF gamma-opt{afp_tag}  P={result['polarization'] * 100:.0f}%  dt={result['dt']}  n_steps≤{result['n_steps']}  Q: {result['initial_q'] * 100:.2f}% -> {result['final_q'] * 100:.2f}% ({delta_q * 100:+.2f}%)  applied={result['n_applied']}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_gamma_profile(result, output_path):
    (f, gamma, q0) = (result['f'], result['gamma_profile'], result['q0'])
    q_final = result['iplus'] - result['iminus']
    (fig, axes) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(f, q0, label='$Q$ (unburned)')
    axes[0].plot(f, q_final, label='$Q$ (after)')
    if result.get('afp_enabled') and 'iplus_pre_afp' in result:
        axes[0].plot(f, result['iplus_pre_afp'] - result['iminus_pre_afp'], color='gray', alpha=0.7, label='$Q$ (pre-AFP)')
    axes[0].axhline(0.0, color='black', linestyle='--')
    axes[0].set_ylabel('$Q(R)$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(f, gamma, color='tab:orange', label='$\\gamma_{\\mathrm{RF}}(R)$')
    axes[1].fill_between(f, gamma, alpha=0.25, color='tab:orange')
    _afp_spans(axes[1], result)
    axes[1].set_ylabel('$\\gamma_{\\mathrm{RF}}$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    applied = [r for r in result['trace'] if not r['skipped']]
    if applied:
        fv = [r['f'] for r in applied]
        axes[2].stem(fv, [r['q_before'] for r in applied], linefmt='C0-', markerfmt='C0o', basefmt=' ', label='$Q_R$ before')
        axes[2].stem(fv, [r['q_after'] for r in applied], linefmt='C1-', markerfmt='C1o', basefmt=' ', label='$Q_R$ after')
    axes[2].axhline(0.0, color='black', linestyle='--')
    axes[2].set_xlabel('$R$')
    axes[2].set_ylabel('$Q$ at burn bin')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    afp_tag = ', AFP on' if result.get('afp_enabled') else ''
    fig.suptitle(f"Optimized $\\gamma_{{\\mathrm{{RF}}}}(R)$ at fixed $dt={result['dt']}$, $\\leq{result['n_steps']}$ steps{afp_tag}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def plot_q_gains(result, output_path):
    applied = [r for r in result['trace'] if not r['skipped']]
    if not applied:
        return
    fv = [r['f'] for r in applied]
    (fig, axes) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].stem(fv, [r['q_gain'] for r in applied], basefmt=' ')
    axes[0].set_ylabel('$\\Delta Q(R)$')
    axes[0].set_title(f"Per-bin local-Q change (dt={result['dt']}, ≤{result['n_steps']} steps)")
    axes[0].grid(True, alpha=0.3)
    axes[1].stem(fv, [r.get('q_total_gain', 0.0) for r in applied], basefmt=' ', linefmt='C1-', markerfmt='C1o')
    axes[1].set_ylabel('$\\Delta Q_{\\mathrm{total}}$')
    axes[1].grid(True, alpha=0.3)
    axes[2].stem(fv, [r['gamma_rf'] for r in applied], basefmt=' ', linefmt='C2-', markerfmt='C2o')
    axes[2].set_ylabel('$\\gamma_{\\mathrm{RF}}$')
    axes[2].grid(True, alpha=0.3)
    axes[3].stem(fv, [r['n_steps'] for r in applied], basefmt=' ', linefmt='C3-', markerfmt='C3o')
    axes[3].axhline(result['n_steps'], color='gray', linestyle=':', label='$N_{\\mathrm{steps}}$ max')
    axes[3].set_xlabel('burn $R$')
    axes[3].set_ylabel('steps used')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    result = optimize_all_bins()
    stem = 'rate_eqs_test_ssrf_all_bins_gamma_opt'
    plot_result(result, OUT_DIR / f'{stem}_lineshape.png')
    plot_q_gains(result, OUT_DIR / f'{stem}_gains.png')
    plot_gamma_profile(result, OUT_DIR / f'{stem}_burn_profile.png')
    save_gamma_profile(result, OUT_DIR / f'{stem}_gamma_profile.csv')
    q_final = result['iplus'] - result['iminus']
    burned = result['gamma_profile'] > 0.0
    if np.any(burned):
        max_abs_q = np.max(np.abs(q_final[burned]))
        mean_abs_q = np.mean(np.abs(q_final[burned]))
        steps_used = result['steps_profile'][burned]
        (mean_steps, max_steps) = (np.mean(steps_used), np.max(steps_used))
        g_min = np.min(result['gamma_profile'][burned])
    else:
        max_abs_q = mean_abs_q = mean_steps = 'nan'
        (max_steps, g_min) = (0, 0.0)
    print()
    print(f"P0={result['polarization']}  dt={result['dt']}  n_steps≤{result['n_steps']}")
    print(f'dynamics: sameθ d+0={D_SAME_PLUS0}, d0-={D_SAME_0MINUS}; neighbors d+0={D_SPEC_PLUS0}, d0-={D_SPEC_0MINUS}; commit burn/mirror; DNP off')
    if result['afp_enabled']:
        print(f"AFP: efficiency={result['afp_efficiency']}  bins={len(result['afp_subset'])}  center_excl=±{result['afp_center_margin']}  Q pre→post AFP: {result['q_pre_afp']:.6f} -> {result['final_q']:.6f}")
    print(f"RF bins applied: {result['n_applied']}/{result['n_candidates']}  skipped={result['n_skipped']}")
    print(f"P total: {result['initial_p']:.8f} -> {result['final_p']:.8f}")
    print(f"Q total: {result['initial_q']:.8f} -> {result['final_q']:.8f}")
    print(f"Q gain:  {result['final_q'] - result['initial_q']:+.8f}")
    print(f"Area loss: {result['area_loss_total']:.8f}")
    print(f'|Q(R)| on burned bins: mean={mean_abs_q:.3e}  max={max_abs_q:.3e}')
    print(f"steps used: mean={mean_steps:.1f}  max={max_steps}/{result['n_steps']}")
    print(f"gamma_rf(R): min={g_min:.6g}  max={np.max(result['gamma_profile']):.6g}")
    print(f'Saved {stem}_lineshape.png, {stem}_gains.png, {stem}_burn_profile.png, {stem}_gamma_profile.csv')
if __name__ == '__main__':
    main()
