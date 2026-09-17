import argparse
from pathlib import Path
import numpy as np
from bin_paths import ssrf_shard_complete, ssrf_shard_path
from shard_store import save_ssrf_shard
from train_bins import organize_ssrf_shards
from bin_setup import equilibrium_lineshape, get_shape_params, resolve_bin_idx, shape_meta, spin1_scale_factors
from common import BURN_R_MAX, BURN_R_MIN, BURN_STEPS_STEP, DIFFUSION_SCALE, DT, F_MAX, F_MIN, GAMMA_RF_MAX, GAMMA_RF_MIN, GAMMA_RF_STEP, MAX_BURN_STEPS, MIN_BURN_STEPS, NUM_BINS, P_MAX, P_MIN, P_STEP, PS_ABS_MIN, RF_GAUSSIAN_FWHM_R, RF_LORENTZIAN_FWHM_R, RF_MODE, RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN, SSRF_SHARD_DIR, SSRF_TRAIN_DIR, burn_steps_grid, gamma_rf_grid, is_burn_bin
from burn_selection import is_manipulation_shard_bin, neighbor_border_offsets, positive_polarization_grid
from model_bridge import build_spin1_model, configure_ssrf_burn, euler_n_sub, full_spectrum_intensities, intensities_at_bins, intensity_at_bin, level_pq, mirror_bin_idx, traj_to_fit_scale
R_MIN = F_MIN
R_MAX = F_MAX
DEFAULT_SHARD_DIR = SSRF_SHARD_DIR
DEFAULT_TRAIN_DIR = SSRF_TRAIN_DIR

def run_one_polarization(bin_idx, polarization, *, num_bins=NUM_BINS, dt=DT, gamma_rf, n_steps, rf_mode=RF_MODE, gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R, lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R, diffusion_scale=DIFFUSION_SCALE, shape_params=None, capture_spectrum=False):
    """Burn exactly ``n_steps`` macro-steps at ``gamma_rf`` from Dulya equilibrium."""
    P = polarization
    n_burn = n_steps
    shape = shape_params if shape_params is not None else get_shape_params()
    f = np.linspace(F_MIN, F_MAX, num_bins)
    (_, ip_fit, im_fit) = equilibrium_lineshape(P, f, shape)
    ip_fit = np.asarray(ip_fit)
    im_fit = np.asarray(im_fit)
    q_eq = ip_fit - im_fit
    (to_spin1, from_spin1) = spin1_scale_factors(P, ip_fit, im_fit)
    iplus0 = ip_fit * to_spin1
    iminus0 = im_fit * to_spin1
    mirror_idx = mirror_bin_idx(num_bins, bin_idx)
    ps0 = ip_fit[bin_idx] + im_fit[bin_idx]
    if abs(ps0) < PS_ABS_MIN:
        return {'polarization': polarization, 'skipped': True, 'n_steps': 0, 'burn_steps': n_burn, 'gamma_rf': gamma_rf, 'ps': np.zeros(0), 'iplus': np.zeros(0), 'iminus': np.zeros(0), 'ps_m': np.zeros(0), 'iplus_m': np.zeros(0), 'iminus_m': np.zeros(0), 'ps0': ps0, 'stop_reason': 'skipped_tiny_ps0', 'rf_mode': str(rf_mode), 'track_lo': False, 'track_hi': False}
    model = build_spin1_model(iplus0, iminus0, polarization=P, num_bins=num_bins, dt=dt, rf_enabled=True, relax_enabled=True, diffusion_scale=diffusion_scale, rf_gaussian_fwhm_R=gaussian_fwhm_R, rf_lorentzian_fwhm_R=lorentzian_fwhm_R)
    used_mode = configure_ssrf_burn(model, bin_idx, gamma_rf, rf_mode=rf_mode, gaussian_fwhm_R=gaussian_fwhm_R, lorentzian_fwhm_R=lorentzian_fwhm_R)
    (p_initial, q_initial) = level_pq(model)
    t_len = n_burn + 1
    ip_spec0 = im_spec0 = ip_spec = im_spec = None
    ps_full = iplus_full = iminus_full = None
    if capture_spectrum:
        (ip_spec0, im_spec0, _) = full_spectrum_intensities(model)
        ps_full = np.empty((t_len, num_bins))
        iplus_full = np.empty((t_len, num_bins))
        iminus_full = np.empty((t_len, num_bins))
    ps = np.empty(t_len)
    iplus = np.empty(t_len)
    iminus = np.empty(t_len)
    ps_m = np.empty(t_len)
    iplus_m = np.empty(t_len)
    iminus_m = np.empty(t_len)
    nb_offsets = neighbor_border_offsets(q_eq, bin_idx, num_bins=num_bins)
    track_lo = -1 in nb_offsets
    track_hi = 1 in nb_offsets
    nb_lo = bin_idx - 1
    nb_hi = bin_idx + 1
    ps_lo = iplus_lo = iminus_lo = None
    ps_hi = iplus_hi = iminus_hi = None
    if track_lo:
        ps_lo = np.empty(t_len)
        iplus_lo = np.empty(t_len)
        iminus_lo = np.empty(t_len)
    if track_hi:
        ps_hi = np.empty(t_len)
        iplus_hi = np.empty(t_len)
        iminus_hi = np.empty(t_len)

    def _record_neighbors(k):
        if track_lo and ps_lo is not None:
            (ip_n, im_n, ps_n) = intensity_at_bin(model, nb_lo)
            (iplus_lo[k], iminus_lo[k], ps_lo[k]) = (ip_n, im_n, ps_n)
        if track_hi and ps_hi is not None:
            (ip_n, im_n, ps_n) = intensity_at_bin(model, nb_hi)
            (iplus_hi[k], iminus_hi[k], ps_hi[k]) = (ip_n, im_n, ps_n)

    def _record_spectrum(k):
        if not capture_spectrum or ps_full is None:
            return
        (ip_s, im_s, ps_s) = full_spectrum_intensities(model)
        iplus_full[k] = ip_s
        iminus_full[k] = im_s
        ps_full[k] = ps_s
    (ip, im, ps0_b, ip_m, im_m, ps_m0) = intensities_at_bins(model, bin_idx, mirror_idx)
    (iplus[0], iminus[0], ps[0]) = (ip, im, ps0_b)
    (iplus_m[0], iminus_m[0], ps_m[0]) = (ip_m, im_m, ps_m0)
    _record_neighbors(0)
    _record_spectrum(0)
    (n_sub, dt_sub) = euler_n_sub(gamma_rf, dt)
    for k in range(1, t_len):
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
        (ip, im, ps_k, ip_m, im_m, ps_mk) = intensities_at_bins(model, bin_idx, mirror_idx)
        (iplus[k], iminus[k], ps[k]) = (ip, im, ps_k)
        (iplus_m[k], iminus_m[k], ps_m[k]) = (ip_m, im_m, ps_mk)
        _record_neighbors(k)
        _record_spectrum(k)
    if capture_spectrum:
        (ip_spec, im_spec, _) = full_spectrum_intensities(model)
    (p_final, q_final) = level_pq(model)
    out = traj_to_fit_scale({'polarization': polarization, 'skipped': False, 'n_steps': t_len, 'burn_steps': n_burn, 'gamma_rf': gamma_rf, 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m, 'ps0': ps0, 'stop_reason': 'fixed_n_steps', 'rf_mode': used_mode, 'ip_spectrum0': ip_spec0, 'im_spectrum0': im_spec0, 'ip_spectrum': ip_spec, 'im_spectrum': im_spec, 'ps_full': ps_full, 'iplus_full': iplus_full, 'iminus_full': iminus_full, 'frequency': f, 'p_initial': p_initial, 'q_initial': q_initial, 'p_final': p_final, 'q_final': q_final, 'center_bin': bin_idx, 'n_burns': 1, 'track_lo': track_lo, 'track_hi': track_hi, 'ps_lo': ps_lo, 'iplus_lo': iplus_lo, 'iminus_lo': iminus_lo, 'ps_hi': ps_hi, 'iplus_hi': iplus_hi, 'iminus_hi': iminus_hi}, from_spin1)
    return out

def run_unmanipulated_polarization(polarization, *, num_bins=NUM_BINS, shape_params=None):
    """Return equilibrium full-spectrum sample (no manipulation)."""
    P = polarization
    shape = shape_params if shape_params is not None else get_shape_params()
    f = np.linspace(F_MIN, F_MAX, num_bins)
    (ps_eq, ip_eq, im_eq) = equilibrium_lineshape(P, f, shape)
    ps_eq = np.asarray(ps_eq)
    ip_eq = np.asarray(ip_eq)
    im_eq = np.asarray(im_eq)
    return {'polarization': P, 'skipped': False, 'n_steps': 1, 'burn_steps': 0, 'gamma_rf': 0.0, 'ps': np.asarray([ps_eq[num_bins // 2]]), 'iplus': np.asarray([ip_eq[num_bins // 2]]), 'iminus': np.asarray([im_eq[num_bins // 2]]), 'ps_m': np.asarray([ps_eq[num_bins // 2]]), 'iplus_m': np.asarray([ip_eq[num_bins // 2]]), 'iminus_m': np.asarray([im_eq[num_bins // 2]]), 'ps0': ps_eq[num_bins // 2], 'stop_reason': 'unmanipulated', 'rf_mode': 'none', 'ps_full': ps_eq.reshape(1, -1), 'iplus_full': ip_eq.reshape(1, -1), 'iminus_full': im_eq.reshape(1, -1), 'frequency': f, 'center_bin': num_bins // 2, 'n_burns': 0}

def _build_combos(p_values, gamma_values, steps_values):
    combos = []
    for p0 in np.asarray(p_values):
        for g in np.asarray(gamma_values):
            for n_burn in np.asarray(steps_values, dtype=np.int32):
                combos.append((p0, g, n_burn))
    return combos

def _run_one_bin_combos(combos, bin_idx, *, t_max, num_bins, dt, rf_mode, gaussian_fwhm_R, lorentzian_fwhm_R, diffusion_scale, capture_spectrum=False):
    """Run ssRF for an explicit combo list (one bin)."""
    bin_idx = bin_idx
    mirror_idx = mirror_bin_idx(num_bins, bin_idx)
    n_samples = len(combos)
    p_out = np.empty(n_samples)
    gamma_out = np.empty(n_samples)
    burn_steps_out = np.empty(n_samples, dtype=np.int32)
    n_steps = np.zeros(n_samples, dtype=np.int32)
    skipped = np.zeros(n_samples, dtype=bool)
    ps = np.full((n_samples, t_max), np.nan)
    iplus = np.full((n_samples, t_max), np.nan)
    iminus = np.full((n_samples, t_max), np.nan)
    ps_m = np.full((n_samples, t_max), np.nan)
    iplus_m = np.full((n_samples, t_max), np.nan)
    iminus_m = np.full((n_samples, t_max), np.nan)
    ps_full = iplus_full = iminus_full = None
    if capture_spectrum:
        ps_full = np.full((n_samples, t_max, num_bins), np.nan)
        iplus_full = np.full((n_samples, t_max, num_bins), np.nan)
        iminus_full = np.full((n_samples, t_max, num_bins), np.nan)
    track_lo = np.zeros(n_samples, dtype=bool)
    track_hi = np.zeros(n_samples, dtype=bool)
    ps_lo = np.full((n_samples, t_max), np.nan)
    iplus_lo = np.full((n_samples, t_max), np.nan)
    iminus_lo = np.full((n_samples, t_max), np.nan)
    ps_hi = np.full((n_samples, t_max), np.nan)
    iplus_hi = np.full((n_samples, t_max), np.nan)
    iminus_hi = np.full((n_samples, t_max), np.nan)
    for (j, (p0, g, n_burn)) in enumerate(combos):
        print(f'  [{j + 1}/{n_samples}] P={p0:+.3f}  gamma={g:.3f}  n_steps={n_burn}', flush=True)
        traj = run_one_polarization(bin_idx, p0, num_bins=num_bins, dt=dt, gamma_rf=g, n_steps=n_burn, rf_mode=rf_mode, gaussian_fwhm_R=gaussian_fwhm_R, lorentzian_fwhm_R=lorentzian_fwhm_R, diffusion_scale=diffusion_scale, capture_spectrum=capture_spectrum)
        p_out[j] = p0
        gamma_out[j] = g
        burn_steps_out[j] = n_burn
        skipped[j] = traj['skipped']
        n = traj['n_steps']
        n_steps[j] = n
        if n <= 0:
            continue
        ps[j, :n] = traj['ps']
        iplus[j, :n] = traj['iplus']
        iminus[j, :n] = traj['iminus']
        ps_m[j, :n] = traj['ps_m']
        iplus_m[j, :n] = traj['iplus_m']
        iminus_m[j, :n] = traj['iminus_m']
        track_lo[j] = traj.get('track_lo', False)
        track_hi[j] = traj.get('track_hi', False)
        if track_lo[j] and traj.get('ps_lo') is not None:
            ps_lo[j, :n] = traj['ps_lo']
            iplus_lo[j, :n] = traj['iplus_lo']
            iminus_lo[j, :n] = traj['iminus_lo']
        if track_hi[j] and traj.get('ps_hi') is not None:
            ps_hi[j, :n] = traj['ps_hi']
            iplus_hi[j, :n] = traj['iplus_hi']
            iminus_hi[j, :n] = traj['iminus_hi']
        if capture_spectrum and ps_full is not None:
            pf = traj.get('ps_full')
            if pf is not None:
                ps_full[j, :n] = np.asarray(pf)[:n]
                iplus_full[j, :n] = np.asarray(traj['iplus_full'])[:n]
                iminus_full[j, :n] = np.asarray(traj['iminus_full'])[:n]
    f = np.linspace(R_MIN, R_MAX, num_bins)
    out = {'bin_idx': bin_idx, 'mirror_idx': mirror_idx, 'R': f[bin_idx], 'num_bins': num_bins, 'dt': dt, 'p_values': p_out, 'gamma_rf': gamma_out, 'burn_steps': burn_steps_out, 'n_steps': n_steps, 'skipped': skipped, 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m, 'track_lo': track_lo, 'track_hi': track_hi, 'ps_lo': ps_lo, 'iplus_lo': iplus_lo, 'iminus_lo': iminus_lo, 'ps_hi': ps_hi, 'iplus_hi': iplus_hi, 'iminus_hi': iminus_hi}
    if capture_spectrum:
        out['ps_full'] = ps_full
        out['iplus_full'] = iplus_full
        out['iminus_full'] = iminus_full
    return out

def run_one_bin(bin_idx, *, p_values, gamma_values=None, steps_values=None, num_bins=NUM_BINS, dt=DT, rf_mode=RF_MODE, gaussian_fwhm_R=RF_GAUSSIAN_FWHM_R, lorentzian_fwhm_R=RF_LORENTZIAN_FWHM_R, diffusion_scale=DIFFUSION_SCALE, capture_spectrum=False):
    """Run ssRF for one burn bin on a Cartesian P × gamma × n_steps grid."""
    bin_idx = bin_idx
    p_values = np.asarray(p_values)
    gamma_values = gamma_rf_grid() if gamma_values is None else np.asarray(gamma_values)
    steps_values = burn_steps_grid() if steps_values is None else np.asarray(steps_values, dtype=np.int32)
    combo_list = _build_combos(p_values, gamma_values, steps_values)
    t_max = np.max(steps_values) + 1
    out = _run_one_bin_combos(combo_list, bin_idx, t_max=t_max, num_bins=num_bins, dt=dt, rf_mode=rf_mode, gaussian_fwhm_R=gaussian_fwhm_R, lorentzian_fwhm_R=lorentzian_fwhm_R, diffusion_scale=diffusion_scale, capture_spectrum=capture_spectrum)
    out['gamma_values'] = np.asarray(gamma_values)
    out['steps_values'] = np.asarray(steps_values, dtype=np.int32)
    out['max_burn_steps'] = np.max(steps_values)
    out['rf_mode'] = str(rf_mode)
    out['gaussian_fwhm_R'] = gaussian_fwhm_R
    out['lorentzian_fwhm_R'] = lorentzian_fwhm_R
    out['diffusion_scale'] = diffusion_scale
    return out

def build_arg_parser():
    p = argparse.ArgumentParser(description='Per-bin ssRF burn worker (writes ssrf_bin_XXXX.npz shards)')
    p.add_argument('--bin-idx', type=int, default=None)
    p.add_argument('--organize', '--combine', dest='organize', action='store_true', help='Organize shards into one training NPZ per bin')
    p.add_argument('--shard-dir', type=Path, default=DEFAULT_SHARD_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument('--num-bins', type=int, default=NUM_BINS)
    p.add_argument('--p-min', type=float, default=P_MIN)
    p.add_argument('--p-max', type=float, default=P_MAX)
    p.add_argument('--p-step', type=float, default=P_STEP)
    p.add_argument('--dt', type=float, default=DT)
    p.add_argument('--gamma-min', type=float, default=GAMMA_RF_MIN)
    p.add_argument('--gamma-max', type=float, default=GAMMA_RF_MAX)
    p.add_argument('--gamma-step', type=float, default=GAMMA_RF_STEP)
    p.add_argument('--steps-min', type=int, default=MIN_BURN_STEPS)
    p.add_argument('--steps-max', type=int, default=MAX_BURN_STEPS)
    p.add_argument('--steps-step', type=int, default=BURN_STEPS_STEP)
    p.add_argument('--rf-mode', choices=(RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN), default=RF_MODE, help='physical_voigt (default) or single_bin RF profile')
    p.add_argument('--gauss-fwhm', type=float, default=RF_GAUSSIAN_FWHM_R)
    p.add_argument('--lorentz-fwhm', type=float, default=RF_LORENTZIAN_FWHM_R)
    p.add_argument('--diffusion-scale', type=float, default=DIFFUSION_SCALE)
    p.add_argument('--skip-if-exists', action='store_true')
    p.add_argument('--strict', action='store_true')
    return p

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.organize:
        organize_ssrf_shards(args.shard_dir, args.output_dir, num_bins=args.num_bins, strict=args.strict)
        return
    bin_idx = resolve_bin_idx(args.bin_idx, num_bins=args.num_bins)
    if bin_idx is None:
        raise SystemExit('Provide --bin-idx <int>, or set SLURM_ARRAY_TASK_ID, or pass --organize')
    out = ssrf_shard_path(args.shard_dir, bin_idx)
    if args.skip_if_exists and ssrf_shard_complete(args.shard_dir, bin_idx):
        return
    if not is_burn_bin(bin_idx):
        print(f'Skipping bin_idx={bin_idx}: outside burn window (R not in ({BURN_R_MIN}, {BURN_R_MAX}))', flush=True)
        return
    shape = get_shape_params()
    p_values = positive_polarization_grid(args.p_min, args.p_max, args.p_step)
    if not is_manipulation_shard_bin(bin_idx):
        print(f'Skipping bin_idx={bin_idx}: outside burn-window manipulation shard list', flush=True)
        return
    result = run_one_bin(bin_idx, p_values=p_values, gamma_values=gamma_rf_grid(args.gamma_min, args.gamma_max, args.gamma_step), steps_values=burn_steps_grid(args.steps_min, args.steps_max, args.steps_step), num_bins=args.num_bins, dt=args.dt, rf_mode=str(args.rf_mode), gaussian_fwhm_R=args.gauss_fwhm, lorentzian_fwhm_R=args.lorentz_fwhm, diffusion_scale=args.diffusion_scale)
    if not np.any(~np.asarray(result['skipped'], dtype=bool)):
        print(f'No samples at bin_idx={bin_idx}: all polarizations skipped; not writing shard', flush=True)
        return
    burn_meta = shape_meta(shape, rf_mode=str(args.rf_mode), gaussian_fwhm_R=args.gauss_fwhm, lorentzian_fwhm_R=args.lorentz_fwhm, diffusion_scale=args.diffusion_scale)
    burn_meta['burn_selection'] = 'all_burn_window_bins'
    burn_meta['polarization_grid'] = 'positive_only'
    save_ssrf_shard(result, out, extra_meta=burn_meta)
if __name__ == '__main__':
    main()
