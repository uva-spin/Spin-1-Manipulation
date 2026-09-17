import argparse
from pathlib import Path
import numpy as np
from bin_paths import afp_shard_path
from shard_store import save_afp_shard
from train_bins import organize_afp_shards
from burn_selection import is_manipulation_shard_bin, neighbor_border_offsets, positive_polarization_grid
from bin_setup import equilibrium_lineshape, get_shape_params, print_shape_banner, resolve_bin_idx, shape_meta, spin1_scale_factors
from common import AFP_CENTER_MARGIN, AFP_EFFICIENCY, AFP_N_RELAX, AFP_SHARD_DIR, AFP_TRAIN_DIR, AFP_WINDOW, BURN_R_MAX, BURN_R_MIN, DIFFUSION_SCALE, DT, F_MAX, F_MIN, NUM_BINS, P_MAX, P_MIN, P_STEP, is_burn_bin
from model_bridge import afp_touched_bins, afp_window_indices, build_spin1_model, commit_touched_bins_only, configure_afp_recovery, full_spectrum_intensities, intensities_at_bins, intensity_at_bin, level_pq, mirror_bin_idx, restore_touched_intensity_area
R_MIN = F_MIN
R_MAX = F_MAX
N_RELAX = AFP_N_RELAX
DEFAULT_SHARD_DIR = AFP_SHARD_DIR
DEFAULT_TRAIN_DIR = AFP_TRAIN_DIR

def run_one_polarization(bin_idx, polarization, *, num_bins=NUM_BINS, dt=DT, n_relax=N_RELAX, afp_window=AFP_WINDOW, afp_efficiency=AFP_EFFICIENCY, diffusion_scale=DIFFUSION_SCALE, shape_params=None, capture_spectrum=False):
    P = polarization
    shape = shape_params if shape_params is not None else get_shape_params()
    f = np.linspace(F_MIN, F_MAX, num_bins)
    (_, ip_fit, im_fit) = equilibrium_lineshape(P, f, shape)
    ip_fit = np.asarray(ip_fit)
    im_fit = np.asarray(im_fit)
    q_eq = ip_fit - im_fit
    subset = afp_window_indices(bin_idx, num_bins, window=afp_window)
    (to_spin1, from_spin1) = spin1_scale_factors(P, ip_fit, im_fit)
    iplus0 = ip_fit * to_spin1
    iminus0 = im_fit * to_spin1
    mirror_idx = mirror_bin_idx(num_bins, bin_idx)
    touched = afp_touched_bins(num_bins, subset)
    area0 = np.sum(iplus0 + iminus0)
    model = build_spin1_model(iplus0, iminus0, polarization=P, num_bins=num_bins, dt=dt, rf_enabled=False, relax_enabled=True, diffusion_scale=diffusion_scale)
    model.params.afp_enabled = True
    model.params.afp_efficiency = afp_efficiency
    model.params.afp_center_margin = AFP_CENTER_MARGIN
    model.params.afp_preserve_intensity_area = True
    model.params.afp_subset_indices = [i for i in subset]
    model._afp_pending = True
    (p_initial, q_initial) = level_pq(model)
    ip_spec0 = im_spec0 = None
    if capture_spectrum:
        (ip_spec0, im_spec0, _) = full_spectrum_intensities(model)
    model.afp_sweep()
    model.params.afp_enabled = False
    model._afp_pending = False
    (ip_sim, im_sim, _) = model.physical_intensities()
    (base_ip, base_im) = commit_touched_bins_only(iplus0, iminus0, ip_sim, im_sim, touched)
    (base_ip, base_im) = restore_touched_intensity_area(base_ip, base_im, touched, area0)
    model.load_from_physical_intensities(base_ip, base_im)
    configure_afp_recovery(model)
    t_len = n_relax + 1
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
    ps_full = iplus_full = iminus_full = None
    if capture_spectrum:
        ps_full = np.empty((t_len, num_bins))
        iplus_full = np.empty((t_len, num_bins))
        iminus_full = np.empty((t_len, num_bins))

    def _record_step(k):
        (ip, im, ps0, ip_m, im_m, ps_m0) = intensities_at_bins(model, bin_idx, mirror_idx)
        (iplus[k], iminus[k], ps[k]) = (ip, im, ps0)
        (iplus_m[k], iminus_m[k], ps_m[k]) = (ip_m, im_m, ps_m0)
        if track_lo and ps_lo is not None:
            (ip_n, im_n, ps_n) = intensity_at_bin(model, nb_lo)
            (iplus_lo[k], iminus_lo[k], ps_lo[k]) = (ip_n, im_n, ps_n)
        if track_hi and ps_hi is not None:
            (ip_n, im_n, ps_n) = intensity_at_bin(model, nb_hi)
            (iplus_hi[k], iminus_hi[k], ps_hi[k]) = (ip_n, im_n, ps_n)
        if capture_spectrum and ps_full is not None:
            (ip_s, im_s, ps_s) = full_spectrum_intensities(model)
            iplus_full[k] = ip_s
            iminus_full[k] = im_s
            ps_full[k] = ps_s
    _record_step(0)
    for k in range(1, t_len):
        model.step_once(dt=dt, rf_on=False, dnp_on=False, copy=False)
        _record_step(k)
    scale = from_spin1
    ip_spec = im_spec = None
    if capture_spectrum:
        ip_spec0 = None if ip_spec0 is None else ip_spec0 * scale
        im_spec0 = None if im_spec0 is None else im_spec0 * scale
        if iplus_full is not None:
            iplus_full *= scale
            iminus_full *= scale
            ps_full *= scale
            ip_spec = iplus_full[-1].copy()
            im_spec = iminus_full[-1].copy()
        else:
            (ip_end, im_end, _) = full_spectrum_intensities(model)
            ip_spec = ip_end * scale
            im_spec = im_end * scale
    (p_final, q_final) = level_pq(model)
    out = {'polarization': polarization, 'skipped': False, 'n_steps': t_len, 'ps': ps * scale, 'iplus': iplus * scale, 'iminus': iminus * scale, 'ps_m': ps_m * scale, 'iplus_m': iplus_m * scale, 'iminus_m': iminus_m * scale, 'afp_subset': subset, 'ip_spectrum0': ip_spec0, 'im_spectrum0': im_spec0, 'ip_spectrum': ip_spec, 'im_spectrum': im_spec, 'ps_full': ps_full, 'iplus_full': iplus_full, 'iminus_full': iminus_full, 'frequency': f, 'diffusion_scale': model.params.diffusion_scale, 'p_initial': p_initial, 'q_initial': q_initial, 'p_final': p_final, 'q_final': q_final, 'center_bin': bin_idx, 'track_lo': track_lo, 'track_hi': track_hi}
    if track_lo and ps_lo is not None:
        out['ps_lo'] = ps_lo * scale
        out['iplus_lo'] = iplus_lo * scale
        out['iminus_lo'] = iminus_lo * scale
    if track_hi and ps_hi is not None:
        out['ps_hi'] = ps_hi * scale
        out['iplus_hi'] = iplus_hi * scale
        out['iminus_hi'] = iminus_hi * scale
    return out

def run_one_bin(bin_idx, *, p_values, num_bins=NUM_BINS, dt=DT, n_relax=N_RELAX, afp_window=AFP_WINDOW, afp_efficiency=AFP_EFFICIENCY, capture_spectrum=False, step_subsample=1):
    bin_idx = bin_idx
    mirror_idx = mirror_bin_idx(num_bins, bin_idx)
    p_values = np.asarray(p_values)
    n_p = p_values.size
    t_len = n_relax + 1
    subset = afp_window_indices(bin_idx, num_bins, window=afp_window)
    n_steps = np.full(n_p, t_len, dtype=np.int32)
    skipped = np.zeros(n_p, dtype=bool)
    ps = np.full((n_p, t_len), np.nan)
    iplus = np.full((n_p, t_len), np.nan)
    iminus = np.full((n_p, t_len), np.nan)
    ps_m = np.full((n_p, t_len), np.nan)
    iplus_m = np.full((n_p, t_len), np.nan)
    iminus_m = np.full((n_p, t_len), np.nan)
    track_lo = np.zeros(n_p, dtype=bool)
    track_hi = np.zeros(n_p, dtype=bool)
    ps_lo = np.full((n_p, t_len), np.nan)
    iplus_lo = np.full((n_p, t_len), np.nan)
    iminus_lo = np.full((n_p, t_len), np.nan)
    ps_hi = np.full((n_p, t_len), np.nan)
    iplus_hi = np.full((n_p, t_len), np.nan)
    iminus_hi = np.full((n_p, t_len), np.nan)
    ps_full = iplus_full = iminus_full = None
    if capture_spectrum:
        ps_full = np.full((n_p, t_len, num_bins), np.nan)
        iplus_full = np.full((n_p, t_len, num_bins), np.nan)
        iminus_full = np.full((n_p, t_len, num_bins), np.nan)
    for (j, p0) in enumerate(p_values):
        print(f'  P={p0:+.3f} ({j + 1}/{n_p})', flush=True)
        traj = run_one_polarization(bin_idx, p0, num_bins=num_bins, dt=dt, n_relax=n_relax, afp_window=afp_window, afp_efficiency=afp_efficiency, capture_spectrum=capture_spectrum)
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
        if capture_spectrum and ps_full is not None and (traj.get('ps_full') is not None):
            ps_full[j, :n] = np.asarray(traj['ps_full'])[:n]
            iplus_full[j, :n] = np.asarray(traj['iplus_full'])[:n]
            iminus_full[j, :n] = np.asarray(traj['iminus_full'])[:n]
    f = np.linspace(R_MIN, R_MAX, num_bins)
    out = {'bin_idx': bin_idx, 'mirror_idx': mirror_idx, 'R': f[bin_idx], 'num_bins': num_bins, 'dt': dt, 'n_relax': n_relax, 'afp_window': afp_window, 'afp_efficiency': afp_efficiency, 'afp_subset': np.asarray(subset, dtype=np.int32), 'p_values': p_values, 'n_steps': n_steps, 'skipped': skipped, 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m, 'track_lo': track_lo, 'track_hi': track_hi, 'ps_lo': ps_lo, 'iplus_lo': iplus_lo, 'iminus_lo': iminus_lo, 'ps_hi': ps_hi, 'iplus_hi': iplus_hi, 'iminus_hi': iminus_hi}
    if capture_spectrum:
        out['ps_full'] = ps_full
        out['iplus_full'] = iplus_full
        out['iminus_full'] = iminus_full
        out['step_subsample'] = step_subsample
    return out

def build_arg_parser():
    p = argparse.ArgumentParser(description='Per-bin AFP + relaxation worker (writes afp_bin_XXXX.npz shards)')
    p.add_argument('--bin-idx', type=int, default=None)
    p.add_argument('--organize', '--combine', dest='organize', action='store_true')
    p.add_argument('--shard-dir', type=Path, default=DEFAULT_SHARD_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument('--num-bins', type=int, default=NUM_BINS)
    p.add_argument('--p-min', type=float, default=P_MIN)
    p.add_argument('--p-max', type=float, default=P_MAX)
    p.add_argument('--p-step', type=float, default=P_STEP)
    p.add_argument('--dt', type=float, default=DT)
    p.add_argument('--n-relax', type=int, default=N_RELAX)
    p.add_argument('--afp-window', type=int, default=AFP_WINDOW)
    p.add_argument('--afp-efficiency', type=float, default=AFP_EFFICIENCY)
    p.add_argument('--skip-if-exists', action='store_true')
    p.add_argument('--strict', action='store_true')
    return p

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.organize:
        result = organize_afp_shards(args.shard_dir, args.output_dir, num_bins=args.num_bins, strict=args.strict)
        print(f"Organized {result['n_samples']} samples from {args.shard_dir} -> {args.output_dir} ({args.num_bins} bin files; missing={result.get('n_missing', 0)})", flush=True)
        return
    bin_idx = resolve_bin_idx(args.bin_idx, num_bins=args.num_bins)
    if bin_idx is None:
        raise SystemExit('Provide --bin-idx <int>, or set SLURM_ARRAY_TASK_ID, or pass --organize')
    out = afp_shard_path(args.shard_dir, bin_idx)
    if args.skip_if_exists and out.is_file():
        print(f'Skipping existing shard {out}', flush=True)
        return
    if not is_burn_bin(bin_idx):
        print(f'Skipping bin_idx={bin_idx}: outside burn window (R not in ({BURN_R_MIN}, {BURN_R_MAX}))', flush=True)
        return
    shape = get_shape_params()
    print_shape_banner(shape, num_bins=args.num_bins)
    p_values = positive_polarization_grid(args.p_min, args.p_max, args.p_step)
    if not is_manipulation_shard_bin(bin_idx):
        print(f'Skipping bin_idx={bin_idx}: outside burn-window manipulation shard list', flush=True)
        return
    print(f'bin_idx={bin_idx}  n_P={p_values.size}  P=[{args.p_min},{args.p_max}] step={args.p_step}  dt={args.dt}  n_relax={args.n_relax}  afp_window={args.afp_window}  eff={args.afp_efficiency}', flush=True)
    result = run_one_bin(bin_idx, p_values=p_values, num_bins=args.num_bins, dt=args.dt, n_relax=args.n_relax, afp_window=args.afp_window, afp_efficiency=args.afp_efficiency)
    save_afp_shard(result, out, extra_meta=shape_meta(shape))
    print(f"Wrote {out}  mirror={result['mirror_idx']}  afp_subset={list(result['afp_subset'])}  steps={result['n_steps'][0]}", flush=True)
if __name__ == '__main__':
    main()
