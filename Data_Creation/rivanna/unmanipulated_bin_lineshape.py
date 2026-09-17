"""
Unmanipulated per-bin lineshape NPZs using the frozen Dulya fit (v2 pipeline tags).

Equilibrium spectra always come from ``GenerateDulyaLineshape``.

Examples (from this directory):
  python unmanipulated_bin_lineshape.py
  python unmanipulated_bin_lineshape.py --bin-idx 172
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from burn_selection import positive_polarization_grid
from bin_setup import LINESHape_MODEL, equilibrium_lineshape, generate_unmanipulated_cube, get_shape_params, polarization_grid, print_shape_banner, resolve_bin_idx, shape_meta
from common import F_MAX, F_MIN, NUM_BINS, P_MAX, P_MIN, P_STEP, UNMANIP_TRAIN_DIR, intensity_pq
from pq_calibration import calibrated_pq_fields, load_pq_calibration, validate_stored_per_bin_pq
SOURCE_UNMANIP = 2

def unmanip_bin_path(output_dir, bin_idx):
    return Path(output_dir) / f'unmanip_bin_{bin_idx:04d}.npz'

def verify_unmanip_train_dir(unmanip_dir, *, num_bins=NUM_BINS, p_min=P_MIN, p_max=P_MAX, p_step=P_STEP):
    """Check that per-bin unmanip NPZs cover ``0..num_bins-1`` with a shared P grid."""
    unmanip_dir = Path(unmanip_dir)
    expected_p = positive_polarization_grid(p_min, p_max, p_step)
    missing = []
    p_mismatch = []
    p_values = None
    for bin_idx in range(num_bins):
        path = unmanip_bin_path(unmanip_dir, bin_idx)
        if not path.is_file():
            missing.append(bin_idx)
            continue
        with np.load(path, allow_pickle=False) as data:
            p0 = np.asarray(data['p0'])
        if p_values is None:
            p_values = p0
            if p0.shape != expected_p.shape or not np.allclose(p0, expected_p, atol=1e-05):
                p_mismatch.append(bin_idx)
        elif p0.shape != p_values.shape or not np.allclose(p0, p_values, atol=1e-05):
            p_mismatch.append(bin_idx)
    ok = not missing and (not p_mismatch) and (p_values is not None)
    return {'ok': ok, 'unmanip_dir': str(unmanip_dir), 'num_bins': num_bins, 'n_present': num_bins - len(missing), 'n_missing': len(missing), 'missing_bins': missing, 'p_mismatch_bins': p_mismatch, 'n_p': p_values.size if p_values is not None else 0, 'p_values': np.asarray(p_values) if p_values is not None else np.zeros(0)}

def save_unmanip_bin(bin_idx, *, p_values, ps, iplus, iminus, amp, R, path, p_min, p_max, p_step, num_bins, shape_params, pq_calibration=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = np.asarray(p_values).size
    ps_arr = np.asarray(ps)
    ip_arr = np.asarray(iplus)
    im_arr = np.asarray(iminus)
    (_, q_arr) = intensity_pq(ip_arr, im_arr)
    row_arrays = {'p0': np.asarray(p_values), 'ps': ps_arr, 'iplus': ip_arr, 'iminus': im_arr, 'q': q_arr}
    pq_calibration = pq_calibration or load_pq_calibration(num_bins=num_bins)
    (p_true, q_true) = calibrated_pq_fields(row_arrays, num_bins=num_bins, calibration=pq_calibration)
    validate_stored_per_bin_pq(row_arrays['ps'], row_arrays['q'], row_arrays['p0'], p_true, q_true, calibration=pq_calibration)
    meta = {'bin_idx': bin_idx, 'n_samples': n, 'num_bins': num_bins, 'R': R, 'p_min': p_min, 'p_max': p_max, 'p_step': p_step, 'source': SOURCE_UNMANIP, 'source_codes': {'ssrf': 0, 'afp': 1, 'unmanipulated': SOURCE_UNMANIP}, 'dataset': 'dulya_unmanip_bin_v2', 'fields': 'ps,q=raw I± sums; P,Q=CC-calibrated true polarizations at this bin (unmanipulated equilibrium)', 'pq_calibrated': True, 'pq_target_scope': 'per_bin', 'pq_cc_scale': 'cc_bin', 'pq_post_correct': True, 'pq_cc_bin': pq_calibration['cc_bin'], 'pq_amp': pq_calibration['amp'], **shape_meta(shape_params)}
    tmp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp_path, meta_json=np.asarray(json.dumps(meta)), bin_idx=np.asarray(bin_idx, dtype=np.int32), p0=np.asarray(p_values), step=np.zeros(n, dtype=np.int32), n_steps=np.zeros(n), gamma_rf=np.zeros(n), center_bin=np.full(n, bin_idx, dtype=np.int32), is_mirror=np.zeros(n, dtype=bool), is_neighbor=np.zeros(n, dtype=bool), source=np.full(n, SOURCE_UNMANIP, dtype=np.uint8), ps=ps_arr, iplus=ip_arr, iminus=im_arr, q=np.asarray(q_arr), amp=np.asarray(amp), P=np.asarray(p_true), Q=np.asarray(q_true))
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

def run_one_bin(bin_idx, *, output_dir, num_bins, p_min, p_max, p_step, skip_if_exists, shape_params):
    out = unmanip_bin_path(output_dir, bin_idx)
    if skip_if_exists and out.is_file():
        print(f'Skipping existing {out}', flush=True)
        return out
    p_values = positive_polarization_grid(p_min, p_max, p_step)
    f = np.linspace(F_MIN, F_MAX, num_bins)
    n_p = p_values.size
    ps_col = np.zeros(n_p)
    ip_col = np.zeros(n_p)
    im_col = np.zeros(n_p)
    for (j, p0) in enumerate(p_values):
        if (j + 1) % 10 == 0 or j == 0 or j == n_p - 1:
            print(f'  bin {bin_idx}  P={p0:+.3f} ({j + 1}/{n_p})', flush=True)
        (signal, ip, im) = equilibrium_lineshape(p0, f, shape_params)
        ps_col[j] = np.asarray(signal)[bin_idx]
        ip_col[j] = np.asarray(ip)[bin_idx]
        im_col[j] = np.asarray(im)[bin_idx]
    save_unmanip_bin(bin_idx, p_values=p_values, ps=ps_col, iplus=ip_col, iminus=im_col, amp=np.abs(ps_col), R=f[bin_idx], path=out, p_min=p_min, p_max=p_max, p_step=p_step, num_bins=num_bins, shape_params=shape_params)
    print(f'Wrote {out}', flush=True)
    return out

def run_all_bins(*, output_dir, num_bins, p_min, p_max, p_step, shape_params):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Generating Dulya unmanipulated lineshapes (v2): P=[{p_min},{p_max}] step={p_step}  bins={num_bins}', flush=True)
    cube = generate_unmanipulated_cube(num_bins=num_bins, p_min=p_min, p_max=p_max, p_step=p_step, shape_params=shape_params)
    p_values = cube['p_values']
    n_p = p_values.size
    print(f'Writing {num_bins} unmanipulated per-bin NPZs to {output_dir}', flush=True)
    pq_calibration = load_pq_calibration(num_bins=num_bins)
    for bin_idx in range(num_bins):
        save_unmanip_bin(bin_idx, p_values=p_values, ps=cube['ps'][:, bin_idx], iplus=cube['iplus'][:, bin_idx], iminus=cube['iminus'][:, bin_idx], amp=cube['amp'][:, bin_idx], R=cube['R'][bin_idx], path=unmanip_bin_path(output_dir, bin_idx), p_min=p_min, p_max=p_max, p_step=p_step, num_bins=num_bins, shape_params=shape_params, pq_calibration=pq_calibration)
        if (bin_idx + 1) % 50 == 0 or bin_idx == num_bins - 1:
            print(f'  wrote through bin {bin_idx}', flush=True)
    return {'output_dir': str(output_dir), 'n_bins': num_bins, 'n_p': n_p, 'n_samples_total': num_bins * n_p, 'p_min': p_min, 'p_max': p_max, 'p_step': p_step, 'dataset': 'dulya_unmanip_bin_v2', 'lineshape_model': LINESHape_MODEL}

def build_arg_parser():
    p = argparse.ArgumentParser(description='Dulya-fit unmanipulated per-bin lineshape generator (v2)')
    p.add_argument('--bin-idx', type=int, default=None, help='Single bin (or SLURM_ARRAY_TASK_ID); omit to write all bins')
    p.add_argument('--output-dir', type=Path, default=UNMANIP_TRAIN_DIR)
    p.add_argument('--num-bins', type=int, default=NUM_BINS)
    p.add_argument('--p-min', type=float, default=P_MIN)
    p.add_argument('--p-max', type=float, default=P_MAX)
    p.add_argument('--p-step', type=float, default=P_STEP)
    p.add_argument('--skip-if-exists', action='store_true')
    p.add_argument('--verify-only', action='store_true', help='Verify existing unmanip_bin_XXXX.npz coverage/P-grid and exit')
    return p

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.verify_only:
        report = verify_unmanip_train_dir(args.output_dir, num_bins=args.num_bins, p_min=args.p_min, p_max=args.p_max, p_step=args.p_step)
        print(f"unmanip verify: ok={report['ok']}  present={report['n_present']}/{report['num_bins']}  n_p={report['n_p']}  missing={report['n_missing']}  p_mismatch={len(report['p_mismatch_bins'])}", flush=True)
        if not report['ok']:
            raise SystemExit(1)
        return
    shape = get_shape_params()
    print_shape_banner(shape, num_bins=args.num_bins)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bin_idx = resolve_bin_idx(args.bin_idx, num_bins=args.num_bins)
    if bin_idx is not None:
        run_one_bin(bin_idx, output_dir=args.output_dir, num_bins=args.num_bins, p_min=args.p_min, p_max=args.p_max, p_step=args.p_step, skip_if_exists=args.skip_if_exists, shape_params=shape)
        return
    result = run_all_bins(output_dir=args.output_dir, num_bins=args.num_bins, p_min=args.p_min, p_max=args.p_max, p_step=args.p_step, shape_params=shape)
    print(f"Wrote {result['n_bins']} files ({result['n_p']} P values each) -> {result['output_dir']}", flush=True)
if __name__ == '__main__':
    main()
