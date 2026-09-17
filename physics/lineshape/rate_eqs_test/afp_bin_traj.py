"""
Per-bin AFP + relaxation trajectories for SLURM array jobs.

Each task applies an AFP window of 3 centered on one R-bin, then relaxes for
N_RELAX steps. Saves (ps, iplus, iminus) and amplitude |ps| at both the center
bin and its mirror at every timestep (post-AFP + each relax step), for each
initial vector polarization.

After all shards exist, ``--organize`` routes samples into one training file per
spectral bin (for 500 independent models): center-bin observations go to the
center bin's file, mirror-bin observations go to the mirror bin's file. There is
no single combined NPZ.

Usage:
  python afp_bin_traj.py --bin-idx 172
  python afp_bin_traj.py --organize --shard-dir afp_shards --output-dir afp_train
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime.model import Spin1Model, Spin1Params
NUM_BINS = 500
R_MIN = -3.0
R_MAX = 3.0
DT = 0.005
N_RELAX = 5000
AFP_WINDOW = 3
AFP_EFFICIENCY = 1.0
P_MIN = -0.7
P_MAX = 0.7
P_STEP = 0.05
DEFAULT_SHARD_DIR = Path(__file__).resolve().parent / 'afp_shards'
DEFAULT_TRAIN_DIR = Path(__file__).resolve().parent / 'afp_train'

def mirror_bin_idx(n_bins, bin_idx):
    return n_bins - 1 - bin_idx

def afp_touched_bins(n_bins, subset):
    """Intensity/packet bins AFP changes: each sweep index i also updates mirror(i)."""
    touched = set()
    for i in subset:
        touched.add(i)
        touched.add(mirror_bin_idx(n_bins, i))
    return sorted(touched)

def commit_touched_bins_only(iplus, iminus, iplus_sim, iminus_sim, touched):
    """Keep baseline intensities except on AFP-touched bins (sweep ∪ mirrors)."""
    out_ip = np.asarray(iplus).copy()
    out_im = np.asarray(iminus).copy()
    ip_sim = np.asarray(iplus_sim)
    im_sim = np.asarray(iminus_sim)
    for k in touched:
        out_ip[k] = ip_sim[k]
        out_im[k] = im_sim[k]
    return (out_ip, out_im)

def restore_touched_intensity_area(iplus, iminus, touched, area_target):
    """
    Restore Σ(I++I−) to ``area_target`` by common-mode Δ on touched bins only
    (same idea as Spin1Model._renormalize_touched_intensity_area / AFP fix).
    """
    out_ip = np.asarray(iplus).copy()
    out_im = np.asarray(iminus).copy()
    if not touched:
        return (out_ip, out_im)
    n = len(out_ip)
    touched_set = set((k for k in touched))
    unt_idx = [i for i in range(n) if i not in touched_set]
    area_unt = np.sum(out_ip[unt_idx] + out_im[unt_idx]) if unt_idx else 0.0
    ps_touch = out_ip[list(touched)] + out_im[list(touched)]
    area_touch = np.sum(ps_touch)
    missing = area_target - area_unt - area_touch
    if abs(missing) < 1e-15:
        return (out_ip, out_im)
    weights = np.maximum(ps_touch, 0.0)
    wsum = np.sum(weights)
    if wsum > 1e-30:
        for (j, k) in enumerate(touched):
            add = missing * (weights[j] / wsum)
            out_ip[k] += 0.5 * add
            out_im[k] += 0.5 * add
    else:
        add = missing / len(touched)
        for k in touched:
            out_ip[k] += 0.5 * add
            out_im[k] += 0.5 * add
    return (out_ip, out_im)

def polarization_grid(p_min, p_max, p_step):
    return np.arange(p_min, p_max + 1e-12, p_step)

def afp_window_indices(bin_idx, n_bins, window=AFP_WINDOW):
    """Centered window of ``window`` bins; shift at edges to keep width when possible."""
    w = max(1, window)
    half = w // 2
    c = bin_idx
    n = n_bins
    lo = c - half
    hi = c + half
    if lo < 0:
        hi = min(n - 1, hi - lo)
        lo = 0
    if hi >= n:
        lo = max(0, lo - (hi - (n - 1)))
        hi = n - 1
    return list(range(lo, hi + 1))

def intensities_at_bins(model, bin_idx, mirror_idx):
    (ip, im, _) = model.physical_intensities()
    iplus = ip[bin_idx]
    iminus = im[bin_idx]
    iplus_m = ip[mirror_idx]
    iminus_m = im[mirror_idx]
    return (iplus, iminus, iplus + iminus, iplus_m, iminus_m, iplus_m + iminus_m)

def run_one_polarization(bin_idx, polarization, *, num_bins=NUM_BINS, dt=DT, n_relax=N_RELAX, afp_window=AFP_WINDOW, afp_efficiency=AFP_EFFICIENCY):
    f = np.linspace(R_MIN, R_MAX, num_bins)
    (_, iplus0, iminus0) = GenerateVectorLineshape(polarization, f)
    iplus0 = np.asarray(iplus0)
    iminus0 = np.asarray(iminus0)
    mirror_idx = mirror_bin_idx(num_bins, bin_idx)
    subset = afp_window_indices(bin_idx, num_bins, window=afp_window)
    touched = afp_touched_bins(num_bins, subset)
    area0 = np.sum(iplus0 + iminus0)
    params = Spin1Params(n_bins=num_bins, r_min=R_MIN, r_max=R_MAX, p0=polarization, q0=0.0, p_dnp_sat=polarization, dnp_enabled=False, rf_enabled=False, relax_enabled=True, afp_enabled=True, afp_efficiency=afp_efficiency, afp_center_margin=0, afp_preserve_intensity_area=True, afp_subset_indices=[i for i in subset], gamma_rf=0.0, dt=dt, initial_polarization=polarization)
    model = Spin1Model(params, initial_polarization=polarization)
    model.load_from_physical_intensities(iplus0, iminus0)
    model.params.afp_enabled = True
    model.params.afp_preserve_intensity_area = True
    model.params.afp_subset_indices = [i for i in subset]
    model._afp_pending = True
    model.afp_sweep()
    model.params.afp_enabled = False
    model._afp_pending = False
    (ip_sim, im_sim, _) = model.physical_intensities()
    (base_ip, base_im) = commit_touched_bins_only(iplus0, iminus0, ip_sim, im_sim, touched)
    (base_ip, base_im) = restore_touched_intensity_area(base_ip, base_im, touched, area0)
    model.load_from_physical_intensities(base_ip, base_im)
    model._active_idx = np.asarray(touched, dtype=int) if touched else None
    t_len = n_relax + 1
    ps = np.empty(t_len)
    iplus = np.empty(t_len)
    iminus = np.empty(t_len)
    ps_m = np.empty(t_len)
    iplus_m = np.empty(t_len)
    iminus_m = np.empty(t_len)
    (ip, im, ps0, ip_m, im_m, ps_m0) = intensities_at_bins(model, bin_idx, mirror_idx)
    (iplus[0], iminus[0], ps[0]) = (ip, im, ps0)
    (iplus_m[0], iminus_m[0], ps_m[0]) = (ip_m, im_m, ps_m0)
    for k in range(1, t_len):
        model.step_once(dt=dt, rf_on=False, dnp_on=False, copy=False)
        (ip, im, ps_k, ip_m, im_m, ps_mk) = intensities_at_bins(model, bin_idx, mirror_idx)
        (iplus[k], iminus[k], ps[k]) = (ip, im, ps_k)
        (iplus_m[k], iminus_m[k], ps_m[k]) = (ip_m, im_m, ps_mk)
    return {'polarization': polarization, 'skipped': False, 'n_steps': t_len, 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m, 'afp_subset': subset}

def run_one_bin(bin_idx, *, p_values, num_bins=NUM_BINS, dt=DT, n_relax=N_RELAX, afp_window=AFP_WINDOW, afp_efficiency=AFP_EFFICIENCY):
    bin_idx = bin_idx
    if bin_idx < 0 or bin_idx >= num_bins:
        raise ValueError(f'bin_idx={bin_idx} out of range for num_bins={num_bins}')
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
    for (j, p0) in enumerate(p_values):
        print(f'  P={p0:+.3f} ({j + 1}/{n_p})', flush=True)
        traj = run_one_polarization(bin_idx, p0, num_bins=num_bins, dt=dt, n_relax=n_relax, afp_window=afp_window, afp_efficiency=afp_efficiency)
        skipped[j] = traj['skipped']
        n = traj['n_steps']
        n_steps[j] = n
        ps[j, :n] = traj['ps']
        iplus[j, :n] = traj['iplus']
        iminus[j, :n] = traj['iminus']
        ps_m[j, :n] = traj['ps_m']
        iplus_m[j, :n] = traj['iplus_m']
        iminus_m[j, :n] = traj['iminus_m']
    f = np.linspace(R_MIN, R_MAX, num_bins)
    return {'bin_idx': bin_idx, 'mirror_idx': mirror_idx, 'R': f[bin_idx], 'num_bins': num_bins, 'dt': dt, 'n_relax': n_relax, 'afp_window': afp_window, 'afp_efficiency': afp_efficiency, 'afp_subset': np.asarray(subset, dtype=np.int32), 'p_values': p_values, 'n_steps': n_steps, 'skipped': skipped, 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m}

def shard_path(output_dir, bin_idx):
    return Path(output_dir) / f'afp_bin_{bin_idx:04d}.npz'

def save_shard(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {'bin_idx': result['bin_idx'], 'mirror_idx': result['mirror_idx'], 'R': result['R'], 'num_bins': result['num_bins'], 'dt': result['dt'], 'n_relax': result['n_relax'], 'afp_window': result['afp_window'], 'afp_efficiency': result['afp_efficiency'], 'afp_subset': [i for i in np.asarray(result['afp_subset']).tolist()], 'dataset': 'afp_bin_traj'}
    ps = np.asarray(result['ps'])
    ps_m = np.asarray(result['ps_m'])
    tmp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp_path, meta_json=np.asarray(json.dumps(meta)), p_values=np.asarray(result['p_values']), n_steps=np.asarray(result['n_steps'], dtype=np.int32), skipped=np.asarray(result['skipped'], dtype=bool), ps=ps, iplus=np.asarray(result['iplus']), iminus=np.asarray(result['iminus']), amp=np.abs(ps), ps_m=ps_m, iplus_m=np.asarray(result['iplus_m']), iminus_m=np.asarray(result['iminus_m']), amp_m=np.abs(ps_m), afp_subset=np.asarray(result['afp_subset'], dtype=np.int32), bin_idx=np.asarray(result['bin_idx'], dtype=np.int32), mirror_idx=np.asarray(result['mirror_idx'], dtype=np.int32), dt=np.asarray(result['dt']))
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

def load_shard(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta_json']))
        ps = np.asarray(data['ps'])
        ps_m = np.asarray(data['ps_m'])
        amp = np.asarray(data['amp']) if 'amp' in data.files else np.abs(ps)
        amp_m = np.asarray(data['amp_m']) if 'amp_m' in data.files else np.abs(ps_m)
        return {**meta, 'p_values': np.asarray(data['p_values']), 'n_steps': np.asarray(data['n_steps'], dtype=np.int32), 'skipped': np.asarray(data['skipped'], dtype=bool), 'ps': ps, 'iplus': np.asarray(data['iplus']), 'iminus': np.asarray(data['iminus']), 'amp': amp, 'ps_m': ps_m, 'iplus_m': np.asarray(data['iplus_m']), 'iminus_m': np.asarray(data['iminus_m']), 'amp_m': amp_m, 'afp_subset': np.asarray(data['afp_subset'], dtype=np.int32)}

def train_bin_path(output_dir, bin_idx):
    return Path(output_dir) / f'afp_train_bin_{bin_idx:04d}.npz'

def _empty_bin_bags(num_bins):
    keys = ('p0', 'step', 'center_bin', 'is_mirror', 'ps', 'iplus', 'iminus', 'amp')
    return [{k: [] for k in keys} for _ in range(num_bins)]

def _append_samples(bag, *, p0, n, center_bin, is_mirror, ps, iplus, iminus, amp):
    bag['p0'].append(np.full(n, p0))
    bag['step'].append(np.arange(n, dtype=np.int32))
    bag['center_bin'].append(np.full(n, center_bin, dtype=np.int32))
    bag['is_mirror'].append(np.full(n, is_mirror, dtype=bool))
    bag['ps'].append(np.asarray(ps))
    bag['iplus'].append(np.asarray(iplus))
    bag['iminus'].append(np.asarray(iminus))
    bag['amp'].append(np.asarray(amp))

def _finalize_bag(bag):
    if not bag['ps']:
        return {'p0': np.zeros(0), 'step': np.zeros(0, dtype=np.int32), 'center_bin': np.zeros(0, dtype=np.int32), 'is_mirror': np.zeros(0, dtype=bool), 'ps': np.zeros(0), 'iplus': np.zeros(0), 'iminus': np.zeros(0), 'amp': np.zeros(0)}
    return {k: np.concatenate(v) for (k, v) in bag.items()}

def save_train_bin(bin_idx, arrays, path, *, n_missing=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = np.asarray(arrays['ps']).size
    meta = {'bin_idx': bin_idx, 'n_samples': n_samples, 'n_missing_shards': n_missing, 'dataset': 'afp_train_bin', 'fields': 'ps,iplus,iminus,amp at this bin; center_bin=AFP center; is_mirror'}
    tmp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp_path, meta_json=np.asarray(json.dumps(meta)), bin_idx=np.asarray(bin_idx, dtype=np.int32), p0=np.asarray(arrays['p0']), step=np.asarray(arrays['step'], dtype=np.int32), center_bin=np.asarray(arrays['center_bin'], dtype=np.int32), is_mirror=np.asarray(arrays['is_mirror'], dtype=bool), ps=np.asarray(arrays['ps']), iplus=np.asarray(arrays['iplus']), iminus=np.asarray(arrays['iminus']), amp=np.asarray(arrays['amp']))
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

def organize_shards(shard_dir, output_dir, *, num_bins=NUM_BINS, strict=True):
    """
    Route shard samples into one training NPZ per spectral bin.

    Each AFP shard records amplitudes at the window center *and* mirror bin.
    Those observations are filed under their respective bin indices so each of
    the ``num_bins`` models can train independently.
    """
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    bags = _empty_bin_bags(num_bins)
    for center in range(num_bins):
        path = shard_path(shard_dir, center)
        if not path.is_file():
            missing.append(center)
            continue
        shard = load_shard(path)
        b = shard['bin_idx']
        m = shard['mirror_idx']
        p_values = np.asarray(shard['p_values'])
        n_steps = np.asarray(shard['n_steps'], dtype=np.int32)
        for (j, p0) in enumerate(p_values):
            n = n_steps[j]
            if n <= 0:
                continue
            _append_samples(bags[b], p0=p0, n=n, center_bin=b, is_mirror=False, ps=shard['ps'][j, :n], iplus=shard['iplus'][j, :n], iminus=shard['iminus'][j, :n], amp=shard['amp'][j, :n])
            if m != b:
                _append_samples(bags[m], p0=p0, n=n, center_bin=b, is_mirror=True, ps=shard['ps_m'][j, :n], iplus=shard['iplus_m'][j, :n], iminus=shard['iminus_m'][j, :n], amp=shard['amp_m'][j, :n])
    if missing and strict:
        raise FileNotFoundError(f'Missing {len(missing)} shard(s) under {shard_dir}; first missing bin_idx={missing[0]}')
    if missing:
        print(f'WARNING: missing {len(missing)} shards; continuing', flush=True)
    samples_per_bin = np.zeros(num_bins, dtype=np.int64)
    for bin_idx in range(num_bins):
        arrays = _finalize_bag(bags[bin_idx])
        samples_per_bin[bin_idx] = arrays['ps'].size
        save_train_bin(bin_idx, arrays, train_bin_path(output_dir, bin_idx), n_missing=len(missing))
        if (bin_idx + 1) % 50 == 0 or bin_idx == num_bins - 1:
            print(f'  wrote through bin {bin_idx}', flush=True)
    return {'output_dir': str(output_dir), 'samples_per_bin': samples_per_bin, 'n_samples': samples_per_bin.sum(), 'n_missing': len(missing), 'dataset': 'afp_train_bin'}

def _resolve_bin_idx(cli_bin_idx):
    if cli_bin_idx is not None:
        return cli_bin_idx
    env_idx = os.environ.get('SLURM_ARRAY_TASK_ID')
    if env_idx is not None and str(env_idx).strip() != '':
        return env_idx
    return None

def build_arg_parser():
    p = argparse.ArgumentParser(description='Per-bin AFP + relaxation trajectory worker / per-bin organizer')
    p.add_argument('--bin-idx', type=int, default=None)
    p.add_argument('--organize', '--combine', dest='organize', action='store_true', help='Organize shards into one training NPZ per bin (alias: --combine)')
    p.add_argument('--shard-dir', type=Path, default=DEFAULT_SHARD_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_TRAIN_DIR, help='Directory for per-bin training NPZs (organize mode)')
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
        result = organize_shards(args.shard_dir, args.output_dir, num_bins=args.num_bins, strict=args.strict)
        print(f"Organized {result['n_samples']} samples from {args.shard_dir} -> {args.output_dir} ({args.num_bins} bin files; missing={result.get('n_missing', 0)})", flush=True)
        return
    bin_idx = _resolve_bin_idx(args.bin_idx)
    if bin_idx is None:
        raise SystemExit('Provide --bin-idx <int>, or set SLURM_ARRAY_TASK_ID, or pass --organize')
    out = shard_path(args.shard_dir, bin_idx)
    if args.skip_if_exists and out.is_file():
        print(f'Skipping existing shard {out}', flush=True)
        return
    p_values = polarization_grid(args.p_min, args.p_max, args.p_step)
    print(f'bin_idx={bin_idx}  n_P={p_values.size}  P=[{args.p_min},{args.p_max}] step={args.p_step}  dt={args.dt}  n_relax={args.n_relax}  afp_window={args.afp_window}  eff={args.afp_efficiency}', flush=True)
    result = run_one_bin(bin_idx, p_values=p_values, num_bins=args.num_bins, dt=args.dt, n_relax=args.n_relax, afp_window=args.afp_window, afp_efficiency=args.afp_efficiency)
    save_shard(result, out)
    print(f"Wrote {out}  mirror={result['mirror_idx']}  afp_subset={list(result['afp_subset'])}  steps={result['n_steps'][0]}", flush=True)
if __name__ == '__main__':
    main()
