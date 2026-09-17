"""
Per-bin ssRF burn trajectories for SLURM array jobs.

Each task burns one R-bin from an unburned lineshape with a smooth Voigt RF profile,
support selected where the envelope exceeds ``PROFILE_REL_THRESHOLD`` of peak
``gamma_rf``. Marches until the mirrored intensity amplitude starts decreasing —
the turnover after maximum semi-saturating RF, when relaxation begins restoring
mirror populations. That decreasing step is discarded and the burn ends. Saves
(ps, iplus, iminus) and amplitude |ps| at both the burn bin and its mirror at
every timestep, for each initial vector polarization.

After all shards exist, ``--organize`` routes samples into one training file per
spectral bin (for 500 independent models): burn-bin observations go to the burn
bin's file, mirror-bin observations go to the mirror bin's file. There is no
single combined NPZ.

Usage:
  python ssrf_bin_traj.py --bin-idx 172
  python ssrf_bin_traj.py --organize --shard-dir ssrf_shards --output-dir ssrf_train
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy.special import wofz
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime.model import Spin1Model, Spin1Params
NUM_BINS = 500
R_MIN = -3.0
R_MAX = 3.0
DT = 0.005
GAMMA_RF = 100.0
SIGMA_BINS = 2.0
VOIGT_GAMMA_BINS = 1.0
HALF_WIDTH = 5
PROFILE_REL_THRESHOLD = 0.05
MAX_STEPS = 5000
P_MIN = -0.7
P_MAX = 0.7
P_STEP = 0.05
PS_ABS_MIN = 1e-12
MIRROR_AMP_EPS = 1e-15
MIRROR_AMP_RTOL = 1e-06
MAX_GDT = 0.05
MAX_NSUB = 20
DEFAULT_SHARD_DIR = Path(__file__).resolve().parent / 'ssrf_shards'
DEFAULT_TRAIN_DIR = Path(__file__).resolve().parent / 'ssrf_train'

def mirror_bin_idx(n_bins, bin_idx):
    return n_bins - 1 - bin_idx

def ssrf_touched_bins(n_bins, subset):
    """Packet/intensity bins ssRF changes: each burn index i also updates mirror(i)."""
    touched = set()
    for i in subset:
        touched.add(i)
        touched.add(mirror_bin_idx(n_bins, i))
    return sorted(touched)

def commit_touched_bins_only(iplus, iminus, iplus_sim, iminus_sim, touched):
    """Keep baseline intensities except on RF-touched bins (burn ∪ mirrors)."""
    out_ip = np.asarray(iplus).copy()
    out_im = np.asarray(iminus).copy()
    ip_sim = np.asarray(iplus_sim)
    im_sim = np.asarray(iminus_sim)
    for k in touched:
        out_ip[k] = ip_sim[k]
        out_im[k] = im_sim[k]
    return (out_ip, out_im)

def polarization_grid(p_min, p_max, p_step):
    return np.arange(p_min, p_max + 1e-12, p_step)

def _voigt_kernel(x, x0, sigma, lorentz_gamma):
    """Discretized Voigt (Faddeeva), same form as ``ssRFMapper._voigt_profile``."""
    sigma = max(sigma, 1e-12)
    x_norm = (np.asarray(x) - x0) / (sigma * np.sqrt(2.0))
    z = x_norm + 1j * (lorentz_gamma / (sigma * np.sqrt(2.0)))
    return np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))

def make_voigt_rf_profile(n_bins, center, gamma_rf, *, sigma=SIGMA_BINS, lorentz_gamma=VOIGT_GAMMA_BINS, half_width=None, rel_threshold=PROFILE_REL_THRESHOLD):
    """
    Compact Voigt RF envelope peaked at ``center``.

    Evaluates a Voigt on the bin grid, then confines it to a local window so
    RF does not bleed across the full lineshape:

    - Optional ``half_width`` hard-caps support to ``center ± half_width``.
    - A raised-cosine taper smooths the window edges (avoids a hard clip).
    - Bins below ``rel_threshold * gamma_rf`` are dropped from support, and the
      returned profile is exactly zero outside the compact support.
    """
    n_bins = n_bins
    c = np.clip(center, 0, n_bins - 1)
    xs = np.arange(n_bins)
    kernel = _voigt_kernel(xs, c, sigma, lorentz_gamma)
    peak = np.max(kernel) if kernel.size else 0.0
    profile = np.zeros(n_bins)
    if peak <= 0.0:
        return (profile, [c])
    profile = gamma_rf * (kernel / peak)
    floor = rel_threshold * abs(gamma_rf)
    support_idx = np.flatnonzero(profile >= floor)
    if support_idx.size == 0:
        profile[:] = 0.0
        profile[c] = gamma_rf
        return (profile, [c])
    if half_width is not None:
        hw = max(0, half_width)
    else:
        hw = max(c - support_idx[0], support_idx[-1] - c)
    lo = max(0, c - hw)
    hi = min(n_bins - 1, c + hw)
    window = np.zeros(n_bins)
    if hw <= 0:
        window[c] = 1.0
        lo = hi = c
    else:
        t = np.abs(np.arange(lo, hi + 1) - c) / hw
        window[lo:hi + 1] = 0.5 * (1.0 + np.cos(np.pi * np.clip(t, 0.0, 1.0)))
    profile *= window
    profile[:lo] = 0.0
    if hi + 1 < n_bins:
        profile[hi + 1:] = 0.0
    support_idx = np.flatnonzero(profile >= floor)
    if support_idx.size == 0:
        profile[:] = 0.0
        profile[c] = gamma_rf
        return (profile, [c])
    compact = np.zeros(n_bins)
    compact[support_idx] = profile[support_idx]
    support = [i for i in support_idx]
    return (compact, support)

def freeze_rf_profile(model, profile):
    """Keep ``params.rf_profile`` fixed; ``ssrf_burn`` always calls ``set_rf_profile``."""
    frozen = np.asarray(profile).copy()
    model.params.rf_profile = frozen.copy()

    def _frozen_set_rf_profile():
        model.params.rf_profile = frozen.copy()
    model.set_rf_profile = _frozen_set_rf_profile
    return _frozen_set_rf_profile

def intensities_at_bins(model, bin_idx, mirror_idx):
    (ip, im, _) = model.physical_intensities()
    iplus = ip[bin_idx]
    iminus = im[bin_idx]
    iplus_m = ip[mirror_idx]
    iminus_m = im[mirror_idx]
    return (iplus, iminus, iplus + iminus, iplus_m, iminus_m, iplus_m + iminus_m)

def mirror_amplitude(ps_m):
    """Scalar amplitude of the mirrored bin intensity (Ps at mirror)."""
    return abs(ps_m)

def mirror_amplitude_decreased(ps_m, ps_m_prev, *, atol=MIRROR_AMP_EPS, rtol=MIRROR_AMP_RTOL):
    """True when mirrored amplitude fell relative to the previous kept step."""
    cur = mirror_amplitude(ps_m)
    prev = mirror_amplitude(ps_m_prev)
    return cur < prev - max(atol, rtol * prev)

def euler_n_sub(gamma_rf, dt):
    """Substep count so |gamma| * dt_sub <= MAX_GDT (stable Euler)."""
    g = abs(gamma_rf)
    dt_f = dt
    if g <= 0.0 or dt_f <= 0.0:
        return (1, dt_f)
    n_sub = min(max(1, np.ceil(g * dt_f / MAX_GDT)), MAX_NSUB)
    return (n_sub, dt_f / n_sub)

def run_one_polarization(bin_idx, polarization, *, num_bins=NUM_BINS, dt=DT, gamma_rf=GAMMA_RF, sigma_bins=SIGMA_BINS, voigt_gamma_bins=VOIGT_GAMMA_BINS, half_width=None, max_steps=MAX_STEPS):
    f = np.linspace(R_MIN, R_MAX, num_bins)
    (_, iplus0, iminus0) = GenerateVectorLineshape(polarization, f)
    iplus0 = np.asarray(iplus0)
    iminus0 = np.asarray(iminus0)
    mirror_idx = mirror_bin_idx(num_bins, bin_idx)
    ps0 = iplus0[bin_idx] + iminus0[bin_idx]
    if abs(ps0) < PS_ABS_MIN:
        return {'polarization': polarization, 'skipped': True, 'n_steps': 0, 'ps': np.zeros(0), 'iplus': np.zeros(0), 'iminus': np.zeros(0), 'ps_m': np.zeros(0), 'iplus_m': np.zeros(0), 'iminus_m': np.zeros(0), 'ps0': ps0, 'stop_reason': 'skipped_tiny_ps0', 'support': []}
    (profile, support) = make_voigt_rf_profile(num_bins, bin_idx, gamma_rf, sigma=sigma_bins, lorentz_gamma=voigt_gamma_bins, half_width=half_width)
    params = Spin1Params(n_bins=num_bins, r_min=R_MIN, r_max=R_MAX, p0=polarization, q0=0.0, p_dnp_sat=polarization, dnp_enabled=False, rf_enabled=True, relax_enabled=True, afp_enabled=False, gamma_rf=gamma_rf, dt=dt, ssrf_subset_indices=[i for i in support], rf_burn_R=f[bin_idx], initial_polarization=polarization)
    model = Spin1Model(params, initial_polarization=polarization)
    model.load_from_physical_intensities(iplus0, iminus0)
    freeze_rf_profile(model, profile)
    touched = ssrf_touched_bins(num_bins, support)
    model._active_idx = np.asarray(touched, dtype=int) if touched else None
    ps_t = []
    ip_t = []
    im_t = []
    ps_m_t = []
    ip_m_t = []
    im_m_t = []
    (ip, im, ps, ip_m, im_m, ps_m) = intensities_at_bins(model, bin_idx, mirror_idx)
    ps_t.append(ps)
    ip_t.append(ip)
    im_t.append(im)
    ps_m_t.append(ps_m)
    ip_m_t.append(ip_m)
    im_m_t.append(im_m)
    (n_sub, dt_sub) = euler_n_sub(gamma_rf, dt)
    steps_done = 0
    stop_reason = 'max_steps'
    while steps_done < max_steps:
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
        steps_done += 1
        (ip, im, ps, ip_m, im_m, ps_m) = intensities_at_bins(model, bin_idx, mirror_idx)
        if mirror_amplitude_decreased(ps_m, ps_m_t[-1]):
            stop_reason = 'mirror_decrease'
            break
        ps_t.append(ps)
        ip_t.append(ip)
        im_t.append(im)
        ps_m_t.append(ps_m)
        ip_m_t.append(ip_m)
        im_m_t.append(im_m)
    return {'polarization': polarization, 'skipped': False, 'n_steps': len(ps_t), 'ps': np.asarray(ps_t), 'iplus': np.asarray(ip_t), 'iminus': np.asarray(im_t), 'ps_m': np.asarray(ps_m_t), 'iplus_m': np.asarray(ip_m_t), 'iminus_m': np.asarray(im_m_t), 'ps0': ps0, 'stop_reason': stop_reason, 'support': support}

def run_one_bin(bin_idx, *, p_values, num_bins=NUM_BINS, dt=DT, gamma_rf=GAMMA_RF, sigma_bins=SIGMA_BINS, voigt_gamma_bins=VOIGT_GAMMA_BINS, half_width=None, max_steps=MAX_STEPS):
    bin_idx = bin_idx
    if bin_idx < 0 or bin_idx >= num_bins:
        raise ValueError(f'bin_idx={bin_idx} out of range for num_bins={num_bins}')
    mirror_idx = mirror_bin_idx(num_bins, bin_idx)
    p_values = np.asarray(p_values)
    n_p = p_values.size
    t_max = max_steps + 1
    n_steps = np.zeros(n_p, dtype=np.int32)
    skipped = np.zeros(n_p, dtype=bool)
    ps = np.full((n_p, t_max), np.nan)
    iplus = np.full((n_p, t_max), np.nan)
    iminus = np.full((n_p, t_max), np.nan)
    ps_m = np.full((n_p, t_max), np.nan)
    iplus_m = np.full((n_p, t_max), np.nan)
    iminus_m = np.full((n_p, t_max), np.nan)
    for (j, p0) in enumerate(p_values):
        print(f'  P={p0:+.3f} ({j + 1}/{n_p})', flush=True)
        traj = run_one_polarization(bin_idx, p0, num_bins=num_bins, dt=dt, gamma_rf=gamma_rf, sigma_bins=sigma_bins, voigt_gamma_bins=voigt_gamma_bins, half_width=half_width, max_steps=max_steps)
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
    f = np.linspace(R_MIN, R_MAX, num_bins)
    (profile, support) = make_voigt_rf_profile(num_bins, bin_idx, gamma_rf, sigma=sigma_bins, lorentz_gamma=voigt_gamma_bins, half_width=half_width)
    support_half_width = 0
    if support:
        support_half_width = max(bin_idx - support[0], support[-1] - bin_idx)
    return {'bin_idx': bin_idx, 'mirror_idx': mirror_idx, 'R': f[bin_idx], 'num_bins': num_bins, 'dt': dt, 'gamma_rf': gamma_rf, 'sigma_bins': sigma_bins, 'voigt_gamma_bins': voigt_gamma_bins, 'half_width': half_width, 'support_half_width': support_half_width, 'max_steps': max_steps, 'p_values': p_values, 'n_steps': n_steps, 'skipped': skipped, 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m}

def shard_path(output_dir, bin_idx):
    return Path(output_dir) / f'ssrf_bin_{bin_idx:04d}.npz'

def save_shard(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {'bin_idx': result['bin_idx'], 'mirror_idx': result['mirror_idx'], 'R': result['R'], 'num_bins': result['num_bins'], 'dt': result['dt'], 'gamma_rf': result['gamma_rf'], 'sigma_bins': result['sigma_bins'], 'voigt_gamma_bins': result.get('voigt_gamma_bins', VOIGT_GAMMA_BINS), 'half_width': result.get('half_width'), 'support_half_width': result.get('support_half_width', 0), 'max_steps': result['max_steps'], 'dataset': 'ssrf_bin_traj'}
    ps = np.asarray(result['ps'])
    ps_m = np.asarray(result['ps_m'])
    tmp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp_path, meta_json=np.asarray(json.dumps(meta)), p_values=np.asarray(result['p_values']), n_steps=np.asarray(result['n_steps'], dtype=np.int32), skipped=np.asarray(result['skipped'], dtype=bool), ps=ps, iplus=np.asarray(result['iplus']), iminus=np.asarray(result['iminus']), amp=np.abs(ps), ps_m=ps_m, iplus_m=np.asarray(result['iplus_m']), iminus_m=np.asarray(result['iminus_m']), amp_m=np.abs(ps_m), bin_idx=np.asarray(result['bin_idx'], dtype=np.int32), mirror_idx=np.asarray(result['mirror_idx'], dtype=np.int32), dt=np.asarray(result['dt']))
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
        return {**meta, 'p_values': np.asarray(data['p_values']), 'n_steps': np.asarray(data['n_steps'], dtype=np.int32), 'skipped': np.asarray(data['skipped'], dtype=bool), 'ps': ps, 'iplus': np.asarray(data['iplus']), 'iminus': np.asarray(data['iminus']), 'amp': amp, 'ps_m': ps_m, 'iplus_m': np.asarray(data['iplus_m']), 'iminus_m': np.asarray(data['iminus_m']), 'amp_m': amp_m}

def train_bin_path(output_dir, bin_idx):
    return Path(output_dir) / f'ssrf_train_bin_{bin_idx:04d}.npz'

def _empty_bin_bags(num_bins):
    keys = ('p0', 'step', 'burn_bin', 'is_mirror', 'ps', 'iplus', 'iminus', 'amp')
    return [{k: [] for k in keys} for _ in range(num_bins)]

def _append_samples(bag, *, p0, n, burn_bin, is_mirror, ps, iplus, iminus, amp):
    bag['p0'].append(np.full(n, p0))
    bag['step'].append(np.arange(n, dtype=np.int32))
    bag['burn_bin'].append(np.full(n, burn_bin, dtype=np.int32))
    bag['is_mirror'].append(np.full(n, is_mirror, dtype=bool))
    bag['ps'].append(np.asarray(ps))
    bag['iplus'].append(np.asarray(iplus))
    bag['iminus'].append(np.asarray(iminus))
    bag['amp'].append(np.asarray(amp))

def _finalize_bag(bag):
    if not bag['ps']:
        return {'p0': np.zeros(0), 'step': np.zeros(0, dtype=np.int32), 'burn_bin': np.zeros(0, dtype=np.int32), 'is_mirror': np.zeros(0, dtype=bool), 'ps': np.zeros(0), 'iplus': np.zeros(0), 'iminus': np.zeros(0), 'amp': np.zeros(0)}
    return {k: np.concatenate(v) for (k, v) in bag.items()}

def save_train_bin(bin_idx, arrays, path, *, n_missing=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = np.asarray(arrays['ps']).size
    meta = {'bin_idx': bin_idx, 'n_samples': n_samples, 'n_missing_shards': n_missing, 'dataset': 'ssrf_train_bin', 'fields': 'ps,iplus,iminus,amp at this bin; burn_bin=RF center; is_mirror'}
    tmp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp_path, meta_json=np.asarray(json.dumps(meta)), bin_idx=np.asarray(bin_idx, dtype=np.int32), p0=np.asarray(arrays['p0']), step=np.asarray(arrays['step'], dtype=np.int32), burn_bin=np.asarray(arrays['burn_bin'], dtype=np.int32), is_mirror=np.asarray(arrays['is_mirror'], dtype=bool), ps=np.asarray(arrays['ps']), iplus=np.asarray(arrays['iplus']), iminus=np.asarray(arrays['iminus']), amp=np.asarray(arrays['amp']))
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

def organize_shards(shard_dir, output_dir, *, num_bins=NUM_BINS, strict=True):
    """
    Route shard samples into one training NPZ per spectral bin.

    Each burn shard records amplitudes at burn bin *and* mirror bin. Those
    observations are filed under their respective bin indices so each of the
    ``num_bins`` models can train independently.
    """
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    bags = _empty_bin_bags(num_bins)
    for burn_bin in range(num_bins):
        path = shard_path(shard_dir, burn_bin)
        if not path.is_file():
            missing.append(burn_bin)
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
            _append_samples(bags[b], p0=p0, n=n, burn_bin=b, is_mirror=False, ps=shard['ps'][j, :n], iplus=shard['iplus'][j, :n], iminus=shard['iminus'][j, :n], amp=shard['amp'][j, :n])
            if m != b:
                _append_samples(bags[m], p0=p0, n=n, burn_bin=b, is_mirror=True, ps=shard['ps_m'][j, :n], iplus=shard['iplus_m'][j, :n], iminus=shard['iminus_m'][j, :n], amp=shard['amp_m'][j, :n])
    if missing and strict:
        raise FileNotFoundError(f'Missing {len(missing)} shard(s) under {shard_dir}; first missing bin_idx={missing[0]}')
    if missing:
        print(f'WARNING: missing {len(missing)} shards; continuing', flush=True)
    samples_per_bin = np.zeros(num_bins, dtype=np.int64)
    for bin_idx in range(num_bins):
        arrays = _finalize_bag(bags[bin_idx])
        samples_per_bin[bin_idx] = arrays['ps'].size
        save_train_bin(bin_idx, arrays, train_bin_path(output_dir, bin_idx), n_missing=len(missing))
    return {'output_dir': str(output_dir), 'samples_per_bin': samples_per_bin, 'n_samples': samples_per_bin.sum(), 'n_missing': len(missing), 'dataset': 'ssrf_train_bin'}

def _resolve_bin_idx(cli_bin_idx):
    if cli_bin_idx is not None:
        return cli_bin_idx
    env_idx = os.environ.get('SLURM_ARRAY_TASK_ID')
    if env_idx is not None and str(env_idx).strip() != '':
        return env_idx
    return None

def build_arg_parser():
    p = argparse.ArgumentParser(description='Per-bin ssRF Voigt burn trajectory worker / per-bin organizer')
    p.add_argument('--bin-idx', type=int, default=None)
    p.add_argument('--organize', '--combine', dest='organize', action='store_true', help='Organize shards into one training NPZ per bin (alias: --combine)')
    p.add_argument('--shard-dir', type=Path, default=DEFAULT_SHARD_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_TRAIN_DIR, help='Directory for per-bin training NPZs (organize mode)')
    p.add_argument('--num-bins', type=int, default=NUM_BINS)
    p.add_argument('--p-min', type=float, default=P_MIN)
    p.add_argument('--p-max', type=float, default=P_MAX)
    p.add_argument('--p-step', type=float, default=P_STEP)
    p.add_argument('--dt', type=float, default=DT)
    p.add_argument('--gamma-rf', type=float, default=GAMMA_RF)
    p.add_argument('--sigma-bins', type=float, default=SIGMA_BINS)
    p.add_argument('--voigt-gamma-bins', type=float, default=VOIGT_GAMMA_BINS)
    p.add_argument('--half-width', type=int, default=None, help='Optional max support cap (±bins); default uses Voigt threshold only')
    p.add_argument('--max-steps', type=int, default=MAX_STEPS)
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
    print(f'bin_idx={bin_idx}  n_P={p_values.size}  P=[{args.p_min},{args.p_max}] step={args.p_step}  dt={args.dt}  gamma={args.gamma_rf}  sigma={args.sigma_bins}  voigt_gamma={args.voigt_gamma_bins}  half_width_cap={args.half_width}  max_steps={args.max_steps}  stop=mirror_turnover', flush=True)
    result = run_one_bin(bin_idx, p_values=p_values, num_bins=args.num_bins, dt=args.dt, gamma_rf=args.gamma_rf, sigma_bins=args.sigma_bins, voigt_gamma_bins=args.voigt_gamma_bins, half_width=args.half_width, max_steps=args.max_steps)
    save_shard(result, out)
    print(f"Wrote {out}  mirror={result['mirror_idx']}  mean_steps={np.mean(result['n_steps']):.1f}", flush=True)
if __name__ == '__main__':
    main()
