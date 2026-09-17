"""
Combine ssRF and AFP trajectory shards into one training NPZ per spectral bin.

Each modality's shards record (ps, iplus, iminus) at the center bin and its
mirror. Samples are routed to the bin they belong to (center → center file,
mirror → mirror file), then concatenated with a ``source`` tag:

  source == 0  → ssRF
  source == 1  → AFP

(Unmanipulated lineshapes are source == 2; see unmanipulated_bin_lineshape.py
and merge_unmanip_into_combined.py.)

Usage:
  python combine_ssrf_afp_train.py \\
      --ssrf-shard-dir ssrf_shards --afp-shard-dir afp_shards \\
      --output-dir combined_train --strict
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
from physics.lineshape.rate_eqs_test import afp_bin_traj as afp_mod
from physics.lineshape.rate_eqs_test import ssrf_bin_traj as ssrf_mod
NUM_BINS = 500
SOURCE_SSRF = 0
SOURCE_AFP = 1
DEFAULT_SSRF_SHARDS = Path(__file__).resolve().parent / 'ssrf_shards'
DEFAULT_AFP_SHARDS = Path(__file__).resolve().parent / 'afp_shards'
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'combined_train'

def combined_bin_path(output_dir, bin_idx):
    return Path(output_dir) / f'train_bin_{bin_idx:04d}.npz'

def _empty_bags(num_bins):
    keys = ('p0', 'step', 'center_bin', 'is_mirror', 'source', 'ps', 'iplus', 'iminus', 'amp')
    return [{k: [] for k in keys} for _ in range(num_bins)]

def _append(bag, *, p0, n, center_bin, is_mirror, source, ps, iplus, iminus, amp):
    bag['p0'].append(np.full(n, p0))
    bag['step'].append(np.arange(n, dtype=np.int32))
    bag['center_bin'].append(np.full(n, center_bin, dtype=np.int32))
    bag['is_mirror'].append(np.full(n, is_mirror, dtype=bool))
    bag['source'].append(np.full(n, source, dtype=np.uint8))
    bag['ps'].append(np.asarray(ps))
    bag['iplus'].append(np.asarray(iplus))
    bag['iminus'].append(np.asarray(iminus))
    bag['amp'].append(np.asarray(amp))

def _finalize(bag):
    if not bag['ps']:
        return {'p0': np.zeros(0), 'step': np.zeros(0, dtype=np.int32), 'center_bin': np.zeros(0, dtype=np.int32), 'is_mirror': np.zeros(0, dtype=bool), 'source': np.zeros(0, dtype=np.uint8), 'ps': np.zeros(0), 'iplus': np.zeros(0), 'iminus': np.zeros(0), 'amp': np.zeros(0)}
    return {k: np.concatenate(v) for (k, v) in bag.items()}

def _amp_pair(shard):
    ps = np.asarray(shard['ps'])
    ps_m = np.asarray(shard['ps_m'])
    amp = np.asarray(shard['amp']) if 'amp' in shard else np.abs(ps)
    amp_m = np.asarray(shard['amp_m']) if 'amp_m' in shard else np.abs(ps_m)
    return (amp, amp_m)

def _ingest_shard(bags, shard, source):
    b = shard['bin_idx']
    m = shard['mirror_idx']
    p_values = np.asarray(shard['p_values'])
    n_steps = np.asarray(shard['n_steps'], dtype=np.int32)
    (amp, amp_m) = _amp_pair(shard)
    for (j, p0) in enumerate(p_values):
        n = n_steps[j]
        if n <= 0:
            continue
        _append(bags[b], p0=p0, n=n, center_bin=b, is_mirror=False, source=source, ps=shard['ps'][j, :n], iplus=shard['iplus'][j, :n], iminus=shard['iminus'][j, :n], amp=amp[j, :n])
        if m != b:
            _append(bags[m], p0=p0, n=n, center_bin=b, is_mirror=True, source=source, ps=shard['ps_m'][j, :n], iplus=shard['iplus_m'][j, :n], iminus=shard['iminus_m'][j, :n], amp=amp_m[j, :n])

def _load_modality(bags, shard_dir, *, num_bins, source, shard_path_fn, load_fn, label, strict):
    missing = []
    shard_dir = Path(shard_dir)
    for bin_idx in range(num_bins):
        path = shard_path_fn(shard_dir, bin_idx)
        if not path.is_file():
            missing.append(bin_idx)
            continue
        print(f'  [{label}] bin {bin_idx}/{num_bins - 1}', flush=True)
        _ingest_shard(bags, load_fn(path), source)
    if missing and strict:
        raise FileNotFoundError(f'Missing {len(missing)} {label} shard(s) under {shard_dir}; first missing bin_idx={missing[0]}')
    if missing:
        print(f'WARNING: missing {len(missing)} {label} shards; continuing', flush=True)
    return missing

def save_combined_bin(bin_idx, arrays, path, *, n_missing_ssrf, n_missing_afp):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = np.asarray(arrays['source'], dtype=np.uint8)
    n_samples = np.asarray(arrays['ps']).size
    n_ssrf = np.count_nonzero(source == SOURCE_SSRF)
    n_afp = np.count_nonzero(source == SOURCE_AFP)
    meta = {'bin_idx': bin_idx, 'n_samples': n_samples, 'n_ssrf': n_ssrf, 'n_afp': n_afp, 'n_missing_ssrf_shards': n_missing_ssrf, 'n_missing_afp_shards': n_missing_afp, 'source_codes': {'ssrf': SOURCE_SSRF, 'afp': SOURCE_AFP}, 'dataset': 'ssrf_afp_train_bin', 'fields': 'ps,iplus,iminus,amp at this bin; center_bin=RF/AFP center; is_mirror; source'}
    tmp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp_path, meta_json=np.asarray(json.dumps(meta)), bin_idx=np.asarray(bin_idx, dtype=np.int32), p0=np.asarray(arrays['p0']), step=np.asarray(arrays['step'], dtype=np.int32), center_bin=np.asarray(arrays['center_bin'], dtype=np.int32), is_mirror=np.asarray(arrays['is_mirror'], dtype=bool), source=source, ps=np.asarray(arrays['ps']), iplus=np.asarray(arrays['iplus']), iminus=np.asarray(arrays['iminus']), amp=np.asarray(arrays['amp']))
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

def combine_ssrf_afp(ssrf_shard_dir, afp_shard_dir, output_dir, *, num_bins=NUM_BINS, strict=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bags = _empty_bags(num_bins)
    print(f'Ingesting ssRF shards from {ssrf_shard_dir}', flush=True)
    missing_ssrf = _load_modality(bags, ssrf_shard_dir, num_bins=num_bins, source=SOURCE_SSRF, shard_path_fn=ssrf_mod.shard_path, load_fn=ssrf_mod.load_shard, label='ssrf', strict=strict)
    print(f'Ingesting AFP shards from {afp_shard_dir}', flush=True)
    missing_afp = _load_modality(bags, afp_shard_dir, num_bins=num_bins, source=SOURCE_AFP, shard_path_fn=afp_mod.shard_path, load_fn=afp_mod.load_shard, label='afp', strict=strict)
    samples_per_bin = np.zeros(num_bins, dtype=np.int64)
    ssrf_per_bin = np.zeros(num_bins, dtype=np.int64)
    afp_per_bin = np.zeros(num_bins, dtype=np.int64)
    print(f'Writing {num_bins} combined per-bin files to {output_dir}', flush=True)
    for bin_idx in range(num_bins):
        arrays = _finalize(bags[bin_idx])
        samples_per_bin[bin_idx] = arrays['ps'].size
        src = arrays['source']
        ssrf_per_bin[bin_idx] = np.count_nonzero(src == SOURCE_SSRF)
        afp_per_bin[bin_idx] = np.count_nonzero(src == SOURCE_AFP)
        save_combined_bin(bin_idx, arrays, combined_bin_path(output_dir, bin_idx), n_missing_ssrf=len(missing_ssrf), n_missing_afp=len(missing_afp))
        if (bin_idx + 1) % 50 == 0 or bin_idx == num_bins - 1:
            print(f'  wrote through bin {bin_idx}', flush=True)
    return {'output_dir': str(output_dir), 'n_samples': samples_per_bin.sum(), 'samples_per_bin': samples_per_bin, 'ssrf_per_bin': ssrf_per_bin, 'afp_per_bin': afp_per_bin, 'n_missing_ssrf': len(missing_ssrf), 'n_missing_afp': len(missing_afp), 'dataset': 'ssrf_afp_train_bin'}

def build_arg_parser():
    p = argparse.ArgumentParser(description='Merge ssRF + AFP shards into per-bin combined training NPZs')
    p.add_argument('--ssrf-shard-dir', type=Path, default=DEFAULT_SSRF_SHARDS)
    p.add_argument('--afp-shard-dir', type=Path, default=DEFAULT_AFP_SHARDS)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--num-bins', type=int, default=NUM_BINS)
    p.add_argument('--strict', action='store_true')
    return p

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = combine_ssrf_afp(args.ssrf_shard_dir, args.afp_shard_dir, args.output_dir, num_bins=args.num_bins, strict=args.strict)
    print(f"Combined {result['n_samples']} samples -> {args.output_dir} ({args.num_bins} bin files; missing_ssrf={result['n_missing_ssrf']} missing_afp={result['n_missing_afp']})", flush=True)
if __name__ == '__main__':
    main()
