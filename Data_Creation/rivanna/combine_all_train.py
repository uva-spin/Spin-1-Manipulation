"""
Merge ssRF shards + AFP shards + unmanipulated bins into one NPZ per spectral bin.

Writes ``combined_train_all/train_bin_XXXX.npz`` with:
  p0, step, center_bin, is_mirror, source, ps, iplus, iminus, q, amp, P, Q
  ps,q = raw I± sums; P,Q = CC-calibrated true polarizations at this bin
  source: 0=ssrf, 1=afp, 2=unmanipulated

Reorganizes burn/mirror trajectory shards so each output file contains samples
at that spectral bin (own burn center + partner mirror side), matching
``train_bins.organize_*``.

Usage (from this directory):
  python combine_all_train.py --strict
  python combine_all_train.py \\
      --ssrf-shard-dir data/ssrf_shards \\
      --afp-shard-dir data/afp_shards \\
      --unmanip-dir data/unmanip_train \\
      --output-dir data/combined_train_all --strict
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from bin_paths import afp_shard_path, ssrf_shard_path
from shard_store import load_afp_shard, load_ssrf_shard
from train_bins import _organize_one_bin
from burn_selection import positive_polarization_grid, manipulation_shard_bins
from common import AFP_SHARD_DIR, BURN_BIN_CHOICES, COMBINED_TRAIN_ALL_DIR, NUM_BINS, P_MAX, P_MIN, P_STEP, PHYSICS_MODEL, SOURCE_AFP, SOURCE_SSRF, SOURCE_UNMANIP, SSRF_SHARD_DIR, UNMANIP_TRAIN_DIR, intensity_pq
from pq_calibration import calibrated_pq_fields, load_pq_calibration, validate_stored_per_bin_pq
COMBINED_KEYS = ('p0', 'step', 'n_steps', 'gamma_rf', 'center_bin', 'is_mirror', 'is_neighbor', 'source', 'ps', 'iplus', 'iminus', 'q', 'amp', 'P', 'Q')

def combined_bin_path(output_dir, bin_idx):
    return Path(output_dir) / f'train_bin_{bin_idx:04d}.npz'

def unmanip_bin_path(output_dir, bin_idx):
    return Path(output_dir) / f'unmanip_bin_{bin_idx:04d}.npz'

def load_unmanip_bin(path, *, num_bins=NUM_BINS, pq_calibration=None):
    with np.load(path, allow_pickle=False) as data:
        out = {}
        for key in COMBINED_KEYS:
            if key in ('P', 'Q') and key not in data.files:
                continue
            if key not in data.files:
                if key == 'q' and 'iplus' in data.files and ('iminus' in data.files):
                    (_, q) = intensity_pq(data['iplus'], data['iminus'])
                    out[key] = np.asarray(q)
                    continue
                if key == 'n_steps' and 'step' in data.files:
                    out[key] = np.asarray(data['step'])
                    continue
                if key == 'gamma_rf':
                    out[key] = np.zeros(np.asarray(data['ps']).size)
                    continue
                raise KeyError(f'{path}: missing required field {key!r}')
            out[key] = np.asarray(data[key])
        if 'P' not in out or 'Q' not in out:
            (p_true, q_true) = calibrated_pq_fields(out, num_bins=num_bins, calibration=pq_calibration)
            out['P'] = p_true
            out['Q'] = q_true
        return out

def _ensure_calibrated_pq(arrays, *, num_bins, pq_calibration):
    if 'P' in arrays and 'Q' in arrays:
        return (_as_dtype('P', arrays['P']), _as_dtype('Q', arrays['Q']))
    (p_true, q_true) = calibrated_pq_fields(arrays, num_bins=num_bins, calibration=pq_calibration)
    return (_as_dtype('P', p_true), _as_dtype('Q', q_true))

def _as_dtype(key, arr):
    if key in ('step', 'center_bin'):
        return np.asarray(arr, dtype=np.int32)
    if key == 'n_steps':
        return np.asarray(arr)
    if key == 'is_mirror':
        return np.asarray(arr, dtype=bool)
    if key == 'source':
        return np.asarray(arr, dtype=np.uint8)
    return np.asarray(arr)

def _modality_to_combined(arrays, *, source, ref_key, num_bins=NUM_BINS, pq_calibration=None):
    """Map a modality bag onto the combined schema (adds ``source``)."""
    n = np.asarray(arrays['ps']).size
    center = np.asarray(arrays[ref_key], dtype=np.int32)
    ps = _as_dtype('ps', arrays['ps'])
    iplus = _as_dtype('iplus', arrays['iplus'])
    iminus = _as_dtype('iminus', arrays['iminus'])
    if 'q' in arrays:
        q = _as_dtype('q', arrays['q'])
    else:
        (_, q) = intensity_pq(iplus, iminus)
    (p_true, q_true) = _ensure_calibrated_pq(arrays, num_bins=num_bins, pq_calibration=pq_calibration)
    step = _as_dtype('step', arrays['step'])
    if source == SOURCE_SSRF and 'gamma_rf' in arrays:
        gamma_rf = _as_dtype('gamma_rf', arrays['gamma_rf'])
    else:
        gamma_rf = np.zeros(n)
    n_steps = step.astype(float, copy=False)
    return {'p0': _as_dtype('p0', arrays['p0']), 'step': step, 'n_steps': n_steps, 'gamma_rf': gamma_rf, 'center_bin': center, 'is_mirror': _as_dtype('is_mirror', arrays['is_mirror']), 'is_neighbor': _as_dtype('is_neighbor', arrays.get('is_neighbor', np.zeros(n, dtype=bool))), 'source': np.full(n, source, dtype=np.uint8), 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'q': q, 'amp': _as_dtype('amp', arrays['amp']), 'P': p_true, 'Q': q_true}

def _concat_combined(parts):
    parts = [p for p in parts if np.asarray(p['ps']).size > 0]
    if not parts:
        return {k: _as_dtype(k, np.zeros(0)) for k in COMBINED_KEYS}
    if len(parts) == 1:
        return {k: _as_dtype(k, parts[0][k]) for k in COMBINED_KEYS}
    return {k: np.concatenate([_as_dtype(k, p[k]) for p in parts]) for k in COMBINED_KEYS}

def save_combined_bin(bin_idx, arrays, path, *, pq_calibration=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source = _as_dtype('source', arrays['source'])
    n_samples = np.asarray(arrays['ps']).size
    cal = pq_calibration or load_pq_calibration(num_bins=NUM_BINS)
    validate_stored_per_bin_pq(arrays['ps'], arrays['q'], arrays['p0'], arrays['P'], arrays['Q'], calibration=cal, post_correct=True)
    meta = {'bin_idx': bin_idx, 'n_samples': n_samples, 'n_ssrf': np.count_nonzero(source == SOURCE_SSRF), 'n_afp': np.count_nonzero(source == SOURCE_AFP), 'n_unmanip': np.count_nonzero(source == SOURCE_UNMANIP), 'source_codes': {'ssrf': SOURCE_SSRF, 'afp': SOURCE_AFP, 'unmanipulated': SOURCE_UNMANIP}, 'physics_model': PHYSICS_MODEL, 'dataset': 'ssrf_afp_unmanip_train_bin_v3', 'fields': 'ps,q=raw I± sums; P,Q=CC-calibrated true polarizations at this bin; gamma_rf,n_steps=global ssRF params (0 for unmanip/AFP); center_bin; is_mirror; source (0=ssrf,1=afp,2=unmanip)', 'pq_calibrated': True, 'pq_target_scope': 'per_bin', 'pq_cc_scale': 'cc_bin', 'pq_post_correct': True, 'pq_cc_bin': cal['cc_bin'], 'pq_amp': cal['amp']}
    tmp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp_path, meta_json=np.asarray(json.dumps(meta)), bin_idx=np.asarray(bin_idx, dtype=np.int32), **{k: _as_dtype(k, arrays[k]) for k in COMBINED_KEYS})
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

def combine_all(ssrf_shard_dir, afp_shard_dir, unmanip_dir, output_dir, *, num_bins=NUM_BINS, p_min=P_MIN, p_max=P_MAX, p_step=P_STEP, strict=True, include_unmanip=True):
    ssrf_shard_dir = Path(ssrf_shard_dir)
    afp_shard_dir = Path(afp_shard_dir)
    unmanip_dir = Path(unmanip_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p_values = positive_polarization_grid(p_min, p_max, p_step)
    required_shards = manipulation_shard_bins(num_bins=num_bins)
    missing_ssrf = []
    missing_afp = []
    missing_unmanip = []
    samples_per_bin = np.zeros(num_bins, dtype=np.int64)
    counts = {'ssrf': np.zeros(num_bins, dtype=np.int64), 'afp': np.zeros(num_bins, dtype=np.int64), 'unmanip': np.zeros(num_bins, dtype=np.int64)}
    pq_calibration = load_pq_calibration(num_bins=num_bins)
    for bin_idx in range(num_bins):
        parts = []
        ssrf = _organize_one_bin(bin_idx, num_bins=num_bins, shard_dir=ssrf_shard_dir, shard_path_fn=ssrf_shard_path, load_shard_fn=load_ssrf_shard, ref_key='burn_bin', with_ssrf_params=True)
        if ssrf['ps'].size > 0:
            parts.append(_modality_to_combined(ssrf, source=SOURCE_SSRF, ref_key='burn_bin', num_bins=num_bins, pq_calibration=pq_calibration))
        elif bin_idx in required_shards and (not ssrf_shard_path(ssrf_shard_dir, bin_idx).is_file()):
            missing_ssrf.append(bin_idx)
        afp = _organize_one_bin(bin_idx, num_bins=num_bins, shard_dir=afp_shard_dir, shard_path_fn=afp_shard_path, load_shard_fn=load_afp_shard, ref_key='center_bin', with_ssrf_params=False)
        if afp['ps'].size > 0:
            parts.append(_modality_to_combined(afp, source=SOURCE_AFP, ref_key='center_bin', num_bins=num_bins, pq_calibration=pq_calibration))
        elif bin_idx in required_shards and (not afp_shard_path(afp_shard_dir, bin_idx).is_file()):
            missing_afp.append(bin_idx)
        u_path = unmanip_bin_path(unmanip_dir, bin_idx)
        if include_unmanip and u_path.is_file():
            parts.append(load_unmanip_bin(u_path, num_bins=num_bins, pq_calibration=pq_calibration))
        elif include_unmanip:
            missing_unmanip.append(bin_idx)
        if not parts:
            continue
        merged = _concat_combined(parts)
        source = merged['source']
        samples_per_bin[bin_idx] = merged['ps'].size
        counts['ssrf'][bin_idx] = np.count_nonzero(source == SOURCE_SSRF)
        counts['afp'][bin_idx] = np.count_nonzero(source == SOURCE_AFP)
        counts['unmanip'][bin_idx] = np.count_nonzero(source == SOURCE_UNMANIP)
        save_combined_bin(bin_idx, merged, combined_bin_path(output_dir, bin_idx), pq_calibration=pq_calibration)
        if (bin_idx + 1) % 50 == 0 or bin_idx + 1 == num_bins:
            print(f'  combined {bin_idx + 1}/{num_bins} bins (running samples={samples_per_bin[:bin_idx + 1].sum()})', flush=True)
    if strict and include_unmanip and (missing_ssrf or missing_afp or missing_unmanip):
        raise FileNotFoundError(f'Missing ssrf={len(missing_ssrf)} afp={len(missing_afp)} unmanip={len(missing_unmanip)} (required burn-window shard centers={len(required_shards)}); first ssrf={missing_ssrf[:5]} afp={missing_afp[:5]} unmanip={missing_unmanip[:5]}')
    if missing_ssrf:
        print(f'WARNING: missing ssRF burn-window shards for {len(missing_ssrf)} bins', flush=True)
    if missing_afp:
        print(f'WARNING: missing AFP burn-window shards for {len(missing_afp)} bins', flush=True)
    if missing_unmanip:
        print(f'WARNING: missing unmanip bins for {len(missing_unmanip)} bins', flush=True)
    return {'output_dir': str(output_dir), 'n_samples': samples_per_bin.sum(), 'samples_per_bin': samples_per_bin, 'ssrf_per_bin': counts['ssrf'], 'afp_per_bin': counts['afp'], 'unmanip_per_bin': counts['unmanip'], 'n_missing_ssrf': len(missing_ssrf), 'n_missing_afp': len(missing_afp), 'n_missing_unmanip': len(missing_unmanip), 'n_required_shards': len(required_shards), 'n_burn_bins': BURN_BIN_CHOICES.size, 'dataset': 'ssrf_afp_unmanip_train_bin_v3'}

def build_arg_parser():
    p = argparse.ArgumentParser(description='Merge ssRF + AFP + unmanipulated into per-bin train_bin_XXXX.npz')
    p.add_argument('--ssrf-shard-dir', type=Path, default=SSRF_SHARD_DIR)
    p.add_argument('--afp-shard-dir', type=Path, default=AFP_SHARD_DIR)
    p.add_argument('--unmanip-dir', type=Path, default=UNMANIP_TRAIN_DIR)
    p.add_argument('--output-dir', type=Path, default=COMBINED_TRAIN_ALL_DIR)
    p.add_argument('--num-bins', type=int, default=NUM_BINS)
    p.add_argument('--p-min', type=float, default=P_MIN)
    p.add_argument('--p-max', type=float, default=P_MAX)
    p.add_argument('--p-step', type=float, default=P_STEP)
    p.add_argument('--strict', action='store_true')
    p.add_argument('--no-unmanip', action='store_true', help='Skip unmanipulated equilibrium rows in per-bin train NPZs')
    return p

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    print(f'Combining ssRF ({args.ssrf_shard_dir}) + AFP ({args.afp_shard_dir}) + unmanip ({args.unmanip_dir}) -> {args.output_dir}  P in [{args.p_min}, {args.p_max}] step={args.p_step}', flush=True)
    result = combine_all(args.ssrf_shard_dir, args.afp_shard_dir, args.unmanip_dir, args.output_dir, num_bins=args.num_bins, p_min=args.p_min, p_max=args.p_max, p_step=args.p_step, strict=args.strict, include_unmanip=not args.no_unmanip)
    print(f"Combined {result['n_samples']} samples -> {args.output_dir} (missing_ssrf={result['n_missing_ssrf']} missing_afp={result['n_missing_afp']} missing_unmanip={result['n_missing_unmanip']})", flush=True)
if __name__ == '__main__':
    main()
