import json
import os
from pathlib import Path
import numpy as np
from bin_paths import afp_shard_path, afp_train_bin_path, format_missing_bins_error, missing_shards, ssrf_shard_path, ssrf_train_bin_path
from common import NUM_BINS, PHYSICS_MODEL
from pq_calibration import calibrated_pq_fields, load_pq_calibration, validate_stored_per_bin_pq
from shard_store import load_afp_shard, load_ssrf_shard
_ORGANIZE_CONFIG = {'ssrf': {'ref_key': 'burn_bin', 'dataset': 'ssrf_train_bin_v2', 'fields': 'ps,q=raw I± sums; P,Q=CC-calibrated true polarizations at this bin; burn_bin=RF center; is_mirror; is_neighbor; gamma_rf; burn_steps; step', 'with_ssrf_params': True}, 'afp': {'ref_key': 'center_bin', 'dataset': 'afp_train_bin_v2', 'fields': 'ps,q=raw I± sums; P,Q=CC-calibrated true polarizations at this bin; center_bin=AFP center; is_mirror; is_neighbor', 'with_ssrf_params': False}}

def _mirror_bin_idx(num_bins, bin_idx):
    return num_bins - 1 - bin_idx

def _empty_arrays(ref_key):
    return {'p0': np.zeros(0), 'step': np.zeros(0, dtype=np.int32), 'gamma_rf': np.zeros(0), 'burn_steps': np.zeros(0, dtype=np.int32), ref_key: np.zeros(0, dtype=np.int32), 'is_mirror': np.zeros(0, dtype=bool), 'is_neighbor': np.zeros(0, dtype=bool), 'ps': np.zeros(0), 'iplus': np.zeros(0), 'iminus': np.zeros(0), 'q': np.zeros(0), 'amp': np.zeros(0), 'P': np.zeros(0), 'Q': np.zeros(0)}

def _arrays_from_shard_side(shard, *, ref_key, is_mirror, is_neighbor=False, side_suffix='', track_mask=None, gamma_rf=None, burn_steps=None):
    p_values = np.asarray(shard['p_values'])
    n_steps = np.asarray(shard['n_steps'], dtype=np.int32)
    n_samp = p_values.shape[0]
    if gamma_rf is None:
        gamma_rf = np.full(n_samp, np.nan)
    else:
        gamma_rf = np.asarray(gamma_rf)
    if burn_steps is None:
        burn_steps = np.full(n_samp, -1, dtype=np.int32)
    else:
        burn_steps = np.asarray(burn_steps, dtype=np.int32)
    lengths = np.maximum(n_steps, 0).astype(np.int64)
    if track_mask is not None:
        active = np.asarray(track_mask, dtype=bool) & (lengths > 0)
        total = lengths[active].sum()
    else:
        total = lengths.sum()
    if total <= 0:
        return _empty_arrays(ref_key)
    ref_bin = shard['bin_idx']
    suffix = side_suffix if side_suffix else '_m' if is_mirror else ''
    ps_src = np.asarray(shard[f'ps{suffix}'])
    ip_src = np.asarray(shard[f'iplus{suffix}'])
    im_src = np.asarray(shard[f'iminus{suffix}'])
    q_key = f'q{suffix}'
    q_src = np.asarray(shard[q_key]) if q_key in shard else ip_src - im_src
    if suffix in ('', '_m') and f'amp{suffix}' in shard:
        amp_src = np.asarray(shard[f'amp{suffix}'])
    else:
        amp_src = np.abs(ps_src)
    out = {'p0': np.empty(total), 'step': np.empty(total, dtype=np.int32), 'gamma_rf': np.empty(total), 'burn_steps': np.empty(total, dtype=np.int32), ref_key: np.empty(total, dtype=np.int32), 'is_mirror': np.empty(total, dtype=bool), 'is_neighbor': np.empty(total, dtype=bool), 'ps': np.empty(total), 'iplus': np.empty(total), 'iminus': np.empty(total), 'q': np.empty(total), 'amp': np.empty(total)}
    offset = 0
    for j in range(n_samp):
        n = lengths[j]
        if n <= 0:
            continue
        if track_mask is not None and (not track_mask[j]):
            continue
        sl = slice(offset, offset + n)
        out['p0'][sl] = p_values[j]
        out['step'][sl] = np.arange(n, dtype=np.int32)
        out['gamma_rf'][sl] = gamma_rf[j]
        out['burn_steps'][sl] = burn_steps[j]
        out[ref_key][sl] = ref_bin
        out['is_mirror'][sl] = is_mirror
        out['is_neighbor'][sl] = is_neighbor
        out['ps'][sl] = ps_src[j, :n]
        out['iplus'][sl] = ip_src[j, :n]
        out['iminus'][sl] = im_src[j, :n]
        out['q'][sl] = q_src[j, :n]
        out['amp'][sl] = amp_src[j, :n]
        offset += n
    if offset <= 0:
        return _empty_arrays(ref_key)
    if offset < total:
        for key in out:
            out[key] = out[key][:offset]
    return out

def _concat_arrays(parts, ref_key):
    parts = [p for p in parts if np.asarray(p['ps']).size > 0]
    if not parts:
        return _empty_arrays(ref_key)
    if len(parts) == 1:
        return parts[0]
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}

def _save_train_bin(bin_idx, arrays, path, *, dataset, ref_key, fields, n_missing=0, num_bins=NUM_BINS, pq_calibration=None, pq_post_correct=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (p_true, q_true) = calibrated_pq_fields(arrays, num_bins=num_bins, calibration=pq_calibration, post_correct=pq_post_correct)
    cal = pq_calibration or load_pq_calibration(num_bins=num_bins)
    validate_stored_per_bin_pq(arrays['ps'], arrays['q'], arrays['p0'], p_true, q_true, calibration=cal, post_correct=pq_post_correct)
    meta = {'bin_idx': bin_idx, 'n_samples': np.asarray(arrays['ps']).size, 'n_missing_shards': n_missing, 'physics_model': PHYSICS_MODEL, 'dataset': dataset, 'fields': fields, 'pq_calibrated': True, 'pq_target_scope': 'per_bin', 'pq_cc_scale': 'cc_bin', 'pq_post_correct': pq_post_correct, 'pq_cc_bin': cal['cc_bin'], 'pq_amp': cal['amp']}
    tmp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp_path, meta_json=np.asarray(json.dumps(meta)), bin_idx=np.asarray(bin_idx, dtype=np.int32), p0=np.asarray(arrays['p0']), step=np.asarray(arrays['step'], dtype=np.int32), gamma_rf=np.asarray(arrays['gamma_rf']), burn_steps=np.asarray(arrays['burn_steps'], dtype=np.int32), **{ref_key: np.asarray(arrays[ref_key], dtype=np.int32)}, is_mirror=np.asarray(arrays['is_mirror'], dtype=bool), is_neighbor=np.asarray(arrays['is_neighbor'], dtype=bool), ps=np.asarray(arrays['ps']), iplus=np.asarray(arrays['iplus']), iminus=np.asarray(arrays['iminus']), q=np.asarray(arrays['q']), amp=np.asarray(arrays['amp']), P=np.asarray(p_true), Q=np.asarray(q_true))
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

def _organize_one_bin(out_bin, *, num_bins, shard_dir, shard_path_fn, load_shard_fn, ref_key, with_ssrf_params):
    parts = []
    own_path = shard_path_fn(shard_dir, out_bin)
    if own_path.is_file():
        shard = load_shard_fn(own_path)
        kwargs = {'gamma_rf': shard['gamma_rf'], 'burn_steps': shard['burn_steps']} if with_ssrf_params else {}
        parts.append(_arrays_from_shard_side(shard, ref_key=ref_key, is_mirror=False, **kwargs))
        del shard
    partner = _mirror_bin_idx(num_bins, out_bin)
    if partner != out_bin:
        partner_path = shard_path_fn(shard_dir, partner)
        if partner_path.is_file():
            shard = load_shard_fn(partner_path)
            if shard['mirror_idx'] == out_bin:
                kwargs = {'gamma_rf': shard['gamma_rf'], 'burn_steps': shard['burn_steps']} if with_ssrf_params else {}
                parts.append(_arrays_from_shard_side(shard, ref_key=ref_key, is_mirror=True, **kwargs))
            del shard
    for (burn_center, side_suffix, track_key) in ((out_bin - 1, '_hi', 'track_hi'), (out_bin + 1, '_lo', 'track_lo')):
        if burn_center < 0 or burn_center >= num_bins:
            continue
        center_path = shard_path_fn(shard_dir, burn_center)
        if not center_path.is_file():
            continue
        shard = load_shard_fn(center_path)
        track = shard.get(track_key)
        if track is None or not np.any(np.asarray(track, dtype=bool)):
            del shard
            continue
        kwargs = {}
        if with_ssrf_params:
            kwargs['gamma_rf'] = shard['gamma_rf']
            kwargs['burn_steps'] = shard['burn_steps']
        parts.append(_arrays_from_shard_side(shard, ref_key=ref_key, is_mirror=False, is_neighbor=True, side_suffix=side_suffix, track_mask=np.asarray(track, dtype=bool), **kwargs))
        del shard
    return _concat_arrays(parts, ref_key)

def _organize_shards(kind, shard_dir, output_dir, *, num_bins, strict, shard_path_fn, train_path_fn, load_shard_fn):
    cfg = _ORGANIZE_CONFIG[kind]
    shard_dir = Path(shard_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_key = cfg['ref_key']
    missing = missing_shards(shard_dir, num_bins, shard_path_fn)
    if missing and strict:
        raise FileNotFoundError(format_missing_bins_error(f'{kind.upper()} shard', shard_dir, missing, num_bins=num_bins, path_fn=shard_path_fn))
    if missing:
        print(f'WARNING: missing {len(missing)} shards; continuing', flush=True)
    pq_calibration = load_pq_calibration(num_bins=num_bins)
    samples_per_bin = np.zeros(num_bins, dtype=np.int64)
    for bin_idx in range(num_bins):
        arrays = _organize_one_bin(bin_idx, num_bins=num_bins, shard_dir=shard_dir, shard_path_fn=shard_path_fn, load_shard_fn=load_shard_fn, ref_key=ref_key, with_ssrf_params=cfg['with_ssrf_params'])
        samples_per_bin[bin_idx] = arrays['ps'].size
        _save_train_bin(bin_idx, arrays, train_path_fn(output_dir, bin_idx), dataset=cfg['dataset'], ref_key=ref_key, fields=cfg['fields'], n_missing=len(missing), num_bins=num_bins, pq_calibration=pq_calibration)
        if kind == 'ssrf' and ((bin_idx + 1) % 50 == 0 or bin_idx + 1 == num_bins):
            print(f'  organized {bin_idx + 1}/{num_bins} bins (running samples={samples_per_bin[:bin_idx + 1].sum()})', flush=True)
    return {'output_dir': str(output_dir), 'samples_per_bin': samples_per_bin, 'n_samples': samples_per_bin.sum(), 'n_missing': len(missing), 'dataset': cfg['dataset']}

def organize_ssrf_shards(shard_dir, output_dir, *, num_bins=NUM_BINS, strict=True):
    return _organize_shards('ssrf', shard_dir, output_dir, num_bins=num_bins, strict=strict, shard_path_fn=ssrf_shard_path, train_path_fn=ssrf_train_bin_path, load_shard_fn=load_ssrf_shard)

def organize_afp_shards(shard_dir, output_dir, *, num_bins=NUM_BINS, strict=True):
    return _organize_shards('afp', shard_dir, output_dir, num_bins=num_bins, strict=strict, shard_path_fn=afp_shard_path, train_path_fn=afp_train_bin_path, load_shard_fn=load_afp_shard)
