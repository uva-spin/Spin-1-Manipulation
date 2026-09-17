import json
import os
from pathlib import Path
import numpy as np
from bin_paths import list_ssrf_shard_paths
from common import PHYSICS_MODEL, RF_MODE, STORE_DTYPE, intensity_pq
_SSRF_SHARD_ARRAY_KEYS = ('p_values', 'gamma_rf', 'burn_steps', 'n_steps', 'skipped', 'ps', 'iplus', 'iminus', 'q', 'amp', 'ps_m', 'iplus_m', 'iminus_m', 'q_m', 'amp_m', 'track_lo', 'track_hi', 'ps_lo', 'iplus_lo', 'iminus_lo', 'q_lo', 'ps_hi', 'iplus_hi', 'iminus_hi', 'q_hi')
_TRAJ_STACK_META_KEYS = ('p_values', 'gamma_rf', 'burn_steps', 'n_steps', 'skipped')

def _save_npz_atomic(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.stem}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise

def _gamma_burn_from_shard(data, meta):
    n_samp = np.asarray(data['p_values']).shape[0]
    if 'gamma_rf' in data.files:
        gamma_rf = np.asarray(data['gamma_rf'])
    else:
        gamma_rf = np.full(n_samp, meta.get('gamma_rf', np.nan))
    if 'burn_steps' in data.files:
        burn_steps = np.asarray(data['burn_steps'], dtype=np.int32)
    else:
        n_steps = np.asarray(data['n_steps'], dtype=np.int32)
        burn_steps = np.maximum(n_steps - 1, 0).astype(np.int32)
    return (gamma_rf, burn_steps)

def load_ssrf_shard_meta(path):
    with np.load(path, allow_pickle=False) as data:
        (gamma_rf, burn_steps) = _gamma_burn_from_shard(data, {})
        return {'p_values': np.asarray(data['p_values']), 'gamma_rf': gamma_rf, 'burn_steps': burn_steps, 'n_steps': np.asarray(data['n_steps'], dtype=np.int32), 'skipped': np.asarray(data['skipped'], dtype=bool)}

def load_afp_shard_meta(path):
    with np.load(path, allow_pickle=False) as data:
        return {'p_values': np.asarray(data['p_values']), 'n_steps': np.asarray(data['n_steps'], dtype=np.int32), 'skipped': np.asarray(data['skipped'], dtype=bool)}

def _concat_shard_parts(parts, keys):
    if not parts:
        raise ValueError('parts must be non-empty')
    if len(parts) == 1:
        return parts[0]
    merged = dict(parts[0])
    for key in keys:
        if key in parts[0]:
            merged[key] = np.concatenate([p[key] for p in parts], axis=0)
    return merged

def load_ssrf_shard_meta_any(shard_dir, bin_idx):
    paths = list_ssrf_shard_paths(shard_dir, bin_idx)
    if not paths:
        return None
    return _concat_shard_parts([load_ssrf_shard_meta(p) for p in paths], _TRAJ_STACK_META_KEYS)

def load_ssrf_shard(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta_json']))
        (gamma_rf, burn_steps) = _gamma_burn_from_shard(data, meta)
        ps = np.asarray(data['ps'])
        ps_m = np.asarray(data['ps_m'])
        amp = np.asarray(data['amp']) if 'amp' in data.files else np.abs(ps)
        amp_m = np.asarray(data['amp_m']) if 'amp_m' in data.files else np.abs(ps_m)
        iplus = np.asarray(data['iplus'])
        iminus = np.asarray(data['iminus'])
        iplus_m = np.asarray(data['iplus_m'])
        iminus_m = np.asarray(data['iminus_m'])
        q = np.asarray(data['q']) if 'q' in data.files else iplus - iminus
        q_m = np.asarray(data['q_m']) if 'q_m' in data.files else iplus_m - iminus_m
        out = {**meta, 'p_values': np.asarray(data['p_values']), 'gamma_rf': gamma_rf, 'burn_steps': burn_steps, 'n_steps': np.asarray(data['n_steps'], dtype=np.int32), 'skipped': np.asarray(data['skipped'], dtype=bool), 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'q': q, 'amp': amp, 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m, 'q_m': q_m, 'amp_m': amp_m}
        for key in ('track_lo', 'track_hi', 'ps_lo', 'iplus_lo', 'iminus_lo', 'q_lo', 'ps_hi', 'iplus_hi', 'iminus_hi', 'q_hi'):
            if key in data.files:
                out[key] = np.asarray(data[key])
        return out

def load_ssrf_shard_any(shard_dir, bin_idx):
    paths = list_ssrf_shard_paths(shard_dir, bin_idx)
    if not paths:
        return None
    return _concat_shard_parts([load_ssrf_shard(p) for p in paths], _SSRF_SHARD_ARRAY_KEYS)

def load_afp_shard(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta_json']))
        ps = np.asarray(data['ps'])
        ps_m = np.asarray(data['ps_m'])
        amp = np.asarray(data['amp']) if 'amp' in data.files else np.abs(ps)
        amp_m = np.asarray(data['amp_m']) if 'amp_m' in data.files else np.abs(ps_m)
        iplus = np.asarray(data['iplus'])
        iminus = np.asarray(data['iminus'])
        iplus_m = np.asarray(data['iplus_m'])
        iminus_m = np.asarray(data['iminus_m'])
        q = np.asarray(data['q']) if 'q' in data.files else iplus - iminus
        q_m = np.asarray(data['q_m']) if 'q_m' in data.files else iplus_m - iminus_m
        out = {**meta, 'p_values': np.asarray(data['p_values']), 'n_steps': np.asarray(data['n_steps'], dtype=np.int32), 'skipped': np.asarray(data['skipped'], dtype=bool), 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'q': q, 'amp': amp, 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m, 'q_m': q_m, 'amp_m': amp_m, 'afp_subset': np.asarray(data['afp_subset'], dtype=np.int32)}
        if 'track_lo' in data.files:
            out['track_lo'] = np.asarray(data['track_lo'], dtype=bool)
        if 'track_hi' in data.files:
            out['track_hi'] = np.asarray(data['track_hi'], dtype=bool)
        for side in ('lo', 'hi'):
            ip_key = f'iplus_{side}'
            if ip_key in data.files:
                ip_side = np.asarray(data[ip_key])
                im_side = np.asarray(data[f'iminus_{side}'])
                out[f'ps_{side}'] = np.asarray(data[f'ps_{side}'])
                out[ip_key] = ip_side
                out[f'iminus_{side}'] = im_side
                out[f'q_{side}'] = np.asarray(data[f'q_{side}']) if f'q_{side}' in data.files else ip_side - im_side
        return out

def save_ssrf_shard(result, path, *, extra_meta=None):
    gamma_values = np.asarray(result.get('gamma_values', []))
    steps_values = np.asarray(result.get('steps_values', []), dtype=np.int32)
    meta = {'bin_idx': result['bin_idx'], 'mirror_idx': result['mirror_idx'], 'R': result['R'], 'num_bins': result['num_bins'], 'dt': result['dt'], 'max_burn_steps': result.get('max_burn_steps', result.get('max_steps', 0)), 'gamma_values': [g for g in gamma_values.tolist()], 'steps_values': [s for s in steps_values.tolist()], 'physics_model': PHYSICS_MODEL, 'rf_mode': str(result.get('rf_mode', RF_MODE)), 'gaussian_fwhm_R': result.get('gaussian_fwhm_R', 0.0), 'lorentzian_fwhm_R': result.get('lorentzian_fwhm_R', 0.0), 'diffusion_scale': result.get('diffusion_scale', 0.0), 'sampling': 'p_x_gamma_x_n_steps', 'dataset': 'ssrf_bin_traj_v2'}
    if extra_meta:
        meta.update(extra_meta)
    ps = np.asarray(result['ps'], dtype=STORE_DTYPE)
    ps_m = np.asarray(result['ps_m'], dtype=STORE_DTYPE)
    iplus = np.asarray(result['iplus'], dtype=STORE_DTYPE)
    iminus = np.asarray(result['iminus'], dtype=STORE_DTYPE)
    iplus_m = np.asarray(result['iplus_m'], dtype=STORE_DTYPE)
    iminus_m = np.asarray(result['iminus_m'], dtype=STORE_DTYPE)
    (_, q) = intensity_pq(iplus, iminus)
    (_, q_m) = intensity_pq(iplus_m, iminus_m)
    payload = {'meta_json': np.asarray(json.dumps(meta)), 'p_values': np.asarray(result['p_values']), 'gamma_rf': np.asarray(result['gamma_rf']), 'burn_steps': np.asarray(result['burn_steps'], dtype=np.int32), 'n_steps': np.asarray(result['n_steps'], dtype=np.int32), 'skipped': np.asarray(result['skipped'], dtype=bool), 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'q': np.asarray(q, dtype=STORE_DTYPE), 'amp': np.abs(ps), 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m, 'q_m': np.asarray(q_m, dtype=STORE_DTYPE), 'amp_m': np.abs(ps_m), 'bin_idx': np.asarray(result['bin_idx'], dtype=np.int32), 'mirror_idx': np.asarray(result['mirror_idx'], dtype=np.int32), 'dt': np.asarray(result['dt'])}
    if 'track_lo' in result:
        payload['track_lo'] = np.asarray(result['track_lo'], dtype=bool)
    if 'track_hi' in result:
        payload['track_hi'] = np.asarray(result['track_hi'], dtype=bool)
    for side in ('lo', 'hi'):
        ip_key = f'iplus_{side}'
        if ip_key in result and result[ip_key] is not None:
            ip_side = np.asarray(result[ip_key], dtype=STORE_DTYPE)
            im_side = np.asarray(result[f'iminus_{side}'], dtype=STORE_DTYPE)
            (_, q_side) = intensity_pq(ip_side, im_side)
            payload[f'ps_{side}'] = np.asarray(result[f'ps_{side}'], dtype=STORE_DTYPE)
            payload[ip_key] = ip_side
            payload[f'iminus_{side}'] = im_side
            payload[f'q_{side}'] = np.asarray(q_side, dtype=STORE_DTYPE)
    _save_npz_atomic(path, **payload)

def save_afp_shard(result, path, *, extra_meta=None):
    meta = {'bin_idx': result['bin_idx'], 'mirror_idx': result['mirror_idx'], 'R': result['R'], 'num_bins': result['num_bins'], 'dt': result['dt'], 'n_relax': result['n_relax'], 'afp_window': result['afp_window'], 'afp_efficiency': result['afp_efficiency'], 'afp_subset': [i for i in np.asarray(result['afp_subset']).tolist()], 'physics_model': PHYSICS_MODEL, 'dataset': 'afp_bin_traj_v2'}
    if extra_meta:
        meta.update(extra_meta)
    ps = np.asarray(result['ps'], dtype=STORE_DTYPE)
    ps_m = np.asarray(result['ps_m'], dtype=STORE_DTYPE)
    iplus = np.asarray(result['iplus'], dtype=STORE_DTYPE)
    iminus = np.asarray(result['iminus'], dtype=STORE_DTYPE)
    iplus_m = np.asarray(result['iplus_m'], dtype=STORE_DTYPE)
    iminus_m = np.asarray(result['iminus_m'], dtype=STORE_DTYPE)
    (_, q) = intensity_pq(iplus, iminus)
    (_, q_m) = intensity_pq(iplus_m, iminus_m)
    payload = {'meta_json': np.asarray(json.dumps(meta)), 'p_values': np.asarray(result['p_values']), 'n_steps': np.asarray(result['n_steps'], dtype=np.int32), 'skipped': np.asarray(result['skipped'], dtype=bool), 'ps': ps, 'iplus': iplus, 'iminus': iminus, 'q': np.asarray(q, dtype=STORE_DTYPE), 'amp': np.abs(ps), 'ps_m': ps_m, 'iplus_m': iplus_m, 'iminus_m': iminus_m, 'q_m': np.asarray(q_m, dtype=STORE_DTYPE), 'amp_m': np.abs(ps_m), 'afp_subset': np.asarray(result['afp_subset'], dtype=np.int32), 'bin_idx': np.asarray(result['bin_idx'], dtype=np.int32), 'mirror_idx': np.asarray(result['mirror_idx'], dtype=np.int32), 'dt': np.asarray(result['dt'])}
    if 'track_lo' in result:
        payload['track_lo'] = np.asarray(result['track_lo'], dtype=bool)
    if 'track_hi' in result:
        payload['track_hi'] = np.asarray(result['track_hi'], dtype=bool)
    for side in ('lo', 'hi'):
        ip_key = f'iplus_{side}'
        if ip_key in result and result[ip_key] is not None:
            ip_side = np.asarray(result[ip_key], dtype=STORE_DTYPE)
            im_side = np.asarray(result[f'iminus_{side}'], dtype=STORE_DTYPE)
            (_, q_side) = intensity_pq(ip_side, im_side)
            payload[f'ps_{side}'] = np.asarray(result[f'ps_{side}'], dtype=STORE_DTYPE)
            payload[ip_key] = ip_side
            payload[f'iminus_{side}'] = im_side
            payload[f'q_{side}'] = np.asarray(q_side, dtype=STORE_DTYPE)
    _save_npz_atomic(path, **payload)
