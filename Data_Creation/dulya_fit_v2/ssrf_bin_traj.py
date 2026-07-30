"""
Per-bin ssRF burn samples for SLURM array jobs (Dulya-fit equilibrium, v2 physics).

For each (polarization, gamma_rf, n_steps) on the discrete grids, starts from the
unburned Dulya lineshape and burns for exactly ``n_steps`` macro-steps (no
continuous burn-to-mirror-turnover). Stores the burn/mirror trajectory at fit
scale.

Run from this directory (self-contained; no parent-repo imports):
  python ssrf_bin_traj.py --bin-idx 172
  python ssrf_bin_traj.py --bin-idx 172 --rf-mode single_bin
  python ssrf_bin_traj.py --organize --shard-dir data/ssrf_shards --output-dir data/ssrf_train
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from bin_io import (
    organize_ssrf_shards,
    save_ssrf_shard,
    save_ssrf_spectrum_shard,
    ssrf_shard_complete,
    ssrf_shard_part_path,
    ssrf_shard_parts_manifest_path,
    ssrf_shard_path,
    ssrf_spectrum_shard_complete,
    ssrf_spectrum_shard_part_path,
    ssrf_spectrum_shard_parts_manifest_path,
    ssrf_spectrum_shard_path,
)
from bin_setup import (
    equilibrium_lineshape,
    get_shape_params,
    polarization_grid,
    print_shape_banner,
    resolve_bin_idx,
    shape_meta,
    spin1_scale_factors,
)
from common import (
    BURN_BIN_CHOICES,
    BURN_STEPS_STEP,
    DEFAULT_RANDOM_SSRF_SAMPLES,
    DEFAULT_SSRF_COMBO_BATCH_SIZE,
    DEFAULT_SSRF_TRAJ_COMBO_BATCH_SIZE,
    DIFFUSION_SCALE,
    DT,
    F_MAX,
    F_MIN,
    GAMMA_RF_MAX,
    GAMMA_RF_MIN,
    GAMMA_RF_STEP,
    MAX_BURN_STEPS,
    MIN_BURN_STEPS,
    MULTI_BURN_MAX,
    MULTI_BURN_MIN,
    NUM_BINS,
    P_MAX,
    P_MIN,
    P_STEP,
    PS_ABS_MIN,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE,
    RF_MODE_PHYSICAL_VOIGT,
    RF_MODE_SINGLE_BIN,
    SEED,
    SPECTRUM_SSRF_SHARD_DIR,
    STORE_DTYPE,
    SSRF_SHARD_DIR,
    SSRF_TRAIN_DIR,
    UNMANIP_TRAIN_FRACTION,
    burn_steps_grid,
    gamma_rf_grid,
)
from model_bridge import (
    build_spin1_model,
    configure_ssrf_burn,
    euler_n_sub,
    full_spectrum_intensities,
    intensities_at_bins,
    level_pq,
    mirror_bin_idx,
    traj_to_fit_scale,
)

R_MIN = F_MIN
R_MAX = F_MAX

DEFAULT_SHARD_DIR = SSRF_SHARD_DIR
DEFAULT_TRAIN_DIR = SSRF_TRAIN_DIR
DEFAULT_SPECTRUM_SHARD_DIR = SPECTRUM_SSRF_SHARD_DIR


def _rng(seed: int | None = None) -> random.Random:
    return random.Random(SEED if seed is None else int(seed))


def _sample_random_burn_params(
    rng: random.Random,
    *,
    gamma_min: float = GAMMA_RF_MIN,
    gamma_max: float = GAMMA_RF_MAX,
    steps_min: int = MIN_BURN_STEPS,
    steps_max: int = MAX_BURN_STEPS,
) -> tuple[float, int]:
    gamma = rng.uniform(float(gamma_min), float(gamma_max))
    n_steps = rng.randint(int(steps_min), int(steps_max))
    return float(gamma), int(n_steps)


def _sample_multi_burn_plan(
    rng: random.Random,
    num_bins: int,
    *,
    n_burns: int,
    gamma_min: float = GAMMA_RF_MIN,
    gamma_max: float = GAMMA_RF_MAX,
    steps_min: int = MIN_BURN_STEPS,
    steps_max: int = MAX_BURN_STEPS,
) -> tuple[list[int], list[float], list[int]]:
    choices = np.asarray(BURN_BIN_CHOICES, dtype=int)
    if choices.size == 0:
        choices = np.arange(int(num_bins), dtype=int)
    burn_bins = [int(rng.choice(choices)) for _ in range(int(n_burns))]
    gammas: list[float] = []
    steps: list[int] = []
    for _ in range(int(n_burns)):
        g, n = _sample_random_burn_params(
            rng,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            steps_min=steps_min,
            steps_max=steps_max,
        )
        gammas.append(g)
        steps.append(n)
    return burn_bins, gammas, steps


def run_one_polarization(
    bin_idx: int,
    polarization: float,
    *,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    gamma_rf: float,
    n_steps: int,
    rf_mode: str = RF_MODE,
    gaussian_fwhm_R: float = RF_GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R: float = RF_LORENTZIAN_FWHM_R,
    diffusion_scale: float = DIFFUSION_SCALE,
    shape_params: dict[str, float] | None = None,
    capture_spectrum: bool = False,
) -> dict:
    """Burn exactly ``n_steps`` macro-steps at ``gamma_rf`` from Dulya equilibrium."""
    P = float(polarization)
    n_burn = int(n_steps)
    if n_burn < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_burn}")

    shape = shape_params if shape_params is not None else get_shape_params()
    f = np.linspace(float(F_MIN), float(F_MAX), int(num_bins))
    _, ip_fit, im_fit = equilibrium_lineshape(P, f, shape)
    ip_fit = np.asarray(ip_fit, dtype=float)
    im_fit = np.asarray(im_fit, dtype=float)
    to_spin1, from_spin1 = spin1_scale_factors(P, ip_fit, im_fit)
    iplus0 = ip_fit * to_spin1
    iminus0 = im_fit * to_spin1
    mirror_idx = mirror_bin_idx(int(num_bins), bin_idx)

    ps0 = float(ip_fit[bin_idx] + im_fit[bin_idx])
    if abs(ps0) < PS_ABS_MIN:
        return {
            "polarization": float(polarization),
            "skipped": True,
            "n_steps": 0,
            "burn_steps": int(n_burn),
            "gamma_rf": float(gamma_rf),
            "ps": np.zeros(0, dtype=float),
            "iplus": np.zeros(0, dtype=float),
            "iminus": np.zeros(0, dtype=float),
            "ps_m": np.zeros(0, dtype=float),
            "iplus_m": np.zeros(0, dtype=float),
            "iminus_m": np.zeros(0, dtype=float),
            "ps0": ps0,
            "stop_reason": "skipped_tiny_ps0",
            "rf_mode": str(rf_mode),
        }

    model = build_spin1_model(
        iplus0,
        iminus0,
        polarization=P,
        num_bins=num_bins,
        dt=dt,
        rf_enabled=True,
        relax_enabled=True,
        diffusion_scale=diffusion_scale,
        rf_gaussian_fwhm_R=gaussian_fwhm_R,
        rf_lorentzian_fwhm_R=lorentzian_fwhm_R,
    )
    used_mode = configure_ssrf_burn(
        model,
        bin_idx,
        float(gamma_rf),
        rf_mode=rf_mode,
        gaussian_fwhm_R=gaussian_fwhm_R,
        lorentzian_fwhm_R=lorentzian_fwhm_R,
    )
    p_initial, q_initial = level_pq(model)

    t_len = n_burn + 1
    ip_spec0 = im_spec0 = ip_spec = im_spec = None
    ps_full = iplus_full = iminus_full = None
    if capture_spectrum:
        ip_spec0, im_spec0, _ = full_spectrum_intensities(model)
        ps_full = np.empty((t_len, int(num_bins)), dtype=float)
        iplus_full = np.empty((t_len, int(num_bins)), dtype=float)
        iminus_full = np.empty((t_len, int(num_bins)), dtype=float)

    ps = np.empty(t_len, dtype=float)
    iplus = np.empty(t_len, dtype=float)
    iminus = np.empty(t_len, dtype=float)
    ps_m = np.empty(t_len, dtype=float)
    iplus_m = np.empty(t_len, dtype=float)
    iminus_m = np.empty(t_len, dtype=float)

    def _record_spectrum(k: int) -> None:
        if not capture_spectrum or ps_full is None:
            return
        ip_s, im_s, ps_s = full_spectrum_intensities(model)
        iplus_full[k] = ip_s
        iminus_full[k] = im_s
        ps_full[k] = ps_s

    ip, im, ps0_b, ip_m, im_m, ps_m0 = intensities_at_bins(model, bin_idx, mirror_idx)
    iplus[0], iminus[0], ps[0] = ip, im, ps0_b
    iplus_m[0], iminus_m[0], ps_m[0] = ip_m, im_m, ps_m0
    _record_spectrum(0)

    n_sub, dt_sub = euler_n_sub(float(gamma_rf), float(dt))
    for k in range(1, t_len):
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
        ip, im, ps_k, ip_m, im_m, ps_mk = intensities_at_bins(model, bin_idx, mirror_idx)
        iplus[k], iminus[k], ps[k] = ip, im, ps_k
        iplus_m[k], iminus_m[k], ps_m[k] = ip_m, im_m, ps_mk
        _record_spectrum(k)

    if capture_spectrum:
        ip_spec, im_spec, _ = full_spectrum_intensities(model)
    p_final, q_final = level_pq(model)

    out = traj_to_fit_scale(
        {
            "polarization": float(polarization),
            "skipped": False,
            "n_steps": t_len,
            "burn_steps": int(n_burn),
            "gamma_rf": float(gamma_rf),
            "ps": ps,
            "iplus": iplus,
            "iminus": iminus,
            "ps_m": ps_m,
            "iplus_m": iplus_m,
            "iminus_m": iminus_m,
            "ps0": ps0,
            "stop_reason": "fixed_n_steps",
            "rf_mode": used_mode,
            "ip_spectrum0": ip_spec0,
            "im_spectrum0": im_spec0,
            "ip_spectrum": ip_spec,
            "im_spectrum": im_spec,
            "ps_full": ps_full,
            "iplus_full": iplus_full,
            "iminus_full": iminus_full,
            "frequency": f,
            "p_initial": float(p_initial),
            "q_initial": float(q_initial),
            "p_final": float(p_final),
            "q_final": float(q_final),
            "center_bin": int(bin_idx),
            "n_burns": 1,
        },
        from_spin1,
    )
    return out


def run_multi_burn_polarization(
    polarization: float,
    burn_bins: list[int],
    gamma_values: list[float],
    n_steps_values: list[int],
    *,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    rf_mode: str = RF_MODE,
    gaussian_fwhm_R: float = RF_GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R: float = RF_LORENTZIAN_FWHM_R,
    diffusion_scale: float = DIFFUSION_SCALE,
    shape_params: dict[str, float] | None = None,
    capture_spectrum: bool = True,
) -> dict:
    """Apply a sequence of ssRF burns and record the full spectrum each macro-step."""
    if len(burn_bins) != len(gamma_values) or len(burn_bins) != len(n_steps_values):
        raise ValueError("burn_bins, gamma_values, and n_steps_values must match")
    if not burn_bins:
        raise ValueError("burn_bins must be non-empty")

    P = float(polarization)
    shape = shape_params if shape_params is not None else get_shape_params()
    f = np.linspace(float(F_MIN), float(F_MAX), int(num_bins))
    _, ip_fit, im_fit = equilibrium_lineshape(P, f, shape)
    ip_fit = np.asarray(ip_fit, dtype=float)
    im_fit = np.asarray(im_fit, dtype=float)
    to_spin1, from_spin1 = spin1_scale_factors(P, ip_fit, im_fit)
    iplus0 = ip_fit * to_spin1
    iminus0 = im_fit * to_spin1
    primary_bin = int(burn_bins[0])
    mirror_idx = mirror_bin_idx(int(num_bins), primary_bin)

    model = build_spin1_model(
        iplus0,
        iminus0,
        polarization=P,
        num_bins=num_bins,
        dt=dt,
        rf_enabled=True,
        relax_enabled=True,
        diffusion_scale=diffusion_scale,
        rf_gaussian_fwhm_R=gaussian_fwhm_R,
        rf_lorentzian_fwhm_R=lorentzian_fwhm_R,
    )
    p_initial, q_initial = level_pq(model)

    total_steps = int(sum(int(n) for n in n_steps_values))
    t_len = total_steps + 1
    ps = np.empty(t_len, dtype=float)
    iplus = np.empty(t_len, dtype=float)
    iminus = np.empty(t_len, dtype=float)
    ps_m = np.empty(t_len, dtype=float)
    iplus_m = np.empty(t_len, dtype=float)
    iminus_m = np.empty(t_len, dtype=float)
    ps_full = iplus_full = iminus_full = None
    if capture_spectrum:
        ps_full = np.empty((t_len, int(num_bins)), dtype=float)
        iplus_full = np.empty((t_len, int(num_bins)), dtype=float)
        iminus_full = np.empty((t_len, int(num_bins)), dtype=float)

    def _record_step(k: int) -> None:
        ip, im, ps_k, ip_m, im_m, ps_mk = intensities_at_bins(
            model, primary_bin, mirror_idx
        )
        iplus[k], iminus[k], ps[k] = ip, im, ps_k
        iplus_m[k], iminus_m[k], ps_m[k] = ip_m, im_m, ps_mk
        if capture_spectrum and ps_full is not None:
            ip_s, im_s, ps_s = full_spectrum_intensities(model)
            iplus_full[k] = ip_s
            iminus_full[k] = im_s
            ps_full[k] = ps_s

    _record_step(0)
    k = 1
    used_mode = str(rf_mode)
    for burn_idx, gamma_rf, n_burn in zip(burn_bins, gamma_values, n_steps_values):
        used_mode = configure_ssrf_burn(
            model,
            int(burn_idx),
            float(gamma_rf),
            rf_mode=rf_mode,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
        )
        n_sub, dt_sub = euler_n_sub(float(gamma_rf), float(dt))
        for _ in range(int(n_burn)):
            for _ in range(n_sub):
                model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
            _record_step(k)
            k += 1

    p_final, q_final = level_pq(model)
    ip_spec0 = None if ps_full is None else ps_full[0].copy()
    return traj_to_fit_scale(
        {
            "polarization": float(polarization),
            "skipped": False,
            "n_steps": t_len,
            "burn_steps": int(total_steps),
            "gamma_rf": float(gamma_values[-1]),
            "ps": ps,
            "iplus": iplus,
            "iminus": iminus,
            "ps_m": ps_m,
            "iplus_m": iplus_m,
            "iminus_m": iminus_m,
            "ps0": float(ps[0]),
            "stop_reason": "multi_burn",
            "rf_mode": used_mode,
            "ip_spectrum0": None if iplus_full is None else iplus_full[0].copy(),
            "im_spectrum0": None if iminus_full is None else iminus_full[0].copy(),
            "ip_spectrum": None if iplus_full is None else iplus_full[-1].copy(),
            "im_spectrum": None if iminus_full is None else iminus_full[-1].copy(),
            "ps_full": ps_full,
            "iplus_full": iplus_full,
            "iminus_full": iminus_full,
            "frequency": f,
            "p_initial": float(p_initial),
            "q_initial": float(q_initial),
            "p_final": float(p_final),
            "q_final": float(q_final),
            "center_bin": primary_bin,
            "n_burns": int(len(burn_bins)),
            "burn_bins": [int(b) for b in burn_bins],
            "gamma_values": [float(g) for g in gamma_values],
            "steps_values": [int(n) for n in n_steps_values],
        },
        from_spin1,
    )


def run_unmanipulated_polarization(
    polarization: float,
    *,
    num_bins: int = NUM_BINS,
    shape_params: dict[str, float] | None = None,
) -> dict:
    """Return equilibrium full-spectrum sample (no manipulation)."""
    P = float(polarization)
    shape = shape_params if shape_params is not None else get_shape_params()
    f = np.linspace(float(F_MIN), float(F_MAX), int(num_bins))
    ps_eq, ip_eq, im_eq = equilibrium_lineshape(P, f, shape)
    ps_eq = np.asarray(ps_eq, dtype=float)
    ip_eq = np.asarray(ip_eq, dtype=float)
    im_eq = np.asarray(im_eq, dtype=float)
    return {
        "polarization": P,
        "skipped": False,
        "n_steps": 1,
        "burn_steps": 0,
        "gamma_rf": 0.0,
        "ps": np.asarray([float(ps_eq[num_bins // 2])], dtype=float),
        "iplus": np.asarray([float(ip_eq[num_bins // 2])], dtype=float),
        "iminus": np.asarray([float(im_eq[num_bins // 2])], dtype=float),
        "ps_m": np.asarray([float(ps_eq[num_bins // 2])], dtype=float),
        "iplus_m": np.asarray([float(ip_eq[num_bins // 2])], dtype=float),
        "iminus_m": np.asarray([float(im_eq[num_bins // 2])], dtype=float),
        "ps0": float(ps_eq[num_bins // 2]),
        "stop_reason": "unmanipulated",
        "rf_mode": "none",
        "ps_full": ps_eq.reshape(1, -1),
        "iplus_full": ip_eq.reshape(1, -1),
        "iminus_full": im_eq.reshape(1, -1),
        "frequency": f,
        "center_bin": int(num_bins // 2),
        "n_burns": 0,
    }


def _build_combos(
    p_values: np.ndarray,
    gamma_values: np.ndarray,
    steps_values: np.ndarray,
) -> list[tuple[float, float, int]]:
    combos: list[tuple[float, float, int]] = []
    for p0 in np.asarray(p_values, dtype=float):
        for g in np.asarray(gamma_values, dtype=float):
            for n_burn in np.asarray(steps_values, dtype=np.int32):
                combos.append((float(p0), float(g), int(n_burn)))
    return combos


def _estimate_spectrum_batch_gib(
    n_samples: int,
    t_max: int,
    num_bins: int,
    *,
    spectrum_only: bool,
    capture_spectrum: bool = True,
) -> float:
    """Rough peak-RAM estimate (GiB) for one combo batch."""
    itemsize = np.dtype(STORE_DTYPE).itemsize
    full_bytes = n_samples * t_max * int(num_bins) * 3 * itemsize if capture_spectrum else 0
    center_bytes = 0 if (capture_spectrum and spectrum_only) else n_samples * t_max * 6 * itemsize
    return (full_bytes + center_bytes) / (1024**3)


def _run_one_bin_combos(
    combos: list[tuple[float, float, int]],
    bin_idx: int,
    *,
    t_max: int,
    num_bins: int,
    dt: float,
    rf_mode: str,
    gaussian_fwhm_R: float,
    lorentzian_fwhm_R: float,
    diffusion_scale: float,
    capture_spectrum: bool,
    spectrum_only: bool = False,
    combo_offset: int = 0,
    combo_total: int | None = None,
) -> dict:
    """Run ssRF for an explicit combo list (one bin)."""
    bin_idx = int(bin_idx)
    mirror_idx = mirror_bin_idx(int(num_bins), bin_idx)
    n_samples = len(combos)
    total = int(combo_total) if combo_total is not None else n_samples

    p_out = np.empty(n_samples, dtype=float)
    gamma_out = np.empty(n_samples, dtype=float)
    burn_steps_out = np.empty(n_samples, dtype=np.int32)
    n_steps = np.zeros(n_samples, dtype=np.int32)
    skipped = np.zeros(n_samples, dtype=bool)

    # STORE_DTYPE (float32): the default grid can reach (n_samples, t_max) in
    # the tens of thousands each way, so these six arrays dominate peak RAM --
    # float64 here is what pushed plain trajectory-mode generation to ~60 GiB
    # and OOM-killed the SLURM task even without --spectrum-mode.
    ps = iplus = iminus = ps_m = iplus_m = iminus_m = None
    if capture_spectrum and not spectrum_only:
        ps = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        iplus = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        iminus = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        ps_m = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        iplus_m = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        iminus_m = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
    elif not capture_spectrum:
        ps = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        iplus = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        iminus = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        ps_m = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        iplus_m = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)
        iminus_m = np.full((n_samples, t_max), np.nan, dtype=STORE_DTYPE)

    ps_full = iplus_full = iminus_full = None
    if capture_spectrum:
        ps_full = np.full((n_samples, t_max, int(num_bins)), np.nan, dtype=STORE_DTYPE)
        iplus_full = np.full((n_samples, t_max, int(num_bins)), np.nan, dtype=STORE_DTYPE)
        iminus_full = np.full((n_samples, t_max, int(num_bins)), np.nan, dtype=STORE_DTYPE)

    for j, (p0, g, n_burn) in enumerate(combos):
        global_j = int(combo_offset) + j
        print(
            f"  [{global_j + 1}/{total}] P={p0:+.3f}  gamma={g:.3f}  n_steps={n_burn}",
            flush=True,
        )
        traj = run_one_polarization(
            bin_idx,
            float(p0),
            num_bins=num_bins,
            dt=dt,
            gamma_rf=float(g),
            n_steps=int(n_burn),
            rf_mode=rf_mode,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
            diffusion_scale=diffusion_scale,
            capture_spectrum=capture_spectrum,
        )
        p_out[j] = float(p0)
        gamma_out[j] = float(g)
        burn_steps_out[j] = int(n_burn)
        skipped[j] = bool(traj["skipped"])
        n = int(traj["n_steps"])
        n_steps[j] = n
        if n <= 0:
            continue
        if ps is not None:
            ps[j, :n] = traj["ps"]
            iplus[j, :n] = traj["iplus"]
            iminus[j, :n] = traj["iminus"]
            ps_m[j, :n] = traj["ps_m"]
            iplus_m[j, :n] = traj["iplus_m"]
            iminus_m[j, :n] = traj["iminus_m"]
        if capture_spectrum and ps_full is not None:
            pf = traj.get("ps_full")
            if pf is not None:
                ps_full[j, :n] = np.asarray(pf, dtype=STORE_DTYPE)[:n]
                iplus_full[j, :n] = np.asarray(traj["iplus_full"], dtype=STORE_DTYPE)[:n]
                iminus_full[j, :n] = np.asarray(traj["iminus_full"], dtype=STORE_DTYPE)[:n]

    f = np.linspace(R_MIN, R_MAX, int(num_bins))
    out: dict = {
        "bin_idx": bin_idx,
        "mirror_idx": mirror_idx,
        "R": float(f[bin_idx]),
        "num_bins": int(num_bins),
        "dt": float(dt),
        "p_values": p_out,
        "gamma_rf": gamma_out,
        "burn_steps": burn_steps_out,
        "n_steps": n_steps,
        "skipped": skipped,
    }
    if ps is not None:
        out.update(
            {
                "ps": ps,
                "iplus": iplus,
                "iminus": iminus,
                "ps_m": ps_m,
                "iplus_m": iplus_m,
                "iminus_m": iminus_m,
            }
        )
    if capture_spectrum:
        out["ps_full"] = ps_full
        out["iplus_full"] = iplus_full
        out["iminus_full"] = iminus_full
    return out


def save_ssrf_spectrum_in_batches(
    bin_idx: int,
    *,
    combos: list[tuple[float, float, int]],
    shard_dir: Path,
    combo_batch_size: int,
    num_bins: int,
    dt: float,
    rf_mode: str,
    gaussian_fwhm_R: float,
    lorentzian_fwhm_R: float,
    diffusion_scale: float,
    gamma_values: np.ndarray,
    steps_values: np.ndarray,
    extra_meta: dict,
) -> dict:
    """Generate spectrum shards in bounded-memory batches; write part NPZs + manifest."""
    shard_dir = Path(shard_dir)
    t_max = int(np.max(steps_values)) + 1
    batch_size = max(1, int(combo_batch_size))
    n_combos = len(combos)
    n_batches = (n_combos + batch_size - 1) // batch_size
    est_gib = _estimate_spectrum_batch_gib(
        min(batch_size, n_combos), t_max, num_bins, spectrum_only=True
    )
    print(
        f"  batching {n_combos} combos into {n_batches} part(s) "
        f"(batch_size={batch_size}, est_peak~{est_gib:.2f} GiB ps_full)",
        flush=True,
    )

    part_files: list[str] = []
    for part_idx, start in enumerate(range(0, n_combos, batch_size)):
        end = min(start + batch_size, n_combos)
        batch = combos[start:end]
        part_result = _run_one_bin_combos(
            batch,
            bin_idx,
            t_max=t_max,
            num_bins=num_bins,
            dt=dt,
            rf_mode=rf_mode,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
            diffusion_scale=diffusion_scale,
            capture_spectrum=True,
            spectrum_only=True,
            combo_offset=start,
            combo_total=n_combos,
        )
        part_result["gamma_values"] = np.asarray(gamma_values, dtype=float)
        part_result["steps_values"] = np.asarray(steps_values, dtype=np.int32)
        part_result["max_burn_steps"] = int(np.max(steps_values))
        part_result["rf_mode"] = str(rf_mode)
        part_result["gaussian_fwhm_R"] = float(gaussian_fwhm_R)
        part_result["lorentzian_fwhm_R"] = float(lorentzian_fwhm_R)
        part_result["diffusion_scale"] = float(diffusion_scale)
        part_result["dataset"] = "ssrf_spectrum_bin_v2"
        part_result["part_index"] = int(part_idx)
        part_result["part_count"] = int(n_batches)

        part_path = ssrf_spectrum_shard_part_path(shard_dir, bin_idx, part_idx)
        save_ssrf_spectrum_shard(part_result, part_path, extra_meta=extra_meta)
        part_files.append(part_path.name)
        print(f"  wrote part {part_idx + 1}/{n_batches} -> {part_path.name}", flush=True)

    manifest_path = ssrf_spectrum_shard_parts_manifest_path(shard_dir, bin_idx)
    manifest = {
        "bin_idx": int(bin_idx),
        "n_samples": int(n_combos),
        "n_parts": int(n_batches),
        "part_files": part_files,
        "combo_batch_size": int(batch_size),
        "max_burn_steps": int(t_max - 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "bin_idx": int(bin_idx),
        "mirror_idx": mirror_bin_idx(int(num_bins), bin_idx),
        "n_samples": int(n_combos),
        "n_parts": int(n_batches),
        "manifest": str(manifest_path),
        "p_values": np.array([c[0] for c in combos], dtype=float),
        "n_steps": np.zeros(n_combos, dtype=np.int32),
    }


def save_ssrf_traj_in_batches(
    bin_idx: int,
    *,
    combos: list[tuple[float, float, int]],
    shard_dir: Path,
    combo_batch_size: int,
    num_bins: int,
    dt: float,
    rf_mode: str,
    gaussian_fwhm_R: float,
    lorentzian_fwhm_R: float,
    diffusion_scale: float,
    gamma_values: np.ndarray,
    steps_values: np.ndarray,
    extra_meta: dict,
) -> dict:
    """Generate plain (non-spectrum) trajectory shards in bounded-memory
    batches; write part NPZs + manifest. Mirrors ``save_ssrf_spectrum_in_batches``
    but for the ps/iplus/iminus (+ mirror) center-bin arrays only -- these are
    what blow up peak RAM once ``steps_values`` reaches into the tens of
    thousands, even without --spectrum-mode.
    """
    shard_dir = Path(shard_dir)
    t_max = int(np.max(steps_values)) + 1
    batch_size = max(1, int(combo_batch_size))
    n_combos = len(combos)
    n_batches = (n_combos + batch_size - 1) // batch_size
    est_gib = _estimate_spectrum_batch_gib(
        min(batch_size, n_combos), t_max, num_bins, spectrum_only=False, capture_spectrum=False
    )
    print(
        f"  batching {n_combos} combos into {n_batches} part(s) "
        f"(batch_size={batch_size}, est_peak~{est_gib:.2f} GiB trajectories)",
        flush=True,
    )

    part_files: list[str] = []
    for part_idx, start in enumerate(range(0, n_combos, batch_size)):
        end = min(start + batch_size, n_combos)
        batch = combos[start:end]
        part_result = _run_one_bin_combos(
            batch,
            bin_idx,
            t_max=t_max,
            num_bins=num_bins,
            dt=dt,
            rf_mode=rf_mode,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
            diffusion_scale=diffusion_scale,
            capture_spectrum=False,
            spectrum_only=False,
            combo_offset=start,
            combo_total=n_combos,
        )
        part_result["gamma_values"] = np.asarray(gamma_values, dtype=float)
        part_result["steps_values"] = np.asarray(steps_values, dtype=np.int32)
        part_result["max_burn_steps"] = int(np.max(steps_values))
        part_result["rf_mode"] = str(rf_mode)
        part_result["gaussian_fwhm_R"] = float(gaussian_fwhm_R)
        part_result["lorentzian_fwhm_R"] = float(lorentzian_fwhm_R)
        part_result["diffusion_scale"] = float(diffusion_scale)

        part_path = ssrf_shard_part_path(shard_dir, bin_idx, part_idx)
        save_ssrf_shard(part_result, part_path, extra_meta=extra_meta)
        part_files.append(part_path.name)
        print(f"  wrote part {part_idx + 1}/{n_batches} -> {part_path.name}", flush=True)

    manifest_path = ssrf_shard_parts_manifest_path(shard_dir, bin_idx)
    manifest = {
        "bin_idx": int(bin_idx),
        "n_samples": int(n_combos),
        "n_parts": int(n_batches),
        "part_files": part_files,
        "combo_batch_size": int(batch_size),
        "max_burn_steps": int(t_max - 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "bin_idx": int(bin_idx),
        "mirror_idx": mirror_bin_idx(int(num_bins), bin_idx),
        "n_samples": int(n_combos),
        "n_parts": int(n_batches),
        "manifest": str(manifest_path),
        "p_values": np.array([c[0] for c in combos], dtype=float),
        "n_steps": np.zeros(n_combos, dtype=np.int32),
    }


def run_one_bin(
    bin_idx: int,
    *,
    p_values: np.ndarray,
    gamma_values: np.ndarray | None = None,
    steps_values: np.ndarray | None = None,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    rf_mode: str = RF_MODE,
    gaussian_fwhm_R: float = RF_GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R: float = RF_LORENTZIAN_FWHM_R,
    diffusion_scale: float = DIFFUSION_SCALE,
    capture_spectrum: bool = False,
) -> dict:
    """Cartesian product over P × gamma_rf × burn_steps for one burn bin."""
    bin_idx = int(bin_idx)
    if bin_idx < 0 or bin_idx >= int(num_bins):
        raise ValueError(f"bin_idx={bin_idx} out of range for num_bins={num_bins}")

    p_values = np.asarray(p_values, dtype=float)
    gamma_values = (
        gamma_rf_grid()
        if gamma_values is None
        else np.asarray(gamma_values, dtype=float)
    )
    steps_values = (
        burn_steps_grid()
        if steps_values is None
        else np.asarray(steps_values, dtype=np.int32)
    )
    if gamma_values.size == 0 or steps_values.size == 0 or p_values.size == 0:
        raise ValueError("p_values, gamma_values, and steps_values must be non-empty")

    combos = _build_combos(p_values, gamma_values, steps_values)
    t_max = int(np.max(steps_values)) + 1
    out = _run_one_bin_combos(
        combos,
        bin_idx,
        t_max=t_max,
        num_bins=num_bins,
        dt=dt,
        rf_mode=rf_mode,
        gaussian_fwhm_R=gaussian_fwhm_R,
        lorentzian_fwhm_R=lorentzian_fwhm_R,
        diffusion_scale=diffusion_scale,
        capture_spectrum=capture_spectrum,
        spectrum_only=False,
    )
    out["gamma_values"] = np.asarray(gamma_values, dtype=float)
    out["steps_values"] = np.asarray(steps_values, dtype=np.int32)
    out["max_burn_steps"] = int(np.max(steps_values))
    out["rf_mode"] = str(rf_mode)
    out["gaussian_fwhm_R"] = float(gaussian_fwhm_R)
    out["lorentzian_fwhm_R"] = float(lorentzian_fwhm_R)
    out["diffusion_scale"] = float(diffusion_scale)
    return out


def run_one_bin_spectrum(
    bin_idx: int,
    *,
    p_values: np.ndarray,
    gamma_values: np.ndarray | None = None,
    steps_values: np.ndarray | None = None,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    rf_mode: str = RF_MODE,
    gaussian_fwhm_R: float = RF_GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R: float = RF_LORENTZIAN_FWHM_R,
    diffusion_scale: float = DIFFUSION_SCALE,
    random_samples: int = DEFAULT_RANDOM_SSRF_SAMPLES,
    multi_burn: bool = False,
    unmanip_fraction: float = UNMANIP_TRAIN_FRACTION,
    seed: int | None = None,
    gamma_min: float = GAMMA_RF_MIN,
    gamma_max: float = GAMMA_RF_MAX,
    steps_min: int = MIN_BURN_STEPS,
    steps_max: int = MAX_BURN_STEPS,
) -> dict:
    """Generate full-spectrum ssRF trajectories with optional random/multi-burn samples."""
    rng = _rng(seed)
    base = run_one_bin(
        bin_idx,
        p_values=p_values,
        gamma_values=gamma_values,
        steps_values=steps_values,
        num_bins=num_bins,
        dt=dt,
        rf_mode=rf_mode,
        gaussian_fwhm_R=gaussian_fwhm_R,
        lorentzian_fwhm_R=lorentzian_fwhm_R,
        diffusion_scale=diffusion_scale,
        capture_spectrum=True,
    )

    extra_trajs: list[dict] = []
    n_random = int(random_samples)
    for _ in range(n_random):
        p0 = float(rng.choice(np.asarray(p_values, dtype=float)))
        g, n_burn = _sample_random_burn_params(
            rng,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            steps_min=steps_min,
            steps_max=steps_max,
        )
        if multi_burn:
            n_burns = rng.randint(MULTI_BURN_MIN, MULTI_BURN_MAX)
            burn_bins, gammas, steps = _sample_multi_burn_plan(
                rng,
                num_bins,
                n_burns=n_burns,
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                steps_min=steps_min,
                steps_max=steps_max,
            )
            burn_bins[0] = int(bin_idx)
            traj = run_multi_burn_polarization(
                p0,
                burn_bins,
                gammas,
                steps,
                num_bins=num_bins,
                dt=dt,
                rf_mode=rf_mode,
                gaussian_fwhm_R=gaussian_fwhm_R,
                lorentzian_fwhm_R=lorentzian_fwhm_R,
                diffusion_scale=diffusion_scale,
                capture_spectrum=True,
            )
        else:
            traj = run_one_polarization(
                bin_idx,
                p0,
                num_bins=num_bins,
                dt=dt,
                gamma_rf=g,
                n_steps=n_burn,
                rf_mode=rf_mode,
                gaussian_fwhm_R=gaussian_fwhm_R,
                lorentzian_fwhm_R=lorentzian_fwhm_R,
                diffusion_scale=diffusion_scale,
                capture_spectrum=True,
            )
        extra_trajs.append(traj)

    n_base = int(base["p_values"].size)
    n_extra = len(extra_trajs)
    n_unmanip = 0
    if float(unmanip_fraction) > 0.0 and n_base + n_extra > 0:
        target = int(round(float(unmanip_fraction) * (n_base + n_extra) / max(1e-12, 1.0 - float(unmanip_fraction))))
        n_unmanip = max(1, target)
        for _ in range(n_unmanip):
            p0 = float(rng.choice(np.asarray(p_values, dtype=float)))
            extra_trajs.append(run_unmanipulated_polarization(p0, num_bins=num_bins))

    if not extra_trajs:
        base["dataset"] = "ssrf_spectrum_bin_v2"
        return base

    def _append_traj(traj: dict) -> None:
        nonlocal base
        n = int(traj["n_steps"])
        if n <= 0 or bool(traj.get("skipped", False)):
            return
        old_n = int(base["p_values"].size)
        t_max_old = int(base["ps"].shape[1])
        t_max_new = max(t_max_old, n)
        num_b = int(base["num_bins"])

        def _pad3(arr: np.ndarray | None) -> np.ndarray | None:
            if arr is None:
                return None
            out = np.full((old_n + 1, t_max_new, num_b), np.nan, dtype=float)
            out[:old_n, : arr.shape[1]] = arr
            return out

        def _pad2(arr: np.ndarray) -> np.ndarray:
            out = np.full((old_n + 1, t_max_new), np.nan, dtype=float)
            out[:old_n, : arr.shape[1]] = arr
            return out

        base["p_values"] = np.concatenate([base["p_values"], [float(traj["polarization"])]])
        base["gamma_rf"] = np.concatenate(
            [base["gamma_rf"], [float(traj.get("gamma_rf", 0.0))]]
        )
        base["burn_steps"] = np.concatenate(
            [base["burn_steps"], [int(traj.get("burn_steps", 0))]]
        )
        base["n_steps"] = np.concatenate([base["n_steps"], [n]])
        base["skipped"] = np.concatenate([base["skipped"], [False]])
        for key in ("ps", "iplus", "iminus", "ps_m", "iplus_m", "iminus_m"):
            base[key] = _pad2(base[key])
            base[key][old_n, :n] = np.asarray(traj[key], dtype=float)
        for key in ("ps_full", "iplus_full", "iminus_full"):
            if key in base:
                padded = _pad3(base[key])
                if padded is not None and traj.get(key) is not None:
                    padded[old_n, :n] = np.asarray(traj[key], dtype=float)[:n]
                    base[key] = padded

    for traj in extra_trajs:
        _append_traj(traj)

    base["dataset"] = "ssrf_spectrum_bin_v2"
    base["n_random_samples"] = n_random
    base["n_unmanip_samples"] = n_unmanip
    base["multi_burn"] = bool(multi_burn)
    return base


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-bin ssRF discrete (gamma, n_steps) burn worker / organizer (v2)"
    )
    p.add_argument("--bin-idx", type=int, default=None)
    p.add_argument(
        "--organize",
        "--combine",
        dest="organize",
        action="store_true",
        help="Organize shards into one training NPZ per bin",
    )
    p.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--p-min", type=float, default=P_MIN)
    p.add_argument("--p-max", type=float, default=P_MAX)
    p.add_argument("--p-step", type=float, default=P_STEP)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--gamma-min", type=float, default=GAMMA_RF_MIN)
    p.add_argument("--gamma-max", type=float, default=GAMMA_RF_MAX)
    p.add_argument("--gamma-step", type=float, default=GAMMA_RF_STEP)
    p.add_argument("--steps-min", type=int, default=MIN_BURN_STEPS)
    p.add_argument("--steps-max", type=int, default=MAX_BURN_STEPS)
    p.add_argument("--steps-step", type=int, default=BURN_STEPS_STEP)
    p.add_argument(
        "--rf-mode",
        choices=(RF_MODE_PHYSICAL_VOIGT, RF_MODE_SINGLE_BIN),
        default=RF_MODE,
        help="physical_voigt (default) or single_bin RF profile",
    )
    p.add_argument("--gauss-fwhm", type=float, default=RF_GAUSSIAN_FWHM_R)
    p.add_argument("--lorentz-fwhm", type=float, default=RF_LORENTZIAN_FWHM_R)
    p.add_argument("--diffusion-scale", type=float, default=DIFFUSION_SCALE)
    p.add_argument(
        "--spectrum-mode",
        action="store_true",
        help="Store full 500-bin spectra at each timestep (spectrum shard format)",
    )
    p.add_argument(
        "--random-samples",
        type=int,
        default=DEFAULT_RANDOM_SSRF_SAMPLES,
        help="Extra random (gamma_rf, n_steps) trajectories per bin (spectrum mode)",
    )
    p.add_argument(
        "--multi-burn",
        action="store_true",
        help="Generate multi-burn (2-5 burns) random trajectories (spectrum mode)",
    )
    p.add_argument(
        "--unmanip-fraction",
        type=float,
        default=UNMANIP_TRAIN_FRACTION,
        help="Fraction of unmanipulated equilibrium samples (spectrum mode)",
    )
    p.add_argument(
        "--combo-batch-size",
        type=int,
        default=None,
        help=(
            "Max P×gamma×steps combos per in-memory batch (writes part NPZs "
            "when the grid is larger). Defaults to "
            f"{DEFAULT_SSRF_COMBO_BATCH_SIZE} in --spectrum-mode (full cube per "
            f"combo) or {DEFAULT_SSRF_TRAJ_COMBO_BATCH_SIZE} otherwise "
            "(center-bin arrays only per combo)"
        ),
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-if-exists", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.organize:
        result = organize_ssrf_shards(
            args.shard_dir,
            args.output_dir,
            num_bins=args.num_bins,
            strict=bool(args.strict),
        )
        print(
            f"Organized {result['n_samples']} samples from {args.shard_dir} -> "
            f"{args.output_dir} ({args.num_bins} bin files; "
            f"missing={result.get('n_missing', 0)})",
            flush=True,
        )
        return

    bin_idx = resolve_bin_idx(args.bin_idx, num_bins=int(args.num_bins))
    if bin_idx is None:
        raise SystemExit(
            "Provide --bin-idx <int>, or set SLURM_ARRAY_TASK_ID, or pass --organize"
        )

    out = (
        ssrf_spectrum_shard_path(args.shard_dir, bin_idx)
        if args.spectrum_mode
        else ssrf_shard_path(args.shard_dir, bin_idx)
    )
    if args.skip_if_exists:
        if args.spectrum_mode and ssrf_spectrum_shard_complete(args.shard_dir, bin_idx):
            print(f"Skipping existing spectrum shard(s) for bin {bin_idx}", flush=True)
            return
        if not args.spectrum_mode and ssrf_shard_complete(args.shard_dir, bin_idx):
            print(f"Skipping existing shard(s) for bin {bin_idx}", flush=True)
            return

    shape = get_shape_params()
    print_shape_banner(shape, num_bins=int(args.num_bins))

    p_values = polarization_grid(args.p_min, args.p_max, args.p_step)
    g_values = gamma_rf_grid(args.gamma_min, args.gamma_max, args.gamma_step)
    s_values = burn_steps_grid(args.steps_min, args.steps_max, args.steps_step)
    n_combos = int(p_values.size * g_values.size * s_values.size)
    if args.combo_batch_size is None:
        default_batch = (
            DEFAULT_SSRF_COMBO_BATCH_SIZE
            if args.spectrum_mode
            else DEFAULT_SSRF_TRAJ_COMBO_BATCH_SIZE
        )
    else:
        default_batch = int(args.combo_batch_size)
    combo_batch_size = max(1, default_batch)
    t_max = int(np.max(s_values)) + 1 if s_values.size else 1
    est_full_gib = _estimate_spectrum_batch_gib(
        n_combos,
        t_max,
        args.num_bins,
        spectrum_only=bool(args.spectrum_mode),
        capture_spectrum=bool(args.spectrum_mode),
    )
    print(
        f"bin_idx={bin_idx}  n_P={p_values.size}  n_gamma={g_values.size}  "
        f"n_steps_grid={s_values.size}  n_combos={n_combos}  "
        f"spectrum_mode={bool(args.spectrum_mode)}  combo_batch_size={combo_batch_size}  "
        f"est_monolithic~{est_full_gib:.1f} GiB  "
        f"random_samples={int(args.random_samples)}  multi_burn={bool(args.multi_burn)}  "
        f"unmanip_fraction={float(args.unmanip_fraction):.3f}  "
        f"P=[{args.p_min},{args.p_max}] step={args.p_step}  "
        f"gamma=[{args.gamma_min},{args.gamma_max}] step={args.gamma_step}  "
        f"burn_steps=[{args.steps_min},{args.steps_max}] step={args.steps_step}  "
        f"dt={args.dt}  rf_mode={args.rf_mode}  "
        f"Gauss={args.gauss_fwhm:.4f} Lorentz={args.lorentz_fwhm:.4f}  "
        f"diffusion={args.diffusion_scale}",
        flush=True,
    )
    extra_meta = shape_meta(
        shape,
        rf_mode=str(args.rf_mode),
        gaussian_fwhm_R=float(args.gauss_fwhm),
        lorentzian_fwhm_R=float(args.lorentz_fwhm),
        diffusion_scale=float(args.diffusion_scale),
    )
    if args.spectrum_mode:
        use_batches = n_combos > combo_batch_size
        if use_batches and (
            int(args.random_samples) > 0 or bool(args.multi_burn) or float(args.unmanip_fraction) > 0
        ):
            print(
                "WARNING: batched spectrum generation supports the base grid only; "
                "random/multi-burn/unmanip extras are skipped. "
                "Lower combo count or increase --combo-batch-size to use run_one_bin_spectrum.",
                flush=True,
            )
        if use_batches:
            combos = _build_combos(p_values, g_values, s_values)
            result = save_ssrf_spectrum_in_batches(
                bin_idx,
                combos=combos,
                shard_dir=args.shard_dir,
                combo_batch_size=combo_batch_size,
                num_bins=args.num_bins,
                dt=args.dt,
                rf_mode=str(args.rf_mode),
                gaussian_fwhm_R=float(args.gauss_fwhm),
                lorentzian_fwhm_R=float(args.lorentz_fwhm),
                diffusion_scale=float(args.diffusion_scale),
                gamma_values=g_values,
                steps_values=s_values,
                extra_meta=extra_meta,
            )
            print(
                f"Wrote {result['n_parts']} part shard(s) + manifest for bin {bin_idx}  "
                f"n_samples={result['n_samples']}",
                flush=True,
            )
            return

        result = run_one_bin_spectrum(
            bin_idx,
            p_values=p_values,
            gamma_values=g_values,
            steps_values=s_values,
            num_bins=args.num_bins,
            dt=args.dt,
            rf_mode=str(args.rf_mode),
            gaussian_fwhm_R=float(args.gauss_fwhm),
            lorentzian_fwhm_R=float(args.lorentz_fwhm),
            diffusion_scale=float(args.diffusion_scale),
            random_samples=int(args.random_samples),
            multi_burn=bool(args.multi_burn),
            unmanip_fraction=float(args.unmanip_fraction),
            seed=int(args.seed),
            gamma_min=float(args.gamma_min),
            gamma_max=float(args.gamma_max),
            steps_min=int(args.steps_min),
            steps_max=int(args.steps_max),
        )
        save_ssrf_spectrum_shard(
            result,
            out,
            extra_meta=extra_meta,
        )
    else:
        if n_combos > combo_batch_size:
            combos = _build_combos(p_values, g_values, s_values)
            result = save_ssrf_traj_in_batches(
                bin_idx,
                combos=combos,
                shard_dir=args.shard_dir,
                combo_batch_size=combo_batch_size,
                num_bins=args.num_bins,
                dt=args.dt,
                rf_mode=str(args.rf_mode),
                gaussian_fwhm_R=float(args.gauss_fwhm),
                lorentzian_fwhm_R=float(args.lorentz_fwhm),
                diffusion_scale=float(args.diffusion_scale),
                gamma_values=g_values,
                steps_values=s_values,
                extra_meta=extra_meta,
            )
            print(
                f"Wrote {result['n_parts']} part shard(s) + manifest for bin {bin_idx}  "
                f"n_samples={result['n_samples']}",
                flush=True,
            )
            return

        result = run_one_bin(
            bin_idx,
            p_values=p_values,
            gamma_values=g_values,
            steps_values=s_values,
            num_bins=args.num_bins,
            dt=args.dt,
            rf_mode=str(args.rf_mode),
            gaussian_fwhm_R=float(args.gauss_fwhm),
            lorentzian_fwhm_R=float(args.lorentz_fwhm),
            diffusion_scale=float(args.diffusion_scale),
        )
        save_ssrf_shard(
            result,
            out,
            extra_meta=shape_meta(
                shape,
                rf_mode=str(args.rf_mode),
                gaussian_fwhm_R=float(args.gauss_fwhm),
                lorentzian_fwhm_R=float(args.lorentz_fwhm),
                diffusion_scale=float(args.diffusion_scale),
            ),
        )
    print(
        f"Wrote {out}  mirror={result['mirror_idx']}  "
        f"n_samples={result['p_values'].size}  "
        f"mean_traj_len={float(np.mean(result['n_steps'])):.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
