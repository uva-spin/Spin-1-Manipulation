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
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from bin_io import (
    organize_ssrf_shards,
    save_ssrf_shard,
    ssrf_shard_path,
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
    BURN_STEPS_STEP,
    DIFFUSION_SCALE,
    DT,
    F_MAX,
    F_MIN,
    GAMMA_RF_MAX,
    GAMMA_RF_MIN,
    GAMMA_RF_STEP,
    MAX_BURN_STEPS,
    MIN_BURN_STEPS,
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
    SSRF_SHARD_DIR,
    SSRF_TRAIN_DIR,
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

    ip_spec0 = im_spec0 = ip_spec = im_spec = None
    if capture_spectrum:
        ip_spec0, im_spec0, _ = full_spectrum_intensities(model)

    t_len = n_burn + 1
    ps = np.empty(t_len, dtype=float)
    iplus = np.empty(t_len, dtype=float)
    iminus = np.empty(t_len, dtype=float)
    ps_m = np.empty(t_len, dtype=float)
    iplus_m = np.empty(t_len, dtype=float)
    iminus_m = np.empty(t_len, dtype=float)

    ip, im, ps0_b, ip_m, im_m, ps_m0 = intensities_at_bins(model, bin_idx, mirror_idx)
    iplus[0], iminus[0], ps[0] = ip, im, ps0_b
    iplus_m[0], iminus_m[0], ps_m[0] = ip_m, im_m, ps_m0

    n_sub, dt_sub = euler_n_sub(float(gamma_rf), float(dt))
    for k in range(1, t_len):
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
        ip, im, ps_k, ip_m, im_m, ps_mk = intensities_at_bins(model, bin_idx, mirror_idx)
        iplus[k], iminus[k], ps[k] = ip, im, ps_k
        iplus_m[k], iminus_m[k], ps_m[k] = ip_m, im_m, ps_mk

    if capture_spectrum:
        ip_spec, im_spec, _ = full_spectrum_intensities(model)
    p_final, q_final = level_pq(model)

    return traj_to_fit_scale(
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
            "frequency": f,
            "p_initial": float(p_initial),
            "q_initial": float(q_initial),
            "p_final": float(p_final),
            "q_final": float(q_final),
        },
        from_spin1,
    )


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
) -> dict:
    """Cartesian product over P × gamma_rf × burn_steps for one burn bin."""
    bin_idx = int(bin_idx)
    if bin_idx < 0 or bin_idx >= int(num_bins):
        raise ValueError(f"bin_idx={bin_idx} out of range for num_bins={num_bins}")

    mirror_idx = mirror_bin_idx(int(num_bins), bin_idx)
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

    combos: list[tuple[float, float, int]] = []
    for p0 in p_values:
        for g in gamma_values:
            for n_burn in steps_values:
                combos.append((float(p0), float(g), int(n_burn)))

    n_samples = len(combos)
    t_max = int(np.max(steps_values)) + 1

    p_out = np.empty(n_samples, dtype=float)
    gamma_out = np.empty(n_samples, dtype=float)
    burn_steps_out = np.empty(n_samples, dtype=np.int32)
    n_steps = np.zeros(n_samples, dtype=np.int32)
    skipped = np.zeros(n_samples, dtype=bool)
    ps = np.full((n_samples, t_max), np.nan, dtype=float)
    iplus = np.full((n_samples, t_max), np.nan, dtype=float)
    iminus = np.full((n_samples, t_max), np.nan, dtype=float)
    ps_m = np.full((n_samples, t_max), np.nan, dtype=float)
    iplus_m = np.full((n_samples, t_max), np.nan, dtype=float)
    iminus_m = np.full((n_samples, t_max), np.nan, dtype=float)

    for j, (p0, g, n_burn) in enumerate(combos):
        print(
            f"  [{j + 1}/{n_samples}] P={p0:+.3f}  gamma={g:.3f}  n_steps={n_burn}",
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
        )
        p_out[j] = float(p0)
        gamma_out[j] = float(g)
        burn_steps_out[j] = int(n_burn)
        skipped[j] = bool(traj["skipped"])
        n = int(traj["n_steps"])
        n_steps[j] = n
        if n <= 0:
            continue
        ps[j, :n] = traj["ps"]
        iplus[j, :n] = traj["iplus"]
        iminus[j, :n] = traj["iminus"]
        ps_m[j, :n] = traj["ps_m"]
        iplus_m[j, :n] = traj["iplus_m"]
        iminus_m[j, :n] = traj["iminus_m"]

    f = np.linspace(R_MIN, R_MAX, int(num_bins))
    return {
        "bin_idx": bin_idx,
        "mirror_idx": mirror_idx,
        "R": float(f[bin_idx]),
        "num_bins": int(num_bins),
        "dt": float(dt),
        "gamma_values": np.asarray(gamma_values, dtype=float),
        "steps_values": np.asarray(steps_values, dtype=np.int32),
        "max_burn_steps": int(np.max(steps_values)),
        "rf_mode": str(rf_mode),
        "gaussian_fwhm_R": float(gaussian_fwhm_R),
        "lorentzian_fwhm_R": float(lorentzian_fwhm_R),
        "diffusion_scale": float(diffusion_scale),
        "p_values": p_out,
        "gamma_rf": gamma_out,
        "burn_steps": burn_steps_out,
        "n_steps": n_steps,
        "skipped": skipped,
        "ps": ps,
        "iplus": iplus,
        "iminus": iminus,
        "ps_m": ps_m,
        "iplus_m": iplus_m,
        "iminus_m": iminus_m,
    }


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

    bin_idx = resolve_bin_idx(args.bin_idx)
    if bin_idx is None:
        raise SystemExit(
            "Provide --bin-idx <int>, or set SLURM_ARRAY_TASK_ID, or pass --organize"
        )

    out = ssrf_shard_path(args.shard_dir, bin_idx)
    if args.skip_if_exists and out.is_file():
        print(f"Skipping existing shard {out}", flush=True)
        return

    shape = get_shape_params()
    print_shape_banner(shape, num_bins=int(args.num_bins))

    p_values = polarization_grid(args.p_min, args.p_max, args.p_step)
    g_values = gamma_rf_grid(args.gamma_min, args.gamma_max, args.gamma_step)
    s_values = burn_steps_grid(args.steps_min, args.steps_max, args.steps_step)
    n_combos = int(p_values.size * g_values.size * s_values.size)
    print(
        f"bin_idx={bin_idx}  n_P={p_values.size}  n_gamma={g_values.size}  "
        f"n_steps_grid={s_values.size}  n_combos={n_combos}  "
        f"P=[{args.p_min},{args.p_max}] step={args.p_step}  "
        f"gamma=[{args.gamma_min},{args.gamma_max}] step={args.gamma_step}  "
        f"burn_steps=[{args.steps_min},{args.steps_max}] step={args.steps_step}  "
        f"dt={args.dt}  rf_mode={args.rf_mode}  "
        f"Gauss={args.gauss_fwhm:.4f} Lorentz={args.lorentz_fwhm:.4f}  "
        f"diffusion={args.diffusion_scale}  mode=fixed_(gamma,n_steps)",
        flush=True,
    )
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
