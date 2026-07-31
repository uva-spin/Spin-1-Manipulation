"""
Per-bin AFP + relaxation trajectories for SLURM array jobs (v2 physics).

Each task applies an AFP window centered on one R-bin, then relaxes toward
Boltzmann equilibrium at the post-AFP vector polarization (Q → Q_boltz(P)),
matching ssrf_realtime AFP recovery. Saves trajectories at fit scale for each
initial polarization.

Usage:
  python afp_bin_traj.py --bin-idx 172
  python afp_bin_traj.py --organize --shard-dir data/afp_shards --output-dir data/afp_train
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from bin_io import (
    afp_shard_path,
    afp_spectrum_shard_path,
    organize_afp_shards,
    save_afp_shard,
    save_afp_spectrum_shard,
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
    AFP_CENTER_MARGIN,
    AFP_EFFICIENCY,
    AFP_N_RELAX,
    AFP_SHARD_DIR,
    AFP_STEP_SUBSAMPLE,
    AFP_TRAIN_DIR,
    AFP_WINDOW,
    DIFFUSION_SCALE,
    DT,
    F_MAX,
    F_MIN,
    NUM_BINS,
    P_MAX,
    P_MIN,
    P_STEP,
    SEED,
    SPECTRUM_AFP_SHARD_DIR,
    SOURCE_AFP,
    SOURCE_UNMANIP,
    UNMANIP_TRAIN_FRACTION,
    effective_afp_step_subsample,
    is_burn_bin,
)
from model_bridge import (
    afp_touched_bins,
    afp_window_indices,
    build_spin1_model,
    commit_touched_bins_only,
    configure_afp_recovery,
    full_spectrum_intensities,
    intensities_at_bins,
    level_pq,
    mirror_bin_idx,
    restore_touched_intensity_area,
)
from ssrf_bin_traj import run_unmanipulated_polarization

R_MIN = F_MIN
R_MAX = F_MAX
N_RELAX = AFP_N_RELAX

DEFAULT_SHARD_DIR = AFP_SHARD_DIR
DEFAULT_TRAIN_DIR = AFP_TRAIN_DIR
DEFAULT_SPECTRUM_SHARD_DIR = SPECTRUM_AFP_SHARD_DIR


def run_one_polarization(
    bin_idx: int,
    polarization: float,
    *,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    n_relax: int = N_RELAX,
    afp_window: int = AFP_WINDOW,
    afp_efficiency: float = AFP_EFFICIENCY,
    diffusion_scale: float = DIFFUSION_SCALE,
    shape_params: dict[str, float] | None = None,
    capture_spectrum: bool = False,
) -> dict:
    P = float(polarization)
    shape = shape_params if shape_params is not None else get_shape_params()
    f = np.linspace(float(F_MIN), float(F_MAX), int(num_bins))
    _, ip_fit, im_fit = equilibrium_lineshape(P, f, shape)
    ip_fit = np.asarray(ip_fit, dtype=float)
    im_fit = np.asarray(im_fit, dtype=float)
    to_spin1, from_spin1 = spin1_scale_factors(P, ip_fit, im_fit)
    iplus0 = ip_fit * to_spin1
    iminus0 = im_fit * to_spin1
    mirror_idx = mirror_bin_idx(int(num_bins), bin_idx)
    subset = afp_window_indices(bin_idx, int(num_bins), window=int(afp_window))
    touched = afp_touched_bins(int(num_bins), subset)
    area0 = float(np.sum(iplus0 + iminus0))

    model = build_spin1_model(
        iplus0,
        iminus0,
        polarization=P,
        num_bins=num_bins,
        dt=dt,
        rf_enabled=False,
        relax_enabled=True,
        diffusion_scale=diffusion_scale,
    )
    model.params.afp_enabled = True
    model.params.afp_efficiency = float(afp_efficiency)
    model.params.afp_center_margin = int(AFP_CENTER_MARGIN)
    model.params.afp_preserve_intensity_area = True
    model.params.afp_subset_indices = [int(i) for i in subset]
    model._afp_pending = True

    p_initial, q_initial = level_pq(model)
    ip_spec0 = im_spec0 = None
    if capture_spectrum:
        ip_spec0, im_spec0, _ = full_spectrum_intensities(model)

    model.afp_sweep()
    model.params.afp_enabled = False
    model._afp_pending = False
    ip_sim, im_sim, _ = model.physical_intensities()
    base_ip, base_im = commit_touched_bins_only(
        iplus0, iminus0, ip_sim, im_sim, touched
    )
    base_ip, base_im = restore_touched_intensity_area(
        base_ip, base_im, touched, area0
    )
    model.load_from_physical_intensities(base_ip, base_im)
    # Boltzmann at manipulated (post-AFP) vector P: Q → Q_boltz(P_AFP).
    configure_afp_recovery(model)

    t_len = int(n_relax) + 1
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
        ip, im, ps0, ip_m, im_m, ps_m0 = intensities_at_bins(model, bin_idx, mirror_idx)
        iplus[k], iminus[k], ps[k] = ip, im, ps0
        iplus_m[k], iminus_m[k], ps_m[k] = ip_m, im_m, ps_m0
        if capture_spectrum and ps_full is not None:
            ip_s, im_s, ps_s = full_spectrum_intensities(model)
            iplus_full[k] = ip_s
            iminus_full[k] = im_s
            ps_full[k] = ps_s

    _record_step(0)

    for k in range(1, t_len):
        model.step_once(dt=float(dt), rf_on=False, dnp_on=False, copy=False)
        _record_step(k)

    scale = float(from_spin1)
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
            ip_end, im_end, _ = full_spectrum_intensities(model)
            ip_spec = ip_end * scale
            im_spec = im_end * scale
    p_final, q_final = level_pq(model)

    return {
        "polarization": float(polarization),
        "skipped": False,
        "n_steps": t_len,
        "ps": ps * scale,
        "iplus": iplus * scale,
        "iminus": iminus * scale,
        "ps_m": ps_m * scale,
        "iplus_m": iplus_m * scale,
        "iminus_m": iminus_m * scale,
        "afp_subset": subset,
        "ip_spectrum0": ip_spec0,
        "im_spectrum0": im_spec0,
        "ip_spectrum": ip_spec,
        "im_spectrum": im_spec,
        "ps_full": ps_full,
        "iplus_full": iplus_full,
        "iminus_full": iminus_full,
        "frequency": f,
        "diffusion_scale": float(model.params.diffusion_scale),
        "p_initial": float(p_initial),
        "q_initial": float(q_initial),
        "p_final": float(p_final),
        "q_final": float(q_final),
        "center_bin": int(bin_idx),
    }


def run_one_bin(
    bin_idx: int,
    *,
    p_values: np.ndarray,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    n_relax: int = N_RELAX,
    afp_window: int = AFP_WINDOW,
    afp_efficiency: float = AFP_EFFICIENCY,
    capture_spectrum: bool = False,
    step_subsample: int = 1,
) -> dict:
    bin_idx = int(bin_idx)
    if bin_idx < 0 or bin_idx >= int(num_bins):
        raise ValueError(f"bin_idx={bin_idx} out of range for num_bins={num_bins}")

    mirror_idx = mirror_bin_idx(int(num_bins), bin_idx)
    p_values = np.asarray(p_values, dtype=float)
    n_p = int(p_values.size)
    t_len = int(n_relax) + 1
    subset = afp_window_indices(bin_idx, int(num_bins), window=int(afp_window))

    n_steps = np.full(n_p, t_len, dtype=np.int32)
    skipped = np.zeros(n_p, dtype=bool)
    ps = np.full((n_p, t_len), np.nan, dtype=float)
    iplus = np.full((n_p, t_len), np.nan, dtype=float)
    iminus = np.full((n_p, t_len), np.nan, dtype=float)
    ps_m = np.full((n_p, t_len), np.nan, dtype=float)
    iplus_m = np.full((n_p, t_len), np.nan, dtype=float)
    iminus_m = np.full((n_p, t_len), np.nan, dtype=float)
    ps_full = iplus_full = iminus_full = None
    if capture_spectrum:
        ps_full = np.full((n_p, t_len, int(num_bins)), np.nan, dtype=float)
        iplus_full = np.full((n_p, t_len, int(num_bins)), np.nan, dtype=float)
        iminus_full = np.full((n_p, t_len, int(num_bins)), np.nan, dtype=float)

    for j, p0 in enumerate(p_values):
        print(f"  P={p0:+.3f} ({j + 1}/{n_p})", flush=True)
        traj = run_one_polarization(
            bin_idx,
            float(p0),
            num_bins=num_bins,
            dt=dt,
            n_relax=n_relax,
            afp_window=afp_window,
            afp_efficiency=afp_efficiency,
            capture_spectrum=capture_spectrum,
        )
        skipped[j] = bool(traj["skipped"])
        n = int(traj["n_steps"])
        n_steps[j] = n
        ps[j, :n] = traj["ps"]
        iplus[j, :n] = traj["iplus"]
        iminus[j, :n] = traj["iminus"]
        ps_m[j, :n] = traj["ps_m"]
        iplus_m[j, :n] = traj["iplus_m"]
        iminus_m[j, :n] = traj["iminus_m"]
        if capture_spectrum and ps_full is not None and traj.get("ps_full") is not None:
            ps_full[j, :n] = np.asarray(traj["ps_full"], dtype=float)[:n]
            iplus_full[j, :n] = np.asarray(traj["iplus_full"], dtype=float)[:n]
            iminus_full[j, :n] = np.asarray(traj["iminus_full"], dtype=float)[:n]

    f = np.linspace(R_MIN, R_MAX, int(num_bins))
    out = {
        "bin_idx": bin_idx,
        "mirror_idx": mirror_idx,
        "R": float(f[bin_idx]),
        "num_bins": int(num_bins),
        "dt": float(dt),
        "n_relax": int(n_relax),
        "afp_window": int(afp_window),
        "afp_efficiency": float(afp_efficiency),
        "afp_subset": np.asarray(subset, dtype=np.int32),
        "p_values": p_values,
        "n_steps": n_steps,
        "skipped": skipped,
        "ps": ps,
        "iplus": iplus,
        "iminus": iminus,
        "ps_m": ps_m,
        "iplus_m": iplus_m,
        "iminus_m": iminus_m,
    }
    if capture_spectrum:
        out["ps_full"] = ps_full
        out["iplus_full"] = iplus_full
        out["iminus_full"] = iminus_full
        out["step_subsample"] = int(step_subsample)
    return out


def run_one_bin_spectrum(
    bin_idx: int,
    *,
    p_values: np.ndarray,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    n_relax: int = N_RELAX,
    afp_window: int = AFP_WINDOW,
    afp_efficiency: float = AFP_EFFICIENCY,
    step_subsample: int = AFP_STEP_SUBSAMPLE,
    unmanip_fraction: float = UNMANIP_TRAIN_FRACTION,
    seed: int = SEED,
) -> dict:
    """Full-spectrum AFP trajectories with step subsampling metadata."""
    rng = random.Random(int(seed))
    effective_sub = effective_afp_step_subsample(int(n_relax), int(step_subsample))
    base = run_one_bin(
        bin_idx,
        p_values=p_values,
        num_bins=num_bins,
        dt=dt,
        n_relax=n_relax,
        afp_window=afp_window,
        afp_efficiency=afp_efficiency,
        capture_spectrum=True,
        step_subsample=effective_sub,
    )
    n_base = int(base["p_values"].size)
    n_unmanip = 0
    if float(unmanip_fraction) > 0.0 and n_base > 0:
        n_unmanip = max(1, int(round(float(unmanip_fraction) * n_base / max(1e-12, 1.0 - float(unmanip_fraction)))))
        for _ in range(n_unmanip):
            p0 = float(rng.choice(np.asarray(p_values, dtype=float)))
            traj = run_unmanipulated_polarization(p0, num_bins=num_bins)
            old_n = int(base["p_values"].size)
            t_max_old = int(base["ps"].shape[1])
            n = 1
            t_max_new = max(t_max_old, n)
            num_b = int(base["num_bins"])

            def _pad2(arr: np.ndarray) -> np.ndarray:
                out = np.full((old_n + 1, t_max_new), np.nan, dtype=float)
                out[:old_n, : arr.shape[1]] = arr
                return out

            def _pad3(arr: np.ndarray) -> np.ndarray:
                out = np.full((old_n + 1, t_max_new, num_b), np.nan, dtype=float)
                out[:old_n, : arr.shape[1]] = arr
                return out

            base["p_values"] = np.concatenate([base["p_values"], [p0]])
            base["n_steps"] = np.concatenate([base["n_steps"], [n]])
            base["skipped"] = np.concatenate([base["skipped"], [False]])
            for key in ("ps", "iplus", "iminus", "ps_m", "iplus_m", "iminus_m"):
                base[key] = _pad2(base[key])
                base[key][old_n, :n] = np.asarray(traj[key], dtype=float)
            for key in ("ps_full", "iplus_full", "iminus_full"):
                if key in base:
                    padded = _pad3(base[key])
                    padded[old_n, :n] = np.asarray(traj[key], dtype=float)[:n]
                    base[key] = padded

    base["dataset"] = "afp_spectrum_bin_v2"
    base["n_unmanip_samples"] = n_unmanip
    base["source_unmanip"] = SOURCE_UNMANIP
    base["source_afp"] = SOURCE_AFP
    return base


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-bin AFP + relaxation trajectory worker / organizer (v2)"
    )
    p.add_argument("--bin-idx", type=int, default=None)
    p.add_argument(
        "--organize",
        "--combine",
        dest="organize",
        action="store_true",
    )
    p.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--num-bins", type=int, default=NUM_BINS)
    p.add_argument("--p-min", type=float, default=P_MIN)
    p.add_argument("--p-max", type=float, default=P_MAX)
    p.add_argument("--p-step", type=float, default=P_STEP)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--n-relax", type=int, default=N_RELAX)
    p.add_argument("--afp-window", type=int, default=AFP_WINDOW)
    p.add_argument("--afp-efficiency", type=float, default=AFP_EFFICIENCY)
    p.add_argument(
        "--spectrum-mode",
        action="store_true",
        help="Store full 500-bin spectra at each timestep",
    )
    p.add_argument(
        "--step-subsample",
        type=int,
        default=AFP_STEP_SUBSAMPLE,
        help=(
            "Keep every Nth AFP relax step when combining (spectrum mode). "
            "Forced to 1 when --n-relax is 0 (instant flip only)."
        ),
    )
    p.add_argument(
        "--unmanip-fraction",
        type=float,
        default=UNMANIP_TRAIN_FRACTION,
        help=(
            "Fraction of unmanipulated equilibrium samples injected into spectrum "
            "shards (default 0; prefer combine_spectrum_train --unmanip-dir)"
        ),
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-if-exists", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.organize:
        result = organize_afp_shards(
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
        afp_spectrum_shard_path(args.shard_dir, bin_idx)
        if args.spectrum_mode
        else afp_shard_path(args.shard_dir, bin_idx)
    )
    if args.skip_if_exists and out.is_file():
        print(f"Skipping existing shard {out}", flush=True)
        return

    if args.spectrum_mode and not is_burn_bin(bin_idx):
        print(
            f"Skipping bin_idx={bin_idx}: outside burn window "
            f"(not in BURN_BIN_CHOICES, R not in burn range)",
            flush=True,
        )
        return

    shape = get_shape_params()
    print_shape_banner(shape, num_bins=int(args.num_bins))

    p_values = polarization_grid(args.p_min, args.p_max, args.p_step)
    effective_sub = effective_afp_step_subsample(int(args.n_relax), int(args.step_subsample))
    print(
        f"bin_idx={bin_idx}  n_P={p_values.size}  spectrum_mode={bool(args.spectrum_mode)}  "
        f"step_subsample={effective_sub} (requested={int(args.step_subsample)})  "
        f"unmanip_fraction={float(args.unmanip_fraction):.3f}  "
        f"P=[{args.p_min},{args.p_max}] step={args.p_step}  "
        f"dt={args.dt}  n_relax={args.n_relax}  "
        f"afp_window={args.afp_window}  eff={args.afp_efficiency}",
        flush=True,
    )
    if args.spectrum_mode:
        result = run_one_bin_spectrum(
            bin_idx,
            p_values=p_values,
            num_bins=args.num_bins,
            dt=args.dt,
            n_relax=args.n_relax,
            afp_window=args.afp_window,
            afp_efficiency=args.afp_efficiency,
            step_subsample=effective_sub,
            unmanip_fraction=float(args.unmanip_fraction),
            seed=int(args.seed),
        )
        save_afp_spectrum_shard(result, out, extra_meta=shape_meta(shape))
    else:
        result = run_one_bin(
            bin_idx,
            p_values=p_values,
            num_bins=args.num_bins,
            dt=args.dt,
            n_relax=args.n_relax,
            afp_window=args.afp_window,
            afp_efficiency=args.afp_efficiency,
        )
        save_afp_shard(result, out, extra_meta=shape_meta(shape))
    print(
        f"Wrote {out}  mirror={result['mirror_idx']}  "
        f"afp_subset={list(result['afp_subset'])}  "
        f"steps={int(result['n_steps'][0])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
