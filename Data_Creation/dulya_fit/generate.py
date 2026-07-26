"""
Generate Dulya-fit *full-spectrum* pickle datasets (legacy random-sample mode).

For per-bin training shards (recommended, matches rate_eqs_test traj layout), use:
  python Data_Creation/dulya_fit/generate_bins.py --mode all --smoke --bin-idx 172
  python Data_Creation/dulya_fit/generate_bins.py --mode ssrf --bin-idx 172
  python Data_Creation/dulya_fit/generate_bins.py --mode unmanipulated

Equilibrium spectra use frozen single-site fit params in ``fit_params.json``,
varying polarization in [P_MIN, P_MAX]. Manipulation physics matches
``Data_Creation/generate_manipulated_test_data.py``.

Examples:
  python Data_Creation/dulya_fit/generate.py --mode all --num-samples 100
  python Data_Creation/dulya_fit/generate.py --mode unmanipulated --num-samples 1000
  python Data_Creation/dulya_fit/generate.py --mode ssrf --num-samples 500
  python Data_Creation/dulya_fit/generate.py --mode afp --num-samples 500
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import (  # noqa: E402
    BURN_BIN_CHOICES,
    DATA_DIR,
    DEFAULT_SAMPLE_COUNT,
    FIT_PARAMS_PATH,
    GAMMA_RF_MAX,
    GAMMA_RF_MIN,
    IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET,
    IPLUS_MIRROR_OVER_IMINUS_BURN_TARGET,
    MAX_BURN_STEPS,
    MAX_SSRF_AREA_RATIO_RETRIES,
    MIN_BURN_STEPS,
    MIRROR_OVER_BURN_AREA_TARGET,
    NUM_BINS,
    P_ABS_MIN,
    P_MAX,
    P_MIN,
    SEED,
)
from lineshape import GenerateDulyaLineshape, load_fit_params, shape_params_from_fit  # noqa: E402
from manipulate import (  # noqa: E402
    RF_HALF_WIDTH,
    RF_SIGMA_BINS,
    RF_VOIGT_GAMMA_BINS,
    make_afp_bins,
    run_event,
    sample_afp_center_bin,
    sample_polarization,
    ssrf_area_ratio_ok,
    ssrf_intensity_ratios_ok,
)
from ssrf_bin_traj import make_voigt_rf_profile, mirror_bin_idx  # noqa: E402

ModeName = Literal["unmanipulated", "ssrf", "afp", "all"]

_WORKER_SHAPE_PARAMS: dict[str, float] | None = None
_WORKER_MODE: str = "unmanipulated"
_WORKER_SEED: int = SEED
_WORKER_P_MIN: float = P_MIN
_WORKER_P_MAX: float = P_MAX
_WORKER_P_ABS_MIN: float = P_ABS_MIN


def _default_output(mode: str, num_samples: int) -> Path:
    return DATA_DIR / f"dulya_{mode}_{num_samples}.pkl"


def _default_example_plot(mode: str) -> Path:
    return DATA_DIR / f"example_dulya_{mode}.png"


def generate_unmanipulated_sample(
    sample_id: int,
    polarization: float,
    shape_params: dict[str, float],
) -> Dict[str, Any]:
    row = run_event(
        polarization=float(polarization),
        mode="none",
        shape_params=shape_params,
    )
    row["sample_id"] = int(sample_id)
    return row


def generate_ssrf_sample(
    sample_id: int,
    rng: np.random.Generator,
    shape_params: dict[str, float],
    *,
    p_min: float = P_MIN,
    p_max: float = P_MAX,
    p_abs_min: float = P_ABS_MIN,
) -> Dict[str, Any]:
    last_area_ratio = float("nan")
    last_im_over_ip = float("nan")
    last_ip_over_im = float("nan")
    for _ in range(MAX_SSRF_AREA_RATIO_RETRIES):
        polarization = sample_polarization(rng, p_min, p_max, p_abs_min)
        burn_bin = int(rng.choice(BURN_BIN_CHOICES))
        gamma_rf = float(rng.uniform(GAMMA_RF_MIN, GAMMA_RF_MAX))
        n_steps = int(rng.integers(MIN_BURN_STEPS, MAX_BURN_STEPS + 1))
        row = run_event(
            polarization=polarization,
            mode="ssrf",
            burn_bin=burn_bin,
            gamma_rf=gamma_rf,
            n_steps=n_steps,
            shape_params=shape_params,
        )
        last_area_ratio = float(row["mirror_over_burn_area"])
        last_im_over_ip = float(row["iminus_mirror_over_iplus_burn"])
        last_ip_over_im = float(row["iplus_mirror_over_iminus_burn"])
        if not ssrf_area_ratio_ok(last_area_ratio):
            continue
        if not ssrf_intensity_ratios_ok(last_im_over_ip, last_ip_over_im):
            continue
        row["sample_id"] = int(sample_id)
        return row

    raise RuntimeError(
        f"Failed to generate ssRF sample {sample_id} after "
        f"{MAX_SSRF_AREA_RATIO_RETRIES} tries "
        f"(area={last_area_ratio}, "
        f"I-_mir/I+_burn={last_im_over_ip}, "
        f"I+_mir/I-_burn={last_ip_over_im})"
    )


def generate_afp_sample(
    sample_id: int,
    rng: np.random.Generator,
    shape_params: dict[str, float],
    *,
    p_min: float = P_MIN,
    p_max: float = P_MAX,
    p_abs_min: float = P_ABS_MIN,
) -> Dict[str, Any]:
    polarization = sample_polarization(rng, p_min, p_max, p_abs_min)
    afp_center_bin = sample_afp_center_bin(NUM_BINS, rng)
    afp_bins = make_afp_bins(afp_center_bin, NUM_BINS)
    row = run_event(
        polarization=polarization,
        mode="afp",
        afp_center_bin=afp_center_bin,
        afp_bins=afp_bins,
        shape_params=shape_params,
    )
    row["sample_id"] = int(sample_id)
    return row


def _init_worker(
    mode: str,
    shape_params: dict[str, float],
    seed: int,
    p_min: float,
    p_max: float,
    p_abs_min: float,
) -> None:
    global _WORKER_MODE, _WORKER_SHAPE_PARAMS, _WORKER_SEED
    global _WORKER_P_MIN, _WORKER_P_MAX, _WORKER_P_ABS_MIN
    _WORKER_MODE = mode
    _WORKER_SHAPE_PARAMS = shape_params
    _WORKER_SEED = int(seed)
    _WORKER_P_MIN = float(p_min)
    _WORKER_P_MAX = float(p_max)
    _WORKER_P_ABS_MIN = float(p_abs_min)


def _generate_sample_task(sample_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(_WORKER_SEED + int(sample_id))
    shape = _WORKER_SHAPE_PARAMS
    assert shape is not None
    if _WORKER_MODE == "unmanipulated":
        p = sample_polarization(rng, _WORKER_P_MIN, _WORKER_P_MAX, _WORKER_P_ABS_MIN)
        return generate_unmanipulated_sample(sample_id, p, shape)
    if _WORKER_MODE == "ssrf":
        return generate_ssrf_sample(
            sample_id,
            rng,
            shape,
            p_min=_WORKER_P_MIN,
            p_max=_WORKER_P_MAX,
            p_abs_min=_WORKER_P_ABS_MIN,
        )
    if _WORKER_MODE == "afp":
        return generate_afp_sample(
            sample_id,
            rng,
            shape,
            p_min=_WORKER_P_MIN,
            p_max=_WORKER_P_MAX,
            p_abs_min=_WORKER_P_ABS_MIN,
        )
    raise ValueError(f"Unknown worker mode: {_WORKER_MODE}")


def generate_dataset(
    mode: Literal["unmanipulated", "ssrf", "afp"],
    num_samples: int,
    *,
    workers: int = 1,
    seed: int = SEED,
    shape_params: dict[str, float] | None = None,
    p_min: float = P_MIN,
    p_max: float = P_MAX,
    p_abs_min: float = P_ABS_MIN,
) -> pd.DataFrame:
    if mode == "ssrf" and BURN_BIN_CHOICES.size == 0:
        raise ValueError("No burn bins available in configured R range.")

    shape_params = shape_params or shape_params_from_fit()
    desc = f"Generating Dulya {mode} lineshapes"

    if workers <= 1:
        rng = np.random.default_rng(seed)
        rows: List[Dict[str, Any]] = []
        for sample_id in tqdm.tqdm(range(num_samples), desc=desc):
            if mode == "unmanipulated":
                p = sample_polarization(rng, p_min, p_max, p_abs_min)
                rows.append(generate_unmanipulated_sample(sample_id, p, shape_params))
            elif mode == "ssrf":
                rows.append(
                    generate_ssrf_sample(
                        sample_id,
                        rng,
                        shape_params,
                        p_min=p_min,
                        p_max=p_max,
                        p_abs_min=p_abs_min,
                    )
                )
            else:
                rows.append(
                    generate_afp_sample(
                        sample_id,
                        rng,
                        shape_params,
                        p_min=p_min,
                        p_max=p_max,
                        p_abs_min=p_abs_min,
                    )
                )
        return pd.DataFrame(rows)

    rows = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(mode, shape_params, seed, p_min, p_max, p_abs_min),
    ) as pool:
        for row in tqdm.tqdm(
            pool.map(_generate_sample_task, range(num_samples), chunksize=16),
            total=num_samples,
            desc=f"{desc} ({workers} workers)",
        ):
            rows.append(row)
    return pd.DataFrame(rows)


def _ssrf_support_and_mirrors(
    row: Dict[str, Any], n_bins: int
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    burn = int(row["burn_bin_idx"])
    mirror = int(row["mirror_bin_idx"])
    _, support = make_voigt_rf_profile(
        int(n_bins),
        burn,
        float(row["gamma_rf"]),
        sigma=float(RF_SIGMA_BINS),
        lorentz_gamma=float(RF_VOIGT_GAMMA_BINS),
        half_width=int(RF_HALF_WIDTH),
    )
    support = np.asarray(support, dtype=int)
    mirrors = np.asarray(
        [mirror_bin_idx(int(n_bins), int(i)) for i in support], dtype=int
    )
    return support, mirrors, burn, mirror


def _shade_burn_mirror(
    ax: Any,
    f: np.ndarray,
    support: np.ndarray,
    mirrors: np.ndarray,
    burn: int,
    mirror: int,
    *,
    label: bool = False,
) -> None:
    ax.axvspan(
        f[int(support.min())],
        f[int(support.max())],
        color="C3",
        alpha=0.15,
        label="burn support" if label else None,
    )
    ax.axvspan(
        f[int(mirrors.min())],
        f[int(mirrors.max())],
        color="C2",
        alpha=0.15,
        label="mirror support" if label else None,
    )
    ax.axvline(f[burn], color="C3", ls="--", lw=1.0)
    ax.axvline(f[mirror], color="C2", ls="--", lw=1.0)


def plot_random_example(
    df: pd.DataFrame,
    output_path: Path,
    *,
    iplus_iminus_path: Path | None = None,
    rng: np.random.Generator | None = None,
    shape_params: dict[str, float] | None = None,
) -> Tuple[Path, Path | None]:
    if len(df) == 0:
        raise ValueError("Cannot plot example from empty dataframe.")
    rng = rng or np.random.default_rng(SEED)
    shape_params = shape_params or shape_params_from_fit()
    row = df.iloc[int(rng.integers(0, len(df)))].to_dict()

    f = np.asarray(row["frequency"], dtype=float)
    ps = np.asarray(row["Ps"], dtype=float)
    qs = np.asarray(row["Qs"], dtype=float)
    ip = np.asarray(row["Iplus"], dtype=float)
    im = np.asarray(row["Iminus"], dtype=float)

    _, ip0, im0 = GenerateDulyaLineshape(float(row["P_initial"]), f, shape_params)
    ip0 = np.asarray(ip0, dtype=float)
    im0 = np.asarray(im0, dtype=float)
    ps0 = ip0 + im0
    qs0 = ip0 - im0
    dip = ip - ip0
    dim = im - im0

    support = mirrors = None
    burn = mirror = -1
    if bool(row.get("ssrf_applied", False)):
        support, mirrors, burn, mirror = _ssrf_support_and_mirrors(row, len(f))

    title_bits = [
        f"dulya example  mode={row['manipulation_mode']}",
        f"P={float(row['P_initial']):.3f}",
        f"sample_id={int(row['sample_id'])}",
    ]
    if bool(row.get("ssrf_applied", False)):
        title_bits.extend(
            [
                f"burn R={float(row['burn_freq']):.3f}",
                f"gamma_rf={float(row['gamma_rf']):.2f}",
                f"steps={int(row['burn_step'])}",
            ]
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(f, ps0, color="0.65", lw=1.2, label="Ps unmanipulated")
    ax.plot(f, ps, color="C0", lw=1.8, label="Ps final")
    ax.plot(f, qs0, color="0.65", lw=1.2, label="Qs unmanipulated")
    ax.plot(f, qs, color="C1", lw=1.8, label="Qs final")
    ax.set_xlabel("R")
    ax.set_ylabel("Ps / Qs")

    if support is not None and mirrors is not None:
        _shade_burn_mirror(ax, f, support, mirrors, burn, mirror, label=True)
        ax.set_title(
            "  ".join(title_bits)
            + "\n"
            + f"mirror/burn area={float(row['mirror_over_burn_area']):.4f}"
        )
    else:
        if bool(row.get("afp_applied", False)) and int(row.get("afp_bin_start", -1)) >= 0:
            lo = int(row["afp_bin_start"])
            hi = max(lo, int(row["afp_bin_stop"]) - 1)
            ax.axvspan(f[lo], f[hi], color="C5", alpha=0.15, label="AFP window")
        ax.set_title("  ".join(title_bits))

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    if iplus_iminus_path is None:
        iplus_iminus_path = output_path.with_name(
            f"{output_path.stem}_iplus_iminus{output_path.suffix}"
        )
    iplus_iminus_path = Path(iplus_iminus_path)
    fig2, ax = plt.subplots(figsize=(12, 8))
    ax.plot(f, ip0, color="0.65", lw=1.2, label="I+ unmanipulated")
    ax.plot(f, ip, color="C0", lw=1.8, label="I+ final")
    ax.plot(f, dip, color="C3", lw=1.2, ls=":", label="dI+")
    ax.plot(f, im0, color="0.65", lw=1.2, label="I- unmanipulated")
    ax.plot(f, im, color="C4", lw=1.8, label="I- final")
    ax.plot(f, dim, color="C2", lw=1.2, ls=":", label="dI-")
    ax.set_xlabel("R")
    ax.set_ylabel("I± / dI±")
    ax.axhline(0.0, color="0.5", lw=0.8)

    if support is not None and mirrors is not None:
        _shade_burn_mirror(ax, f, support, mirrors, burn, mirror, label=True)
        ax.set_title(
            "  ".join(title_bits)
            + "\n"
            + (
                f"I-_mirror / I+_burn={float(row['iminus_mirror_over_iplus_burn']):.4f}  "
                f"I+_mirror / I-_burn={float(row['iplus_mirror_over_iminus_burn']):.4f}  "
                f"(target={IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET})"
            )
        )
    else:
        ax.set_title("  ".join(title_bits) + "\nI+/- comparison")

    ax.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    iplus_iminus_path.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(iplus_iminus_path, dpi=160)
    plt.close(fig2)
    return output_path, iplus_iminus_path


def _print_summary(df: pd.DataFrame, mode: str) -> None:
    print(f"[{mode}] Saved {len(df)} samples")
    print(
        "P_initial:",
        f"min={df['P_initial'].min():.3f}, max={df['P_initial'].max():.3f}, "
        f"mean={df['P_initial'].mean():.3f}",
    )
    print(
        "true_P:",
        f"min={df['true_P'].min():.3f}, max={df['true_P'].max():.3f}, "
        f"mean={df['true_P'].mean():.3f}",
    )
    if mode == "ssrf":
        ratios = df["mirror_over_burn_area"].astype(float)
        print(
            "mirror_over_burn_area:",
            f"median={ratios.median():.4f}, mean={ratios.mean():.4f} "
            f"(target={MIRROR_OVER_BURN_AREA_TARGET})",
        )
        im_over_ip = df["iminus_mirror_over_iplus_burn"].astype(float)
        ip_over_im = df["iplus_mirror_over_iminus_burn"].astype(float)
        print(
            "I-_mir/I+_burn:",
            f"median={im_over_ip.median():.4f} (target={IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET})",
        )
        print(
            "I+_mir/I-_burn:",
            f"median={ip_over_im.median():.4f} (target={IPLUS_MIRROR_OVER_IMINUS_BURN_TARGET})",
        )


def run_mode(
    mode: Literal["unmanipulated", "ssrf", "afp"],
    *,
    num_samples: int,
    workers: int,
    output: Path | None,
    example_plot: Path | None,
    no_example_plot: bool,
    seed: int,
    shape_params: dict[str, float],
) -> Path:
    out = Path(output) if output is not None else _default_output(mode, num_samples)
    print(
        f"Generating {num_samples} Dulya-fit {mode} lineshapes "
        f"(P in [{P_MIN}, {P_MAX}], |P| >= {P_ABS_MIN})"
    )
    print(f"Fit params: {FIT_PARAMS_PATH}")
    print(f"Workers: {workers}")

    df = generate_dataset(
        mode,
        num_samples,
        workers=workers,
        seed=seed,
        shape_params=shape_params,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out)
    print(f"Wrote {out}")
    _print_summary(df, mode)

    if not no_example_plot:
        plot_path = (
            Path(example_plot) if example_plot is not None else _default_example_plot(mode)
        )
        iplus_path = plot_path.with_name(f"{plot_path.stem}_iplus_iminus{plot_path.suffix}")
        saved, saved_ip = plot_random_example(
            df, plot_path, iplus_iminus_path=iplus_path, shape_params=shape_params
        )
        print(f"Saved example plot: {saved}")
        if saved_ip is not None:
            print(f"Saved I+/- plot: {saved_ip}")
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Dulya-fit unmanipulated / ssRF / AFP lineshape data."
    )
    parser.add_argument(
        "--mode",
        choices=("unmanipulated", "ssrf", "afp", "all"),
        default="all",
        help="Which dataset(s) to generate (default: all)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Samples per mode (default: {DEFAULT_SAMPLE_COUNT})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel worker processes (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output pickle path (single-mode only; default under dulya_fit/data/)",
    )
    parser.add_argument(
        "--example-plot",
        type=Path,
        default=None,
        help="Example plot path (single-mode only)",
    )
    parser.add_argument(
        "--no-example-plot",
        action="store_true",
        help="Skip example plots",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"RNG seed (default: {SEED})",
    )
    parser.add_argument(
        "--fit-params",
        type=Path,
        default=FIT_PARAMS_PATH,
        help=f"Path to fit_params.json (default: {FIT_PARAMS_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fit_blob = load_fit_params(args.fit_params)
    shape_params = shape_params_from_fit(fit_blob)
    print(
        "Frozen shape params:",
        ", ".join(f"{k}={v:.6g}" for k, v in shape_params.items()),
    )

    modes: Tuple[Literal["unmanipulated", "ssrf", "afp"], ...]
    if args.mode == "all":
        modes = ("unmanipulated", "ssrf", "afp")
        if args.output is not None or args.example_plot is not None:
            raise SystemExit(
                "--output / --example-plot only apply to a single --mode "
                "(not --mode all)."
            )
    else:
        modes = (args.mode,)  # type: ignore[assignment]

    for mode in modes:
        # Offset seeds so the three modes are not identical polarization draws.
        mode_seed = int(args.seed) + {
            "unmanipulated": 0,
            "ssrf": 10_000,
            "afp": 20_000,
        }[mode]
        run_mode(
            mode,
            num_samples=int(args.num_samples),
            workers=int(args.workers),
            output=args.output if args.mode != "all" else None,
            example_plot=args.example_plot if args.mode != "all" else None,
            no_example_plot=bool(args.no_example_plot),
            seed=mode_seed,
            shape_params=shape_params,
        )


if __name__ == "__main__":
    main()
