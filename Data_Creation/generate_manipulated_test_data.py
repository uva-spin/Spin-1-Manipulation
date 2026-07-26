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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physics.lineshape.Lineshape import GenerateVectorLineshape  # noqa: E402
from physics.lineshape.rate_eqs_test.afp_bin_traj import (  # noqa: E402
    afp_touched_bins,
    afp_window_indices,
    commit_touched_bins_only,
    restore_touched_intensity_area,
)
from physics.lineshape.rate_eqs_test.ssrf_afp import (  # noqa: E402
    commit_ssrf_bins_only,
    mirror_bin_idx,
    resolve_burn_bin,
    ssrf_touched_bins,
)
from physics.lineshape.rate_eqs_test.ssrf_bin_traj import (  # noqa: E402
    HALF_WIDTH,
    SIGMA_BINS,
    VOIGT_GAMMA_BINS,
    freeze_rf_profile,
    make_voigt_rf_profile,
)
from physics.ssrf_realtime.model import Spin1Model, Spin1Params  # noqa: E402

ManipulationMode = Literal["ssrf", "afp", "both"]

NUM_SAMPLES = 10_000
SEED = 42
OUTPUT_PATH = REPO_ROOT / "data" / "manipulated_test_10000.pkl"
EXAMPLE_PLOT_PATH = REPO_ROOT / "data" / "example_manipulated_lineshape.png"
EXAMPLE_IPLUS_IMINUS_PLOT_PATH = (
    REPO_ROOT / "data" / "example_manipulated_lineshape_iplus_iminus.png"
)

NUM_BINS = 500
F_MIN = -3.0
F_MAX = 3.0
FREQUENCY = np.linspace(F_MIN, F_MAX, NUM_BINS, dtype=np.float32)

P_MIN = 0.25
P_MAX = 0.50

BURN_R_MIN = -2.0
BURN_R_MAX = 2.0
GAMMA_RF_MIN = 1.0
GAMMA_RF_MAX = 2.0
MIN_BURN_STEPS = 20
MAX_BURN_STEPS = 100
DT = 0.005

# ssRF burn∪mirror intensity-change areas: |ΔA_mirror| / |ΔA_burn| ≈ 1/2.
MIRROR_OVER_BURN_AREA_TARGET = 0.5
MIRROR_OVER_BURN_AREA_RTOL = 0.10
# Manipulated I±: |ΔI-_mirror| / |ΔI+_burn| ≈ 1/2 and |ΔI+_mirror| / |ΔI-_burn| ≈ 1/2.
IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET = 0.5
IPLUS_MIRROR_OVER_IMINUS_BURN_TARGET = 0.5
SSRF_INTENSITY_RATIO_RTOL = 0.10
MAX_SSRF_AREA_RATIO_RETRIES = 32

# Match afp_bin_traj.py / combined_train AFP shards.
AFP_WINDOW = 8
AFP_EFFICIENCY = 1.0
AFP_CENTER_MARGIN = 0

RF_SIGMA_BINS = SIGMA_BINS
RF_VOIGT_GAMMA_BINS = VOIGT_GAMMA_BINS
# Compact Voigt support: burn only within center ± RF_HALF_WIDTH (smoothed edges).
RF_HALF_WIDTH = HALF_WIDTH

MANIPULATION_MODE_WEIGHTS = (1.00, 0.00, 0.00)
STORE_DTYPE = np.float32

_BURN_BIN_CHOICES = np.flatnonzero(
    (FREQUENCY > BURN_R_MIN) & (FREQUENCY < BURN_R_MAX)
).astype(int)


def _sample_afp_center_bin(n_bins: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, int(n_bins)))


def apply_afp_to_lineshape(
    iplus: np.ndarray,
    iminus: np.ndarray,
    afp_subset: List[int],
    *,
    polarization: float,
    num_bins: int,
    afp_efficiency: float,
    afp_center_margin: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous AFP on ``afp_subset`` with training commit/restore (no relaxation)."""
    iplus0 = np.asarray(iplus, dtype=float).copy()
    iminus0 = np.asarray(iminus, dtype=float).copy()
    subset = [int(i) for i in afp_subset]
    touched = afp_touched_bins(int(num_bins), subset)
    area0 = float(np.sum(iplus0 + iminus0))

    params = Spin1Params(
        n_bins=int(num_bins),
        r_min=F_MIN,
        r_max=F_MAX,
        p0=float(polarization),
        initial_polarization=float(polarization),
        q0=0.0,
        p_dnp_sat=float(polarization),
        dnp_enabled=False,
        rf_enabled=False,
        relax_enabled=False,
        afp_enabled=True,
        afp_efficiency=float(afp_efficiency),
        afp_center_margin=int(afp_center_margin),
        afp_preserve_intensity_area=True,
        afp_subset_indices=subset,
        gamma_rf=0.0,
    )
    model = Spin1Model(params, initial_polarization=float(polarization))
    model.load_from_physical_intensities(iplus0, iminus0)
    model.params.afp_enabled = True
    model.params.afp_preserve_intensity_area = True
    model.params.afp_subset_indices = subset
    model._afp_pending = True

    model.afp_sweep()
    ip_sim, im_sim, _ = model.physical_intensities()
    out_ip, out_im = commit_touched_bins_only(
        iplus0, iminus0, ip_sim, im_sim, touched
    )
    return restore_touched_intensity_area(out_ip, out_im, touched, area0)


def _ssrf_burn_mirror_region_metrics(
    baseline_ip: np.ndarray,
    baseline_im: np.ndarray,
    burned_ip: np.ndarray,
    burned_im: np.ndarray,
    burn_subset: List[int] | np.ndarray,
    num_bins: int,
) -> Dict[str, float]:
    """
    Integrated |ΔPs| / |ΔI±| over the compact Voigt burn support and mirrors only.

    Burn metrics use RF-active burn bins; mirror metrics use mirrored counterparts.
    Overlapping bins (near R≈0) are excluded from both so each region is disjoint.

    Expected:
      mirror_area / burn_area ≈ 1/2
      |ΔI-_mirror| / |ΔI+_burn| ≈ 1/2
      |ΔI+_mirror| / |ΔI-_burn| ≈ 1/2
    """
    nan = float("nan")
    empty = {
        "burn_region_area": 0.0,
        "mirror_region_area": 0.0,
        "mirror_over_burn_area": nan,
        "iminus_mirror_over_iplus_burn": nan,
        "iplus_mirror_over_iminus_burn": nan,
    }
    baseline_ip = np.asarray(baseline_ip, dtype=float)
    baseline_im = np.asarray(baseline_im, dtype=float)
    burned_ip = np.asarray(burned_ip, dtype=float)
    burned_im = np.asarray(burned_im, dtype=float)
    dps = (burned_ip + burned_im) - (baseline_ip + baseline_im)
    dip = burned_ip - baseline_ip
    dim = burned_im - baseline_im

    burn_set = {int(i) for i in burn_subset}
    mirror_set = {mirror_bin_idx(int(num_bins), int(i)) for i in burn_set}
    burn_only = sorted(burn_set - mirror_set)
    mirror_only = sorted(mirror_set - burn_set)
    if not burn_only or not mirror_only:
        return empty

    burn_area = float(np.sum(np.abs(dps[burn_only])))
    mirror_area = float(np.sum(np.abs(dps[mirror_only])))
    iplus_burn = float(np.sum(np.abs(dip[burn_only])))
    iminus_burn = float(np.sum(np.abs(dim[burn_only])))
    iplus_mirror = float(np.sum(np.abs(dip[mirror_only])))
    iminus_mirror = float(np.sum(np.abs(dim[mirror_only])))

    return {
        "burn_region_area": burn_area,
        "mirror_region_area": mirror_area,
        "mirror_over_burn_area": (
            float(mirror_area / burn_area) if burn_area > 1e-30 else nan
        ),
        "iminus_mirror_over_iplus_burn": (
            float(iminus_mirror / iplus_burn) if iplus_burn > 1e-30 else nan
        ),
        "iplus_mirror_over_iminus_burn": (
            float(iplus_mirror / iminus_burn) if iminus_burn > 1e-30 else nan
        ),
    }


def _ratio_near_target(ratio: float, target: float, rtol: float) -> bool:
    if not np.isfinite(ratio):
        return False
    return abs(float(ratio) - float(target)) <= float(rtol) * abs(float(target))


def _ssrf_area_ratio_ok(ratio: float) -> bool:
    return _ratio_near_target(
        ratio, MIRROR_OVER_BURN_AREA_TARGET, MIRROR_OVER_BURN_AREA_RTOL
    )


def _ssrf_intensity_ratios_ok(
    iminus_mirror_over_iplus_burn: float,
    iplus_mirror_over_iminus_burn: float,
) -> bool:
    return _ratio_near_target(
        iminus_mirror_over_iplus_burn,
        IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET,
        SSRF_INTENSITY_RATIO_RTOL,
    ) and _ratio_near_target(
        iplus_mirror_over_iminus_burn,
        IPLUS_MIRROR_OVER_IMINUS_BURN_TARGET,
        SSRF_INTENSITY_RATIO_RTOL,
    )


def _run_ssrf_burn(
    *,
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_idx: int,
    gamma_rf: float,
    n_steps: int,
    num_bins: int,
    dt: float,
    polarization: float,
) -> Tuple[np.ndarray, np.ndarray, int, List[int], Dict[str, float]]:
    """ssRF Voigt burn + relaxation on touched bins; returns committed spectrum."""
    f = np.asarray(FREQUENCY[:num_bins], dtype=float)
    profile, ssrf_subset = make_voigt_rf_profile(
        num_bins,
        int(burn_idx),
        float(gamma_rf),
        sigma=float(RF_SIGMA_BINS),
        lorentz_gamma=float(RF_VOIGT_GAMMA_BINS),
        half_width=int(RF_HALF_WIDTH),
    )
    ssrf_subset = [int(i) for i in ssrf_subset]
    # Profile must be zero outside the burned support (compact Voigt window).
    compact_profile = np.zeros(num_bins, dtype=float)
    compact_profile[ssrf_subset] = np.asarray(profile, dtype=float)[ssrf_subset]
    profile = compact_profile
    touched = ssrf_touched_bins(num_bins, ssrf_subset)
    baseline_ip = np.asarray(iplus, dtype=float).copy()
    baseline_im = np.asarray(iminus, dtype=float).copy()

    params = Spin1Params(
        n_bins=num_bins,
        r_min=F_MIN,
        r_max=F_MAX,
        p0=float(polarization),
        initial_polarization=float(polarization),
        q0=0.0,
        p_dnp_sat=float(polarization),
        dnp_enabled=False,
        rf_enabled=True,
        relax_enabled=True,
        afp_enabled=False,
        dt=float(dt),
    )
    model = Spin1Model(params)
    model.load_from_physical_intensities(baseline_ip, baseline_im)
    model.params.gamma_rf = float(gamma_rf)
    model.params.ssrf_subset_indices = list(ssrf_subset)
    model.params.rf_burn_R = float(f[int(burn_idx)])
    freeze_rf_profile(model, profile)
    model._active_idx = np.asarray(touched, dtype=int) if touched else None

    step_count = max(1, int(n_steps))
    model.step(n_steps=step_count)
    ip_sim, im_sim, _ = model.physical_intensities()
    out_ip, out_im = commit_ssrf_bins_only(
        baseline_ip, baseline_im, ip_sim, im_sim, touched
    )
    metrics = _ssrf_burn_mirror_region_metrics(
        baseline_ip, baseline_im, out_ip, out_im, ssrf_subset, num_bins
    )
    return out_ip, out_im, step_count, ssrf_subset, metrics


def _sample_manipulation_mode(rng: np.random.Generator) -> ManipulationMode:
    modes: Tuple[ManipulationMode, ...] = ("ssrf", "afp", "both")
    idx = int(rng.choice(len(modes), p=MANIPULATION_MODE_WEIGHTS))
    return modes[idx]


def run_manipulation_event(
    *,
    polarization: float,
    mode: ManipulationMode,
    burn_bin: int | None,
    afp_center_bin: int | None,
    afp_bins: List[int] | None,
    gamma_rf: float,
    n_steps: int,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    afp_efficiency: float = AFP_EFFICIENCY,
    afp_center_margin: int = AFP_CENTER_MARGIN,
) -> Dict[str, Any]:
    """Build one manipulated lineshape; AFP path matches afp_bin_traj.py."""
    apply_ssrf = mode in ("ssrf", "both")
    apply_afp = mode in ("afp", "both")

    f = np.asarray(FREQUENCY[:num_bins], dtype=float)
    _, iplus0, iminus0 = GenerateVectorLineshape(float(polarization), f)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    iplus_unburned = iplus.copy()
    iminus_unburned = iminus.copy()

    q_signal = iplus_unburned - iminus_unburned
    burn_idx = resolve_burn_bin(q_signal, burn_bin if apply_ssrf else None)
    ps0_at_burn = (
        float(iplus_unburned[burn_idx] + iminus_unburned[burn_idx])
        if apply_ssrf
        else np.nan
    )

    step_count = 0
    burn_region_area = np.nan
    mirror_region_area = np.nan
    mirror_over_burn_area = np.nan
    iminus_mirror_over_iplus_burn = np.nan
    iplus_mirror_over_iminus_burn = np.nan
    if apply_ssrf:
        (
            iplus,
            iminus,
            step_count,
            _ssrf_subset,
            ssrf_metrics,
        ) = _run_ssrf_burn(
            iplus=iplus,
            iminus=iminus,
            burn_idx=int(burn_idx),
            gamma_rf=float(gamma_rf),
            n_steps=int(n_steps),
            num_bins=num_bins,
            dt=float(dt),
            polarization=float(polarization),
        )
        burn_region_area = float(ssrf_metrics["burn_region_area"])
        mirror_region_area = float(ssrf_metrics["mirror_region_area"])
        mirror_over_burn_area = float(ssrf_metrics["mirror_over_burn_area"])
        iminus_mirror_over_iplus_burn = float(
            ssrf_metrics["iminus_mirror_over_iplus_burn"]
        )
        iplus_mirror_over_iminus_burn = float(
            ssrf_metrics["iplus_mirror_over_iminus_burn"]
        )

    if apply_afp:
        if not afp_bins:
            raise ValueError("AFP mode requires afp_bins (3-bin window indices).")
        iplus, iminus = apply_afp_to_lineshape(
            iplus,
            iminus,
            afp_bins,
            polarization=float(polarization),
            num_bins=num_bins,
            afp_efficiency=float(afp_efficiency),
            afp_center_margin=int(afp_center_margin),
        )

    ps = (iplus + iminus).astype(STORE_DTYPE)
    qs = (iplus - iminus).astype(STORE_DTYPE)
    ps_at_burn = float(iplus[burn_idx] + iminus[burn_idx]) if apply_ssrf else np.nan
    if apply_ssrf and abs(ps0_at_burn) > 1e-12:
        ps_ratio = float(ps_at_burn / ps0_at_burn)
    else:
        ps_ratio = 1.0

    afp_lo = int(afp_bins[0]) if afp_bins else -1
    afp_hi = int(afp_bins[-1]) + 1 if afp_bins else -1
    afp_center = int(afp_center_bin) if afp_center_bin is not None else -1

    return {
        "P_initial": float(polarization),
        "true_P_initial": STORE_DTYPE(float(np.sum(iplus_unburned + iminus_unburned))),
        "true_P": STORE_DTYPE(float(np.sum(ps))),
        "true_Q": STORE_DTYPE(float(np.sum(qs))),
        "manipulation_mode": mode,
        "ssrf_applied": apply_ssrf,
        "afp_applied": apply_afp,
        "burn_bin_idx": int(burn_idx) if apply_ssrf else None,
        "burn_freq": float(f[burn_idx]) if apply_ssrf else np.nan,
        "mirror_bin_idx": int(mirror_bin_idx(num_bins, burn_idx)) if apply_ssrf else None,
        "gamma_rf": float(gamma_rf) if apply_ssrf else np.nan,
        "burn_step_requested": int(n_steps) if apply_ssrf else 0,
        "burn_step": int(step_count) if apply_ssrf else 0,
        "burn_step_norm": float(step_count / max(MAX_BURN_STEPS, 1)) if apply_ssrf else 0.0,
        "ps0_at_burn_bin": STORE_DTYPE(ps0_at_burn) if apply_ssrf else np.nan,
        "ps_at_burn_bin": STORE_DTYPE(ps_at_burn) if apply_ssrf else np.nan,
        "ps_ratio": STORE_DTYPE(ps_ratio),
        "burn_progress": STORE_DTYPE(float(1.0 - abs(ps_ratio))),
        "burn_region_area": STORE_DTYPE(burn_region_area) if apply_ssrf else np.nan,
        "mirror_region_area": STORE_DTYPE(mirror_region_area) if apply_ssrf else np.nan,
        "mirror_over_burn_area": (
            STORE_DTYPE(mirror_over_burn_area) if apply_ssrf else np.nan
        ),
        "iminus_mirror_over_iplus_burn": (
            STORE_DTYPE(iminus_mirror_over_iplus_burn) if apply_ssrf else np.nan
        ),
        "iplus_mirror_over_iminus_burn": (
            STORE_DTYPE(iplus_mirror_over_iminus_burn) if apply_ssrf else np.nan
        ),
        "afp_center_bin_idx": afp_center if apply_afp else None,
        "afp_mirror_bin_idx": (
            int(mirror_bin_idx(num_bins, afp_center)) if apply_afp and afp_center >= 0 else None
        ),
        "afp_bin_start": afp_lo,
        "afp_bin_stop": afp_hi,
        "afp_sweep_width": afp_hi - afp_lo if apply_afp and afp_bins else 0,
        "afp_freq_start": float(f[afp_lo]) if apply_afp and afp_bins else np.nan,
        "afp_freq_stop": float(f[afp_hi - 1]) if apply_afp and afp_bins else np.nan,
        "afp_efficiency": AFP_EFFICIENCY if apply_afp else np.nan,
        "frequency": FREQUENCY.astype(STORE_DTYPE),
        "Ps": ps,
        "Qs": qs,
        "Iplus": iplus.astype(STORE_DTYPE),
        "Iminus": iminus.astype(STORE_DTYPE),
    }


def generate_sample(sample_id: int, rng: np.random.Generator) -> Dict[str, Any]:
    last_area_ratio = float("nan")
    last_im_over_ip = float("nan")
    last_ip_over_im = float("nan")
    for _attempt in range(MAX_SSRF_AREA_RATIO_RETRIES):
        polarization = float(rng.uniform(P_MIN, P_MAX))
        mode = _sample_manipulation_mode(rng)

        burn_bin: int | None = None
        gamma_rf = np.nan
        n_steps = 0
        if mode in ("ssrf", "both"):
            burn_bin = int(rng.choice(_BURN_BIN_CHOICES))
            gamma_rf = float(rng.uniform(GAMMA_RF_MIN, GAMMA_RF_MAX))
            n_steps = int(rng.integers(MIN_BURN_STEPS, MAX_BURN_STEPS + 1))

        afp_center_bin: int | None = None
        afp_bins: List[int] | None = None
        if mode in ("afp", "both"):
            afp_center_bin = _sample_afp_center_bin(NUM_BINS, rng)
            afp_bins = afp_window_indices(int(afp_center_bin), NUM_BINS, window=AFP_WINDOW)

        row = run_manipulation_event(
            polarization=polarization,
            mode=mode,
            burn_bin=burn_bin,
            afp_center_bin=afp_center_bin,
            afp_bins=afp_bins,
            gamma_rf=gamma_rf,
            n_steps=n_steps,
        )
        if mode in ("ssrf", "both"):
            last_area_ratio = float(row["mirror_over_burn_area"])
            last_im_over_ip = float(row["iminus_mirror_over_iplus_burn"])
            last_ip_over_im = float(row["iplus_mirror_over_iminus_burn"])
            if not _ssrf_area_ratio_ok(last_area_ratio):
                # Near R=0, burn∪mirror supports overlap and the 1/2 area rule fails.
                continue
            if not _ssrf_intensity_ratios_ok(last_im_over_ip, last_ip_over_im):
                continue
        row["sample_id"] = int(sample_id)
        return row

    raise RuntimeError(
        f"Failed to generate sample {sample_id} with ssRF burn/mirror ratios "
        f"after {MAX_SSRF_AREA_RATIO_RETRIES} tries "
        f"(area={last_area_ratio}, "
        f"I-_mir/I+_burn={last_im_over_ip}, "
        f"I+_mir/I-_burn={last_ip_over_im})"
    )


def _generate_sample_task(sample_id: int) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED + sample_id)
    return generate_sample(sample_id, rng)


def generate_dataset(num_samples: int, workers: int) -> pd.DataFrame:
    if _BURN_BIN_CHOICES.size == 0:
        raise ValueError(
            f"No burn bins in R range ({BURN_R_MIN}, {BURN_R_MAX}) for {NUM_BINS} bins."
        )

    if workers <= 1:
        rng = np.random.default_rng(SEED)
        rows = [
            generate_sample(sample_id, rng)
            for sample_id in tqdm.tqdm(range(num_samples), desc="Generating manipulated lineshapes")
        ]
        return pd.DataFrame(rows)

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in tqdm.tqdm(
            pool.map(_generate_sample_task, range(num_samples), chunksize=32),
            total=num_samples,
            desc=f"Generating manipulated lineshapes ({workers} workers)",
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
) -> Tuple[Path, Path | None]:
    """Pick a random generated event and save Ps/Qs plus I+/- comparison plots."""
    if len(df) == 0:
        raise ValueError("Cannot plot example from empty dataframe.")
    rng = rng or np.random.default_rng(SEED)
    # Prefer an ssRF (or both) event when available so burn/mirror spans show.
    ssrf_mask = df["ssrf_applied"].astype(bool)
    pool = df[ssrf_mask] if ssrf_mask.any() else df
    row = pool.iloc[int(rng.integers(0, len(pool)))].to_dict()

    f = np.asarray(row["frequency"], dtype=float)
    ps = np.asarray(row["Ps"], dtype=float)
    qs = np.asarray(row["Qs"], dtype=float)
    ip = np.asarray(row["Iplus"], dtype=float)
    im = np.asarray(row["Iminus"], dtype=float)

    _, ip0, im0 = GenerateVectorLineshape(float(row["P_initial"]), f)
    ip0 = np.asarray(ip0, dtype=float)
    im0 = np.asarray(im0, dtype=float)
    ps0 = ip0 + im0
    qs0 = ip0 - im0
    dip = ip - ip0
    dim = im - im0

    support = mirrors = None
    burn = mirror = -1
    if bool(row["ssrf_applied"]):
        support, mirrors, burn, mirror = _ssrf_support_and_mirrors(row, len(f))

    title_bits = [
        f"random example  mode={row['manipulation_mode']}",
        f"P={float(row['P_initial']):.3f}",
        f"sample_id={int(row['sample_id'])}",
    ]
    if bool(row["ssrf_applied"]):
        title_bits.extend(
            [
                f"burn R={float(row['burn_freq']):.3f}",
                f"gamma_rf={float(row['gamma_rf']):.2f}",
                f"steps={int(row['burn_step'])}",
            ]
        )

    # --- Ps / Qs figure ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(f, ps0, color="0.65", lw=1.2, label="Ps unburned")
    ax.plot(f, ps, color="C0", lw=1.8, label="Ps manipulated")
    ax.set_ylabel("Ps")

    ax.plot(f, qs0, color="0.65", lw=1.2, label="Qs unburned")
    ax.plot(f, qs, color="C1", lw=1.8, label="Qs manipulated")
    ax.set_xlabel("R")
    ax.set_ylabel("Qs")

    if support is not None and mirrors is not None:
        _shade_burn_mirror(ax, f, support, mirrors, burn, mirror, label=True)
        ax.set_title(
            "  ".join(title_bits)
            + "\n"
            + f"mirror/burn area={float(row['mirror_over_burn_area']):.4f}"
        )
    else:
        if bool(row["afp_applied"]) and row.get("afp_bin_start", -1) >= 0:
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

    # --- Separate I+ / I- comparison figure ---
    if iplus_iminus_path is None:
        iplus_iminus_path = output_path.with_name(
            f"{output_path.stem}_iplus_iminus{output_path.suffix}"
        )
    iplus_iminus_path = Path(iplus_iminus_path)
    fig2, ax = plt.subplots(figsize=(10, 7))

    ax.plot(f, ip0, color="0.65", lw=1.2, label="I+ unburned")
    ax.plot(f, ip, color="C0", lw=1.8, label="I+ manipulated")
    ax.plot(f, dip, color="C3", lw=1.2, ls=":", label="dI+")
    ax.set_ylabel("I+ / dI+")
    ax.axhline(0.0, color="0.5", lw=0.8)

    ax.plot(f, im0, color="0.65", lw=1.2, label="I- unburned")
    ax.plot(f, im, color="C4", lw=1.8, label="I- manipulated")
    ax.plot(f, dim, color="C2", lw=1.2, ls=":", label="dI-")
    ax.set_xlabel("R")
    ax.set_ylabel("I- / dI-")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Spin1Model manipulated test data.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel worker processes (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help=f"Number of samples to generate (default: {NUM_SAMPLES})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output pickle path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--example-plot",
        type=Path,
        default=EXAMPLE_PLOT_PATH,
        help=f"Path for random-example Ps/Qs plot (default: {EXAMPLE_PLOT_PATH})",
    )
    parser.add_argument(
        "--example-iplus-iminus-plot",
        type=Path,
        default=EXAMPLE_IPLUS_IMINUS_PLOT_PATH,
        help=(
            "Path for random-example I+/- comparison plot "
            f"(default: {EXAMPLE_IPLUS_IMINUS_PLOT_PATH})"
        ),
    )
    parser.add_argument(
        "--no-example-plot",
        action="store_true",
        help="Skip saving random example plots",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        f"Generating {args.num_samples} manipulated lineshapes "
        f"(P in [{P_MIN}, {P_MAX}]) via Spin1Model / ssrf_afp physics"
    )
    print(
        "Manipulation mix (ssRF / AFP / both): "
        f"{MANIPULATION_MODE_WEIGHTS[0]:.0%} / "
        f"{MANIPULATION_MODE_WEIGHTS[1]:.0%} / "
        f"{MANIPULATION_MODE_WEIGHTS[2]:.0%}"
    )
    print(f"AFP window: {AFP_WINDOW} bins (matches afp_bin_traj.py; no post-AFP relaxation)")
    print(
        f"ssRF Voigt: sigma={RF_SIGMA_BINS}, lorentz_gamma={RF_VOIGT_GAMMA_BINS}, "
        f"half_width=+/-{RF_HALF_WIDTH} bins (compact, cosine-tapered)"
    )
    print(f"Workers: {args.workers}")

    df = generate_dataset(args.num_samples, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.output)

    print(f"Saved {len(df)} samples to {args.output}")
    print("Mode counts:")
    print(df["manipulation_mode"].value_counts().to_string())
    print(
        "P_initial:",
        f"min={df['P_initial'].min():.3f}, max={df['P_initial'].max():.3f}, "
        f"mean={df['P_initial'].mean():.3f}",
    )
    print(
        "true_P (final integrated):",
        f"min={df['true_P'].min():.3f}, max={df['true_P'].max():.3f}, "
        f"mean={df['true_P'].mean():.3f}",
    )
    print(
        "true_P shift:",
        f"mean={(df['true_P'] - df['true_P_initial']).mean():.4f}",
    )
    ssrf_rows = df[df["ssrf_applied"].astype(bool)]
    if len(ssrf_rows) > 0:
        burn_areas = ssrf_rows["burn_region_area"].astype(float)
        mirror_areas = ssrf_rows["mirror_region_area"].astype(float)
        ratios = ssrf_rows["mirror_over_burn_area"].astype(float)
        print(f"ssRF samples with area check: {len(ssrf_rows)}")
        print(
            "burn_region_area:",
            f"mean={burn_areas.mean():.6g}, "
            f"min={burn_areas.min():.6g}, max={burn_areas.max():.6g}",
        )
        print(
            "mirror_region_area:",
            f"mean={mirror_areas.mean():.6g}, "
            f"min={mirror_areas.min():.6g}, max={mirror_areas.max():.6g}",
        )
        print(
            "mirror_over_burn_area:",
            f"median={ratios.median():.4f}, "
            f"mean={ratios.mean():.4f}, "
            f"min={ratios.min():.4f}, max={ratios.max():.4f} "
            f"(target={MIRROR_OVER_BURN_AREA_TARGET})",
        )
        im_over_ip = ssrf_rows["iminus_mirror_over_iplus_burn"].astype(float)
        ip_over_im = ssrf_rows["iplus_mirror_over_iminus_burn"].astype(float)
        print(
            "iminus_mirror / iplus_burn:",
            f"median={im_over_ip.median():.4f}, "
            f"mean={im_over_ip.mean():.4f}, "
            f"min={im_over_ip.min():.4f}, max={im_over_ip.max():.4f} "
            f"(target={IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET})",
        )
        print(
            "iplus_mirror / iminus_burn:",
            f"median={ip_over_im.median():.4f}, "
            f"mean={ip_over_im.mean():.4f}, "
            f"min={ip_over_im.min():.4f}, max={ip_over_im.max():.4f} "
            f"(target={IPLUS_MIRROR_OVER_IMINUS_BURN_TARGET})",
        )

    if not args.no_example_plot:
        plot_path, iplus_iminus_path = plot_random_example(
            df,
            args.example_plot,
            iplus_iminus_path=args.example_iplus_iminus_plot,
        )
        print(f"Saved random example plot to {plot_path}")
        if iplus_iminus_path is not None:
            print(f"Saved I+/- comparison plot to {iplus_iminus_path}")


if __name__ == "__main__":
    main()
