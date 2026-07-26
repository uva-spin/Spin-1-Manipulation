"""ssRF / AFP manipulation of Dulya-fit equilibrium lineshapes."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from afp_bin_traj import (  # noqa: E402
    afp_touched_bins,
    afp_window_indices,
    commit_touched_bins_only,
    restore_touched_intensity_area,
)
from ssrf_bin_traj import (  # noqa: E402
    HALF_WIDTH,
    SIGMA_BINS,
    VOIGT_GAMMA_BINS,
    commit_ssrf_bins_only,
    freeze_rf_profile,
    make_voigt_rf_profile,
    mirror_bin_idx,
    resolve_burn_bin,
    ssrf_touched_bins,
)
from physics.ssrf_realtime.model import Spin1Model, Spin1Params  # noqa: E402

from common import (  # noqa: E402
    AFP_CENTER_MARGIN,
    AFP_EFFICIENCY,
    AFP_WINDOW,
    DT,
    F_MAX,
    F_MIN,
    FREQUENCY,
    IMINUS_MIRROR_OVER_IPLUS_BURN_TARGET,
    IPLUS_MIRROR_OVER_IMINUS_BURN_TARGET,
    MAX_BURN_STEPS,
    MIRROR_OVER_BURN_AREA_RTOL,
    MIRROR_OVER_BURN_AREA_TARGET,
    NUM_BINS,
    SSRF_INTENSITY_RATIO_RTOL,
    STORE_DTYPE,
)
from lineshape import GenerateDulyaLineshape, shape_params_from_fit  # noqa: E402
from polarization import (  # noqa: E402
    build_amp_integration_calibration,
    integrated_polarizations,
)

ManipulationMode = Literal["ssrf", "afp", "none"]

RF_SIGMA_BINS = SIGMA_BINS
RF_VOIGT_GAMMA_BINS = VOIGT_GAMMA_BINS
RF_HALF_WIDTH = HALF_WIDTH


def sample_polarization(
    rng: np.random.Generator,
    p_min: float,
    p_max: float,
    p_abs_min: float,
) -> float:
    """Uniform P in [p_min, p_max], rejecting |P| < p_abs_min."""
    for _ in range(10_000):
        p = float(rng.uniform(p_min, p_max))
        if abs(p) >= float(p_abs_min):
            return p
    raise RuntimeError(
        f"Failed to sample |P| >= {p_abs_min} in [{p_min}, {p_max}]"
    )


def apply_afp_to_lineshape(
    iplus: np.ndarray,
    iminus: np.ndarray,
    afp_subset: List[int],
    *,
    polarization: float,
    num_bins: int,
    afp_efficiency: float = AFP_EFFICIENCY,
    afp_center_margin: int = AFP_CENTER_MARGIN,
) -> Tuple[np.ndarray, np.ndarray]:
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


def ssrf_area_ratio_ok(ratio: float) -> bool:
    return _ratio_near_target(
        ratio, MIRROR_OVER_BURN_AREA_TARGET, MIRROR_OVER_BURN_AREA_RTOL
    )


def ssrf_intensity_ratios_ok(
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


def run_event(
    *,
    polarization: float,
    mode: ManipulationMode,
    burn_bin: int | None = None,
    afp_center_bin: int | None = None,
    afp_bins: List[int] | None = None,
    gamma_rf: float = np.nan,
    n_steps: int = 0,
    num_bins: int = NUM_BINS,
    dt: float = DT,
    shape_params: dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Build one unmanipulated / ssRF / AFP lineshape from the Dulya fit kernel."""
    apply_ssrf = mode == "ssrf"
    apply_afp = mode == "afp"
    P = float(polarization)
    shape = shape_params if shape_params is not None else shape_params_from_fit()
    amp = float(shape["amp"])
    n_bins_i = int(num_bins)

    f = np.asarray(FREQUENCY[:num_bins], dtype=float)
    calibration = build_amp_integration_calibration(
        f, shape, p_min=-0.9, p_max=0.9, p_step=0.01
    )
    _, iplus0, iminus0 = GenerateDulyaLineshape(P, f, shape)
    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    iplus_unburned = iplus.copy()
    iminus_unburned = iminus.copy()

    # Fit-scale equilibrium. Spin1 dynamics expect sum(I+ + I-) ~ P, so convert
    # for manipulation and map deltas back to fit units afterward.
    area_fit = float(np.sum(iplus_unburned + iminus_unburned))
    if abs(area_fit) > 1e-30 and abs(P) > 1e-15:
        to_spin1 = P / area_fit
        from_spin1 = area_fit / P
    else:
        to_spin1 = 1.0
        from_spin1 = 1.0

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
        iplus_u = iplus * to_spin1
        iminus_u = iminus * to_spin1
        (
            iplus_u,
            iminus_u,
            step_count,
            _ssrf_subset,
            ssrf_metrics,
        ) = _run_ssrf_burn(
            iplus=iplus_u,
            iminus=iminus_u,
            burn_idx=int(burn_idx),
            gamma_rf=float(gamma_rf),
            n_steps=int(n_steps),
            num_bins=num_bins,
            dt=float(dt),
            polarization=P,
        )
        iplus = iplus_u * from_spin1
        iminus = iminus_u * from_spin1
        burn_region_area = float(ssrf_metrics["burn_region_area"]) * abs(from_spin1)
        mirror_region_area = float(ssrf_metrics["mirror_region_area"]) * abs(from_spin1)
        mirror_over_burn_area = float(ssrf_metrics["mirror_over_burn_area"])
        iminus_mirror_over_iplus_burn = float(
            ssrf_metrics["iminus_mirror_over_iplus_burn"]
        )
        iplus_mirror_over_iminus_burn = float(
            ssrf_metrics["iplus_mirror_over_iminus_burn"]
        )

    if apply_afp:
        if not afp_bins:
            raise ValueError("AFP mode requires afp_bins.")
        iplus_u = iplus * to_spin1
        iminus_u = iminus * to_spin1
        iplus_u, iminus_u = apply_afp_to_lineshape(
            iplus_u,
            iminus_u,
            afp_bins,
            polarization=P,
            num_bins=num_bins,
        )
        iplus = iplus_u * from_spin1
        iminus = iminus_u * from_spin1

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
        "P_initial": P,
        "true_P_initial": STORE_DTYPE(
            integrated_polarizations(
                iplus_unburned,
                iminus_unburned,
                amp=amp,
                n_bins=n_bins_i,
                post_correct=True,
                calibration=calibration,
            )[0]
        ),
        "true_P": STORE_DTYPE(
            integrated_polarizations(
                iplus,
                iminus,
                amp=amp,
                n_bins=n_bins_i,
                post_correct=True,
                calibration=calibration,
            )[0]
        ),
        "true_Q": STORE_DTYPE(
            integrated_polarizations(
                iplus,
                iminus,
                amp=amp,
                n_bins=n_bins_i,
                post_correct=True,
                calibration=calibration,
            )[1]
        ),
        "lineshape_model": "dulya_fit",
        "manipulation_mode": mode,
        "ssrf_applied": apply_ssrf,
        "afp_applied": apply_afp,
        "burn_bin_idx": int(burn_idx) if apply_ssrf else None,
        "burn_freq": float(f[burn_idx]) if apply_ssrf else np.nan,
        "mirror_bin_idx": (
            int(mirror_bin_idx(num_bins, burn_idx)) if apply_ssrf else None
        ),
        "gamma_rf": float(gamma_rf) if apply_ssrf else np.nan,
        "burn_step_requested": int(n_steps) if apply_ssrf else 0,
        "burn_step": int(step_count) if apply_ssrf else 0,
        "burn_step_norm": (
            float(step_count / max(MAX_BURN_STEPS, 1)) if apply_ssrf else 0.0
        ),
        "ps0_at_burn_bin": STORE_DTYPE(ps0_at_burn) if apply_ssrf else np.nan,
        "ps_at_burn_bin": STORE_DTYPE(ps_at_burn) if apply_ssrf else np.nan,
        "ps_ratio": STORE_DTYPE(ps_ratio),
        "burn_progress": STORE_DTYPE(float(1.0 - abs(ps_ratio))),
        "burn_region_area": STORE_DTYPE(burn_region_area) if apply_ssrf else np.nan,
        "mirror_region_area": (
            STORE_DTYPE(mirror_region_area) if apply_ssrf else np.nan
        ),
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
            int(mirror_bin_idx(num_bins, afp_center))
            if apply_afp and afp_center >= 0
            else None
        ),
        "afp_bin_start": afp_lo if apply_afp else -1,
        "afp_bin_stop": afp_hi if apply_afp else -1,
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


def sample_afp_center_bin(n_bins: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, int(n_bins)))


def make_afp_bins(center_bin: int, n_bins: int = NUM_BINS) -> List[int]:
    return afp_window_indices(int(center_bin), int(n_bins), window=AFP_WINDOW)
