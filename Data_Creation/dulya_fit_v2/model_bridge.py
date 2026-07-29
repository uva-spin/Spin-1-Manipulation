"""Bridge Dulya equilibrium lineshapes to vendored ssrf_realtime_v2."""

from __future__ import annotations

from typing import Optional

import numpy as np

import _bootstrap  # noqa: F401
from ssrf_realtime_v2 import Spin1Model, Spin1Params
from ssrf_realtime_v2.rate_equations_realtime import (
    configure_physical_voigt_ssrf,
    configure_single_bin_ssrf,
    configure_voigt_burn_spectral_recovery,
)

from common import (
    D_SAME_0MINUS,
    D_SAME_PLUS0,
    D_SPEC_0MINUS,
    D_SPEC_PLUS0,
    DIFFUSION_SCALE,
    F_MAX,
    F_MIN,
    MAX_GDT,
    MAX_NSUB,
    MIRROR_AMP_EPS,
    MIRROR_AMP_RTOL,
    RF_GAUSSIAN_FWHM_R,
    RF_LORENTZIAN_FWHM_R,
    RF_MODE_PHYSICAL_VOIGT,
    RF_MODE_SINGLE_BIN,
    ZQ_WIDTH_R,
)


PROFILE_REL_THRESHOLD = 0.01
KERNEL_CUTOFF_WIDTHS = 3.0
INTENSITY_DELTA_ABS_TOL = 1e-15


def mirror_bin_idx(n_bins: int, bin_idx: int) -> int:
    return int(n_bins) - 1 - int(bin_idx)


def afp_touched_bins(n_bins: int, subset: list[int] | np.ndarray) -> list[int]:
    touched: set[int] = set()
    for i in subset:
        touched.add(int(i))
        touched.add(mirror_bin_idx(n_bins, int(i)))
    return sorted(touched)


def ssrf_touched_bins(n_bins: int, subset: list[int] | np.ndarray) -> list[int]:
    """Packet/intensity bins ssRF changes: each burn index i also updates mirror(i)."""
    return afp_touched_bins(n_bins, subset)


def bins_within_R_radius(
    R: np.ndarray, seed_bins: list[int] | np.ndarray, radius_R: float
) -> list[int]:
    grid = np.asarray(R, dtype=float)
    radius = float(radius_R)
    out: set[int] = set()
    for i in seed_bins:
        ri = float(grid[int(i)])
        for j in range(grid.size):
            if abs(float(grid[j]) - ri) <= radius:
                out.add(int(j))
    return sorted(out)


def diffusion_spillover_bins(
    R: np.ndarray,
    seed_bins: list[int] | np.ndarray,
    *,
    zq_width_R: float = ZQ_WIDTH_R,
    kernel_cutoff_widths: float = KERNEL_CUTOFF_WIDTHS,
) -> list[int]:
    """Bins within the spin-diffusion kernel reach of ``seed_bins``."""
    return bins_within_R_radius(
        R, seed_bins, float(kernel_cutoff_widths) * float(zq_width_R)
    )


def bins_with_intensity_delta(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_sim: np.ndarray,
    iminus_sim: np.ndarray,
    candidates: list[int] | np.ndarray,
    *,
    abs_tol: float = INTENSITY_DELTA_ABS_TOL,
) -> list[int]:
    changed: list[int] = []
    for i in candidates:
        idx = int(i)
        if (
            abs(float(iplus_sim[idx]) - float(iplus[idx])) > abs_tol
            or abs(float(iminus_sim[idx]) - float(iminus[idx])) > abs_tol
        ):
            changed.append(idx)
    return changed


def physical_voigt_rf_support_bins(
    R: np.ndarray,
    burn_idx: int,
    *,
    gaussian_fwhm_R: float,
    lorentzian_fwhm_R: float,
    rel_threshold: float = PROFILE_REL_THRESHOLD,
) -> list[int]:
    """Bins with non-negligible physical-R Voigt RF at the burn center."""
    from ssrf_realtime_v2.voigt_physical import bin_averaged_voigt

    grid = np.asarray(R, dtype=float)
    burn_idx = int(burn_idx)
    if burn_idx < 0 or burn_idx >= grid.size:
        raise ValueError(f"burn_idx={burn_idx} out of range for n_bins={grid.size}")
    dR = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
    profile = bin_averaged_voigt(
        grid,
        center_R=float(grid[burn_idx]),
        bin_width_R=dR,
        gaussian_fwhm_R=float(gaussian_fwhm_R),
        lorentzian_fwhm_R=float(lorentzian_fwhm_R),
        normalization="center_bin",
    )
    peak = float(np.max(profile)) if profile.size else 0.0
    if peak <= 0.0:
        return [burn_idx]
    floor = float(rel_threshold) * peak
    support = [int(i) for i in np.flatnonzero(np.asarray(profile, dtype=float) >= floor)]
    return support if support else [burn_idx]


def burn_commit_touched_bins(
    n_bins: int,
    burn_idx: int,
    *,
    rf_mode: str,
    R: np.ndarray | None = None,
    gaussian_fwhm_R: float = RF_GAUSSIAN_FWHM_R,
    lorentzian_fwhm_R: float = RF_LORENTZIAN_FWHM_R,
    iplus: np.ndarray | None = None,
    iminus: np.ndarray | None = None,
    iplus_sim: np.ndarray | None = None,
    iminus_sim: np.ndarray | None = None,
    include_diffusion_spillover: bool = True,
    zq_width_R: float = ZQ_WIDTH_R,
) -> list[int]:
    """Bins whose intensities should be committed after a burn trial."""
    burn_idx = int(burn_idx)
    if rf_mode == RF_MODE_SINGLE_BIN:
        return ssrf_touched_bins(n_bins, [burn_idx])
    if rf_mode == RF_MODE_PHYSICAL_VOIGT:
        if R is None:
            raise ValueError("R grid required for physical Voigt burn commit")
        support = physical_voigt_rf_support_bins(
            R,
            burn_idx,
            gaussian_fwhm_R=gaussian_fwhm_R,
            lorentzian_fwhm_R=lorentzian_fwhm_R,
        )
        touched: set[int] = set(ssrf_touched_bins(n_bins, support))
        if include_diffusion_spillover:
            spill_candidates = diffusion_spillover_bins(
                R, support, zq_width_R=zq_width_R
            )
            if (
                iplus is not None
                and iminus is not None
                and iplus_sim is not None
                and iminus_sim is not None
            ):
                spill = bins_with_intensity_delta(
                    iplus, iminus, iplus_sim, iminus_sim, spill_candidates
                )
            else:
                spill = spill_candidates
            for i in spill:
                touched.add(int(i))
                touched.add(mirror_bin_idx(n_bins, int(i)))
        return sorted(touched)
    raise ValueError(
        f"unknown rf_mode={rf_mode!r}; expected "
        f"{RF_MODE_PHYSICAL_VOIGT!r} or {RF_MODE_SINGLE_BIN!r}"
    )


def commit_touched_bins_only(
    iplus: np.ndarray,
    iminus: np.ndarray,
    iplus_sim: np.ndarray,
    iminus_sim: np.ndarray,
    touched: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep baseline intensities except on touched bins."""
    out_ip = np.asarray(iplus, dtype=float).copy()
    out_im = np.asarray(iminus, dtype=float).copy()
    ip_sim = np.asarray(iplus_sim, dtype=float)
    im_sim = np.asarray(iminus_sim, dtype=float)
    for k in touched:
        out_ip[k] = float(ip_sim[k])
        out_im[k] = float(im_sim[k])
    return out_ip, out_im


def restore_touched_intensity_area(
    iplus: np.ndarray,
    iminus: np.ndarray,
    touched: list[int],
    area_target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore total area on touched bins via common-mode offset."""
    out_ip = np.asarray(iplus, dtype=float).copy()
    out_im = np.asarray(iminus, dtype=float).copy()
    if not touched:
        return out_ip, out_im
    n = len(out_ip)
    touched_set = set(int(k) for k in touched)
    unt_idx = [i for i in range(n) if i not in touched_set]
    area_unt = float(np.sum(out_ip[unt_idx] + out_im[unt_idx])) if unt_idx else 0.0
    ps_touch = out_ip[list(touched)] + out_im[list(touched)]
    area_touch = float(np.sum(ps_touch))
    missing = float(area_target) - area_unt - area_touch
    if abs(missing) < 1e-15:
        return out_ip, out_im
    weights = np.maximum(ps_touch, 0.0)
    wsum = float(np.sum(weights))
    if wsum > 1e-30:
        for j, k in enumerate(touched):
            add = missing * (float(weights[j]) / wsum)
            out_ip[k] += 0.5 * add
            out_im[k] += 0.5 * add
    else:
        add = missing / float(len(touched))
        for k in touched:
            out_ip[k] += 0.5 * add
            out_im[k] += 0.5 * add
    return out_ip, out_im


def afp_window_indices(bin_idx: int, n_bins: int, window: int) -> list[int]:
    w = max(1, int(window))
    half = w // 2
    c = int(bin_idx)
    n = int(n_bins)
    lo = c - half
    hi = c + half
    if lo < 0:
        hi = min(n - 1, hi - lo)
        lo = 0
    if hi >= n:
        lo = max(0, lo - (hi - (n - 1)))
        hi = n - 1
    return list(range(lo, hi + 1))


def build_spin1_model(
    iplus: np.ndarray,
    iminus: np.ndarray,
    *,
    polarization: float,
    num_bins: int,
    dt: float,
    rf_enabled: bool = False,
    relax_enabled: bool = True,
    diffusion_scale: float = DIFFUSION_SCALE,
    rf_gaussian_fwhm_R: float = RF_GAUSSIAN_FWHM_R,
    rf_lorentzian_fwhm_R: float = RF_LORENTZIAN_FWHM_R,
    r_min: float | None = None,
    r_max: float | None = None,
) -> Spin1Model:
    """Build a capacity-weighted Spin1 model loaded from physical intensities."""
    P = float(polarization)
    # display_cal == p0 when plot_signal_units; avoid divide-by-zero at P≈0.
    near_zero_p = abs(P) < 1e-12
    params = Spin1Params(
        n_bins=int(num_bins),
        r_min=float(F_MIN if r_min is None else r_min),
        r_max=float(F_MAX if r_max is None else r_max),
        p0=P if not near_zero_p else 0.0,
        q0=None,
        p_dnp_sat=P if not near_zero_p else 0.0,
        dnp_enabled=False,
        rf_enabled=bool(rf_enabled),
        relax_enabled=bool(relax_enabled),
        afp_enabled=False,
        gamma_rf=0.0,
        dt=float(dt),
        capacity_rate_power=1.0,
        plot_signal_units=not near_zero_p,
        display_scale=1.0,
        use_physical_voigt_rf=False,
        rf_gaussian_fwhm_R=float(rf_gaussian_fwhm_R),
        rf_lorentzian_fwhm_R=float(rf_lorentzian_fwhm_R),
        rf_profile_normalization="center_bin",
        diffusion_scale=float(diffusion_scale),
        zq_width_R=float(ZQ_WIDTH_R),
        d_same_plus0=float(D_SAME_PLUS0) if relax_enabled else 0.0,
        d_same_0minus=float(D_SAME_0MINUS) if relax_enabled else 0.0,
        d_spec_plus0=float(D_SPEC_PLUS0) if relax_enabled else 0.0,
        d_spec_0minus=float(D_SPEC_0MINUS) if relax_enabled else 0.0,
    )
    model = Spin1Model(params)
    model.load_from_physical_intensities(
        np.asarray(iplus, dtype=float),
        np.asarray(iminus, dtype=float),
    )
    if relax_enabled:
        apply_shared_spectral_recovery(model)
    return model


def apply_shared_spectral_recovery(model: Spin1Model) -> None:
    """Enable the same v15 spectral recovery used by voigt_burn demos."""
    configure_voigt_burn_spectral_recovery(model)
    # Keep rates aligned with dulya_fit_v2 constants (helper uses the same defaults).
    model.params.d_same_plus0 = float(D_SAME_PLUS0)
    model.params.d_same_0minus = float(D_SAME_0MINUS)
    model.params.d_spec_plus0 = float(D_SPEC_PLUS0)
    model.params.d_spec_0minus = float(D_SPEC_0MINUS)


def configure_ssrf_burn(
    model: Spin1Model,
    burn_idx: int,
    gamma_rf: float,
    *,
    rf_mode: str = RF_MODE_PHYSICAL_VOIGT,
    gaussian_fwhm_R: Optional[float] = None,
    lorentzian_fwhm_R: Optional[float] = None,
) -> str:
    """
    Install RF for one burn bin with shared spectral recovery.

    ``physical_voigt`` uses bin-averaged physical-R Voigt RF (``_rf_population_term``).
    ``single_bin`` uses a frozen one-bin profile (``ssrf_burn``).
    """
    mode = str(rf_mode)
    apply_shared_spectral_recovery(model)
    burn_idx = int(burn_idx)
    gamma = float(gamma_rf)
    if mode == RF_MODE_SINGLE_BIN:
        configure_single_bin_ssrf(
            model, burn_idx, gamma, apply_demo_recovery=False
        )
        apply_shared_spectral_recovery(model)
    elif mode == RF_MODE_PHYSICAL_VOIGT:
        g_fwhm = (
            RF_GAUSSIAN_FWHM_R if gaussian_fwhm_R is None else float(gaussian_fwhm_R)
        )
        l_fwhm = (
            RF_LORENTZIAN_FWHM_R
            if lorentzian_fwhm_R is None
            else float(lorentzian_fwhm_R)
        )
        configure_physical_voigt_ssrf(
            model,
            burn_idx,
            gamma,
            gaussian_fwhm_R=g_fwhm,
            lorentzian_fwhm_R=l_fwhm,
            full_spectrum_recovery=True,
        )
        # configure_physical_voigt_ssrf clears relax; restore shared recovery.
        apply_shared_spectral_recovery(model)
    else:
        raise ValueError(
            f"unknown rf_mode={mode!r}; expected "
            f"{RF_MODE_PHYSICAL_VOIGT!r} or {RF_MODE_SINGLE_BIN!r}"
        )
    # ssRF: keep Dulya event ``n_ref`` for hole filling; if P drifts under RF,
    # rebuild Boltzmann at the *initial* polarization (not the reduced P).
    model._sync_level_populations(capture_initial=False)
    p_init = float(model.n_plus - model.n_minus)
    model.set_recovery_boltzmann_P(p_init)
    return mode


def configure_ssrf_single_bin(model: Spin1Model, burn_idx: int, gamma_rf: float) -> None:
    """Backward-compatible alias for single-bin RF with shared recovery."""
    configure_ssrf_burn(
        model, burn_idx, gamma_rf, rf_mode=RF_MODE_SINGLE_BIN
    )


def configure_afp_recovery(model: Spin1Model) -> None:
    """Post-AFP relaxation: Boltzmann at the manipulated (post-AFP) vector P.

    Recovery drives Q → Q_boltz(P_AFP). Uses the same ``D_SAME_*`` / ``D_SPEC_*``
    rates as ssRF, no spin diffusion, and uniform capacity weighting.
    """
    model.params.relax_enabled = True
    model.params.d_same_plus0 = float(D_SAME_PLUS0)
    model.params.d_same_0minus = float(D_SAME_0MINUS)
    model.params.d_spec_plus0 = float(D_SPEC_PLUS0)
    model.params.d_spec_0minus = float(D_SPEC_0MINUS)
    model.params.diffusion_scale = 0.0
    model.params.capacity_rate_power = 0.0
    model._active_idx = None
    model.install_boltzmann_recovery_at_current_P()


def level_pq(model: Spin1Model) -> tuple[float, float]:
    """Return current vector P and tensor Q from stored level populations."""
    lp = model.level_populations()
    return float(lp["P"]), float(lp["Q"])


def intensities_at_bins(
    model: Spin1Model, bin_idx: int, mirror_idx: int
) -> tuple[float, float, float, float, float, float]:
    ip, im, _ = model.physical_intensities()
    iplus = float(ip[bin_idx])
    iminus = float(im[bin_idx])
    iplus_m = float(ip[mirror_idx])
    iminus_m = float(im[mirror_idx])
    return iplus, iminus, iplus + iminus, iplus_m, iminus_m, iplus_m + iminus_m


def full_spectrum_intensities(model: Spin1Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ip, im, total = model.physical_intensities()
    return np.asarray(ip, dtype=float), np.asarray(im, dtype=float), np.asarray(total, dtype=float)


def mirror_amplitude(ps_m: float) -> float:
    return abs(float(ps_m))


def mirror_amplitude_decreased(
    ps_m: float,
    ps_m_prev: float,
    *,
    atol: float = MIRROR_AMP_EPS,
    rtol: float = MIRROR_AMP_RTOL,
) -> bool:
    cur = mirror_amplitude(ps_m)
    prev = mirror_amplitude(ps_m_prev)
    return cur < prev - max(float(atol), float(rtol) * prev)


def euler_n_sub(gamma_rf: float, dt: float) -> tuple[int, float]:
    g = abs(float(gamma_rf))
    dt_f = float(dt)
    if g <= 0.0 or dt_f <= 0.0:
        return 1, dt_f
    n_sub = min(max(1, int(np.ceil(g * dt_f / MAX_GDT))), int(MAX_NSUB))
    return n_sub, dt_f / float(n_sub)


def traj_to_fit_scale(traj: dict, from_spin1: float) -> dict:
    scale = float(from_spin1)
    for key in ("ps", "iplus", "iminus", "ps_m", "iplus_m", "iminus_m"):
        if key in traj and np.asarray(traj[key]).size:
            traj[key] = np.asarray(traj[key], dtype=float) * scale
    if "ps0" in traj:
        traj["ps0"] = float(traj["ps0"]) * scale
    for key in ("ip_spectrum0", "im_spectrum0", "ip_spectrum", "im_spectrum"):
        if key in traj and traj[key] is not None:
            traj[key] = np.asarray(traj[key], dtype=float) * scale
    for key in ("ps_full", "iplus_full", "iminus_full"):
        if key in traj and traj[key] is not None:
            arr = np.asarray(traj[key], dtype=float)
            if arr.size:
                traj[key] = arr * scale
    return traj
