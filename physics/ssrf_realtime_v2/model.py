"""
Spin-1 Pake-doublet ss-RF real-time model (v2).

Capacity-weighted version of the spin-1 ss-RF model.  Local Pake spin-packet
density scales RF, DNP, same-theta recovery, and spectral-neighbor diffusion.
AFP, multi-burn ssRF, and intensity loading follow the v1 conventions.

With DNP off, RF is the only vector-polarization sink.  Internal recovery and
neighbor diffusion conserve the current reduced P(t).  With DNP on, a separate
external reservoir builds toward P_DNP_sat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .conversions import physical_intensities_to_packet_n, packet_n_to_physical_intensities
from .voigt_burn_physics import VoigtBurnPhysicsMixin
from .lineshape import (
    boltzmann_Q,
    boltzmann_branch_ratio,
    level_populations_from_PQ,
    normalized_component,
    pake_component_raw,
    trapezoid_integral,
)

PLUS, ZERO, MINUS = 0, 1, 2


def _clamp_p(P: float) -> float:
    """Keep P inside the physically safe open interval for numeric use."""
    return float(np.clip(float(P), -0.999999, 0.999999))


@dataclass
class Spin1Params:
    """Numerical and phenomenological parameters for the spin-1 model."""

    n_bins: int = 500
    r_min: float = -3.0
    r_max: float = 3.0

    line_gamma: float = 0.05
    line_asym: float = 0.04

    # Intensity display scale: ``p0`` (initial vector polarization) when
    # ``plot_signal_units`` is true; otherwise ``display_scale``.
    plot_signal_units: bool = True
    plot_divisor: float = 10.0
    display_scale: float = 1.0
    calibration_p: float = 0.50

    p0: float = 0.60
    q0: Optional[float] = None

    rf_burn_R: float = -0.92
    rf_enabled: bool = True
    gamma_rf: float = 2.0
    ssrf_subset_indices: Optional[List[int]] = None
    rf_profile: Optional[np.ndarray] = None
    # Multi-bin Voigt capacity: "center_shared" or "separate_branch" (w[kp], w[km] per pair).
    ssrf_multi_bin_capacity: str = "center_shared"

    # Physical-R Voigt RF (spin1_ssrf_realtime_voigt_burn). Legacy discrete profiles
    # from freeze_rf_profile / ssrf_subset_indices take precedence when installed.
    use_physical_voigt_rf: bool = False
    rf_gaussian_fwhm_R: float = 0.030
    rf_lorentzian_fwhm_R: float = 0.015
    rf_profile_normalization: str = "center_bin"
    rf_profile_quadrature_order: int = 0

    # Population-dependent spin-diffusion kernel (voigt_burn). diffusion_scale=0
    # disables it. Legacy mode recovery can run alongside spin diffusion when
    # ``relax_enabled`` is true (physical Voigt or discrete single-bin RF).
    diffusion_scale: float = 0.0
    zq_width_R: float = 0.05
    cross_branch_ratio: float = 0.0
    orientation_corr_fraction: float = 0.0
    orientation_corr_width_deg: float = 20.0
    kernel_cutoff_widths: float = 4.0
    microwave_diffusion_factor: float = 1.0

    relax_enabled: bool = True
    d_same_plus0: float = 0.18
    d_same_0minus: float = 0.10
    d_spec_plus0: float = 2.0
    d_spec_0minus: float = 1.0
    t2_width_R: float = 0.05

    # Local Pake-density / spin-packet capacity weighting.
    capacity_rate_power: float = 1.0
    capacity_rate_clip: float = 12.0

    ### DNP build/rebuild reservoir. ###
    dnp_enabled: bool = False
    p_dnp_sat: float = 0.58
    dnp_rate: float = 0.05

    t1_rate: float = 0.0
    t1_p_eq: float = 0.0

    dt: float = 0.0015
    noise_sigma: float = 0.0

    steps: int = 50

    # Instantaneous AFP: fired once before time stepping (apply_pending_afp / step), then cleared.
    afp_enabled: bool = False
    afp_efficiency: float = 1.0
    afp_center_margin: int = 0
    afp_preserve_intensity_area: bool = False
    afp_subset_indices: Optional[List[int]] = None


class Spin1Model(VoigtBurnPhysicsMixin):
    """Stateful spin-1 population model with ideal-bin ss-RF and optional DNP."""

    def __init__(
        self,
        params: Optional[Spin1Params] = None,
    ):
        self.params = params or Spin1Params()
        self.reset()

    def reset(self) -> None:
        p = self.params
        if p.n_bins < 5:
            raise ValueError("n_bins must be at least 5")
        if p.r_min >= p.r_max:
            raise ValueError("r_min must be less than r_max")
        self.Rplus = np.linspace(p.r_min, p.r_max, p.n_bins)
        self.dR = float(self.Rplus[1] - self.Rplus[0])

        density = normalized_component(self.Rplus, +1, gamma=p.line_gamma, asym=p.line_asym)
        mu = density * self.dR
        self.mu = mu / max(float(mu.sum()), 1e-30)
        self.base_density = self.mu / self.dR

        self.pref_initial = level_populations_from_PQ(_clamp_p(p.p0), p.q0)
        self.n_ref = self.mu[:, None] * self.pref_initial[None, :]
        self.n = self.n_ref.copy()
        self.n_initial = self.n.copy()
        self.t = 0.0

        self.display_cal = self._compute_display_calibration()
        self._populations_from_intensities = False
        self.n_plus = self.n_zero = self.n_minus = 0.0
        self.n_plus_initial = self.n_zero_initial = self.n_minus_initial = 0.0
        self._sync_level_populations(capture_initial=True)

        self._invalidate_branch_cache()
        self._capacity_cache_key = None
        self._capacity_cache = None
        self._rf_profile_cache_key = None
        self._rf_profile_cache = None
        self._diffusion_kernel_key = None
        self._same_i = np.empty(0, dtype=np.int64)
        self._same_j = np.empty(0, dtype=np.int64)
        self._same_base = np.empty(0, dtype=float)
        self._cross_i = np.empty(0, dtype=np.int64)
        self._cross_j = np.empty(0, dtype=np.int64)
        self._cross_base = np.empty(0, dtype=float)
        self._rf_profile_frozen = False
        self._active_idx: Optional[np.ndarray] = None
        self._window_radius: Optional[int] = None
        # Optional fixed vector P for Boltzmann rebuilds when P has moved (ssRF).
        self._recovery_boltzmann_P: Optional[float] = None
        # If True, always recover toward Boltzmann at ``_recovery_boltzmann_P``
        # (AFP). If False, keep event ``n_ref`` while P≈P₀ (ssRF hole filling).
        self._force_boltzmann_recovery: bool = False
        self._afp_pending: bool = bool(self.params.afp_enabled)
        self._afp_last_subset: List[int] = []

        self.ip_afp = None
        self.im_afp = None

        if not self.params.relax_enabled:
            self.params.d_same_plus0 = 0.0
            self.params.d_same_0minus = 0.0
            self.params.d_spec_plus0 = 0.0
            self.params.d_spec_0minus = 0.0

        if self.params.rf_enabled:
            self.set_rf_profile()

    def set_rf_profile(self) -> None:
        """Per-bin RF rate. Q-shaped profile peaks at deepest Q<0."""
        ip, im, _ = self.physical_intensities(self.n_initial)
        q = ip - im
        q_min = np.min(q)
        if q_min >= 0.0:
            self.params.rf_profile = np.zeros_like(q)
        else:
            self.params.rf_profile = self.params.gamma_rf * np.clip(q / q_min, 0.0, 1.0)

    def _compute_display_calibration(self) -> float:
        """Scale packet differences to displayed intensities using initial ``p0``."""
        p = self.params
        if not p.plot_signal_units:
            return float(p.display_scale)
        return float(_clamp_p(p.p0))

    def _plot_signal_reference_calibration(self) -> float:
        """Plot_Signal-style scale for ``static_plot_signal_reference`` comparisons only."""
        p = self.params
        if not p.plot_signal_units:
            return float(p.display_scale)
        Pcal = _clamp_p(p.calibration_p)
        pref_cal = level_populations_from_PQ(Pcal, None)
        minor_diff = abs(float(pref_cal[ZERO] - pref_cal[MINUS]))
        if minor_diff < 1e-15:
            pref_cal = level_populations_from_PQ(0.50, None)
            minor_diff = abs(float(pref_cal[ZERO] - pref_cal[MINUS]))
        raw = pake_component_raw(self.Rplus, +1, gamma=p.line_gamma, asym=p.line_asym)
        raw_area = trapezoid_integral(raw, self.Rplus)
        if raw_area <= 0 or not np.isfinite(raw_area):
            return float(p.display_scale)
        return float(p.display_scale * raw_area / (max(p.plot_divisor, 1e-15) * minor_diff))

    def set_params(self, **kwargs) -> None:
        rf_profile_changed = False
        diffusion_changed = False
        for key, value in kwargs.items():
            if not hasattr(self.params, key):
                raise AttributeError(f"Unknown parameter: {key}")
            setattr(self.params, key, value)
            if key in {
                "rf_burn_R",
                "rf_gaussian_fwhm_R",
                "rf_lorentzian_fwhm_R",
                "rf_profile_normalization",
                "rf_profile_quadrature_order",
                "use_physical_voigt_rf",
            }:
                rf_profile_changed = True
            if key in {
                "diffusion_scale",
                "zq_width_R",
                "cross_branch_ratio",
                "orientation_corr_fraction",
                "orientation_corr_width_deg",
                "kernel_cutoff_widths",
                "microwave_diffusion_factor",
                "line_asym",
                "n_bins",
                "r_min",
                "r_max",
            }:
                diffusion_changed = True
        if rf_profile_changed:
            self.invalidate_rf_profile()
        if diffusion_changed:
            self._diffusion_kernel_key = None

    def as_dict(self) -> Dict[str, float]:
        return asdict(self.params)

    def set_rf_enabled(self, enabled: bool) -> None:
        self.params.rf_enabled = bool(enabled)

    def set_dnp_enabled(self, enabled: bool) -> None:
        self.params.dnp_enabled = bool(enabled)

    def load_from_physical_intensities(self, Iplus: np.ndarray, Iminus: np.ndarray) -> None:
        """Set ``self.n`` from physical R-grid intensities using model conversions."""
        self.n = physical_intensities_to_packet_n(
            Iplus,
            Iminus,
            self.mu,
            display_cal=self.display_cal,
            dR=self.dR,
        )
        self.n_initial = self.n.copy()
        self.n_ref = self.n.copy()
        self._populations_from_intensities = True
        self._recovery_boltzmann_P = None
        self._force_boltzmann_recovery = False
        self._sync_level_populations(capture_initial=True)
        self._afp_pending = bool(self.params.afp_enabled)
        self._afp_last_subset = []
        if self.params.rf_enabled:
            self.set_rf_profile()

    @staticmethod
    def _resolve_afp_subset(
        n_bins: int,
        subset_indices: Optional[List[int]] = None,
        center_margin: int = 0,
    ) -> List[int]:
        if subset_indices is None:
            subset = list(range(n_bins))
        else:
            subset = [int(i) for i in subset_indices]
        if center_margin > 0:
            c = n_bins // 2
            forbidden = set(range(max(0, c - center_margin), min(n_bins, c + center_margin + 1)))
            subset = [i for i in subset if i not in forbidden]
        return subset

    @staticmethod
    def _perform_afp_on_populations(
        rho_plus: np.ndarray,
        rho_zero: np.ndarray,
        rho_minus: np.ndarray,
        sweep: List[int],
        efficiency: float = 1.0,
    ) -> None:
        """AFP sweep on per-bin populations (in place): n+↔n0 at i, n0↔n- at mirror."""
        n = len(rho_plus)
        eff = float(efficiency)
        for i in sweep:
            m = n - 1 - i
            rho_plus[i], rho_zero[i] = (
                eff * rho_zero[i] + (1.0 - eff) * rho_plus[i],
                eff * rho_plus[i] + (1.0 - eff) * rho_zero[i],
            )
            if m == i:
                continue
            rho_zero[m], rho_minus[m] = (
                eff * rho_minus[m] + (1.0 - eff) * rho_zero[m],
                eff * rho_zero[m] + (1.0 - eff) * rho_minus[m],
            )

    def afp_target_state(
        self,
        n: Optional[np.ndarray] = None,
        subset_indices: Optional[List[int]] = None,
        *,
        efficiency: float = 1.0,
        center_margin: int = 0,
    ) -> Tuple[np.ndarray, List[int]]:
        """Return the AFP-swapped packet state without mutating ``self.n``."""
        state = self.n if n is None else n
        out = np.array(state, dtype=float, copy=True)
        subset = self._resolve_afp_subset(len(out), subset_indices, center_margin)
        if subset:
            self._perform_afp_on_populations(
                out[:, PLUS],
                out[:, ZERO],
                out[:, MINUS],
                subset,
                efficiency=float(efficiency),
            )
        return out, subset

    def afp_sweep(self) -> List[int]:
        """
        Instantaneous AFP on ``subset_indices`` (sweep frequencies only).

        Each index i swaps n+↔n0 at i and n0↔n- at mirror(i). Do not also pass
        mirror indices in ``subset_indices`` or AFP is applied twice. Only those
        packets are written; all other bins are left unchanged.
        """
        ip_before, im_before, _ = self.physical_intensities()
        area_before = float(np.sum(ip_before + im_before))

        target, subset = self.afp_target_state(
            self.n,
            self.params.afp_subset_indices,
            efficiency=self.params.afp_efficiency,
            center_margin=self.params.afp_center_margin,
        )

        n_bins = len(self.n)
        touched = sorted(
            {int(i) for i in subset} | {n_bins - 1 - int(i) for i in subset}
        )
        for k in touched:
            self.n[k] = target[k]

        if self.params.afp_preserve_intensity_area and touched and abs(area_before) > 1e-30:
            self._renormalize_touched_intensity_area(touched, area_before)

        self._sync_level_populations(capture_initial=False)
        self._afp_last_subset = list(subset)
        self.ip_afp, self.im_afp, _ = self.physical_intensities()
        return subset

    def _ssrf_burn_pairs(self) -> List[Tuple[Optional[int], Optional[int]]]:
        subset = self.params.ssrf_subset_indices
        if subset is None:
            return [self.cached_branch_indices(self.params.rf_burn_R)]
        n_bins = len(self.n)
        pairs: List[Tuple[Optional[int], Optional[int]]] = []
        for raw in subset:
            i = int(raw)
            if 0 <= i < n_bins:
                pairs.append((i, n_bins - 1 - i))
        return pairs

    def ssrf_burn(self) -> np.ndarray:
        """Return RF population rates for each ssRF burn bin in params."""
        dn_rf = np.zeros_like(self.n)
        if float(self.params.gamma_rf) == 0.0:
            return dn_rf

        self.set_rf_profile()
        profile = np.asarray(self.params.rf_profile, dtype=float)
        w = self.capacity_rate_weights()

        subset = self.params.ssrf_subset_indices
        multi_bin = subset is not None
        center_kp, center_km = self.cached_branch_indices(self.params.rf_burn_R)
        w_center = (
            float(w[center_kp])
            if center_kp is not None
            else float(w[center_km])
            if center_km is not None
            else 1.0
        )

        for kp, km in self._ssrf_burn_pairs():
            if kp is not None:
                gamma_base = float(profile[kp])
            elif km is not None:
                gamma_base = float(profile[km])
            else:
                continue
            if gamma_base == 0.0:
                continue
            if multi_bin:
                mode = str(getattr(self.params, "ssrf_multi_bin_capacity", "center_shared"))
                if mode == "separate_branch":
                    gamma_plus = gamma_base * float(w[kp]) if kp is not None else 0.0
                    gamma_minus = gamma_base * float(w[km]) if km is not None else 0.0
                elif kp == center_kp and km == center_km:
                    gamma_plus = gamma_base * float(w[center_kp]) if center_kp is not None else 0.0
                    gamma_minus = gamma_base * float(w[center_km]) if center_km is not None else 0.0
                else:
                    gamma_plus = gamma_base * w_center
                    gamma_minus = gamma_base * w_center
            else:
                gamma_plus = gamma_base * w[kp] if kp is not None else 0.0
                gamma_minus = gamma_base * w[km] if km is not None else 0.0
            if kp is not None and gamma_plus != 0.0:
                J = gamma_plus * (self.n[kp, PLUS] - self.n[kp, ZERO])
                dn_rf[kp, PLUS] -= J
                dn_rf[kp, ZERO] += J
            if km is not None and gamma_minus != 0.0:
                J = gamma_minus * (self.n[km, ZERO] - self.n[km, MINUS])
                dn_rf[km, ZERO] -= J
                dn_rf[km, MINUS] += J
        return dn_rf

    def _renormalize_touched_intensity_area(
        self,
        touched: List[int],
        area_target: float,
    ) -> None:
        """Restore Σ(I⁺+I⁻) to ``area_target`` on ``touched`` bins without changing Q."""
        ip, im, _ = self.physical_intensities()
        ip_new = np.asarray(ip, dtype=float).copy()
        im_new = np.asarray(im, dtype=float).copy()
        touched_list = [int(k) for k in touched]
        touched_set = set(touched_list)

        area_unt = 0.0
        area_touch = 0.0
        weights = np.zeros(len(touched_list), dtype=float)
        for j, k in enumerate(touched_list):
            s = float(ip_new[k] + im_new[k])
            weights[j] = max(s, 0.0)
            area_touch += s
        for k in range(len(ip_new)):
            if k not in touched_set:
                area_unt += float(ip_new[k] + im_new[k])

        missing = float(area_target) - area_unt - area_touch
        if abs(missing) < 1e-15:
            return

        wsum = float(np.sum(weights))
        if wsum > 1e-30:
            for j, k in enumerate(touched_list):
                add = missing * (weights[j] / wsum)
                ip_new[k] += 0.5 * add
                im_new[k] += 0.5 * add
        else:
            add = missing / float(len(touched_list))
            for k in touched_list:
                ip_new[k] += 0.5 * add
                im_new[k] += 0.5 * add

        inv_scale = float(self.dR) / float(self.display_cal)
        n_bins = len(self.n)
        for k in touched_set:
            mirror = n_bins - 1 - k
            a = float(ip_new[k]) * inv_scale
            b = float(im_new[mirror]) * inv_scale
            n_zero = (float(self.mu[k]) - a + b) / 3.0
            self.n[k, PLUS] = n_zero + a
            self.n[k, ZERO] = n_zero
            self.n[k, MINUS] = n_zero - b
        self.n = np.maximum(self.n, 1e-30)
        for k in touched_set:
            row = float(self.n[k].sum())
            if row > 1e-30:
                self.n[k] *= float(self.mu[k]) / row

    def step(self, n_steps: int = 1, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None) -> None:
        if self.params.afp_enabled:
            self.afp_sweep()
            self.params.afp_enabled = False

        dt = float(self.params.dt)
        if rf_on is None:
            rf_on = bool(self.params.rf_enabled)
        if dnp_on is None:
            dnp_on = bool(self.params.dnp_enabled)
        for _ in range(max(1, int(n_steps))):
            self.step_once(dt=dt, rf_on=rf_on, dnp_on=dnp_on, copy=False)

    def step_once(
        self,
        dt: Optional[float] = None,
        rf_on: Optional[bool] = None,
        dnp_on: Optional[bool] = None,
        *,
        copy: bool = False,
    ) -> np.ndarray:
        """One Euler macro-step of RF / relaxation / DNP (no AFP)."""
        step_dt = float(self.params.dt if dt is None else dt)
        if rf_on is None:
            rf_on = bool(self.params.rf_enabled)
        if dnp_on is None:
            dnp_on = bool(self.params.dnp_enabled)

        dn = self.derivative(rf_on=rf_on, dnp_on=dnp_on, breakdown=False)
        active = self._active_idx
        if active is None:
            self.n = self.n + step_dt * dn
            self.n = np.maximum(self.n, 1e-30)
            sums = self.n.sum(axis=1, keepdims=True)
            self.n *= self.mu[:, None] / np.maximum(sums, 1e-30)
        elif active.size > 0:
            self.n[active] = self.n[active] + step_dt * dn[active]
            self.n[active] = np.maximum(self.n[active], 1e-30)
            sums = self.n[active].sum(axis=1, keepdims=True)
            self.n[active] *= self.mu[active, None] / np.maximum(sums, 1e-30)
        self.t += step_dt
        self._sync_level_populations(capture_initial=False)
        return self.n.copy() if copy else self.n

    def _sync_level_populations(self, *, capture_initial: bool = False) -> None:
        """Refresh stored global level fractions ``n_plus``, ``n_zero``, ``n_minus``."""
        if self._populations_from_intensities:
            ip, im, _ = self.physical_intensities()
            p_raw = float(np.sum(ip + im))
            q_raw = float(np.sum(ip - im))
            if abs(p_raw) <= 1.0 + 1e-9:
                p_vec, q_ten = p_raw, q_raw
            else:
                scale = float(self.dR) / max(abs(float(self.display_cal)), 1e-30)
                p_vec, q_ten = p_raw * scale, q_raw * scale
            try:
                pref = level_populations_from_PQ(p_vec, q_ten)
            except ValueError:
                pref = level_populations_from_PQ(p_vec, None)
        else:
            pops = np.sum(self.n, axis=0)
            total = max(float(pops.sum()), 1e-30)
            pref = pops / total
        self.n_plus = float(pref[PLUS])
        self.n_zero = float(pref[ZERO])
        self.n_minus = float(pref[MINUS])
        if capture_initial:
            self.n_plus_initial = self.n_plus
            self.n_zero_initial = self.n_zero
            self.n_minus_initial = self.n_minus

    def level_populations(self) -> Dict[str, float]:
        """Return stored global level fractions and derived P, Q (and initials)."""
        return {
            "n_plus": self.n_plus,
            "n_zero": self.n_zero,
            "n_minus": self.n_minus,
            "P": self.n_plus - self.n_minus,
            "Q": self.n_plus - 2.0 * self.n_zero + self.n_minus,
            "n_plus_initial": self.n_plus_initial,
            "n_zero_initial": self.n_zero_initial,
            "n_minus_initial": self.n_minus_initial,
            "P_initial": self.n_plus_initial - self.n_minus_initial,
            "Q_initial": self.n_plus_initial - 2.0 * self.n_zero_initial + self.n_minus_initial,
        }

    def _invalidate_branch_cache(self) -> None:
        self._cached_rf_burn_R: Optional[float] = None
        self._cached_kp: Optional[int] = None
        self._cached_km: Optional[int] = None

    def cached_branch_indices(self, R: Optional[float] = None) -> Tuple[Optional[int], Optional[int]]:
        """Like ``branch_indices``, but cached while ``rf_burn_R`` (or ``R``) is unchanged."""
        if R is None:
            R = self.params.rf_burn_R
        R = float(R)
        if self._cached_rf_burn_R is not None and abs(self._cached_rf_burn_R - R) <= 1e-15:
            return self._cached_kp, self._cached_km
        kp, km = self.branch_indices(R)
        self._cached_rf_burn_R = R
        self._cached_kp = kp
        self._cached_km = km
        return kp, km

    def recovery_reference(self) -> np.ndarray:
        """Packet state that defines the recovery null manifold (initial event shape)."""
        return self.n_initial

    def equilibrium_reference(self, P: Optional[float] = None) -> np.ndarray:
        """Boltzmann-shaped packet state with the current grid weights (v15)."""
        if P is None:
            P = self.polarizations()["P"]
        pref = level_populations_from_PQ(_clamp_p(float(P)), None)
        return self.mu[:, None] * pref[None, :]

    def recovery_dynamic_reference(self) -> np.ndarray:
        """Reference for mode recovery (Boltzmann at P(t), or loaded event shape)."""
        if self._populations_from_intensities:
            return self.recovery_equilibrium_reference()
        return self.equilibrium_reference()

    def _boltzmann_packet_at_vector_p(self, P: float) -> np.ndarray:
        """Boltzmann packet state at vector ``P`` in the current intensity basis."""
        pref = level_populations_from_PQ(float(P), None)
        if not self._populations_from_intensities:
            return self.mu[:, None] * pref[None, :]

        n_ideal = self.mu[:, None] * pref[None, :]
        ip, im, _ = packet_n_to_physical_intensities(
            n_ideal, self.Rplus, display_cal=1.0, dR=self.dR
        )
        area = float(np.sum(ip + im))
        if abs(area) > 1e-30:
            scale = float(P) / area
            ip = ip * scale
            im = im * scale
        return physical_intensities_to_packet_n(
            ip, im, self.mu, display_cal=self.display_cal, dR=self.dR
        )

    def set_recovery_boltzmann_P(self, P: float) -> float:
        """Use vector ``P`` when rebuilding Boltzmann after P drifts (ssRF).

        Does not replace event ``n_ref`` — hole filling still targets the loaded
        lineshape while P≈P₀.
        """
        self._recovery_boltzmann_P = float(P)
        self._force_boltzmann_recovery = False
        return self._recovery_boltzmann_P

    def install_boltzmann_recovery_at_P(self, P: float) -> float:
        """Always recover toward Boltzmann at vector ``P``; leave ``n`` unchanged.

        Used after AFP (manipulated P). Overwrites ``n_ref`` with that Boltzmann.
        """
        P = float(P)
        self._recovery_boltzmann_P = P
        self._force_boltzmann_recovery = True
        self.n_ref = self._boltzmann_packet_at_vector_p(P)
        return P

    def install_boltzmann_recovery_at_current_P(self) -> float:
        """Set recovery to Boltzmann at the current (post-manipulation) vector P."""
        self._sync_level_populations(capture_initial=False)
        return self.install_boltzmann_recovery_at_P(float(self.n_plus - self.n_minus))

    def recovery_equilibrium_reference(self) -> np.ndarray:
        """Null manifold for mode recovery.

        AFP (``_force_boltzmann_recovery``): always Boltzmann at the fixed
        (manipulated) vector P so Q → Q_boltz(P).

        ssRF / intensity-loaded events: always the loaded event shape ``n_ref``
        (Dulya at the initial polarization). That way RF-mode recovery only
        fills burn holes; unburned bins are already on the null manifold and
        do not drift toward a global Boltzmann reshape when P dips under RF.
        """
        if self._force_boltzmann_recovery and self._recovery_boltzmann_P is not None:
            return self._boltzmann_packet_at_vector_p(float(self._recovery_boltzmann_P))

        if self._populations_from_intensities:
            return self.n_ref

        return self.equilibrium_reference(self.n_plus - self.n_minus)

    def capacity_rate_weights(self) -> np.ndarray:
        """Return Pake-density rate multipliers with mu-weighted average one."""
        p = self.params
        power = max(0.0, float(p.capacity_rate_power))
        clip = max(1.0, float(p.capacity_rate_clip))
        key = (power, clip)
        if getattr(self, "_capacity_cache_key", None) == key and getattr(self, "_capacity_cache", None) is not None:
            return self._capacity_cache
        if power == 0.0:
            w = np.ones_like(self.mu)
            self._capacity_cache_key = key
            self._capacity_cache = w
            return w

        density = np.maximum(self.base_density, 0.0)
        avg_density = float(np.average(density, weights=self.mu))
        if avg_density <= 0.0 or not np.isfinite(avg_density):
            return np.ones_like(self.mu)

        with np.errstate(divide="ignore", invalid="ignore"):
            w = (density / avg_density) ** power
        w = np.nan_to_num(w, nan=0.0, posinf=clip, neginf=0.0)
        w = np.clip(w, 1.0 / clip, clip)

        norm = float(np.average(w, weights=self.mu))
        if norm > 0.0 and np.isfinite(norm):
            w = w / norm
        self._capacity_cache_key = key
        self._capacity_cache = w
        return w

    def local_capacity_factors(self, R: Optional[float] = None) -> Dict[str, float]:
        """Return local Pake-density capacities/rate weights at a physical R."""
        if R is None:
            R = self.params.rf_burn_R
        kp, km = self.branch_indices(R)
        w = self.capacity_rate_weights()
        out = {
            "R": float(R),
            "w_Iplus_R": np.nan,
            "w_Iminus_R": np.nan,
            "density_Iplus_R": np.nan,
            "density_Iminus_R": np.nan,
            "mu_Iplus_R": np.nan,
            "mu_Iminus_R": np.nan,
        }
        if kp is not None:
            out["w_Iplus_R"] = float(w[kp])
            out["density_Iplus_R"] = float(self.base_density[kp])
            out["mu_Iplus_R"] = float(self.mu[kp])
        if km is not None:
            out["w_Iminus_R"] = float(w[km])
            out["density_Iminus_R"] = float(self.base_density[km])
            out["mu_Iminus_R"] = float(self.mu[km])
        return out

    def effective_local_rates(self, R: Optional[float] = None) -> Dict[str, float]:
        """Return actual local rate multipliers at the selected physical R."""
        if self._uses_physical_voigt_rf():
            return self._physical_effective_local_rates(R)
        cap = self.local_capacity_factors(R)
        p = self.params
        wp = cap["w_Iplus_R"]
        wm = cap["w_Iminus_R"]
        return {
            **cap,
            "gamma_rf_Iplus_R": float(p.gamma_rf * wp) if np.isfinite(wp) else np.nan,
            "gamma_rf_Iminus_R": float(p.gamma_rf * wm) if np.isfinite(wm) else np.nan,
            "dnp_Iplus_R": float(p.dnp_rate * wp) if np.isfinite(wp) else np.nan,
            "dnp_Iminus_R": float(p.dnp_rate * wm) if np.isfinite(wm) else np.nan,
            "same_plus0_Iplus_R": float(p.d_same_plus0 * wp) if np.isfinite(wp) else np.nan,
            "same_0minus_Iminus_R": float(p.d_same_0minus * wm) if np.isfinite(wm) else np.nan,
        }

    def branch_areas(self, n: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Return display-calibrated integrated branch areas and total area (v15)."""
        if n is None:
            n = self.n
        a_plus = float(self.display_cal * np.sum(n[:, PLUS] - n[:, ZERO]))
        a_minus = float(self.display_cal * np.sum(n[:, ZERO] - n[:, MINUS]))
        return {
            "A_plus": a_plus,
            "A_minus": a_minus,
            "A_total": a_plus + a_minus,
            "A_diff": a_plus - a_minus,
        }

    def static_plot_signal_reference(self):
        """Return the Plot_Signal-style static reference for comparison (v15)."""
        from .lineshape import plot_signal_reference

        return plot_signal_reference(
            self.Rplus,
            P=self.params.p0,
            gamma=self.params.line_gamma,
            asym=self.params.line_asym,
            divisor=self.params.plot_divisor,
        )

    def polarizations(self, n: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Return current dimensionless level populations, vector P, and tensor Q."""
        if n is None:
            n = self.n
        pops = np.sum(n, axis=0)
        P = float(pops[PLUS] - pops[MINUS])
        Q = float(pops[PLUS] - 2.0 * pops[ZERO] + pops[MINUS])
        return {
            "n_plus": float(pops[PLUS]),
            "n_zero": float(pops[ZERO]),
            "n_minus": float(pops[MINUS]),
            "P": P,
            "Q": Q,
            "Q_boltz_at_P": boltzmann_Q(_clamp_p(P)),
        }

    def branch_indices(self, R: Optional[float] = None) -> Tuple[Optional[int], Optional[int]]:
        """Return packet indices for the two components at physical R."""
        if R is None:
            R = self.params.rf_burn_R
        R = float(R)
        kp = int(np.argmin(np.abs(self.Rplus - R))) if self.Rplus[0] <= R <= self.Rplus[-1] else None
        km = int(np.argmin(np.abs(self.Rplus + R))) if self.Rplus[0] <= -R <= self.Rplus[-1] else None
        return kp, km

    def burn_index(self, R: Optional[float] = None) -> int:
        """Index of physical +R on the symmetric grid (nearest bin to ``R``)."""
        if R is None:
            R = self.params.rf_burn_R
        return int(np.argmin(np.abs(self.Rplus - float(R))))

    def _transition_differences(self, n: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if n is None:
            n = self.n
        return n[:, PLUS] - n[:, ZERO], n[:, ZERO] - n[:, MINUS]

    def packet_intensities(self, use_reference: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Return display-calibrated I_plus(x) and I_minus(x-packet) arrays."""
        n = self.n_ref if use_reference else self.n
        Iplus, Iminus = self._transition_differences(n)
        Iplus = self.display_cal * Iplus / self.dR
        Iminus = self.display_cal * Iminus / self.dR
        return Iplus, Iminus

    def physical_intensities(self, n: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return I+(R), I-(R), and total on the model's physical R grid."""
        state = self.n if n is None else n
        return packet_n_to_physical_intensities(
            state, self.Rplus, display_cal=self.display_cal, dR=self.dR
        )

    def local_intensities(self, R: Optional[float] = None, use_reference: bool = False) -> Dict[str, float]:
        kp, km = self.branch_indices(R)
        Iplus_packet, Iminus_packet = self.packet_intensities(use_reference=use_reference)
        Iplus = np.nan if kp is None else float(Iplus_packet[kp])
        Iminus = np.nan if km is None else float(Iminus_packet[km])
        total = 0.0
        if np.isfinite(Iplus):
            total += Iplus
        if np.isfinite(Iminus):
            total += Iminus
        return {
            "Iplus": Iplus,
            "Iminus": Iminus,
            "total": float(total),
            "k_plus": -1 if kp is None else int(kp),
            "k_minus": -1 if km is None else int(km),
        }

    def pair_intensities(self, R: Optional[float] = None, use_reference: bool = False) -> Dict[str, float]:
        if R is None:
            R = self.params.rf_burn_R
        direct = self.local_intensities(R, use_reference=use_reference)
        mirror = self.local_intensities(-float(R), use_reference=use_reference)
        return {
            "Iplus_R": float(direct["Iplus"]),
            "Iminus_R": float(direct["Iminus"]),
            "Itotal_R": float(direct["total"]),
            "Iplus_minusR": float(mirror["Iplus"]),
            "Iminus_minusR": float(mirror["Iminus"]),
            "Itotal_minusR": float(mirror["total"]),
            "k_plus_R": int(direct["k_plus"]),
            "k_minus_R": int(direct["k_minus"]),
        }

    def response_values(self, R: Optional[float] = None) -> Dict[str, float]:
        if R is None:
            R = self.params.rf_burn_R
        now = self.pair_intensities(R, use_reference=False)
        ref = self.pair_intensities(R, use_reference=True)
        out = dict(now)
        out.update({
            "R": float(R),
            "dIplus_R": now["Iplus_R"] - ref["Iplus_R"],
            "dIminus_R": now["Iminus_R"] - ref["Iminus_R"],
            "dIplus_minusR": now["Iplus_minusR"] - ref["Iplus_minusR"],
            "dIminus_minusR": now["Iminus_minusR"] - ref["Iminus_minusR"],
            "Iplus_R_ref": ref["Iplus_R"],
            "Iminus_R_ref": ref["Iminus_R"],
            "Iplus_minusR_ref": ref["Iplus_minusR"],
            "Iminus_minusR_ref": ref["Iminus_minusR"],
        })
        return out

    def spectrum_from_state(self, n: np.ndarray):
        Iplus_packet, Iminus_packet = self._transition_differences(n)
        Iplus_packet = self.display_cal * Iplus_packet / self.dR
        Iminus_packet = self.display_cal * Iminus_packet / self.dR
        R_axis = self.Rplus.copy()
        Iplus = Iplus_packet.copy()
        Rminus_phys = -self.Rplus
        order = np.argsort(Rminus_phys)
        Iminus = np.interp(R_axis, Rminus_phys[order], Iminus_packet[order], left=0.0, right=0.0)
        return R_axis, Iplus, Iminus, Iplus + Iminus

    def spectrum(self, noise_sigma: Optional[float] = None):
        R_axis, Iplus, Iminus, total = self.spectrum_from_state(self.n)
        if noise_sigma is None:
            noise_sigma = self.params.noise_sigma
        if noise_sigma and noise_sigma > 0:
            rng = np.random.default_rng()
            Iplus = Iplus + rng.normal(0.0, noise_sigma, size=Iplus.shape)
            Iminus = Iminus + rng.normal(0.0, noise_sigma, size=Iminus.shape)
            total = Iplus + Iminus
        return R_axis, Iplus, Iminus, total

    def reference_spectrum(self):
        return self.spectrum_from_state(self.n_ref)

    def packet_spectrum(self, use_reference: bool = False, noise_sigma: Optional[float] = None):
        """Return branch component spectra at their physical R bin centers."""
        Iplus_packet, Iminus_packet = self.packet_intensities(use_reference=use_reference)
        Rplus_phys = self.Rplus.copy()
        Rminus_phys = -self.Rplus.copy()
        order = np.argsort(Rminus_phys)
        Rminus_ordered = Rminus_phys[order]
        Iminus_ordered = Iminus_packet[order]
        if noise_sigma is None:
            noise_sigma = self.params.noise_sigma
        if noise_sigma and noise_sigma > 0:
            rng = np.random.default_rng()
            Iplus_packet = Iplus_packet + rng.normal(0.0, noise_sigma, size=Iplus_packet.shape)
            Iminus_ordered = Iminus_ordered + rng.normal(0.0, noise_sigma, size=Iminus_ordered.shape)
        return Rplus_phys, Iplus_packet, Rminus_ordered, Iminus_ordered

    def _rf_mode_amplitudes(self, reference: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if reference is None:
            reference = self.recovery_equilibrium_reference()
        delta = self.n - reference
        a_plus0 = -2.0 * delta[:, PLUS]
        b_0minus = 2.0 * delta[:, MINUS]
        return a_plus0, b_0minus

    def _mode_to_population_derivative(self, da_dt: np.ndarray, db_dt: np.ndarray) -> np.ndarray:
        dn = np.zeros_like(self.n)
        dn[:, PLUS] += -0.5 * da_dt
        dn[:, ZERO] += 0.5 * da_dt - 0.5 * db_dt
        dn[:, MINUS] += 0.5 * db_dt
        return dn

    def _project_conserve_vector(self, dn: np.ndarray) -> np.ndarray:
        dP = float(np.sum(dn[:, PLUS] - dn[:, MINUS]))
        if abs(dP) < 1e-18:
            return dn
        correction = np.zeros_like(dn)
        correction[:, PLUS] -= 0.5 * dP * self.mu
        correction[:, MINUS] += 0.5 * dP * self.mu
        return dn + correction

    def _mode_relax_reference(self, which: str, rate: float, reference: np.ndarray) -> np.ndarray:
        """Same-bin backpath: decay an RF-created mode toward the current reference."""
        if rate == 0.0:
            return np.zeros_like(self.n)
        a_plus0, b_0minus = self._rf_mode_amplitudes(reference)
        w = self.capacity_rate_weights()
        if which == "plus0":
            return self._mode_to_population_derivative(-rate * w * a_plus0, np.zeros_like(b_0minus))
        if which == "0minus":
            return self._mode_to_population_derivative(np.zeros_like(a_plus0), -rate * w * b_0minus)
        raise ValueError("which must be 'plus0' or '0minus'")

    def _mode_diffuse_delta(self, which: str, rate: float, reference: np.ndarray) -> np.ndarray:
        """Conservative nearest-neighbor diffusion of an RF-created hole mode."""
        if rate == 0.0 or len(self.Rplus) < 2:
            return np.zeros_like(self.n)
        a_plus0, b_0minus = self._rf_mode_amplitudes(reference)
        if which == "plus0":
            mode = a_plus0
        elif which == "0minus":
            mode = b_0minus
        else:
            raise ValueError("which must be 'plus0' or '0minus'")

        rho_mode = mode / np.maximum(self.mu, 1e-30)
        dR_edges = self.Rplus[1:] - self.Rplus[:-1]
        width = max(self.params.t2_width_R, 1e-12)
        overlap = np.exp(-0.5 * (dR_edges / width) ** 2)
        edge_mass = np.sqrt(self.mu[:-1] * self.mu[1:])
        w = self.capacity_rate_weights()
        edge_rate_weight = np.sqrt(w[:-1] * w[1:])

        flux = rate * edge_rate_weight * overlap * edge_mass * (rho_mode[1:] - rho_mode[:-1])
        dmode_dt = np.zeros_like(mode)
        dmode_dt[:-1] += flux
        dmode_dt[1:] -= flux

        if which == "plus0":
            return self._mode_to_population_derivative(dmode_dt, np.zeros_like(mode))
        return self._mode_to_population_derivative(np.zeros_like(mode), dmode_dt)

    def _spectral_edge_rates(self, which: str) -> np.ndarray:
        """Per-edge spectral diffusion rates for observability."""
        p = self.params
        rate = p.d_spec_plus0 if which == "plus0" else p.d_spec_0minus
        if rate == 0.0 or len(self.Rplus) < 2:
            return np.zeros(max(0, len(self.Rplus) - 1), dtype=float)
        dR_edges = self.Rplus[1:] - self.Rplus[:-1]
        width = max(float(p.t2_width_R), 1e-12)
        overlap = np.exp(-0.5 * (dR_edges / width) ** 2)
        edge_mass = np.sqrt(self.mu[:-1] * self.mu[1:])
        w = self.capacity_rate_weights()
        edge_rate_weight = np.sqrt(w[:-1] * w[1:])
        return rate * edge_rate_weight * overlap * edge_mass

    def local_recovery_rates(self, R: Optional[float] = None) -> Dict[str, float]:
        if R is None:
            R = self.params.rf_burn_R
        kp, km = self.branch_indices(R)
        w = self.capacity_rate_weights()
        p = self.params

        def edge(a: np.ndarray, k: Optional[int], side: str) -> float:
            if k is None:
                return float("nan")
            idx = k - 1 if side == "left" else k
            if idx < 0 or idx >= len(a):
                return 0.0
            return float(a[idx] / max(self.mu[k], 1e-30))

        edge_plus = self._spectral_edge_rates("plus0")
        edge_minus = self._spectral_edge_rates("0minus")
        _, parts = self.derivative(rf_on=False, dnp_on=self.params.dnp_enabled, breakdown=True)

        def same(k: Optional[int], base: float) -> float:
            if k is None:
                return float("nan")
            return float(base * w[k])

        return {
            "R": float(R),
            "k_Iplus": -1 if kp is None else int(kp),
            "k_Iminus": -1 if km is None else int(km),
            "Iplus_same_theta": same(kp, p.d_same_plus0),
            "Iplus_neighbor_left": edge(edge_plus, kp, "left"),
            "Iplus_neighbor_right": edge(edge_plus, kp, "right"),
            "Iplus_capacity_weight": float(w[kp]) if kp is not None else float("nan"),
            "Iplus_refill_dt_no_rf": float(parts["net"].get("dIplus_R_dt", float("nan"))),
            "Iminus_same_theta": same(km, p.d_same_0minus),
            "Iminus_neighbor_left": edge(edge_minus, km, "left"),
            "Iminus_neighbor_right": edge(edge_minus, km, "right"),
            "Iminus_capacity_weight": float(w[km]) if km is not None else float("nan"),
            "Iminus_refill_dt_no_rf": float(parts["net"].get("dIminus_R_dt", float("nan"))),
        }

    def recovery_pathway_rates(self, R: Optional[float] = None) -> Dict[str, float]:
        local = self.local_recovery_rates(R)
        return {
            "R": float(local["R"]),
            "k_plus": int(local["k_Iplus"]),
            "k_minus": int(local["k_Iminus"]),
            "same_plus0_eff": float(local["Iplus_same_theta"]),
            "same_0minus_eff": float(local["Iminus_same_theta"]),
            "neighbor_plus_left_eff": float(local["Iplus_neighbor_left"]),
            "neighbor_plus_right_eff": float(local["Iplus_neighbor_right"]),
            "neighbor_0minus_left_eff": float(local["Iminus_neighbor_left"]),
            "neighbor_0minus_right_eff": float(local["Iminus_neighbor_right"]),
            "left_plus0_eff": float(local["Iplus_neighbor_left"]),
            "right_plus0_eff": float(local["Iplus_neighbor_right"]),
            "left_0minus_eff": float(local["Iminus_neighbor_left"]),
            "right_0minus_eff": float(local["Iminus_neighbor_right"]),
            "capacity_weight_plus0": float(local["Iplus_capacity_weight"]),
            "capacity_weight_0minus": float(local["Iminus_capacity_weight"]),
            "Iplus_refill_dt_no_rf": float(local["Iplus_refill_dt_no_rf"]),
            "Iminus_refill_dt_no_rf": float(local["Iminus_refill_dt_no_rf"]),
        }

    def derivative(self, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None, breakdown: bool = False):
        if rf_on is None:
            rf_on = bool(self.params.rf_enabled)
        if dnp_on is None:
            dnp_on = bool(self.params.dnp_enabled)
        dn_terms: Dict[str, np.ndarray] = {}
        if rf_on:
            if self._uses_physical_voigt_rf():
                dn_rf = self._rf_population_term(True)
            else:
                dn_rf = self.ssrf_burn()
        else:
            dn_rf = np.zeros_like(self.n)
        dn_terms["RF"] = dn_rf

        if float(self.params.diffusion_scale) > 0.0:
            dn_terms.update(self._spin_diffusion_terms(bool(dnp_on)))
        if self.params.relax_enabled:
            dynamic_ref = self.recovery_dynamic_reference()

            dn_same = np.zeros_like(self.n)
            dn_same += self._mode_relax_reference("plus0", self.params.d_same_plus0, dynamic_ref)
            dn_same += self._mode_relax_reference("0minus", self.params.d_same_0minus, dynamic_ref)
            dn_same = self._project_conserve_vector(dn_same)
            dn_terms["spin_temp_redistribution"] = dn_same

            dn_spec = np.zeros_like(self.n)
            dn_spec += self._mode_diffuse_delta("plus0", self.params.d_spec_plus0, dynamic_ref)
            dn_spec += self._mode_diffuse_delta("0minus", self.params.d_spec_0minus, dynamic_ref)
            dn_spec = self._project_conserve_vector(dn_spec)
            dn_terms["spectral_neighbors"] = dn_spec

        dn_dnp = np.zeros_like(self.n)
        if dnp_on and self.params.dnp_rate != 0.0:
            dnp_target = self.equilibrium_reference(_clamp_p(self.params.p_dnp_sat))
            w = self.capacity_rate_weights()[:, None]
            dn_dnp = self.params.dnp_rate * w * (dnp_target - self.n)
        dn_terms["DNP_sat"] = dn_dnp

        dn_t1 = np.zeros_like(self.n)
        if self.params.t1_rate != 0.0:
            t1_target = self.equilibrium_reference(_clamp_p(self.params.t1_p_eq))
            dn_t1 = self.params.t1_rate * (t1_target - self.n)
        dn_terms["T1"] = dn_t1

        dn = sum(dn_terms.values())
        active = self._active_idx
        if active is not None:
            masked = np.zeros_like(dn)
            if active.size > 0:
                masked[active] = dn[active]
            dn = masked
            if not breakdown:
                return dn
            for name in dn_terms:
                term = np.zeros_like(dn_terms[name])
                if active.size > 0:
                    term[active] = dn_terms[name][active]
                dn_terms[name] = term

        if not breakdown:
            return dn

        kp, km = self.cached_branch_indices(self.params.rf_burn_R)
        scale = self.display_cal / self.dR

        def obs_for_term(term: np.ndarray) -> Dict[str, float]:
            d = {
                "dIplus_R_dt": np.nan,
                "dIminus_R_dt": np.nan,
                "dIplus_minusR_dt": np.nan,
                "dIminus_minusR_dt": np.nan,
                "dP_dt": float(np.sum(term[:, PLUS] - term[:, MINUS])),
            }
            if kp is not None:
                d["dIplus_R_dt"] = float(scale * (term[kp, PLUS] - term[kp, ZERO]))
                d["dIminus_minusR_dt"] = float(scale * (term[kp, ZERO] - term[kp, MINUS]))
            if km is not None:
                d["dIminus_R_dt"] = float(scale * (term[km, ZERO] - term[km, MINUS]))
                d["dIplus_minusR_dt"] = float(scale * (term[km, PLUS] - term[km, ZERO]))
            return d

        obs_terms: Dict[str, Dict[str, float]] = {name: obs_for_term(term) for name, term in dn_terms.items()}
        obs_terms["net"] = obs_for_term(dn)
        return dn, obs_terms

    def rf_balance_estimate(self, R: Optional[float] = None) -> Dict[str, float]:
        """Estimate common RF rate needed to hold the two direct components."""
        old_R = self.params.rf_burn_R
        if R is not None:
            self.params.rf_burn_R = float(R)
            self._invalidate_branch_cache()
            self.invalidate_rf_profile()
        try:
            _, parts = self.derivative(rf_on=False, dnp_on=self.params.dnp_enabled, breakdown=True)
            if self._uses_physical_voigt_rf():
                unit_term = self._rf_population_term(True, self.params.rf_burn_R, gamma_rf=1.0)
                kp, km = self.branch_indices(self.params.rf_burn_R)
                scale = self.display_cal / self.dR

                u_plus = (
                    np.nan
                    if kp is None
                    else float(scale * (unit_term[kp, PLUS] - unit_term[kp, ZERO]))
                )
                u_minus = (
                    np.nan
                    if km is None
                    else float(scale * (unit_term[km, ZERO] - unit_term[km, MINUS]))
                )
                refill_plus = float(parts["net"]["dIplus_R_dt"])
                refill_minus = float(parts["net"]["dIminus_R_dt"])

                def required(refill: float, unit_slope: float) -> float:
                    if not np.isfinite(refill) or not np.isfinite(unit_slope) or abs(unit_slope) < 1e-20:
                        return np.nan
                    value = -refill / unit_slope
                    return float(max(0.0, value)) if np.isfinite(value) else np.nan

                gp = required(refill_plus, u_plus)
                gm = required(refill_minus, u_minus)
            else:
                loc = self.local_intensities(self.params.rf_burn_R)
                Ip = loc["Iplus"]
                Im = loc["Iminus"]
                refill_p = parts["net"]["dIplus_R_dt"]
                refill_m = parts["net"]["dIminus_R_dt"]
                gp = max(0.0, refill_p / (2.0 * Ip)) if np.isfinite(Ip) and abs(Ip) > 0 else np.nan
                gm = max(0.0, refill_m / (2.0 * Im)) if np.isfinite(Im) and abs(Im) > 0 else np.nan
            vals = [v for v in [gp, gm] if np.isfinite(v)]
            common = max(vals) if vals else np.nan
            return {
                "gamma_hold_Iplus": float(gp),
                "gamma_hold_Iminus": float(gm),
                "gamma_common_suggested": float(common),
            }
        finally:
            self.params.rf_burn_R = old_R
            self._invalidate_branch_cache()
            self.invalidate_rf_profile()

    @property
    def branch_ratio(self) -> float:
        return boltzmann_branch_ratio(self.polarizations()["P"])

    @property
    def initial_branch_ratio(self) -> float:
        return boltzmann_branch_ratio(self.params.p0)
