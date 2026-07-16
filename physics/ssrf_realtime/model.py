"""
Spin-1 Pake-doublet ss-RF real-time model.

Position-dependent recovery pathways recompute local coefficients at every step
from the selected physical R bin, mirror branch, initial event reference distance,
neighboring bins, and population availability.

With DNP off, RF is the only vector-polarization sink.  Internal recovery and
neighbor diffusion conserve the current reduced P(t).  With DNP on, a separate
external reservoir builds toward P_DNP_sat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .conversions import physical_intensities_to_packet_n, packet_n_to_physical_intensities
from .lineshape import (
    boltzmann_Q,
    boltzmann_branch_ratio,
    level_populations_from_PQ,
    normalized_component,
)

PLUS, ZERO, MINUS = 0, 1, 2





@dataclass
class Spin1Params:
    """Numerical and phenomenological parameters for the spin-1 model."""

    n_bins: int = 500
    r_min: float = -3.0
    r_max: float = 3.0

    line_gamma: float = 0.05
    line_asym: float = 0.04

    p0: float = 0.60
    q0: Optional[float] = None
    initial_polarization: Optional[float] = None

    rf_burn_R: float = -0.92
    rf_enabled: bool = True
    gamma_rf: float = 2.0
    ssrf_subset_indices: Optional[List[int]] = None
    rf_profile: Optional[np.ndarray] = None

    relax_enabled: bool = True
    d_same_plus0: float = 0.25
    d_same_0minus: float = 0.15
    d_spec_plus0: float = 1.5
    d_spec_0minus: float = 0.8
    same_theta_mirror_gain: float = 1.5
    
    boltzmann_distance_gain: float = 0.5
    population_availability: float = 1.0
    t2_width_R: float = 0.05

    r_dependent_recovery: bool = True
    recovery_position_power: float = 1.0
    recovery_position_floor: float = 0.03
    recovery_rate_clip: float = 20.0

    ### DNP build/rebuild reservoir. ###
    dnp_enabled: bool = False
    p_dnp_sat: float = 0.58
    dnp_rate: float = 0.05

    t1_rate: float = 0.0
    t1_p_eq: float = 0.0

    dt: float = 0.015
    noise_sigma: float = 0.0

    steps: int = 50

    # Instantaneous AFP: fired once before time stepping (apply_pending_afp / step), then cleared.
    # Not a rate term — applied as a map before the Euler update that step.
    afp_enabled: bool = False
    afp_efficiency: float = 1.0
    afp_center_margin: int = 0
    afp_preserve_intensity_area: bool = False
    afp_subset_indices: Optional[List[int]] = None


class Spin1Model:
    """Stateful spin-1 population model with ideal-bin ss-RF and optional DNP."""

    def __init__(
        self,
        params: Optional[Spin1Params] = None,
        *,
        initial_polarization: Optional[float] = None,
    ):
        self.params = params or Spin1Params()
        if initial_polarization is not None:
            self.params.initial_polarization = float(initial_polarization)
        self.reset()

    def reset(self) -> None:
        p = self.params
        self.Rplus = np.linspace(p.r_min, p.r_max, p.n_bins)
        self.dR = float(self.Rplus[1] - self.Rplus[0])

        density = normalized_component(self.Rplus, +1, gamma=p.line_gamma, asym=p.line_asym)
        mu = density * self.dR
        self.mu = mu / max(float(mu.sum()), 1e-30)
        self.base_density = self.mu / self.dR

        self.pref_initial = level_populations_from_PQ(p.p0, p.q0)
        self.n_ref = self.mu[:, None] * self.pref_initial[None, :]
        self.n = self.n_ref.copy()
        self.n_initial = self.n.copy()
        self.t = 0.0

        self.display_cal = self._event_display_calibration()
        # Global spin-1 level fractions (sum=1, n+−n− = vector P). Packet ``self.n``
        # is a different representation; these scalars are the physical levels.
        self._populations_from_intensities = False
        self.n_plus = self.n_zero = self.n_minus = 0.0
        self.n_plus_initial = self.n_zero_initial = self.n_minus_initial = 0.0
        self._sync_level_populations(capture_initial=True)

        self._invalidate_rate_cache()
        self._invalidate_branch_cache()
        self._active_idx: Optional[np.ndarray] = None
        self._window_radius: Optional[int] = None
        # One-shot AFP before the next step() / apply_pending_afp when params.afp_enabled.
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
        """Per-bin RF rate from initial Q via ``rf_burn_profile`` (``gamma_rf`` = peak)."""
        ip, im, _ = self.physical_intensities(self.n_initial)
        q = ip - im
        q_min = np.min(q)
        if q_min >= 0.0:
            self.params.rf_profile = np.zeros_like(q)
        else:
            self.params.rf_profile = self.params.gamma_rf * np.clip(q / q_min, 0.0, 1.0)

    def _event_display_calibration(self) -> float:
        """Intensity scale matching ``GenerateVectorLineshape`` event normalization (P₀)."""
        p = self.params
        return float(p.initial_polarization if p.initial_polarization is not None else p.p0)

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
        self._sync_level_populations(capture_initial=True)
        self._invalidate_rate_cache()
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

        By default Σ(I⁺+I⁻) is left as the post-swap area (vector P can change).
        When ``preserve_intensity_area`` is True, that sum is restored to the
        pre-AFP event area by adding *common-mode* intensity on touched bins
        (equal Δ to I⁺ and I⁻), leaving Q = Σ(I⁺−I⁻) unchanged.
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
        self._invalidate_rate_cache()
        self._afp_last_subset = list(subset)
        self.ip_afp, self.im_afp, _ = self.physical_intensities()
        return subset

    def ssrf_burn(self) -> np.ndarray:
        """Return RF population rates for each ssRF burn bin in params.

        When ``params.ssrf_subset_indices`` is set, each index ``i`` is a burn
        frequency: +↔0 at ``i`` and 0↔- at mirror(``i``). When it is ``None``,
        burns the single ``rf_burn_R`` branch pair (legacy).

        Per-bin rates come from ``rf_profile`` (Q-shaped; peak ``gamma_rf`` at deepest Q<0).
        """
        dn_rf = np.zeros_like(self.n)
        if float(self.params.gamma_rf) == 0.0:
            return dn_rf

        self.set_rf_profile()
        profile = np.asarray(self.params.rf_profile, dtype=float)

        subset = self.params.ssrf_subset_indices
        if subset is None:
            pairs: List[Tuple[Optional[int], Optional[int]]] = [
                self.cached_branch_indices(self.params.rf_burn_R)
            ]
        else:
            n_bins = len(self.n)
            pairs = []
            for raw in subset:
                i = int(raw)
                if 0 <= i < n_bins:
                    pairs.append((i, n_bins - 1 - i))

        for kp, km in pairs:
            if kp is not None:
                gamma = float(profile[kp])
            elif km is not None:
                gamma = float(profile[km])
            else:
                continue
            if gamma == 0.0:
                continue
            if kp is not None:
                J = gamma * (self.n[kp, PLUS] - self.n[kp, ZERO])
                dn_rf[kp, PLUS] -= J
                dn_rf[kp, ZERO] += J
            if km is not None:
                J = gamma * (self.n[km, ZERO] - self.n[km, MINUS])
                dn_rf[km, ZERO] -= J
                dn_rf[km, MINUS] += J
        return dn_rf

    def _renormalize_touched_intensity_area(
        self,
        touched: List[int],
        area_target: float,
    ) -> None:
        """
        Restore Σ(I⁺+I⁻) to ``area_target`` on ``touched`` bins without changing Q.

        Adds a common-mode offset (equal ΔI⁺ and ΔI⁻), weighted by local Ps on
        touched bins. Untouched bins stay unchanged. Packet row sums stay ``mu``.
        """
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
            self.params.afp_enabled = False ### only apply AFP once, set bool to false afterwards

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
        """
        One Euler macro-step of RF / relaxation / DNP (no AFP).

        AFP is instantaneous and must run before time stepping via ``afp_sweep``,
        ``apply_pending_afp``, or ``step`` (which flushes pending AFP first).
        """
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
        """
        Refresh stored global level fractions ``n_plus``, ``n_zero``, ``n_minus``.

        When the state came from physical I±, invert via ``level_populations_from_PQ``
        using event-normalized Σ(I++I−)=P when |ΣI|≤1, otherwise convert plot-unit
        intensities with the ``dR/display_cal`` factor.  Boltzmann ``mu*pref``
        initialization uses packet-integrated fractions instead.
        """
        if self._populations_from_intensities:
            ip, im, _ = self.physical_intensities()
            p_raw = float(np.sum(ip + im))
            q_raw = float(np.sum(ip - im))
            if abs(p_raw) <= 1.0 + 1e-9:
                p_vec, q_ten = p_raw, q_raw
            else:
                scale = float(self.dR) / max(abs(float(self.display_cal)), 1e-30)
                p_vec, q_ten = p_raw * scale, q_raw * scale
            pref = level_populations_from_PQ(p_vec, q_ten)
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

    def _invalidate_rate_cache(self) -> None:
        self._rate_cache_ref_id: Optional[int] = None
        self._cached_ref_plus_norm: Optional[float] = None
        self._cached_ref_minus_norm: Optional[float] = None
        self._cached_t2_overlap: Optional[np.ndarray] = None
        self._cached_edge_mass: Optional[np.ndarray] = None

    def _invalidate_branch_cache(self) -> None:
        self._cached_rf_burn_R: Optional[float] = None
        self._cached_kp: Optional[int] = None
        self._cached_km: Optional[int] = None

    def _ensure_static_rate_cache(self, reference: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Cache reference norms and geometric edge factors (independent of ``self.n``)."""
        ref_id = id(reference)
        if (
            self._rate_cache_ref_id == ref_id
            and self._cached_ref_plus_norm is not None
            and self._cached_t2_overlap is not None
            and self._cached_edge_mass is not None
        ):
            return (
                float(self._cached_ref_plus_norm),
                float(self._cached_ref_minus_norm),
                self._cached_t2_overlap,
                self._cached_edge_mass,
            )

        ref_plus = self._transition_density_abs("plus0", reference)
        ref_minus = self._transition_density_abs("0minus", reference)
        finite_p = ref_plus[np.isfinite(ref_plus)]
        finite_m = ref_minus[np.isfinite(ref_minus)]
        plus_norm = max(finite_p)
        minus_norm = max(finite_m)
        if len(self.Rplus) < 2:
            overlap = np.zeros(0, dtype=float)
            edge_mass = np.zeros(0, dtype=float)
        else:
            dR_edges = self.Rplus[1:] - self.Rplus[:-1]
            width = max(float(self.params.t2_width_R), 1e-12)
            overlap = np.exp(-0.5 * (dR_edges / width) ** 2)
            edge_mass = np.sqrt(self.mu[:-1] * self.mu[1:])

        self._rate_cache_ref_id = ref_id
        self._cached_ref_plus_norm = plus_norm
        self._cached_ref_minus_norm = minus_norm
        self._cached_t2_overlap = overlap
        self._cached_edge_mass = edge_mass
        return plus_norm, minus_norm, overlap, edge_mass

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
        """Boltzmann packet state at ``P``.

        With no ``P``, return the fixed initial-event baseline ``n_ref`` (display /
        response reference). Recovery pathways should pass the current vector
        polarization so the null manifold tracks post-manipulation P, not P₀.
        """
        if P is None:
            return self.n_ref
        pref = level_populations_from_PQ(float(P), None)
        return self.mu[:, None] * pref[None, :]

    def recovery_equilibrium_reference(self) -> np.ndarray:

        P = self.n_plus - self.n_minus
        if not self._populations_from_intensities:
            return self.equilibrium_reference(P)

        P0 = self.n_plus_initial - self.n_minus_initial
        if abs(P - P0) <= 1e-10:
            return self.n_ref

        # Boltzmann at P in event intensity units, then into the loaded packet basis.
        pref = level_populations_from_PQ(P, None)
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
            "Q_boltz_at_P": boltzmann_Q(P),
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

    def physical_intensities(self, n: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return I+(R), I-(R), and total on the model's physical R grid."""
        state = self.n if n is None else n
        return packet_n_to_physical_intensities(
            state, self.Rplus, display_cal=self.display_cal, dR=self.dR
        )

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
        # Always use full-spectrum μ weights. Active-window reweighting dumps the
        # P-conservation correction onto burn/mirror packets and can flip the
        # apparent mirror-peak sign (normalization artifact).
        dP = float(np.sum(dn[:, PLUS] - dn[:, MINUS]))
        if abs(dP) < 1e-18:
            return dn
        correction = np.zeros_like(dn)
        correction[:, PLUS] -= 0.5 * dP * self.mu
        correction[:, MINUS] += 0.5 * dP * self.mu
        return dn + correction

    def _level_fractions(self, n: Optional[np.ndarray] = None) -> np.ndarray:
        if n is None:
            n = self.n
        return n / np.maximum(self.mu[:, None], 1e-30)

    def _availability_factor(self, a: int, b: int, reference: Optional[np.ndarray] = None) -> np.ndarray:
        weight = float(np.clip(self.params.population_availability, 0.0, 1.0))
        if weight == 0.0:
            return np.ones(len(self.Rplus))
        p_cur = self._level_fractions(self.n)
        p_ref = p_cur if reference is None else self._level_fractions(reference)
        raw = np.sqrt(np.maximum(p_cur[:, a] * p_cur[:, b], 0.0) * np.maximum(p_ref[:, a] * p_ref[:, b], 0.0))
        raw = np.clip(raw / (1.0 / 9.0), 0.0, 3.0)
        return (1.0 - weight) + weight * raw

    def _edge_availability_factor(self, a: int, b: int) -> np.ndarray:
        weight = float(np.clip(self.params.population_availability, 0.0, 1.0))
        if weight == 0.0 or len(self.Rplus) < 2:
            return np.ones(max(0, len(self.Rplus) - 1))
        p_cur = self._level_fractions(self.n)
        raw = np.sqrt(np.maximum(p_cur[:-1, a] * p_cur[:-1, b], 0.0) * np.maximum(p_cur[1:, a] * p_cur[1:, b], 0.0))
        raw = np.clip(raw / (1.0 / 9.0), 0.0, 3.0)
        return (1.0 - weight) + weight * raw

    def _transition_density_abs(self, which: str, state: Optional[np.ndarray] = None) -> np.ndarray:
        if state is None:
            state = self.n
        if which == "plus0":
            diff = state[:, PLUS] - state[:, ZERO]
        elif which == "0minus":
            diff = state[:, ZERO] - state[:, MINUS]
        else:
            raise ValueError("which must be 'plus0' or '0minus'")
        return np.abs(diff) / max(float(self.dR), 1e-30)

    def _position_scale(self, values: np.ndarray) -> np.ndarray:
        if not bool(self.params.r_dependent_recovery):
            return np.ones_like(np.asarray(values, dtype=float))
        x = np.abs(np.asarray(values, dtype=float))
        finite = x[np.isfinite(x) & (x > 1e-30)]
        if finite.size == 0:
            return np.ones_like(x, dtype=float)
        norm = max(float(np.nanpercentile(finite, 90.0)), 1e-30)
        power = max(0.0, float(self.params.recovery_position_power))
        raw = np.power(x / norm, power) if power != 0.0 else np.ones_like(x)
        floor = float(np.clip(self.params.recovery_position_floor, 0.0, 1.0))
        cap = max(1.0, float(self.params.recovery_rate_clip))
        return np.clip(floor + (1.0 - floor) * raw, floor, cap)

    def _position_rate_arrays(self, reference: np.ndarray) -> Dict[str, np.ndarray]:
        a_plus0, b_0minus = self._rf_mode_amplitudes(reference)
        ref_plus = self._transition_density_abs("plus0", reference)
        ref_minus = self._transition_density_abs("0minus", reference)
        cur_plus = self._transition_density_abs("plus0", self.n)
        cur_minus = self._transition_density_abs("0minus", self.n)
        plus_norm, minus_norm, overlap, edge_mass = self._ensure_static_rate_cache(reference)
        mode_plus_density = np.abs(a_plus0) / max(float(self.dR), 1e-30)
        mode_minus_density = np.abs(b_0minus) / max(float(self.dR), 1e-30)
        dev_plus = np.clip(mode_plus_density / (ref_plus + 0.02 * plus_norm + 1e-30), 0.0, 50.0)
        dev_minus = np.clip(mode_minus_density / (ref_minus + 0.02 * minus_norm + 1e-30), 0.0, 50.0)

        # Same-theta recovery of one transition is weighted by the corresponding mirror branch.
        # Norms are taken from the *current* arrays (not cached ref norms) so mirror
        # weighting keeps the correct sign/scale after burns.
        mirror_for_plus = self._position_scale(cur_minus)
        mirror_for_minus = self._position_scale(cur_plus)
        neighbor_plus_packet = self._position_scale(cur_plus)
        neighbor_minus_packet = self._position_scale(cur_minus)
        avail_plus = self._availability_factor(PLUS, ZERO, reference)
        avail_minus = self._availability_factor(ZERO, MINUS, reference)
        gain = max(0.0, float(self.params.same_theta_mirror_gain))
        bgain = max(0.0, float(self.params.boltzmann_distance_gain))
        cap = max(1.0, float(self.params.recovery_rate_clip))
        same_plus = self.params.d_same_plus0 * mirror_for_plus * avail_plus * (1.0 + gain * dev_plus) * (1.0 + bgain * dev_plus)
        same_minus = self.params.d_same_0minus * mirror_for_minus * avail_minus * (1.0 + gain * dev_minus) * (1.0 + bgain * dev_minus)
        same_plus = np.clip(same_plus, 0.0, cap * max(abs(self.params.d_same_plus0), 1e-30))
        same_minus = np.clip(same_minus, 0.0, cap * max(abs(self.params.d_same_0minus), 1e-30))

        if len(self.Rplus) < 2:
            edge_plus = np.zeros(0, dtype=float)
            edge_minus = np.zeros(0, dtype=float)
            edge_plus_weight = np.zeros(0, dtype=float)
            edge_minus_weight = np.zeros(0, dtype=float)
        else:
            edge_avail_plus = self._edge_availability_factor(PLUS, ZERO)
            edge_avail_minus = self._edge_availability_factor(ZERO, MINUS)
            edge_plus_weight = np.sqrt(neighbor_plus_packet[:-1] * neighbor_plus_packet[1:])
            edge_minus_weight = np.sqrt(neighbor_minus_packet[:-1] * neighbor_minus_packet[1:])
            edge_dev_plus = 0.5 * (dev_plus[:-1] + dev_plus[1:])
            edge_dev_minus = 0.5 * (dev_minus[:-1] + dev_minus[1:])
            edge_plus = self.params.d_spec_plus0 * overlap * edge_avail_plus * edge_plus_weight * edge_mass * (1.0 + 0.5 * bgain * edge_dev_plus)
            edge_minus = self.params.d_spec_0minus * overlap * edge_avail_minus * edge_minus_weight * edge_mass * (1.0 + 0.5 * bgain * edge_dev_minus)
            edge_plus = np.clip(edge_plus, 0.0, cap * max(abs(self.params.d_spec_plus0), 1e-30))
            edge_minus = np.clip(edge_minus, 0.0, cap * max(abs(self.params.d_spec_0minus), 1e-30))

        # Local-window burn: zero rates outside active packets (edges kept only if both ends active).
        active = self._active_idx
        if active is not None and active.size > 0:
            mask = np.zeros(len(self.Rplus), dtype=bool)
            mask[active] = True
            same_plus = np.where(mask, same_plus, 0.0)
            same_minus = np.where(mask, same_minus, 0.0)
            if len(self.Rplus) >= 2:
                edge_mask = mask[:-1] & mask[1:]
                edge_plus = np.where(edge_mask, edge_plus, 0.0)
                edge_minus = np.where(edge_mask, edge_minus, 0.0)

        return {
            "same_plus0": same_plus,
            "same_0minus": same_minus,
            "edge_plus0": edge_plus,
            "edge_0minus": edge_minus,
            "dev_plus0": dev_plus,
            "dev_0minus": dev_minus,
            "mirror_weight_plus0": mirror_for_plus,
            "mirror_weight_0minus": mirror_for_minus,
            "neighbor_weight_plus0": edge_plus_weight,
            "neighbor_weight_0minus": edge_minus_weight,
            "t2_overlap": overlap,
            "availability_plus0": avail_plus,
            "availability_0minus": avail_minus,
        }

    def local_recovery_rates(self, R: Optional[float] = None) -> Dict[str, float]:
        if R is None:
            R = self.params.rf_burn_R
        reference = self.recovery_equilibrium_reference()
        rates = self._position_rate_arrays(reference)
        kp, km = self.branch_indices(R)

        def arr(a: np.ndarray, k: Optional[int]) -> float:
            if k is None or k < 0 or k >= len(a):
                return float("nan")
            return float(a[k])

        def edge(a: np.ndarray, k: Optional[int], side: str) -> float:
            if k is None:
                return float("nan")
            idx = k - 1 if side == "left" else k
            if idx < 0 or idx >= len(a):
                return 0.0
            return float(a[idx] / max(self.mu[k], 1e-30))

        _, parts = self.derivative(rf_on=False, dnp_on=self.params.dnp_enabled, breakdown=True)
        return {
            "R": float(R),
            "k_Iplus": -1 if kp is None else int(kp),
            "k_Iminus": -1 if km is None else int(km),
            "Iplus_same_theta": arr(rates["same_plus0"], kp),
            "Iplus_neighbor_left": edge(rates["edge_plus0"], kp, "left"),
            "Iplus_neighbor_right": edge(rates["edge_plus0"], kp, "right"),
            "Iplus_mirror_factor": arr(rates["mirror_weight_plus0"], kp),
            "Iplus_deviation": arr(rates["dev_plus0"], kp),
            "Iplus_refill_dt_no_rf": float(parts["net"].get("dIplus_R_dt", float("nan"))),
            "Iminus_same_theta": arr(rates["same_0minus"], km),
            "Iminus_neighbor_left": edge(rates["edge_0minus"], km, "left"),
            "Iminus_neighbor_right": edge(rates["edge_0minus"], km, "right"),
            "Iminus_mirror_factor": arr(rates["mirror_weight_0minus"], km),
            "Iminus_deviation": arr(rates["dev_0minus"], km),
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
            "mirror_plus0_factor": float(local["Iplus_mirror_factor"]),
            "mirror_0minus_factor": float(local["Iminus_mirror_factor"]),
            "dev_plus0": float(local["Iplus_deviation"]),
            "dev_0minus": float(local["Iminus_deviation"]),
            "Iplus_refill_dt_no_rf": float(local["Iplus_refill_dt_no_rf"]),
            "Iminus_refill_dt_no_rf": float(local["Iminus_refill_dt_no_rf"]),
        }

    def _mode_relax_reference(
        self,
        which: str,
        rate: float,
        reference: np.ndarray,
        *,
        rates: Optional[Dict[str, np.ndarray]] = None,
        amplitudes: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        if rate == 0.0:
            return np.zeros_like(self.n)
        a_plus0, b_0minus = amplitudes if amplitudes is not None else self._rf_mode_amplitudes(reference)
        if rates is None:
            rates = self._position_rate_arrays(reference)
        if which == "plus0":
            dmode_dt = -rates["same_plus0"] * a_plus0
            return self._mode_to_population_derivative(dmode_dt, np.zeros_like(a_plus0))
        if which == "0minus":
            dmode_dt = -rates["same_0minus"] * b_0minus
            return self._mode_to_population_derivative(np.zeros_like(b_0minus), dmode_dt)
        raise ValueError("which must be 'plus0' or '0minus'")

    def _mode_diffuse_delta(
        self,
        which: str,
        rate: float,
        reference: np.ndarray,
        *,
        rates: Optional[Dict[str, np.ndarray]] = None,
        amplitudes: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        if rate == 0.0 or len(self.Rplus) < 2:
            return np.zeros_like(self.n)
        a_plus0, b_0minus = amplitudes if amplitudes is not None else self._rf_mode_amplitudes(reference)
        if rates is None:
            rates = self._position_rate_arrays(reference)
        if which == "plus0":
            mode = a_plus0
            edge_rate = rates["edge_plus0"]
        elif which == "0minus":
            mode = b_0minus
            edge_rate = rates["edge_0minus"]
        else:
            raise ValueError("which must be 'plus0' or '0minus'")
        rho_mode = mode / np.maximum(self.mu, 1e-30)
        flux = edge_rate * (rho_mode[1:] - rho_mode[:-1])
        dmode_dt = np.zeros_like(mode)
        dmode_dt[:-1] += flux
        dmode_dt[1:] -= flux
        if which == "plus0":
            return self._mode_to_population_derivative(dmode_dt, np.zeros_like(mode))
        return self._mode_to_population_derivative(np.zeros_like(mode), dmode_dt)

    def derivative(self, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None, breakdown: bool = False):
        if rf_on is None:
            rf_on = bool(self.params.rf_enabled)
        if dnp_on is None:
            dnp_on = bool(self.params.dnp_enabled)
        dn_terms: Dict[str, np.ndarray] = {}
        dn_rf = self.ssrf_burn() if rf_on else np.zeros_like(self.n)
        dn_terms["RF"] = dn_rf

        # Recover toward Boltzmann at current P (post-manipulation), not initial n_ref.
        dynamic_ref = self.recovery_equilibrium_reference()
        need_same = self.params.d_same_plus0 != 0.0 or self.params.d_same_0minus != 0.0
        need_spec = self.params.d_spec_plus0 != 0.0 or self.params.d_spec_0minus != 0.0
        rates = None
        amplitudes = None
        if need_same or need_spec:
            rates = self._position_rate_arrays(dynamic_ref)
            amplitudes = self._rf_mode_amplitudes(dynamic_ref)

        if need_same:
            dn_same = (
                self._mode_relax_reference(
                    "plus0", self.params.d_same_plus0, dynamic_ref, rates=rates, amplitudes=amplitudes
                )
                + self._mode_relax_reference(
                    "0minus", self.params.d_same_0minus, dynamic_ref, rates=rates, amplitudes=amplitudes
                )
            )
            dn_same = self._project_conserve_vector(dn_same)
        else:
            dn_same = np.zeros_like(self.n)
        dn_terms["same_theta_mirror_backpath"] = dn_same

        if need_spec:
            dn_spec = (
                self._mode_diffuse_delta(
                    "plus0", self.params.d_spec_plus0, dynamic_ref, rates=rates, amplitudes=amplitudes
                )
                + self._mode_diffuse_delta(
                    "0minus", self.params.d_spec_0minus, dynamic_ref, rates=rates, amplitudes=amplitudes
                )
            )
            dn_spec = self._project_conserve_vector(dn_spec)
        else:
            dn_spec = np.zeros_like(self.n)
        dn_terms["spectral_neighbor_diffusion"] = dn_spec

        dn_dnp = np.zeros_like(self.n)
        if dnp_on and self.params.dnp_rate != 0.0:
            dnp_target = self.equilibrium_reference(self.params.p_dnp_sat)
            dn_dnp = self.params.dnp_rate * (dnp_target - self.n)
        dn_terms["DNP_sat"] = dn_dnp

        dn_t1 = np.zeros_like(self.n)
        if self.params.t1_rate != 0.0:
            t1_target = self.equilibrium_reference(self.params.t1_p_eq)
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
        try:
            _, parts = self.derivative(rf_on=False, dnp_on=self.params.dnp_enabled, breakdown=True)
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

    @property
    def branch_ratio(self) -> float:
        return boltzmann_branch_ratio(self.polarizations()["P"])

    @property
    def initial_branch_ratio(self) -> float:
        return boltzmann_branch_ratio(self.params.p0)


