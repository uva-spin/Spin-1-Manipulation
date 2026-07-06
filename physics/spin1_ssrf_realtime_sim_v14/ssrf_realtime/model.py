"""
Spin-1 Pake-doublet ss-RF real-time model, v14.

This version keeps the v11/v12 signed-polarization, DNP, and realistic
analytic Pake-doublet lineshape machinery, but makes the recovery pathways
explicitly position dependent.

The GUI recovery controls are base material scales.  The model recomputes the
actual local recovery coefficients at every step from:

* the selected physical R bin and the corresponding mirror -R packet;
* the local/mirror branch populations;
* the distance from the current-P Boltzmann-shaped state;
* the left/right neighboring bins and a T2-like spectral overlap;
* population availability for flip-flop-like recovery.

With DNP off, RF is the only vector-polarization sink.  Internal recovery and
neighbor diffusion conserve the current reduced P(t), so holes can fill only by
redistributing remaining area.  With DNP on, a separate external reservoir
builds toward P_DNP_sat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np

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
    return float(np.clip(float(P), -0.999999, 0.999999))


@dataclass
class Spin1Params:
    """Numerical and phenomenological parameters for the spin-1 model."""

    # Packet/display grid in dimensionless physical R units.
    n_bins: int = 701
    r_min: float = -3.0
    r_max: float = 3.0

    # Realistic analytic Pake branch parameters. Defaults match Plot_Signal.py.
    line_gamma: float = 0.05
    line_asym: float = 0.04

    # Display calibration.
    plot_signal_units: bool = True
    plot_divisor: float = 10.0
    display_scale: float = 1.0
    calibration_p: float = 0.50

    # Initial vector polarization and optional tensor polarization.
    p0: float = 0.45
    q0: Optional[float] = None

    # RF location is a physical R coordinate in the plotted spectrum.
    rf_burn_R: float = 0.40
    rf_enabled: bool = False
    gamma_rf: float = 2.0

    # Recovery base material scales.  Effective values are R-dependent.
    d_same_plus0: float = 0.25
    d_same_0minus: float = 0.15
    d_spec_plus0: float = 1.5
    d_spec_0minus: float = 0.8
    same_theta_mirror_gain: float = 1.5
    # Additional multiplier for distance from current-P Boltzmann reference.
    boltzmann_distance_gain: float = 0.5
    population_availability: float = 1.0
    t2_width_R: float = 0.05

    # Position-dependent recovery weighting.
    r_dependent_recovery: bool = True
    recovery_position_power: float = 1.0
    recovery_position_floor: float = 0.03
    recovery_rate_clip: float = 20.0

    # DNP build/rebuild reservoir.
    dnp_enabled: bool = False
    p_dnp_sat: float = 0.58
    dnp_rate: float = 0.05

    # Optional ordinary T1-like relaxation.
    t1_rate: float = 0.0
    t1_p_eq: float = 0.0

    # Integration and display.
    dt: float = 0.0015
    noise_sigma: float = 0.0


class Spin1Model:
    """Stateful spin-1 population model with ideal-bin ss-RF and optional DNP."""

    def __init__(self, params: Optional[Spin1Params] = None):
        self.params = params or Spin1Params()
        self.reset()

    # ------------------------------------------------------------------
    # Setup and static references
    # ------------------------------------------------------------------
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
        self.t = 0.0
        self.display_cal = self._compute_display_calibration()

    def _compute_display_calibration(self) -> float:
        p = self.params
        if not p.plot_signal_units:
            return float(p.display_scale)
        pref_cal = level_populations_from_PQ(_clamp_p(p.calibration_p), None)
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
        for key, value in kwargs.items():
            if not hasattr(self.params, key):
                raise AttributeError(f"Unknown parameter: {key}")
            setattr(self.params, key, value)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self.params)

    def set_rf_enabled(self, enabled: bool) -> None:
        self.params.rf_enabled = bool(enabled)

    def set_dnp_enabled(self, enabled: bool) -> None:
        self.params.dnp_enabled = bool(enabled)

    def equilibrium_reference(self, P: Optional[float] = None) -> np.ndarray:
        if P is None:
            P = self.polarizations()["P"]
        pref = level_populations_from_PQ(_clamp_p(float(P)), None)
        return self.mu[:, None] * pref[None, :]

    # ------------------------------------------------------------------
    # Observables and spectra
    # ------------------------------------------------------------------
    def polarizations(self, n: Optional[np.ndarray] = None) -> Dict[str, float]:
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

    def branch_areas(self, n: Optional[np.ndarray] = None) -> Dict[str, float]:
        if n is None:
            n = self.n
        a_plus = float(self.display_cal * np.sum(n[:, PLUS] - n[:, ZERO]))
        a_minus = float(self.display_cal * np.sum(n[:, ZERO] - n[:, MINUS]))
        return {"A_plus": a_plus, "A_minus": a_minus, "A_total": a_plus + a_minus, "A_diff": a_plus - a_minus}

    def branch_indices(self, R: Optional[float] = None) -> Tuple[Optional[int], Optional[int]]:
        if R is None:
            R = self.params.rf_burn_R
        R = float(R)
        kp = int(np.argmin(np.abs(self.Rplus - R))) if self.Rplus[0] <= R <= self.Rplus[-1] else None
        km = int(np.argmin(np.abs(self.Rplus + R))) if self.Rplus[0] <= -R <= self.Rplus[-1] else None
        return kp, km

    def _transition_differences(self, n: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if n is None:
            n = self.n
        return n[:, PLUS] - n[:, ZERO], n[:, ZERO] - n[:, MINUS]

    def packet_intensities(self, use_reference: bool = False, density: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n_ref if use_reference else self.n
        Iplus, Iminus = self._transition_differences(n)
        scale = self.display_cal / self.dR if density else self.display_cal
        return scale * Iplus, scale * Iminus

    def local_intensities(self, R: Optional[float] = None, use_reference: bool = False) -> Dict[str, float]:
        kp, km = self.branch_indices(R)
        Iplus_packet, Iminus_packet = self.packet_intensities(use_reference=use_reference, density=True)
        Iplus = np.nan if kp is None else float(Iplus_packet[kp])
        Iminus = np.nan if km is None else float(Iminus_packet[km])
        total = 0.0
        if np.isfinite(Iplus):
            total += Iplus
        if np.isfinite(Iminus):
            total += Iminus
        return {"Iplus": Iplus, "Iminus": Iminus, "total": float(total), "k_plus": -1 if kp is None else int(kp), "k_minus": -1 if km is None else int(km)}

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

    def static_plot_signal_reference(self):
        from .lineshape import plot_signal_reference
        return plot_signal_reference(self.Rplus, P=self.params.p0, gamma=self.params.line_gamma, asym=self.params.line_asym, divisor=self.params.plot_divisor)

    def packet_spectrum(self, use_reference: bool = False, noise_sigma: Optional[float] = None):
        Iplus_packet, Iminus_packet = self.packet_intensities(use_reference=use_reference, density=True)
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

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------
    def _rf_mode_amplitudes(self, reference: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if reference is None:
            reference = self.equilibrium_reference()
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
        par = self.params
        a_plus0, b_0minus = self._rf_mode_amplitudes(reference)
        ref_plus = self._transition_density_abs("plus0", reference)
        ref_minus = self._transition_density_abs("0minus", reference)
        cur_plus = self._transition_density_abs("plus0", self.n)
        cur_minus = self._transition_density_abs("0minus", self.n)
        plus_norm = max(float(np.nanpercentile(ref_plus[np.isfinite(ref_plus)], 90.0)), 1e-12)
        minus_norm = max(float(np.nanpercentile(ref_minus[np.isfinite(ref_minus)], 90.0)), 1e-12)
        mode_plus_density = np.abs(a_plus0) / max(float(self.dR), 1e-30)
        mode_minus_density = np.abs(b_0minus) / max(float(self.dR), 1e-30)
        dev_plus = np.clip(mode_plus_density / (ref_plus + 0.02 * plus_norm + 1e-30), 0.0, 50.0)
        dev_minus = np.clip(mode_minus_density / (ref_minus + 0.02 * minus_norm + 1e-30), 0.0, 50.0)

        # Same-theta recovery of one transition is weighted by the corresponding mirror branch.
        mirror_for_plus = self._position_scale(cur_minus)  # I+ burn has I-(-R) mirror proxy
        mirror_for_minus = self._position_scale(cur_plus)  # I- burn has I+(-R) mirror proxy
        neighbor_plus_packet = self._position_scale(cur_plus)
        neighbor_minus_packet = self._position_scale(cur_minus)
        avail_plus = self._availability_factor(PLUS, ZERO, reference)
        avail_minus = self._availability_factor(ZERO, MINUS, reference)
        gain = max(0.0, float(par.same_theta_mirror_gain))
        bgain = max(0.0, float(par.boltzmann_distance_gain))
        cap = max(1.0, float(par.recovery_rate_clip))
        same_plus = par.d_same_plus0 * mirror_for_plus * avail_plus * (1.0 + gain * dev_plus) * (1.0 + bgain * dev_plus)
        same_minus = par.d_same_0minus * mirror_for_minus * avail_minus * (1.0 + gain * dev_minus) * (1.0 + bgain * dev_minus)
        same_plus = np.clip(same_plus, 0.0, cap * max(abs(par.d_same_plus0), 1e-30))
        same_minus = np.clip(same_minus, 0.0, cap * max(abs(par.d_same_0minus), 1e-30))

        if len(self.Rplus) < 2:
            edge_plus = np.zeros(0, dtype=float)
            edge_minus = np.zeros(0, dtype=float)
            edge_plus_weight = np.zeros(0, dtype=float)
            edge_minus_weight = np.zeros(0, dtype=float)
            overlap = np.zeros(0, dtype=float)
        else:
            dR_edges = self.Rplus[1:] - self.Rplus[:-1]
            width = max(float(par.t2_width_R), 1e-12)
            overlap = np.exp(-0.5 * (dR_edges / width) ** 2)
            edge_avail_plus = self._edge_availability_factor(PLUS, ZERO)
            edge_avail_minus = self._edge_availability_factor(ZERO, MINUS)
            edge_plus_weight = np.sqrt(neighbor_plus_packet[:-1] * neighbor_plus_packet[1:])
            edge_minus_weight = np.sqrt(neighbor_minus_packet[:-1] * neighbor_minus_packet[1:])
            edge_dev_plus = 0.5 * (dev_plus[:-1] + dev_plus[1:])
            edge_dev_minus = 0.5 * (dev_minus[:-1] + dev_minus[1:])
            edge_mass = np.sqrt(self.mu[:-1] * self.mu[1:])
            edge_plus = par.d_spec_plus0 * overlap * edge_avail_plus * edge_plus_weight * edge_mass * (1.0 + 0.5 * bgain * edge_dev_plus)
            edge_minus = par.d_spec_0minus * overlap * edge_avail_minus * edge_minus_weight * edge_mass * (1.0 + 0.5 * bgain * edge_dev_minus)
            edge_plus = np.clip(edge_plus, 0.0, cap * max(abs(par.d_spec_plus0), 1e-30))
            edge_minus = np.clip(edge_minus, 0.0, cap * max(abs(par.d_spec_0minus), 1e-30))

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
        reference = self.equilibrium_reference()
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

    def _mode_relax_reference(self, which: str, rate: float, reference: np.ndarray) -> np.ndarray:
        if rate == 0.0:
            return np.zeros_like(self.n)
        a_plus0, b_0minus = self._rf_mode_amplitudes(reference)
        rates = self._position_rate_arrays(reference)
        if which == "plus0":
            dmode_dt = -rates["same_plus0"] * a_plus0
            return self._mode_to_population_derivative(dmode_dt, np.zeros_like(a_plus0))
        if which == "0minus":
            dmode_dt = -rates["same_0minus"] * b_0minus
            return self._mode_to_population_derivative(np.zeros_like(b_0minus), dmode_dt)
        raise ValueError("which must be 'plus0' or '0minus'")

    def _mode_diffuse_delta(self, which: str, rate: float, reference: np.ndarray) -> np.ndarray:
        if rate == 0.0 or len(self.Rplus) < 2:
            return np.zeros_like(self.n)
        a_plus0, b_0minus = self._rf_mode_amplitudes(reference)
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

    # ------------------------------------------------------------------
    # Time derivative and integration
    # ------------------------------------------------------------------
    def derivative(self, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None, breakdown: bool = False):
        p = self.params
        if rf_on is None:
            rf_on = bool(p.rf_enabled)
        if dnp_on is None:
            dnp_on = bool(p.dnp_enabled)
        dn_terms: Dict[str, np.ndarray] = {}

        dn_rf = np.zeros_like(self.n)
        if rf_on and p.gamma_rf != 0.0:
            kp, km = self.branch_indices(p.rf_burn_R)
            if kp is not None:
                J = p.gamma_rf * (self.n[kp, PLUS] - self.n[kp, ZERO])
                dn_rf[kp, PLUS] -= J
                dn_rf[kp, ZERO] += J
            if km is not None:
                J = p.gamma_rf * (self.n[km, ZERO] - self.n[km, MINUS])
                dn_rf[km, ZERO] -= J
                dn_rf[km, MINUS] += J
        dn_terms["RF"] = dn_rf

        dynamic_ref = self.equilibrium_reference()
        dn_same = self._mode_relax_reference("plus0", p.d_same_plus0, dynamic_ref) + self._mode_relax_reference("0minus", p.d_same_0minus, dynamic_ref)
        dn_same = self._project_conserve_vector(dn_same)
        dn_terms["same_theta_mirror_backpath"] = dn_same

        dn_spec = self._mode_diffuse_delta("plus0", p.d_spec_plus0, dynamic_ref) + self._mode_diffuse_delta("0minus", p.d_spec_0minus, dynamic_ref)
        dn_spec = self._project_conserve_vector(dn_spec)
        dn_terms["spectral_neighbor_diffusion"] = dn_spec

        dn_dnp = np.zeros_like(self.n)
        if dnp_on and p.dnp_rate != 0.0:
            dnp_target = self.equilibrium_reference(_clamp_p(p.p_dnp_sat))
            dn_dnp = p.dnp_rate * (dnp_target - self.n)
        dn_terms["DNP_sat"] = dn_dnp

        dn_t1 = np.zeros_like(self.n)
        if p.t1_rate != 0.0:
            t1_target = self.equilibrium_reference(_clamp_p(p.t1_p_eq))
            dn_t1 = p.t1_rate * (t1_target - self.n)
        dn_terms["T1"] = dn_t1

        dn = sum(dn_terms.values())
        if not breakdown:
            return dn

        kp, km = self.branch_indices(p.rf_burn_R)
        scale = self.display_cal / self.dR

        def obs_for_term(term: np.ndarray) -> Dict[str, float]:
            d = {"dIplus_R_dt": np.nan, "dIminus_R_dt": np.nan, "dIplus_minusR_dt": np.nan, "dIminus_minusR_dt": np.nan, "dP_dt": float(np.sum(term[:, PLUS] - term[:, MINUS]))}
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

    def step(self, n_steps: int = 1, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None) -> None:
        dt = float(self.params.dt)
        if rf_on is None:
            rf_on = bool(self.params.rf_enabled)
        if dnp_on is None:
            dnp_on = bool(self.params.dnp_enabled)
        for _ in range(max(1, int(n_steps))):
            dn = self.derivative(rf_on=rf_on, dnp_on=dnp_on, breakdown=False)
            self.n = self.n + dt * dn
            self.n = np.maximum(self.n, 1e-30)
            sums = self.n.sum(axis=1, keepdims=True)
            self.n *= self.mu[:, None] / np.maximum(sums, 1e-30)
            self.t += dt

    def rf_balance_estimate(self, R: Optional[float] = None) -> Dict[str, float]:
        old_R = self.params.rf_burn_R
        if R is not None:
            self.params.rf_burn_R = float(R)
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
            return {"gamma_hold_Iplus": float(gp), "gamma_hold_Iminus": float(gm), "gamma_common_suggested": float(common)}
        finally:
            self.params.rf_burn_R = old_R

    @property
    def branch_ratio(self) -> float:
        return boltzmann_branch_ratio(_clamp_p(self.polarizations()["P"]))

    @property
    def initial_branch_ratio(self) -> float:
        return boltzmann_branch_ratio(_clamp_p(self.params.p0))
