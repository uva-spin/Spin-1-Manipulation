"""
Spin-1 Pake-doublet ss-RF real-time model, v11 (headless).

Headless copy of ``physics/simulations/spin1_ssrf_realtime/ssrf_realtime/model.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .conversions import physical_intensities_to_packet_n, packet_n_to_physical_intensities
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

    n_bins: int = 701
    r_min: float = -3.0
    r_max: float = 3.0

    line_gamma: float = 0.05
    line_asym: float = 0.04

    plot_signal_units: bool = True
    plot_divisor: float = 10.0
    display_scale: float = 1.0
    calibration_p: float = 0.50

    p0: float = 0.45
    q0: Optional[float] = None

    rf_burn_R: float = 0.40
    rf_enabled: bool = False

    gamma_rf: float = 2.0

    d_same_plus0: float = 0.18
    d_same_0minus: float = 0.10
    d_spec_plus0: float = 2.0
    d_spec_0minus: float = 1.0

    t2_width_R: float = 0.05

    dnp_enabled: bool = False
    p_dnp_sat: float = 0.58
    dnp_rate: float = 0.05

    t1_rate: float = 0.0
    t1_p_eq: float = 0.0

    dt: float = 0.0015
    noise_sigma: float = 0.0


class Spin1Model:
    """Stateful spin-1 population model with ideal-bin ss-RF and optional DNP."""

    def __init__(self, params: Optional[Spin1Params] = None):
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
        self.mu = mu / mu.sum()
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

    def load_from_physical_intensities(self, Iplus: np.ndarray, Iminus: np.ndarray) -> None:
        """Set ``self.n`` from physical R-grid intensities using model conversions."""
        self.n = physical_intensities_to_packet_n(
            Iplus,
            Iminus,
            self.mu,
            display_cal=self.display_cal,
            dR=self.dR,
        )

    def equilibrium_reference(self, P: Optional[float] = None) -> np.ndarray:
        """Boltzmann-shaped packet state with the current grid weights."""
        if P is None:
            P = self.polarizations()["P"]
        pref = level_populations_from_PQ(_clamp_p(float(P)), None)
        return self.mu[:, None] * pref[None, :]

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

    def branch_areas(self, n: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Return display-calibrated integrated branch areas and total area."""
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

    def branch_indices(self, R: Optional[float] = None) -> Tuple[Optional[int], Optional[int]]:
        """Return packet indices for the two components at physical R."""
        if R is None:
            R = self.params.rf_burn_R
        R = float(R)
        kp: Optional[int] = None
        km: Optional[int] = None
        if self.Rplus[0] <= R <= self.Rplus[-1]:
            kp = int(np.argmin(np.abs(self.Rplus - R)))
        if self.Rplus[0] <= -R <= self.Rplus[-1]:
            km = int(np.argmin(np.abs(self.Rplus + R)))
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

    def packet_intensities(self, use_reference: bool = False, density: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return display-calibrated I_plus(x) and I_minus(x-packet) arrays."""
        n = self.n_ref if use_reference else self.n
        Iplus, Iminus = self._transition_differences(n)
        if density:
            Iplus = self.display_cal * Iplus / self.dR
            Iminus = self.display_cal * Iminus / self.dR
        else:
            Iplus = self.display_cal * Iplus
            Iminus = self.display_cal * Iminus
        return Iplus, Iminus

    def physical_intensities(self, n: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return I+(R), I-(R), and total on the model's physical R grid."""
        state = self.n if n is None else n
        return packet_n_to_physical_intensities(
            state, self.Rplus, display_cal=self.display_cal, dR=self.dR
        )

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
        return {
            "Iplus": Iplus,
            "Iminus": Iminus,
            "total": float(total),
            "k_plus": -1 if kp is None else int(kp),
            "k_minus": -1 if km is None else int(km),
        }

    def pair_intensities(self, R: Optional[float] = None, use_reference: bool = False) -> Dict[str, float]:
        """Return direct (+R) and mirror (-R) local components in display units."""
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
        """Return direct/mirror values and changes from the initial reference."""
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
        """Project a provided packet-population state to physical R bin centers."""
        Iplus, Iminus, total = packet_n_to_physical_intensities(
            n, self.Rplus, display_cal=self.display_cal, dR=self.dR
        )
        return self.Rplus.copy(), Iplus, Iminus, total

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
        """Return the exact positive-P Plot_Signal-style static reference for comparison."""
        from .lineshape import plot_signal_reference
        return plot_signal_reference(
            self.Rplus,
            P=self.params.p0,
            gamma=self.params.line_gamma,
            asym=self.params.line_asym,
            divisor=self.params.plot_divisor,
        )

    def packet_spectrum(self, use_reference: bool = False, noise_sigma: Optional[float] = None):
        """Return branch component spectra at their physical R bin centers."""
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

    def _mode_relax_reference(self, which: str, rate: float, reference: np.ndarray) -> np.ndarray:
        if rate == 0.0:
            return np.zeros_like(self.n)
        a_plus0, b_0minus = self._rf_mode_amplitudes(reference)
        if which == "plus0":
            return self._mode_to_population_derivative(-rate * a_plus0, np.zeros_like(b_0minus))
        if which == "0minus":
            return self._mode_to_population_derivative(np.zeros_like(a_plus0), -rate * b_0minus)
        raise ValueError("which must be 'plus0' or '0minus'")

    def _mode_diffuse_delta(self, which: str, rate: float, reference: np.ndarray) -> np.ndarray:
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

        flux = rate * overlap * edge_mass * (rho_mode[1:] - rho_mode[:-1])
        dmode_dt = np.zeros_like(mode)
        dmode_dt[:-1] += flux
        dmode_dt[1:] -= flux

        if which == "plus0":
            return self._mode_to_population_derivative(dmode_dt, np.zeros_like(mode))
        return self._mode_to_population_derivative(np.zeros_like(mode), dmode_dt)

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

        dn_same = np.zeros_like(self.n)
        dn_same += self._mode_relax_reference("plus0", p.d_same_plus0, dynamic_ref)
        dn_same += self._mode_relax_reference("0minus", p.d_same_0minus, dynamic_ref)
        dn_same = self._project_conserve_vector(dn_same)
        dn_terms["spin_temp_redistribution"] = dn_same

        dn_spec = np.zeros_like(self.n)
        dn_spec += self._mode_diffuse_delta("plus0", p.d_spec_plus0, dynamic_ref)
        dn_spec += self._mode_diffuse_delta("0minus", p.d_spec_0minus, dynamic_ref)
        dn_spec = self._project_conserve_vector(dn_spec)
        dn_terms["spectral_neighbors"] = dn_spec

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

    def _apply_euler_update(self, dn: np.ndarray, dt: float) -> None:
        self.n = self.n + dt * dn
        self.n = np.maximum(self.n, 1e-30)
        sums = self.n.sum(axis=1, keepdims=True)
        self.n *= self.mu[:, None] / np.maximum(sums, 1e-30)

    def step(self, n_steps: int = 1, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None) -> None:
        """Advance the model by n_steps using positivity-preserving Euler steps."""
        dt = float(self.params.dt)
        if rf_on is None:
            rf_on = bool(self.params.rf_enabled)
        if dnp_on is None:
            dnp_on = bool(self.params.dnp_enabled)
        for _ in range(max(1, int(n_steps))):
            dn = self.derivative(rf_on=rf_on, dnp_on=dnp_on, breakdown=False)
            self._apply_euler_update(dn, dt)
            self.t += dt

    def step_once(
        self,
        dt: Optional[float] = None,
        *,
        rf_on: Optional[bool] = None,
        dnp_on: Optional[bool] = None,
        rf_only: bool = False,
    ) -> np.ndarray:
        """
        Advance one Euler step and return the updated packet populations.

        When ``rf_only`` is True, only the ideal-bin RF term is applied.
        """
        step_dt = float(self.params.dt if dt is None else dt)
        if rf_only:
            p = self.params
            dn = np.zeros_like(self.n)
            if rf_on is not False and p.gamma_rf != 0.0:
                kp, km = self.branch_indices(p.rf_burn_R)
                if kp is not None:
                    J = p.gamma_rf * (self.n[kp, PLUS] - self.n[kp, ZERO])
                    dn[kp, PLUS] -= J
                    dn[kp, ZERO] += J
                if km is not None:
                    J = p.gamma_rf * (self.n[km, ZERO] - self.n[km, MINUS])
                    dn[km, ZERO] -= J
                    dn[km, MINUS] += J
        else:
            dn = self.derivative(rf_on=rf_on, dnp_on=dnp_on, breakdown=False)
        self._apply_euler_update(dn, step_dt)
        self.t += step_dt
        return self.n.copy()

    def rf_balance_estimate(self, R: Optional[float] = None) -> Dict[str, float]:
        """Estimate common RF rate needed to hold the two direct components."""
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
            return {
                "gamma_hold_Iplus": float(gp),
                "gamma_hold_Iminus": float(gm),
                "gamma_common_suggested": float(common),
            }
        finally:
            self.params.rf_burn_R = old_R

    @property
    def branch_ratio(self) -> float:
        return boltzmann_branch_ratio(_clamp_p(self.polarizations()["P"]))

    @property
    def initial_branch_ratio(self) -> float:
        return boltzmann_branch_ratio(_clamp_p(self.params.p0))
