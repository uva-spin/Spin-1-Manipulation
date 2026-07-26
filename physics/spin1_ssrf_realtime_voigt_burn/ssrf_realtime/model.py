"""
Spin-1 Pake-doublet ss-RF real-time model with an adjustable Voigt RF-rate profile.

This package preserves the established equal-width R-bin population state,
DNP source, analytic Pake line shape, GUI observables, and numerical time scale.
The recovery operator is population dependent, and the ideal one-bin RF selector
is generalized to an adjustable bin-averaged Voigt transition-rate profile.  Spin diffusion
is now computed from transition-resolved, population-dependent packet-pair
currents filtered by spectral overlap and optional orientation correlation:

* The user can initialize any signed vector polarization P0.
* DNP is optional.  When enabled, it drives the population distribution toward
  a user-selected saturation polarization P_DNP_sat at a finite rate; it does
  not maximize beyond that selected condition.
* With DNP disabled, RF equalization is a depolarizing sink.  Subsequent spin
  diffusion/recovery redistributes the remaining polarization without restoring
  the vector area removed by RF.  Same-branch exchange conserves the global
  level populations; the optional cross-branch channel can additionally relax
  tensor order.
* RF and DNP retain the accepted Pake-density rate weighting from the previous
  package.
* Internal spin diffusion uses packet capacities and nonlinear population
  availability directly; it does not multiply the microscopic exchange scale by
  the Pake line height a second time.
* Setting diffusion_scale=0 and both RF widths to zero reproduces the previous ideal-bin RF/DNP dynamics exactly.

State convention
----------------
The dynamic state is stored in packet space using an x coordinate identified
with the I_plus branch coordinate R_plus.  The same packet appears at two
mirror-related physical positions:

    I_plus  at physical R =  x, with n_plus <-> n_zero driven by RF
    I_minus at physical R = -x, with n_zero <-> n_minus driven by RF

At one physical RF bin R_RF, the two overlapping visible components are:

    I_plus(R_RF)  : packet x= R_RF
    I_minus(R_RF) : packet x=-R_RF

One common RF spectral field is applied to both transitions.  Mirror changes are never imposed separately; they follow from the shared populations.
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
from .rf_profile import (
    approximate_voigt_fwhm,
    bin_averaged_voigt,
    implementation_name as voigt_implementation_name,
)

PLUS, ZERO, MINUS = 0, 1, 2


def _clamp_p(P: float) -> float:
    """Keep P inside the physically safe open interval for numeric use."""
    return float(np.clip(float(P), -0.999999, 0.999999))


@dataclass
class Spin1Params:
    """Numerical and phenomenological parameters for the spin-1 model."""

    # Packet/display grid in dimensionless physical R units.
    n_bins: int = 701
    r_min: float = -3.0
    r_max: float = 3.0

    # Realistic analytic Pake branch parameters.  Defaults match Plot_Signal.py.
    line_gamma: float = 0.05
    line_asym: float = 0.04

    # Display calibration.  The analytic branch shape is used for the spectral
    # envelope.  calibration_p fixes the displayed scale; by default P=0.50
    # matches the Plot_Signal.py convention for the minor branch when divisor=10.
    plot_signal_units: bool = True
    plot_divisor: float = 10.0
    display_scale: float = 1.0
    calibration_p: float = 0.50

    # Initial vector polarization and optional tensor polarization.  If q0 is
    # None, the Boltzmann relation is used.  Signed P is allowed.
    p0: float = 0.45
    q0: Optional[float] = None

    # RF location is a physical R coordinate in the plotted spectrum.
    rf_burn_R: float = -0.94
    rf_enabled: bool = False

    # One common RF equalization-rate scale for both overlapping transitions.
    # With center-bin normalization, gamma_rf is the strongest bin-averaged
    # rate before the optional Pake-density multiplier.
    gamma_rf: float = 2.0

    # Adjustable Voigt RF-rate profile in dimensionless physical R units.
    # Widths are FWHM and are distinct from both the static Pake broadening and
    # the zero-quantum diffusion width.  Both zero restores exact one-bin RF.
    rf_gaussian_fwhm_R: float = 0.030
    rf_lorentzian_fwhm_R: float = 0.015
    rf_profile_normalization: str = "center_bin"

    # Zero selects an automatic Gauss-Legendre order based on width/bin size.
    rf_profile_quadrature_order: int = 0

    # Transition-resolved spin-diffusion kernel.  One common bare exchange scale
    # is used for both allowed ladder transitions; any difference between the
    # observed I+ and I- recovery emerges from populations, packet capacities,
    # spectral mismatch, and geometry rather than an imposed 2:1 constant.
    diffusion_scale: float = 5.0

    # Spectral overlap width in dimensionless R units.  This is an effective
    # zero-quantum / flip-flop bandwidth and must ultimately be calibrated from
    # position-resolved recovery data.
    zq_width_R: float = 0.05

    # Optional cross-branch pair channel 0_k+0_l <-> +_k+-_l.  It is disabled by
    # default so the accepted ss-RF direct/mirror response is preserved unless
    # data require this extra tensor-order relaxation channel.
    cross_branch_ratio: float = 0.0

    # The real-space dipolar network is not known from the powder spectrum.
    # orientation_corr_fraction=0 assumes spatial proximity is independent of
    # EFG orientation.  A nonzero value gives extra weight to similar theta
    # packets with the Gaussian width below.
    orientation_corr_fraction: float = 0.0
    orientation_corr_width_deg: float = 20.0

    # Sparse-kernel truncation in units of zq_width_R.
    kernel_cutoff_widths: float = 4.0

    # Microwaves can alter nuclear diffusion near paramagnetic centers.  Keep
    # this at one unless DNP-on/off recovery measurements support a difference.
    microwave_diffusion_factor: float = 1.0

    # Local Pake-density rate weighting retained for RF and DNP only.  The
    # diffusion kernel already contains packet capacities mu_k and mu_l, so the
    # Pake line height is not applied to diffusion a second time.
    capacity_rate_power: float = 1.0
    capacity_rate_clip: float = 12.0

    # DNP build/rebuild.  If enabled, this is an external reservoir that drives
    # the full line toward P_DNP_sat at rate dnp_rate.
    dnp_enabled: bool = False
    p_dnp_sat: float = 0.58
    dnp_rate: float = 0.05

    # Optional ordinary spin-lattice relaxation toward a user-specified thermal
    # vector polarization.  Default is off so no-DNP tests isolate RF loss and
    # spin diffusion redistribution.
    t1_rate: float = 0.0
    t1_p_eq: float = 0.0

    # Integration and display parameters.
    dt: float = 0.0015
    noise_sigma: float = 0.0


class Spin1Model:
    """Stateful spin-1 population model with Voigt-profile ss-RF and optional DNP."""

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

        # The analytic branch density replaces the toy 1/sqrt(1-R) powder law.
        density = normalized_component(self.Rplus, +1, gamma=p.line_gamma, asym=p.line_asym)
        mu = density * self.dR
        self.mu = mu / mu.sum()
        self.base_density = self.mu / self.dR

        self.pref_initial = level_populations_from_PQ(_clamp_p(p.p0), p.q0)
        self.n_ref = self.mu[:, None] * self.pref_initial[None, :]
        self.n = self.n_ref.copy()
        self.t = 0.0

        # Display calibration only.  The state/rates remain population-based.
        self.display_cal = self._compute_display_calibration()
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

    def _compute_display_calibration(self) -> float:
        p = self.params
        if not p.plot_signal_units:
            return float(p.display_scale)
        Pcal = _clamp_p(p.calibration_p)
        pref_cal = level_populations_from_PQ(Pcal, None)
        # Use the absolute minor-branch population difference so signed P gives
        # signed spectra instead of flipping the calibration sign.
        minor_diff = abs(float(pref_cal[ZERO] - pref_cal[MINUS]))
        if minor_diff < 1e-15:
            # Fall back to a well-behaved calibration point.
            pref_cal = level_populations_from_PQ(0.50, None)
            minor_diff = abs(float(pref_cal[ZERO] - pref_cal[MINUS]))
        raw = pake_component_raw(self.Rplus, +1, gamma=p.line_gamma, asym=p.line_asym)
        raw_area = trapezoid_integral(raw, self.Rplus)
        if raw_area <= 0 or not np.isfinite(raw_area):
            return float(p.display_scale)
        return float(p.display_scale * raw_area / (max(p.plot_divisor, 1e-15) * minor_diff))

    def set_params(self, **kwargs) -> None:
        rf_profile_changed = False
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
            }:
                rf_profile_changed = True
        if rf_profile_changed:
            self.invalidate_rf_profile()

    def as_dict(self) -> Dict[str, float]:
        return asdict(self.params)

    def set_rf_enabled(self, enabled: bool) -> None:
        self.params.rf_enabled = bool(enabled)

    def set_dnp_enabled(self, enabled: bool) -> None:
        self.params.dnp_enabled = bool(enabled)

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
        """
        Return packet indices for the two components at physical R.

        I_plus(R) exists if packet x=R exists.
        I_minus(R) exists if packet x=-R exists.
        """
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

    # ------------------------------------------------------------------
    # Adjustable Voigt RF rate profile on the established physical-R grid
    # ------------------------------------------------------------------
    def _rf_profile_key(self, center_R: float) -> Tuple[object, ...]:
        p = self.params
        return (
            float(center_R),
            float(p.rf_gaussian_fwhm_R),
            float(p.rf_lorentzian_fwhm_R),
            str(p.rf_profile_normalization),
            int(p.rf_profile_quadrature_order),
            float(self.dR),
            int(len(self.Rplus)),
            float(self.Rplus[0]),
            float(self.Rplus[-1]),
        )

    def invalidate_rf_profile(self) -> None:
        self._rf_profile_cache_key = None
        self._rf_profile_cache = None

    def rf_profile_physical(self, center_R: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return the common bin-averaged RF rate profile versus physical R.

        If both Voigt widths are zero, the exact legacy one-bin selector is
        returned.  Otherwise a continuous Voigt is integrated across every
        finite R bin.  This is a transition-rate field; it is not subtracted
        from the plotted spectrum.
        """
        if center_R is None:
            center_R = self.params.rf_burn_R
        center_R = float(center_R)
        key = self._rf_profile_key(center_R)
        if self._rf_profile_cache_key == key and self._rf_profile_cache is not None:
            return self.Rplus.copy(), self._rf_profile_cache.copy()

        p = self.params
        profile = bin_averaged_voigt(
            self.Rplus,
            center_R=center_R,
            bin_width_R=self.dR,
            gaussian_fwhm_R=p.rf_gaussian_fwhm_R,
            lorentzian_fwhm_R=p.rf_lorentzian_fwhm_R,
            normalization=str(p.rf_profile_normalization),  # type: ignore[arg-type]
            quadrature_order=int(p.rf_profile_quadrature_order),
        )
        self._rf_profile_cache_key = key
        self._rf_profile_cache = np.asarray(profile, dtype=float)
        return self.Rplus.copy(), self._rf_profile_cache.copy()

    def rf_profile_arrays(self, center_R: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Profile weights at every packet's two physical transition frequencies.

        The + transition of packet k is at physical R=x_k, and its - transition
        is at R=-x_k.  Both arrays are evaluations of one common RF field.
        """
        R, physical = self.rf_profile_physical(center_R)
        if (
            float(self.params.rf_gaussian_fwhm_R) == 0.0
            and float(self.params.rf_lorentzian_fwhm_R) == 0.0
        ):
            minus = np.zeros_like(physical)
            center = self.params.rf_burn_R if center_R is None else float(center_R)
            if self.Rplus[0] <= -center <= self.Rplus[-1]:
                minus[int(np.argmin(np.abs(self.Rplus + center)))] = 1.0
        elif np.allclose(-self.Rplus, R[::-1], rtol=0.0, atol=5e-14):
            minus = physical[::-1].copy()
        else:
            minus = np.interp(-self.Rplus, R, physical, left=0.0, right=0.0)
        return physical.copy(), minus

    def rf_profile_summary(self, center_R: Optional[float] = None) -> Dict[str, float | str]:
        _, profile = self.rf_profile_physical(center_R)
        peak = float(np.max(profile)) if profile.size else 0.0
        threshold = 0.01 * peak if peak > 0.0 else np.inf
        return {
            "equivalent_width_R": float(np.sum(profile) * self.dR),
            "equivalent_bins": float(np.sum(profile)),
            "bins_above_half": float(np.count_nonzero(profile >= 0.5 * peak)) if peak > 0.0 else 0.0,
            "bins_above_1pct": float(np.count_nonzero(profile >= threshold)) if peak > 0.0 else 0.0,
            "profile_peak": peak,
            "approx_fwhm_R": approximate_voigt_fwhm(
                self.params.rf_gaussian_fwhm_R,
                self.params.rf_lorentzian_fwhm_R,
            ),
            "backend": voigt_implementation_name(),
            "normalization": str(self.params.rf_profile_normalization),
        }

    def rf_rate_fields(
        self,
        center_R: Optional[float] = None,
        gamma_rf: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-packet +<->0 and 0<->- equalization rates."""
        if gamma_rf is None:
            gamma_rf = self.params.gamma_rf
        v_plus, v_minus = self.rf_profile_arrays(center_R)
        capacity = self.capacity_rate_weights()
        scale = float(gamma_rf)
        return scale * capacity * v_plus, scale * capacity * v_minus

    def _rf_population_term(
        self,
        rf_on: bool,
        center_R: Optional[float] = None,
        gamma_rf: Optional[float] = None,
    ) -> np.ndarray:
        """Calculate RF dynamics directly from all three packet populations.

        No direct-hole or mirror-gain relation is imposed.  Both transition
        rates can act simultaneously on a packet, which is required for broad
        profiles and carriers near the overlap center.
        """
        dn = np.zeros_like(self.n)
        if not rf_on:
            return dn
        gamma_plus, gamma_minus = self.rf_rate_fields(center_R, gamma_rf)
        delta_plus = self.n[:, PLUS] - self.n[:, ZERO]
        delta_minus = self.n[:, ZERO] - self.n[:, MINUS]
        J_plus = gamma_plus * delta_plus
        J_minus = gamma_minus * delta_minus
        dn[:, PLUS] -= J_plus
        dn[:, ZERO] += J_plus - J_minus
        dn[:, MINUS] += J_minus
        return dn

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
        """Return the exact positive-P Plot_Signal-style static reference for comparison."""
        from .lineshape import plot_signal_reference
        return plot_signal_reference(
            self.Rplus,
            P=self.params.p0,
            gamma=self.params.line_gamma,
            asym=self.params.line_asym,
            divisor=self.params.plot_divisor,
        )

    def capacity_rate_weights(self) -> np.ndarray:
        """Return Pake-density rate multipliers with mu-weighted average one.

        The packet density is the broadened Pake branch density used to initialize
        the line.  It is a proxy for how many spins and how much transition
        strength reside in an equal-width R bin.  Applying it as a rate multiplier
        makes DNP fill and RF depolarization fastest at the horns/theta=90 deg
        regions while leaving the GUI rate knobs as material-scale averages.
        """
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
        """Return actual local RF and DNP rates at the selected burn center."""
        if R is None:
            R = self.params.rf_burn_R
        cap = self.local_capacity_factors(R)
        p = self.params
        kp, km = self.branch_indices(R)
        v_plus, v_minus = self.rf_profile_arrays(R)
        w = self.capacity_rate_weights()

        def at(arr: np.ndarray, idx: Optional[int]) -> float:
            return np.nan if idx is None else float(arr[idx])

        vp = at(v_plus, kp)
        vm = at(v_minus, km)
        opposite_plus_packet = at(v_minus, kp)
        opposite_minus_packet = at(v_plus, km)
        wp = cap["w_Iplus_R"]
        wm = cap["w_Iminus_R"]
        return {
            **cap,
            "rf_profile_Iplus_R": vp,
            "rf_profile_Iminus_R": vm,
            "rf_profile_opposite_on_Iplus_packet": opposite_plus_packet,
            "rf_profile_opposite_on_Iminus_packet": opposite_minus_packet,
            "gamma_rf_Iplus_R": float(p.gamma_rf * wp * vp) if np.isfinite(wp) and np.isfinite(vp) else np.nan,
            "gamma_rf_Iminus_R": float(p.gamma_rf * wm * vm) if np.isfinite(wm) and np.isfinite(vm) else np.nan,
            "gamma_rf_opposite_on_Iplus_packet": (
                float(p.gamma_rf * wp * opposite_plus_packet)
                if np.isfinite(wp) and np.isfinite(opposite_plus_packet) else np.nan
            ),
            "gamma_rf_opposite_on_Iminus_packet": (
                float(p.gamma_rf * wm * opposite_minus_packet)
                if np.isfinite(wm) and np.isfinite(opposite_minus_packet) else np.nan
            ),
            "dnp_Iplus_R": float(p.dnp_rate * wp) if np.isfinite(wp) else np.nan,
            "dnp_Iminus_R": float(p.dnp_rate * wm) if np.isfinite(wm) else np.nan,
        }

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

    # ------------------------------------------------------------------
    # Population-dependent spin diffusion on the established R-packet grid
    # ------------------------------------------------------------------
    def _effective_theta(self) -> np.ndarray:
        """Infer the packet EFG angle from the I+ branch coordinate.

        The dynamic grid remains the equal-width R grid used by the accepted
        RF/DNP model.  This inverse map is used only to construct an optional
        orientation-correlation factor in the diffusion kernel.  Broadening
        tails outside the first-order support are clipped to the nearest
        physical orientation; their small packet capacities suppress them.
        """
        s = float(self.params.line_asym)
        cos2 = np.clip((1.0 - s - self.Rplus) / 3.0, 0.0, 1.0)
        return np.arccos(np.sqrt(cos2))

    def _diffusion_key(self) -> Tuple[float, ...]:
        p = self.params
        return (
            float(p.zq_width_R),
            float(p.cross_branch_ratio),
            float(p.orientation_corr_fraction),
            float(p.orientation_corr_width_deg),
            float(p.kernel_cutoff_widths),
            float(p.line_asym),
            float(p.n_bins),
            float(p.r_min),
            float(p.r_max),
        )

    def _spectral_overlap(self, delta_R: np.ndarray) -> np.ndarray:
        """Gaussian spectral-overlap factor used by the reduced kernel."""
        width = max(float(self.params.zq_width_R), 1e-12)
        x = np.asarray(delta_R, dtype=float) / width
        out = np.exp(-0.5 * x * x)
        cutoff = max(float(self.params.kernel_cutoff_widths), 1.0)
        out[np.abs(x) > cutoff] = 0.0
        return out

    def _orientation_factor(self, i: np.ndarray, j: np.ndarray, theta: np.ndarray) -> np.ndarray:
        f = float(np.clip(self.params.orientation_corr_fraction, 0.0, 1.0))
        if f == 0.0:
            return np.ones_like(i, dtype=float)
        sigma = np.deg2rad(max(float(self.params.orientation_corr_width_deg), 1e-6))
        dtheta = theta[i] - theta[j]
        gaussian = np.exp(-0.5 * (dtheta / sigma) ** 2)
        return (1.0 - f) + f * gaussian

    def _ensure_diffusion_kernel(self) -> None:
        """Build sparse same-branch and cross-branch packet-pair kernels.

        The base pair weight is

            mu_i mu_j * spatial_orientation_factor * spectral_overlap.

        The common diffusion_scale and microwave factor are applied at run time.
        Same-branch pairs are stored once with i<j.  Cross-branch pairs are
        ordered because +_i coupled to -_j is a different channel from +_j to
        -_i.
        """
        key = self._diffusion_key()
        if self._diffusion_kernel_key == key:
            return

        N = len(self.Rplus)
        theta = self._effective_theta()
        width = max(float(self.params.zq_width_R), 1e-12)
        cutoff = max(float(self.params.kernel_cutoff_widths), 1.0)
        max_delta = cutoff * width

        # Same-transition exchange: R_sigma(i) ~= R_sigma(j).  Since the grid is
        # uniform in R, only a narrow band around the diagonal is required.
        max_offset = max(1, int(np.ceil(max_delta / max(self.dR, 1e-15))))
        same_i_parts = []
        same_j_parts = []
        same_w_parts = []
        for off in range(1, min(max_offset, N - 1) + 1):
            i = np.arange(0, N - off, dtype=np.int64)
            j = i + off
            delta = self.Rplus[i] - self.Rplus[j]
            ov = self._spectral_overlap(delta)
            keep = ov > 0.0
            if not np.any(keep):
                continue
            i = i[keep]
            j = j[keep]
            ov = ov[keep]
            orient = self._orientation_factor(i, j, theta)
            base = self.mu[i] * self.mu[j] * orient * ov
            positive = base > 0.0
            same_i_parts.append(i[positive])
            same_j_parts.append(j[positive])
            same_w_parts.append(base[positive])

        if same_i_parts:
            self._same_i = np.concatenate(same_i_parts)
            self._same_j = np.concatenate(same_j_parts)
            self._same_base = np.concatenate(same_w_parts)
        else:
            self._same_i = np.empty(0, dtype=np.int64)
            self._same_j = np.empty(0, dtype=np.int64)
            self._same_base = np.empty(0, dtype=float)

        # Cross-branch exchange: R_plus(i) ~= R_minus(j) = -Rplus(j), hence
        # Rplus(i)+Rplus(j) ~= 0.  Build a sparse anti-diagonal band.
        cross_i_parts = []
        cross_j_parts = []
        cross_w_parts = []
        if float(self.params.cross_branch_ratio) != 0.0:
            R = self.Rplus
            for ii in range(N):
                target = -R[ii]
                lo = int(np.searchsorted(R, target - max_delta, side="left"))
                hi = int(np.searchsorted(R, target + max_delta, side="right"))
                if hi <= lo:
                    continue
                jj = np.arange(lo, hi, dtype=np.int64)
                ii_arr = np.full(jj.size, ii, dtype=np.int64)
                delta = R[ii] + R[jj]
                ov = self._spectral_overlap(delta)
                keep = ov > 0.0
                if not np.any(keep):
                    continue
                jj = jj[keep]
                ii_arr = ii_arr[keep]
                ov = ov[keep]
                orient = self._orientation_factor(ii_arr, jj, theta)
                base = self.mu[ii_arr] * self.mu[jj] * orient * ov
                # For two spins drawn from the same packet, use half the naive
                # ordered-pair count.  This matters only near R=0.
                base = np.where(ii_arr == jj, 0.5 * base, base)
                positive = base > 0.0
                cross_i_parts.append(ii_arr[positive])
                cross_j_parts.append(jj[positive])
                cross_w_parts.append(base[positive])

        if cross_i_parts:
            self._cross_i = np.concatenate(cross_i_parts)
            self._cross_j = np.concatenate(cross_j_parts)
            self._cross_base = np.concatenate(cross_w_parts)
        else:
            self._cross_i = np.empty(0, dtype=np.int64)
            self._cross_j = np.empty(0, dtype=np.int64)
            self._cross_base = np.empty(0, dtype=float)

        self._diffusion_kernel_key = key

    def diffusion_connectivity(self, dnp_on: Optional[bool] = None) -> Dict[str, np.ndarray]:
        """Return population-independent maximum connectivity per packet.

        Connectivity is the sum of pair weights divided by packet capacity.  It
        is not an instantaneous recovery rate; actual currents also contain the
        nonlinear forward-minus-reverse population products.
        """
        self._ensure_diffusion_kernel()
        if dnp_on is None:
            dnp_on = bool(self.params.dnp_enabled)
        mw = float(self.params.microwave_diffusion_factor) if dnp_on else 1.0
        scale = float(self.params.diffusion_scale) * mw
        conn_same = np.zeros(len(self.Rplus), dtype=float)
        if self._same_base.size:
            np.add.at(conn_same, self._same_i, self._same_base)
            np.add.at(conn_same, self._same_j, self._same_base)
        conn_cross_plus = np.zeros_like(conn_same)
        conn_cross_minus = np.zeros_like(conn_same)
        if self._cross_base.size:
            np.add.at(conn_cross_plus, self._cross_i, self._cross_base)
            np.add.at(conn_cross_minus, self._cross_j, self._cross_base)
        mu_safe = np.maximum(self.mu, 1e-30)
        return {
            "same_plus0": scale * conn_same / mu_safe,
            "same_0minus": scale * conn_same / mu_safe,
            "cross_plus": scale * float(self.params.cross_branch_ratio) * conn_cross_plus / mu_safe,
            "cross_minus": scale * float(self.params.cross_branch_ratio) * conn_cross_minus / mu_safe,
        }

    def _spin_diffusion_terms(self, dnp_on: bool) -> Dict[str, np.ndarray]:
        """Compute conservative, population-dependent spin-diffusion currents."""
        self._ensure_diffusion_kernel()
        if float(self.params.diffusion_scale) == 0.0:
            z = np.zeros_like(self.n)
            return {
                "diff_plus0": z.copy(),
                "diff_0minus": z.copy(),
                "diff_cross": z.copy(),
            }

        frac = self.n / np.maximum(self.mu[:, None], 1e-30)
        pp = frac[:, PLUS]
        p0 = frac[:, ZERO]
        pm = frac[:, MINUS]
        mw = float(self.params.microwave_diffusion_factor) if dnp_on else 1.0
        scale = float(self.params.diffusion_scale) * mw

        # Same-branch + <-> 0 exchange between packet pairs.
        dn_p = np.zeros_like(self.n)
        if self._same_base.size:
            i = self._same_i
            j = self._same_j
            K = scale * self._same_base
            Jp = K * (p0[i] * pp[j] - pp[i] * p0[j])
            np.add.at(dn_p[:, PLUS], i, Jp)
            np.add.at(dn_p[:, ZERO], i, -Jp)
            np.add.at(dn_p[:, PLUS], j, -Jp)
            np.add.at(dn_p[:, ZERO], j, Jp)

        # Same-branch 0 <-> - exchange between the same packet pairs.
        dn_m = np.zeros_like(self.n)
        if self._same_base.size:
            i = self._same_i
            j = self._same_j
            K = scale * self._same_base
            Jm = K * (pm[i] * p0[j] - p0[i] * pm[j])
            np.add.at(dn_m[:, ZERO], i, Jm)
            np.add.at(dn_m[:, MINUS], i, -Jm)
            np.add.at(dn_m[:, ZERO], j, -Jm)
            np.add.at(dn_m[:, MINUS], j, Jm)

        # Optional cross-branch pair process 0_i+0_j <-> +_i+-_j.
        dn_x = np.zeros_like(self.n)
        cross_ratio = float(self.params.cross_branch_ratio)
        if cross_ratio != 0.0 and self._cross_base.size:
            i = self._cross_i
            j = self._cross_j
            Kx = scale * cross_ratio * self._cross_base
            Jx = Kx * (p0[i] * p0[j] - pp[i] * pm[j])
            np.add.at(dn_x[:, PLUS], i, Jx)
            np.add.at(dn_x[:, ZERO], i, -Jx)
            np.add.at(dn_x[:, MINUS], j, Jx)
            np.add.at(dn_x[:, ZERO], j, -Jx)

        return {
            "diff_plus0": dn_p,
            "diff_0minus": dn_m,
            "diff_cross": dn_x,
        }

    def local_diffusion_diagnostics(self, R: Optional[float] = None, dnp_on: Optional[bool] = None) -> Dict[str, float]:
        """Return live connectivity and RF-off diffusion slopes at a selected R."""
        if R is None:
            R = self.params.rf_burn_R
        if dnp_on is None:
            dnp_on = bool(self.params.dnp_enabled)
        kp, km = self.branch_indices(R)
        conn = self.diffusion_connectivity(dnp_on=dnp_on)
        terms = self._spin_diffusion_terms(bool(dnp_on))
        net = terms["diff_plus0"] + terms["diff_0minus"] + terms["diff_cross"]
        scale_obs = self.display_cal / self.dR

        def val(arr: np.ndarray, idx: Optional[int]) -> float:
            return np.nan if idx is None else float(arr[idx])

        out = {
            "R": float(R),
            "conn_Iplus": val(conn["same_plus0"] + conn["cross_plus"], kp),
            "conn_Iminus": val(conn["same_0minus"] + conn["cross_minus"], km),
            "dIplus_diff_dt": np.nan,
            "dIminus_diff_dt": np.nan,
            "lambda_Iplus": np.nan,
            "lambda_Iminus": np.nan,
        }
        if kp is not None:
            slope = scale_obs * (net[kp, PLUS] - net[kp, ZERO])
            out["dIplus_diff_dt"] = float(slope)
            Ip = self.local_intensities(R)["Iplus"]
            # An operational instantaneous rate relative to the current direct
            # signal.  The signed slope itself is the primary diagnostic.
            if np.isfinite(Ip) and abs(Ip) > 1e-15:
                out["lambda_Iplus"] = float(slope / Ip)
        if km is not None:
            slope = scale_obs * (net[km, ZERO] - net[km, MINUS])
            out["dIminus_diff_dt"] = float(slope)
            Im = self.local_intensities(R)["Iminus"]
            if np.isfinite(Im) and abs(Im) > 1e-15:
                out["lambda_Iminus"] = float(slope / Im)
        return out

    def derivative(self, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None, breakdown: bool = False):
        p = self.params
        if rf_on is None:
            rf_on = bool(p.rf_enabled)
        if dnp_on is None:
            dnp_on = bool(p.dnp_enabled)
        dn_terms: Dict[str, np.ndarray] = {}

        dn_rf = self._rf_population_term(
            bool(rf_on), center_R=p.rf_burn_R, gamma_rf=p.gamma_rf
        )
        dn_terms["RF"] = dn_rf

        # Only the recovery operator differs from the previous package.  Pair
        # currents conserve packet mass and total vector polarization by
        # construction; no after-the-fact projection toward a reference line is
        # used.
        diff_terms = self._spin_diffusion_terms(bool(dnp_on))
        dn_terms.update(diff_terms)

        dn_dnp = np.zeros_like(self.n)
        if dnp_on and p.dnp_rate != 0.0:
            dnp_target = self.equilibrium_reference(_clamp_p(p.p_dnp_sat))
            # DNP fills toward a Boltzmann-like Pake distribution over theta.
            # Multiplying by the local Pake-density rate factor makes absolute
            # and fractional build visibly faster at the horns while preserving
            # the selected P_DNP_sat as the asymptotic state.
            w = self.capacity_rate_weights()[:, None]
            dn_dnp = p.dnp_rate * w * (dnp_target - self.n)
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

    def step(self, n_steps: int = 1, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None) -> None:
        """Advance the model by n_steps using positivity-preserving Euler steps."""
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
        """Estimate the common RF scale needed to balance non-RF refill.

        The estimate uses the full Voigt population operator at unit gamma_rf;
        it does not assume a factor-of-two direct/mirror relation.
        """
        old_R = self.params.rf_burn_R
        if R is not None:
            self.params.rf_burn_R = float(R)
            self.invalidate_rf_profile()
        try:
            _, nonrf = self.derivative(rf_on=False, dnp_on=self.params.dnp_enabled, breakdown=True)
            unit_term = self._rf_population_term(True, self.params.rf_burn_R, gamma_rf=1.0)
            kp, km = self.branch_indices(self.params.rf_burn_R)
            scale = self.display_cal / self.dR

            u_plus = np.nan if kp is None else float(scale * (unit_term[kp, PLUS] - unit_term[kp, ZERO]))
            u_minus = np.nan if km is None else float(scale * (unit_term[km, ZERO] - unit_term[km, MINUS]))
            refill_plus = float(nonrf["net"]["dIplus_R_dt"])
            refill_minus = float(nonrf["net"]["dIminus_R_dt"])

            def required(refill: float, unit_slope: float) -> float:
                if not np.isfinite(refill) or not np.isfinite(unit_slope) or abs(unit_slope) < 1e-20:
                    return np.nan
                value = -refill / unit_slope
                return float(max(0.0, value)) if np.isfinite(value) else np.nan

            gp = required(refill_plus, u_plus)
            gm = required(refill_minus, u_minus)
            vals = [v for v in (gp, gm) if np.isfinite(v)]
            return {
                "gamma_hold_Iplus": float(gp),
                "gamma_hold_Iminus": float(gm),
                "gamma_common_suggested": float(max(vals)) if vals else np.nan,
            }
        finally:
            self.params.rf_burn_R = old_R
            self.invalidate_rf_profile()

    @property
    def branch_ratio(self) -> float:
        return boltzmann_branch_ratio(_clamp_p(self.polarizations()["P"]))

    @property
    def initial_branch_ratio(self) -> float:
        return boltzmann_branch_ratio(_clamp_p(self.params.p0))
