"""
ss-RF simulation with a ping-pong sweep counter (Clement & Keller, NIM A 1050).

Implements the simulation rule of Section 7.0.1: when the sweep counter is aimed
at a bin, that bin follows the differential equation for **growth then ss-RF
decay, in that order**.

* Growth (enhancement): each bin relaxes toward the reference lineshape,
  ``I → I + r·dt·(I_ref − I)`` (exponential saturation toward the target).
* ss-RF decay: the ss-RF drives the two populations in the bin toward each
  other with power proportional to the product of a user power constant and the
  difference between the two populations. Because ``I_+ = C(ρ_+ − ρ_0)`` and
  ``I_- = C(ρ_0 − ρ_-)``, each intensity is exactly that population difference,
  so equalizing the populations by a fraction ``ξ`` removes a fraction
  ``min(2ξ, 1)`` of each intensity at the burn bin. The opposing absorption line
  at the mirror bin gains half of what is lost, matching the rates response
  ``A_gained = ½ A_lost`` (Eq. 38).

``ξ = power_constant × profile(ω) × dt`` is the per-step ss-RF power; the burn
shape across bins comes from the discretized Voigt/Gaussian profile.

Populations (``ρ_+, ρ_0, ρ_-``) are recovered from the resulting intensities via
the ``AFP`` conversions in ``physics/afp.py`` and returned for inspection.

Also supports instantaneous single-bin burns (``apply_bin_burn``).
"""

from __future__ import annotations

from importlib import util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import wofz
from Lineshape import GenerateVectorLineshape

_afp_path = Path(__file__).resolve().parent.parent / "afp.py"
_afp_spec = util.spec_from_file_location("_spin1_afp", str(_afp_path))
_afp_mod = util.module_from_spec(_afp_spec)
_afp_spec.loader.exec_module(_afp_mod)
AFP = _afp_mod.AFP

BinRange = Union[Tuple[int, int], List[int], np.ndarray]


def build_ping_pong_cycle(low: int, high: int) -> List[int]:
    """
    Build one ping-pong sweep: low → high (inclusive), then high-1 → low.

    The upper edge is visited only once per cycle (one time step at the top).
    """
    low = int(low)
    high = int(high)
    if low > high:
        raise ValueError(f"bin_range requires low <= high; got ({low}, {high})")
    up = list(range(low, high + 1))
    down = list(range(high - 1, low - 1, -1))
    return up + down


class SweepCounter:
    """
    Counter that sweeps a bin range ping-pong fashion at a controllable rate.

    ``sweep_rate`` is the number of bins advanced per simulation time step
    (fractional rates accumulate until a whole bin step is taken).
    """

    def __init__(self, low: int, high: int, sweep_rate: float = 1.0):
        self.low = int(low)
        self.high = int(high)
        self.cycle = build_ping_pong_cycle(self.low, self.high)
        self.sweep_rate = float(sweep_rate)
        self._accum = 0.0
        self._cycle_pos = 0
        self.current_bin = self.cycle[0]

    def reset(self) -> None:
        self._accum = 0.0
        self._cycle_pos = 0
        self.current_bin = self.cycle[0]

    def advance(self) -> List[int]:
        """
        Apply growth/decay at the current bin, then move along the cycle.

        Returns every bin visited during this time step (always includes the
        starting bin; additional bins when ``sweep_rate`` exceeds one).
        """
        visited: List[int] = [self.current_bin]
        self._accum += self.sweep_rate
        while self._accum >= 1.0:
            self._cycle_pos = (self._cycle_pos + 1) % len(self.cycle)
            self.current_bin = self.cycle[self._cycle_pos]
            visited.append(self.current_bin)
            self._accum -= 1.0
        return visited


class ssRFMapper:
    """
    Simulate ss-RF via per-bin spin-1 populations and a sweeping counter.

    Parameters
    ----------
    f, sigma, gamma, x0, amp :
        Frequency grid and discretized Voigt/Gaussian burn profile parameters.
        power_constant :
        User-set ss-RF strength scaling each burn step.
        growth_rate :
        Enhancement rate toward reference populations during the growth phase.
        center_freq :
        Frequency axis centre used for mirror mapping.
        calibration_constant :
        ``C`` in the burn intensity formulas (default 1).
    """

    def __init__(
        self,
        f: np.ndarray,
        sigma: float,
        gamma: float,
        x0: float,
        amp: float,
        power_constant: Optional[float] = None,
        growth_rate: float = 0.0,
        center_freq: float = 0.0,
        calibration_constant: float = 1.0,
    ):
        self.f = np.asarray(f, dtype=float)
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.x0 = float(x0)
        self.amp = float(amp)
        self.power_constant = float(amp if power_constant is None else power_constant)
        self.growth_rate = float(growth_rate)
        self.center_freq = float(center_freq)
        self.calibration_constant = float(calibration_constant)
        self._st_p_grid: Optional[np.ndarray] = None
        self._st_ps_grid: Optional[np.ndarray] = None
        self._st_iplus_grid: Optional[np.ndarray] = None
        self._st_iminus_grid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Burn profile (Voigt / Gaussian)
    # ------------------------------------------------------------------

    def _voigt_profile(self, x: np.ndarray, x0: float) -> np.ndarray:
        x_norm = (x - x0) / (self.sigma * np.sqrt(2))
        z = x_norm + 1j * (self.gamma / (self.sigma * np.sqrt(2)))
        return np.real(wofz(z)) / (self.sigma * np.sqrt(2 * np.pi))

    def _gaussian_profile(self, x: np.ndarray, x0: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - x0) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))

    def _discretize_profile(self, profile_type: str = "voigt") -> np.ndarray:
        if profile_type.lower() == "voigt":
            profile = self._voigt_profile(self.f, self.x0)
        elif profile_type.lower() == "gaussian":
            profile = self._gaussian_profile(self.f, self.x0)
        else:
            raise ValueError(f"Unknown profile_type: {profile_type}. Use 'voigt' or 'gaussian'")

        max_val = np.max(profile)
        if max_val > 0:
            profile = profile / max_val
        return profile * self.amp

    @staticmethod
    def _normalize_populations(
        rho_plus: np.ndarray,
        rho_zero: np.ndarray,
        rho_minus: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total = rho_plus.sum() + rho_zero.sum() + rho_minus.sum()
        if total <= 0:
            raise ValueError("Population sum must be positive.")
        return rho_plus / total, rho_zero / total, rho_minus / total

    def _mirror_freq_index(self, bin_idx: int) -> Optional[int]:
        """Frequency-bin index symmetric about ``center_freq``, or None if off-grid."""
        freq = self.f[bin_idx]
        mirrored_freq = 2 * self.center_freq - freq
        freq_diff = np.abs(self.f - mirrored_freq)
        mirrored_idx = int(np.argmin(freq_diff))
        bin_width = (self.f[1] - self.f[0]) / 2 if len(self.f) > 1 else np.inf
        if freq_diff[mirrored_idx] < bin_width:
            return mirrored_idx
        return None

    @staticmethod
    def _population_slice_index(freq_bin: int, n_bins: int) -> int:
        """Map frequency bin to AFP population slice (θ index)."""
        return int(freq_bin)

    @staticmethod
    def _neg_theta_index(theta_idx: int, n_bins: int) -> int:
        """Mirror θ index: ``-θ`` maps to the paired slice."""
        return int(n_bins - 1 - theta_idx)

    @staticmethod
    def _r_to_theta_component(
        rho_r: np.ndarray,
        idx: int,
        mirror_idx: int,
    ) -> float:
        """``f(θ) = f(R) + f(-R)`` for one population component."""
        return float(rho_r[idx] + rho_r[mirror_idx])

    @staticmethod
    def _theta_to_r_component(
        rho_theta: np.ndarray,
        idx: int,
        neg_theta_idx: int,
    ) -> Tuple[float, float]:
        """
        Split a θ-space component total across ``R`` and ``-R``.

        ``f(R) = f(θ) + f(-θ)`` with each R bin carrying half the paired sum.
        """
        total = float(rho_theta[idx] + rho_theta[neg_theta_idx])
        return 0.5 * total, 0.5 * total

    def _theta_bin_indices_for_burn(
        self,
        freq_bin: int,
        n_bins: int,
    ) -> Tuple[int, int, Optional[int]]:
        """
        Return ``(theta_2, theta_1, mirror_bin)`` for a burn at ``freq_bin``.

        ``δθ₂`` hosts the ``ρ_+ ↔ ρ_0`` transition (``I_+`` region).
        ``δθ₁`` hosts the ``ρ_0 ↔ ρ_-`` transition (``I_-`` region).
        """
        mirror_bin = self._mirror_freq_index(freq_bin)
        theta_2 = self._population_slice_index(freq_bin, n_bins)
        if mirror_bin is None or mirror_bin == freq_bin:
            return theta_2, theta_2, None
        theta_1 = self._population_slice_index(mirror_bin, n_bins)
        return theta_2, theta_1, mirror_bin

    def _theta_space_populations(
        self,
        rho_plus: np.ndarray,
        rho_zero: np.ndarray,
        rho_minus: np.ndarray,
        freq_bin: int,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Raw and normalized θ-space totals for the symmetric pair at ``freq_bin``.

        Uses ``ρ(θ) = ρ(R) + ρ(-R)`` for each spin component.
        """
        mirror_bin = self._mirror_freq_index(freq_bin)
        eps = np.finfo(float).eps

        if mirror_bin is None or mirror_bin == freq_bin:
            rp_raw = float(rho_plus[freq_bin])
            rz_raw = float(rho_zero[freq_bin])
            rm_raw = float(rho_minus[freq_bin])
            return rp_raw, rz_raw, rm_raw, 1.0, 1.0, 1.0

        rp_raw = self._r_to_theta_component(rho_plus, freq_bin, mirror_bin)
        rz_raw = self._r_to_theta_component(rho_zero, freq_bin, mirror_bin)
        rm_raw = self._r_to_theta_component(rho_minus, freq_bin, mirror_bin)
        total_raw = rp_raw + rz_raw + rm_raw
        if total_raw <= eps:
            raise ValueError(
                f"θ-space population sum must be positive at freq_bin={freq_bin}; got {total_raw}."
            )

        return (
            rp_raw, rz_raw, rm_raw,
            rp_raw / total_raw, rz_raw / total_raw, rm_raw / total_raw,
        )

    @staticmethod
    def _r_space_theta_intensity_fractions(
        rho_plus: np.ndarray,
        rho_zero: np.ndarray,
        rho_minus: np.ndarray,
        freq_bin: int,
        mirror_bin: int,
    ) -> Tuple[float, float]:
        """
        Fractions of θ intensities ``I_+(θ)`` and ``I_-(θ)`` carried by ``R``.

        Uses AFP R-space coupling: ``I_+(θ) = I_+(R) + I_-(-R)``,
        ``I_-(θ) = I_-(R) + I_+(-R)``.
        """
        eps = np.finfo(float).eps
        R, M = int(freq_bin), int(mirror_bin)

        iplus_r = rho_plus[R] - rho_zero[R]
        iminus_m = rho_zero[R] - rho_minus[R]
        iplus_theta = iplus_r + iminus_m

        iminus_r = rho_zero[M] - rho_minus[M]
        iplus_m = rho_plus[M] - rho_zero[M]
        iminus_theta = iminus_r + iplus_m

        iplus_frac_r = iplus_r / iplus_theta if abs(iplus_theta) > eps else 0.5
        iminus_frac_r = iminus_r / iminus_theta if abs(iminus_theta) > eps else 0.5
        return float(iplus_frac_r), float(iminus_frac_r)

    def _theta_to_r_space_populations(
        self,
        rho_plus: np.ndarray,
        rho_zero: np.ndarray,
        rho_minus: np.ndarray,
        freq_bin: int,
        mirror_bin: int,
        rp_raw: float,
        rz_raw: float,
        rm_raw: float,
        iplus_frac_r: float,
        iminus_frac_r: float,
        tot_area: float,
    ) -> None:
        """
        Convert updated θ-space totals to per-frequency R-space populations.

        Reconstructs ``I_+(θ) = ρ_+(θ) - ρ_0(θ)`` and ``I_-(θ) = ρ_0(θ) - ρ_-(θ)``,
        splits each across ``R`` / ``-R`` using pre-burn intensity fractions, then
        inverts ``AFP.intensities_to_populations`` independently at each slice.
        """
        if tot_area <= np.finfo(float).eps:
            raise ValueError("tot_area must be positive for θ→R conversion.")

        R, M = int(freq_bin), int(mirror_bin)
        iplus_theta = rp_raw - rz_raw
        iminus_theta = rz_raw - rm_raw

        iplus_r = iplus_frac_r * iplus_theta
        iminus_m = (1.0 - iplus_frac_r) * iplus_theta
        iminus_r = iminus_frac_r * iminus_theta
        iplus_m = (1.0 - iminus_frac_r) * iminus_theta

        mu_r = (iplus_r + iminus_m) / tot_area
        rho_zero[R] = mu_r / 3.0
        rho_plus[R] = rho_zero[R] + iplus_r / tot_area
        rho_minus[R] = rho_zero[R] - iminus_m / tot_area

        mu_m = (iplus_m + iminus_r) / tot_area
        rho_zero[M] = mu_m / 3.0
        rho_plus[M] = rho_zero[M] + iplus_m / tot_area
        rho_minus[M] = rho_zero[M] - iminus_r / tot_area

    @staticmethod
    def _transfer_zero_to_minus(
        rho_zero: np.ndarray,
        rho_minus: np.ndarray,
        slice_idx: int,
        rate: float,
    ) -> float:
        """Cascade population from m=0 to m=-1 at one θ slice (``δθ₁``)."""
        if rate <= 0.0:
            return 0.0
        source = rho_zero[slice_idx]
        if source <= np.finfo(float).eps:
            return 0.0
        transfer = min(rate * source, source)
        rho_zero[slice_idx] -= transfer
        rho_minus[slice_idx] += transfer
        return float(transfer)

    @staticmethod
    def _transfer_plus_to_zero(
        rho_plus: np.ndarray,
        rho_zero: np.ndarray,
        slice_idx: int,
        rate: float,
    ) -> float:
        """Cascade population from m=+1 to m=0 at one θ slice (``δθ₂``)."""
        if rate <= 0.0:
            return 0.0
        source = rho_plus[slice_idx]
        if source <= np.finfo(float).eps:
            return 0.0
        transfer = min(rate * source, source)
        rho_plus[slice_idx] -= transfer
        rho_zero[slice_idx] += transfer
        return float(transfer)

    def _apply_theta_space_burn(
        self,
        rho_plus: np.ndarray,
        rho_zero: np.ndarray,
        rho_minus: np.ndarray,
        freq_bin: int,
        xi: float,
        apply_mirror: bool = True,
    ) -> Dict[str, float]:
        """
        Apply population swaps in ``δθ₂`` and ``δθ₁``, then convert back to R-space.
        """
        C = self.calibration_constant
        n_bins = len(rho_plus)
        mirror_bin = self._mirror_freq_index(freq_bin) if apply_mirror else None
        theta_2, theta_1, _ = self._theta_bin_indices_for_burn(freq_bin, n_bins)
        slice_idx = self._population_slice_index(freq_bin, n_bins)

        if mirror_bin is None or mirror_bin == freq_bin:
            transfer_primary = self._transfer_plus_to_zero(
                rho_plus, rho_zero, theta_2, C * xi,
            )
            transfer_mirror = 0.0
            if apply_mirror:
                transfer_mirror = self._transfer_zero_to_minus(
                    rho_zero, rho_minus, theta_1, 0.5 * C * xi,
                )
            return {
                "freq_bin": float(freq_bin),
                "slice_idx": float(slice_idx),
                "theta_2": float(theta_2),
                "theta_1": float(theta_1),
                "xi": float(xi),
                "mirror_bin": np.nan,
                "primary_transfer": float(transfer_primary),
                "mirror_transfer": float(transfer_mirror),
            }

        iplus_frac_r, iminus_frac_r = self._r_space_theta_intensity_fractions(
            rho_plus, rho_zero, rho_minus, freq_bin, mirror_bin,
        )
        iplus, iminus = AFP.populations_to_intensities(rho_plus, rho_zero, rho_minus)
        tot_area = float(np.sum(iplus + iminus))

        (
            rp_raw, rz_raw, rm_raw,
            rp_theta, rz_theta, rm_theta,
        ) = self._theta_space_populations(rho_plus, rho_zero, rho_minus, freq_bin)

        transfer_primary = min(C * xi * rp_raw, rp_raw) if rp_raw > 0.0 else 0.0
        rp_raw -= transfer_primary
        rz_raw += transfer_primary

        transfer_mirror = 0.0
        if apply_mirror and rz_raw > 0.0:
            transfer_mirror = min(0.5 * C * xi * rz_raw, rz_raw)
            rz_raw -= transfer_mirror
            rm_raw += transfer_mirror

        self._theta_to_r_space_populations(
            rho_plus, rho_zero, rho_minus,
            freq_bin, mirror_bin,
            rp_raw, rz_raw, rm_raw,
            iplus_frac_r, iminus_frac_r,
            tot_area,
        )

        return {
            "freq_bin": float(freq_bin),
            "slice_idx": float(slice_idx),
            "theta_2": float(theta_2),
            "theta_1": float(theta_1),
            "xi": float(xi),
            "rho_plus_theta": float(rp_theta),
            "rho_zero_theta": float(rz_theta),
            "rho_minus_theta": float(rm_theta),
            "mirror_bin": float(mirror_bin),
            "primary_transfer": float(transfer_primary),
            "mirror_transfer": float(transfer_mirror),
        }

    # def _apply_intensity_burn(
    #     self,
    #     iplus: np.ndarray,
    #     iminus: np.ndarray,
    #     rho_plus: np.ndarray,
    #     rho_zero: np.ndarray,
    #     rho_minus: np.ndarray,
    #     freq_bin: int,
    #     xi: float,
    #     apply_mirror: bool = True,
    # ) -> Dict[str, float]:
    #     """
    #     Apply ss-RF burn directly to ``I_+`` and ``I_-`` at ``freq_bin`` and its mirror.
    #
    #     Populations enter only through the burn-rate factors ``ρ_+`` and ``ρ_0``.
    #     """
    #     C = self.calibration_constant
    #     n_bins = len(iplus)
    #     slice_idx = self._population_slice_index(freq_bin, n_bins)
    #
    #     d_iplus_primary = -2.0 * C * xi * rho_plus[slice_idx]
    #     d_iminus_primary = -2.0 * C * xi * rho_zero[slice_idx]
    #     iplus[freq_bin] += d_iplus_primary
    #     iminus[freq_bin] += d_iminus_primary
    #
    #     mirror_bin = self._mirror_freq_index(freq_bin) if apply_mirror else None
    #     d_iplus_mirror = 0.0
    #     d_iminus_mirror = 0.0
    #     if mirror_bin is not None and mirror_bin != freq_bin:
    #         d_iplus_mirror = C * xi * rho_zero[slice_idx]
    #         d_iminus_mirror = C * xi * rho_plus[slice_idx]
    #         iplus[mirror_bin] += d_iplus_mirror
    #         iminus[mirror_bin] += d_iminus_mirror
    #
    #     return {
    #         "freq_bin": float(freq_bin),
    #         "slice_idx": float(slice_idx),
    #         "xi": float(xi),
    #         "d_iplus_primary": float(d_iplus_primary),
    #         "d_iminus_primary": float(d_iminus_primary),
    #         "mirror_bin": float(mirror_bin) if mirror_bin is not None else np.nan,
    #         "d_iplus_mirror": float(d_iplus_mirror),
    #         "d_iminus_mirror": float(d_iminus_mirror),
    #     }

    # ------------------------------------------------------------------
    # Differential-equation steps (growth then decay)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_growth_at_bin(
        iplus: np.ndarray,
        iminus: np.ndarray,
        ref_iplus: np.ndarray,
        ref_iminus: np.ndarray,
        bin_idx: int,
        growth_rate: float,
        dt: float,
    ) -> None:
        """Enhancement: relax the bin intensities toward the reference lineshape.

        Implements the differential growth step ``I → I + r·dt·(I_ref − I)``,
        the exponential saturation toward the target lineshape ("microwave on")
        applied before ss-RF decay at each swept bin.
        """
        if growth_rate <= 0.0:
            return
        scale = min(max(growth_rate * dt, 0.0), 1.0)
        iplus[bin_idx] += scale * (ref_iplus[bin_idx] - iplus[bin_idx])
        iminus[bin_idx] += scale * (ref_iminus[bin_idx] - iminus[bin_idx])

    def _apply_ssrf_decay(
        self,
        iplus: np.ndarray,
        iminus: np.ndarray,
        freq_bin: int,
        xi: float,
        apply_mirror: bool = True,
    ) -> Dict[str, float]:
        """Drive the two populations in ``freq_bin`` toward each other.

        The ss-RF equalizes ``ρ_+ ↔ ρ_0`` (the ``I_+`` transition) and
        ``ρ_0 ↔ ρ_-`` (the ``I_-`` transition) at a rate proportional to the
        product of the user power constant (carried in ``ξ``) and the difference
        between the two populations being driven together.

        Since ``I_+ = C(ρ_+ − ρ_0)`` and ``I_- = C(ρ_0 − ρ_-)``, each intensity
        equals that population difference (up to ``C``), so equalizing the
        populations by a fraction ``ξ`` removes a fraction ``min(2ξ, 1)`` of each
        intensity at the burn bin. The opposing absorption line at the mirror bin
        gains half of what is lost, the rates response ``A_gained = ½ A_lost``
        (Eq. 38). Clamping the fraction to ``[0, 1]`` keeps a single application
        from driving an intensity past full saturation (sign flip).
        """
        sat = min(max(2.0 * xi, 0.0), 1.0)

        iplus_burn = float(iplus[freq_bin])
        iminus_burn = float(iminus[freq_bin])
        loss_iplus = sat * iplus_burn
        loss_iminus = sat * iminus_burn
        iplus[freq_bin] = iplus_burn - loss_iplus
        iminus[freq_bin] = iminus_burn - loss_iminus

        mirror_bin = self._mirror_freq_index(freq_bin) if apply_mirror else None
        gain_iplus_mirror = 0.0
        gain_iminus_mirror = 0.0
        if mirror_bin is not None and mirror_bin != freq_bin:
            gain_iplus_mirror = 0.5 * loss_iminus
            gain_iminus_mirror = 0.5 * loss_iplus
            iplus[mirror_bin] += gain_iplus_mirror
            iminus[mirror_bin] += gain_iminus_mirror

        return {
            "freq_bin": float(freq_bin),
            "xi": float(xi),
            "saturation": float(sat),
            "loss_iplus": float(loss_iplus),
            "loss_iminus": float(loss_iminus),
            "mirror_bin": (
                float(mirror_bin)
                if mirror_bin is not None and mirror_bin != freq_bin
                else np.nan
            ),
            "gain_iplus_mirror": float(gain_iplus_mirror),
            "gain_iminus_mirror": float(gain_iminus_mirror),
        }

    def _step_bin(
        self,
        iplus: np.ndarray,
        iminus: np.ndarray,
        ref_iplus: np.ndarray,
        ref_iminus: np.ndarray,
        freq_bin: int,
        strength: float,
        dt: float,
        apply_mirror: bool = True,
    ) -> Dict[str, float]:
        """Growth then ss-RF decay at ``freq_bin`` (and its mirror), in that order.

        Mutates ``iplus`` / ``iminus`` in place and returns the decay info dict.
        """
        self._apply_growth_at_bin(
            iplus, iminus, ref_iplus, ref_iminus, freq_bin, self.growth_rate, dt,
        )

        mirror_bin = self._mirror_freq_index(freq_bin) if apply_mirror else None
        if mirror_bin is not None and mirror_bin != freq_bin:
            self._apply_growth_at_bin(
                iplus, iminus, ref_iplus, ref_iminus, mirror_bin, self.growth_rate, dt,
            )

        xi = self.power_constant * strength * dt
        burn_info = self._apply_ssrf_decay(
            iplus, iminus, freq_bin, xi, apply_mirror=apply_mirror,
        )
        burn_info["strength"] = float(strength)
        return burn_info

    def _populations_to_outputs(
        self,
        rho_plus: np.ndarray,
        rho_zero: np.ndarray, 
        rho_minus: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        iplus, iminus = AFP.populations_to_intensities(rho_plus, rho_zero, rho_minus)
        ps = iplus + iminus
        return ps, iplus, iminus

    def _load_state_from_intensities(
        self,
        iplus: np.ndarray,
        iminus: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return AFP.intensities_to_populations(
            np.asarray(iplus, dtype=float),
            np.asarray(iminus, dtype=float),
        )

    # def _ensure_spin_temperature_manifold(
    #     self,
    #     p_min: float = -0.95,
    #     p_max: float = 0.95,
    #     p_step: float = 0.001,
    # ) -> None:
    #     """Build Boltzmann-consistent manifold used for per-bin projection."""
    #     if self._st_ps_grid is not None:
    #         return

    #     p_grid = np.arange(p_min, p_max + 0.5 * p_step, p_step, dtype=float)
    #     n_p = len(p_grid)
    #     n_bins = len(self.f)
    #     ps_grid = np.empty((n_p, n_bins), dtype=float)
    #     iplus_grid = np.empty((n_p, n_bins), dtype=float)
    #     iminus_grid = np.empty((n_p, n_bins), dtype=float)

    #     for i, p in enumerate(p_grid):
    #         ps, iplus, iminus = GenerateVectorLineshape(float(p), self.f)
    #         ps_grid[i] = np.asarray(ps, dtype=float)
    #         iplus_grid[i] = np.asarray(iplus, dtype=float)
    #         iminus_grid[i] = np.asarray(iminus, dtype=float)

    #     self._st_p_grid = p_grid
    #     self._st_ps_grid = ps_grid
    #     self._st_iplus_grid = iplus_grid
    #     self._st_iminus_grid = iminus_grid

    # def enforce_spin_temperature_consistency(
    #     self,
    #     iplus: np.ndarray,
    #     iminus: np.ndarray,
    #     indices: Optional[np.ndarray] = None,
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Enforce local Boltzmann consistency in affected bins.

    #     For each selected bin, project (I+, I-) onto the nearest point of the
    #     precomputed spin-temperature manifold, using linear interpolation to avoid
    #     quantization artifacts.
    #     """
    #     self._ensure_spin_temperature_manifold()
    #     iplus_out = np.asarray(iplus, dtype=float).copy()
    #     iminus_out = np.asarray(iminus, dtype=float).copy()
    #     ps_out = iplus_out + iminus_out

    #     if indices is None:
    #         work_idx = np.arange(len(ps_out), dtype=int)
    #     else:
    #         work_idx = np.unique(np.asarray(indices, dtype=int))
    #         work_idx = work_idx[(work_idx >= 0) & (work_idx < len(ps_out))]

    #     eps = np.finfo(float).eps
    #     for idx in work_idx:
    #         target_ps = float(ps_out[idx])
    #         ps_column = self._st_ps_grid[:, idx]

    #         if target_ps > eps:
    #             mask = ps_column > 0.0
    #         elif target_ps < -eps:
    #             mask = ps_column < 0.0
    #         else:
    #             mask = np.ones_like(ps_column, dtype=bool)

    #         candidate_ps = ps_column[mask]
    #         if candidate_ps.size == 0:
    #             candidate_ps = ps_column
    #             cand_indices = np.arange(len(ps_column), dtype=int)
    #         else:
    #             cand_indices = np.flatnonzero(mask)

    #         sort_order = np.argsort(candidate_ps)
    #         ps_sorted = candidate_ps[sort_order]
    #         idx_sorted = cand_indices[sort_order]
    #         ip_sorted = self._st_iplus_grid[idx_sorted, idx]
    #         im_sorted = self._st_iminus_grid[idx_sorted, idx]

    #         if target_ps <= ps_sorted[0]:
    #             iplus_out[idx] = ip_sorted[0]
    #             iminus_out[idx] = im_sorted[0]
    #         elif target_ps >= ps_sorted[-1]:
    #             iplus_out[idx] = ip_sorted[-1]
    #             iminus_out[idx] = im_sorted[-1]
    #         else:
    #             hi = int(np.searchsorted(ps_sorted, target_ps, side="right"))
    #             lo = hi - 1
    #             p0 = float(ps_sorted[lo])
    #             p1 = float(ps_sorted[hi])
    #             if abs(p1 - p0) <= eps:
    #                 w = 0.0
    #             else:
    #                 w = (target_ps - p0) / (p1 - p0)
    #             iplus_out[idx] = (1.0 - w) * ip_sorted[lo] + w * ip_sorted[hi]
    #             iminus_out[idx] = (1.0 - w) * im_sorted[lo] + w * im_sorted[hi]

    #     ps_out = iplus_out + iminus_out
    #     return ps_out, iplus_out, iminus_out

    @staticmethod
    def _resolve_bin_range(
        n_bins: int,
        bin_range: Optional[BinRange] = None,
        profile_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        if bin_range is None:
            if profile_mask is not None and np.any(profile_mask):
                active = np.where(profile_mask)[0]
                return int(active[0]), int(active[-1])
            return 0, n_bins - 1

        if isinstance(bin_range, (list, np.ndarray)):
            indices = [int(b) for b in bin_range]
            if not indices:
                raise ValueError("bin_range list must not be empty.")
            return min(indices), max(indices)

        low, high = int(bin_range[0]), int(bin_range[1])
        if low < 0 or high >= n_bins or low > high:
            raise ValueError(
                f"bin_range must satisfy 0 <= low <= high < n_bins ({n_bins}); got ({low}, {high})"
            )
        return low, high


    def apply_bin_burn(
        self,
        ps: np.ndarray,
        iplus: np.ndarray,
        iminus: np.ndarray,
        bin_idx: int,
        amp: float,
        dt: float = 1.0,
        return_burn_info: bool = False,
    ) -> Tuple:
        """
        Apply ss-RF at a single frequency bin (no profile shape).

        Input arrays are copied; callers keep their originals unchanged.
        Mirrored decay is applied at the symmetric frequency when it exists.
        """
        iplus_out = np.asarray(iplus, dtype=float).copy()
        iminus_out = np.asarray(iminus, dtype=float).copy()
        ref_iplus = iplus_out.copy()
        ref_iminus = iminus_out.copy()

        step_info = self._step_bin(
            iplus_out, iminus_out,
            ref_iplus, ref_iminus,
            int(bin_idx),
            abs(float(amp)),
            dt,
            apply_mirror=True,
        )
        ps_out = iplus_out + iminus_out
        rho_plus, rho_zero, rho_minus = self._load_state_from_intensities(
            iplus_out, iminus_out,
        )

        if return_burn_info:
            burn_info = {
                "burn_indices": np.array([int(bin_idx)], dtype=int),
                "mirror_indices": (
                    np.array([int(step_info["mirror_bin"])], dtype=int)
                    if np.isfinite(step_info["mirror_bin"])
                    else None
                ),
                "strength": abs(float(amp)),
                "step_info": step_info,
                "rho_plus": rho_plus,
                "rho_zero": rho_zero,
                "rho_minus": rho_minus,
            }
            return ps_out, iplus_out, iminus_out, rho_plus, rho_zero, rho_minus, burn_info

        return ps_out, iplus_out, iminus_out, rho_plus, rho_zero, rho_minus

    def apply_ssRF(
        self,
        ps: np.ndarray,
        iplus: np.ndarray,
        iminus: np.ndarray,
        bin_range: Optional[BinRange] = None,
        n_cycles: int = 1,
        n_steps: Optional[int] = None,
        sweep_rate: float = 1.0,
        dt: float = 1.0,
        profile_type: str = "voigt",
        burn_threshold: Optional[float] = None,
        return_burn_info: bool = False,
        record_history: bool = False,
    ) -> Tuple:
        """
        Simulate ss-RF with a ping-pong sweeping counter over ``bin_range``.

        At each visited bin: growth (enhancement) then ss-RF decay, in that order.
        Profile strength at each bin comes from the discretized Voigt/Gaussian
        shape centred at ``self.x0``.

        Input arrays are copied; callers keep their originals unchanged.

        Returns
        -------
        ps, iplus, iminus, rho_plus, rho_zero, rho_minus
        and optionally ``burn_info`` / population ``history``.
        """
        iplus_out = np.asarray(iplus, dtype=float).copy()
        iminus_out = np.asarray(iminus, dtype=float).copy()
        n_bins = len(iplus_out)

        profile = self._discretize_profile(profile_type=profile_type)
        threshold = self.amp * 1e-3 if burn_threshold is None else float(burn_threshold)
        profile_mask = profile > threshold

        low, high = self._resolve_bin_range(n_bins, bin_range, profile_mask)

        ref_iplus = iplus_out.copy()
        ref_iminus = iminus_out.copy()

        counter = SweepCounter(low, high, sweep_rate=sweep_rate)
        cycle_len = len(counter.cycle)
        total_steps = (
            int(n_steps)
            if n_steps is not None
            else max(1, int(n_cycles) * cycle_len)
        )

        history: List[Dict] = []
        visited_bins: List[int] = []

        for _ in range(total_steps):
            bins_this_step = counter.advance()
            for freq_bin in bins_this_step:
                strength = float(profile[freq_bin])
                step_info = self._step_bin(
                    iplus_out, iminus_out,
                    ref_iplus, ref_iminus,
                    freq_bin, strength, dt,
                )
                visited_bins.append(freq_bin)
                if record_history:
                    history.append({
                        **step_info,
                        "iplus": iplus_out.copy(),
                        "iminus": iminus_out.copy(),
                    })

        ps_out = iplus_out + iminus_out
        rho_plus, rho_zero, rho_minus = self._load_state_from_intensities(
            iplus_out, iminus_out,
        )

        if return_burn_info:
            burn_info = {
                "bin_range": (low, high),
                "n_steps": total_steps,
                "sweep_rate": sweep_rate,
                "visited_bins": np.asarray(visited_bins, dtype=int),
                "profile": profile,
                "rho_plus": rho_plus,
                "rho_zero": rho_zero,
                "rho_minus": rho_minus,
                "history": history if record_history else None,
            }
            return (
                ps_out, iplus_out, iminus_out,
                rho_plus, rho_zero, rho_minus,
                burn_info,
            )

        if record_history:
            return (
                ps_out, iplus_out, iminus_out,
                rho_plus, rho_zero, rho_minus,
                history,
            )

        return ps_out, iplus_out, iminus_out, rho_plus, rho_zero, rho_minus
