"""Physical Voigt RF profiles and population-dependent spin diffusion (voigt_burn port)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .voigt_physical import (
    approximate_voigt_fwhm,
    bin_averaged_voigt,
    implementation_name as voigt_implementation_name,
)

PLUS, ZERO, MINUS = 0, 1, 2


class VoigtBurnPhysicsMixin:
    """RF profile and spin-diffusion methods ported from spin1_ssrf_realtime_voigt_burn."""

    def _uses_legacy_discrete_rf(self) -> bool:
        if getattr(self, "_rf_profile_frozen", False):
            return True
        return self.params.ssrf_subset_indices is not None

    def _uses_physical_voigt_rf(self) -> bool:
        if self._uses_legacy_discrete_rf():
            return False
        p = self.params
        if bool(getattr(p, "use_physical_voigt_rf", False)):
            return True
        return float(p.rf_gaussian_fwhm_R) > 0.0 or float(p.rf_lorentzian_fwhm_R) > 0.0

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
        """Bin-averaged RF rate profile versus physical R."""
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
        """Profile weights at each packet's + and - transition frequencies."""
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

    def _effective_theta(self) -> np.ndarray:
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
        key = self._diffusion_key()
        if self._diffusion_kernel_key == key:
            return

        N = len(self.Rplus)
        theta = self._effective_theta()
        width = max(float(self.params.zq_width_R), 1e-12)
        cutoff = max(float(self.params.kernel_cutoff_widths), 1.0)
        max_delta = cutoff * width

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
            if np.isfinite(Ip) and abs(Ip) > 1e-15:
                out["lambda_Iplus"] = float(slope / Ip)
        if km is not None:
            slope = scale_obs * (net[km, ZERO] - net[km, MINUS])
            out["dIminus_diff_dt"] = float(slope)
            Im = self.local_intensities(R)["Iminus"]
            if np.isfinite(Im) and abs(Im) > 1e-15:
                out["lambda_Iminus"] = float(slope / Im)
        return out

    def _physical_effective_local_rates(self, R: Optional[float] = None) -> Dict[str, float]:
        if R is None:
            R = self.params.rf_burn_R
        cap = self.local_capacity_factors(R)
        p = self.params
        kp, km = self.branch_indices(R)
        v_plus, v_minus = self.rf_profile_arrays(R)

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
                if np.isfinite(wp) and np.isfinite(opposite_plus_packet)
                else np.nan
            ),
            "gamma_rf_opposite_on_Iminus_packet": (
                float(p.gamma_rf * wm * opposite_minus_packet)
                if np.isfinite(wm) and np.isfinite(opposite_minus_packet)
                else np.nan
            ),
            "dnp_Iplus_R": float(p.dnp_rate * wp) if np.isfinite(wp) else np.nan,
            "dnp_Iminus_R": float(p.dnp_rate * wm) if np.isfinite(wm) else np.nan,
            "same_plus0_Iplus_R": float(p.d_same_plus0 * wp) if np.isfinite(wp) else np.nan,
            "same_0minus_Iminus_R": float(p.d_same_0minus * wm) if np.isfinite(wm) else np.nan,
        }
