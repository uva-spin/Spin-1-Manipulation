"""Incoherent spin-1 RF saturation with optional empirical packet response (ssRF-beta).

U_j denotes power in carrier j (proportional to B1_j**2), not amplitude.
For exponentially decaying transverse coherence, eliminating that coherence
gives W_ij = strength * Gamma / (Gamma**2 + (R_i-R_j)**2) * U_j.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft


class RFRateOperator:
    def __init__(self, grid, weights, *, strength=None, hwhm=None, packet_response=None):
        self.weights = np.asarray(weights, dtype=float).copy()
        self.N = len(grid)
        self.spectral = strength is not None
        self.dense = False
        self.packet_response = (
            np.ones(self.N)
            if packet_response is None
            else np.array(packet_response, dtype=float, copy=True)
        )
        if (
            self.packet_response.shape != (self.N,)
            or not np.isfinite(self.packet_response).all()
            or np.any(self.packet_response <= 0)
        ):
            raise ValueError(
                "Packet RF response must be positive and finite on the packet grid"
            )
        if self.spectral:
            if not np.isfinite(strength) or strength < 0:
                raise ValueError("rf_spectral_strength must be finite and nonnegative")
            if np.ndim(hwhm) == 0 and (not np.isfinite(hwhm) or hwhm <= 0):
                raise ValueError("rf_homogeneous_hwhm_R must be finite and positive")
            hwhm_array = np.asarray(hwhm, dtype=float)
            self.dense = hwhm_array.ndim > 0
            if self.dense:
                if (
                    hwhm_array.shape != (self.N,)
                    or not np.all(np.isfinite(hwhm_array))
                    or np.any(hwhm_array <= 0)
                ):
                    raise ValueError(
                        "Per-packet RF HWHM must be positive and finite on the packet grid"
                    )
                grid_r = np.asarray(grid, dtype=float)
                width = hwhm_array[:, None]
                self.Kplus = strength * width / (
                    width * width + (grid_r[:, None] - grid_r[None, :]) ** 2
                )
                self.Kminus = strength * width / (
                    width * width + (-grid_r[:, None] - grid_r[None, :]) ** 2
                )
            else:
                delta = np.asarray(grid) - grid[0]
                self.column = strength * hwhm / (hwhm * hwhm + delta * delta)
                self.L = next_fast_len(2 * self.N - 1)
                embed = np.zeros(self.L)
                embed[: self.N] = self.column
                embed[-self.N + 1 :] = self.column[1:][::-1]
                self.kernel_fft = rfft(embed)

    def _convolve(self, value):
        return irfft(self.kernel_fft * rfft(value, n=self.L), n=self.L)[: self.N]

    def rates(self, command):
        if self.spectral:
            if getattr(self, "_last_command", None) is not None and np.array_equal(
                command, self._last_command
            ):
                return self._last_rates
            if self.dense:
                plus = self.packet_response * (self.Kplus @ command)
                minus = self.packet_response * (self.Kminus @ command)
            else:
                field = self._convolve(command)
                plus = self.packet_response * field
                minus = self.packet_response * field[::-1]
            plus.setflags(write=False)
            minus.setflags(write=False)
            self._last_command = np.array(command, copy=True)
            self._last_rates = (plus, minus)
            return self._last_rates
        return self.weights * command, self.weights * command[::-1]

    def transpose(self, plus_gradient, minus_gradient):
        if self.spectral:
            if self.dense:
                return self.Kplus.T @ (self.packet_response * plus_gradient) + self.Kminus.T @ (
                    self.packet_response * minus_gradient
                )
            return self._convolve(
                self.packet_response * plus_gradient
                + (self.packet_response * minus_gradient)[::-1]
            )
        return self.weights * plus_gradient + (self.weights * minus_gradient)[::-1]