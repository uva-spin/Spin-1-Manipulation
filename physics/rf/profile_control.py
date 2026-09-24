"""Ideal-bin RF profile codec and one-shot burn environment (no torch).

Maps a lineshape to a simultaneous ``PulseProgram`` (per-bin rates U_j on
[0, T)) and plays it through ``IdealBinModel``. Candidate bins follow the
ssRF-beta union of negative signed tensor density and positive initial RF-gain.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .ideal_model import IdealBinModel, IdealBinParams
from .pulse_program import BinPulse, PulseProgram, RFProfile

DEFAULT_N_BINS = 249
DEFAULT_R_MIN = -3.0
DEFAULT_R_MAX = 3.0
DEFAULT_LINE_GAMMA = 0.05
DEFAULT_LINE_ASYM = 0.04
DEFAULT_P0 = 0.45
U_MAX = 20.0
T_MIN = 0.01
T_MAX = 2.0
CANDIDATE_THRESHOLD = 1e-10


def candidate_mask(model, *, mode="union", threshold=CANDIDATE_THRESHOLD):
    """Boolean mask of bins allowed to receive RF (ssRF-beta candidate rule)."""
    n = np.asarray(model.n)
    w = np.asarray(model.capacity_rate_weights())
    dp = n[:, 0] - n[:, 1]
    dm = (n[:, 1] - n[:, 2])[::-1]
    signed_tensor = (dp - dm) / float(model.dR)
    gain = 3.0 * (w[::-1] * dm - w * dp)
    t = float(threshold)
    deficit = signed_tensor < -t * max(float(np.max(np.abs(signed_tensor))), 1e-30)
    favorable = gain > t * max(float(np.max(np.abs(gain))), 1e-30)
    masks = {
        "tensor_deficit": deficit,
        "rf_gain": favorable,
        "union": deficit | favorable,
        "all": np.ones(len(model.Rplus), dtype=bool),
    }
    if mode not in masks:
        raise ValueError(f"unknown candidate mode {mode!r}")
    return np.asarray(masks[mode], dtype=bool)


def q_shaped_manual_rates(iplus, iminus, gamma_rf=2.0):
    """Legacy Q-shaped envelope (the previous ``set_rf_profile`` formula)."""
    q = np.asarray(iplus) - np.asarray(iminus)
    q_min = float(np.min(q))
    if q_min >= 0.0:
        return np.zeros_like(q, dtype=float)
    return float(gamma_rf) * np.clip(q / q_min, 0.0, 1.0)


def rates_and_duration_to_program(
    rates,
    duration,
    *,
    r_min=DEFAULT_R_MIN,
    r_max=DEFAULT_R_MAX,
    mask=None,
    name="RL profile",
):
    """Build a simultaneous-start ``PulseProgram`` from a per-bin rate vector."""
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1:
        raise ValueError("rates must be 1-D")
    n_bins = int(rates.size)
    if mask is not None:
        rates = rates * np.asarray(mask, dtype=float)
    duration = float(duration)
    pulses = [
        BinPulse(int(i), float(u), 0.0, duration)
        for i, u in enumerate(rates)
        if u > 0.0 and duration > 0.0
    ]
    program = PulseProgram(
        n_bins=n_bins,
        r_min=float(r_min),
        r_max=float(r_max),
        gain=1.0,
        profiles=[RFProfile(name=name, pulses=pulses)],
    )
    program.validate()
    return program


def decode_unit_action(action, *, u_max=U_MAX, t_min=T_MIN, t_max=T_MAX):
    """Map unit-interval action ``[U_norm..., T_norm]`` to rates and duration."""
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.size < 2:
        raise ValueError("action must be per-bin rates plus duration")
    u_norm = np.clip(action[:-1], 0.0, 1.0)
    t_norm = float(np.clip(action[-1], 0.0, 1.0))
    rates = u_norm * float(u_max)
    duration = float(t_min) + t_norm * (float(t_max) - float(t_min))
    return rates, duration


def observation_from_model(model):
    """Concatenate I+, I-, P, and Q_total."""
    ip, im, _ = model.physical_intensities()
    pol = model.polarizations()
    return np.concatenate(
        [
            np.asarray(ip, dtype=float),
            np.asarray(im, dtype=float),
            np.array([pol["P"], pol["Q"]], dtype=float),
        ]
    )


def play_program(model, program, *, dnp_on=False):
    """Install and run ``program`` from the current state. Returns ΔQ, Q, P."""
    q0 = float(model.polarizations()["Q"])
    compiled = program.compile()
    if compiled.end_time <= 0.0:
        return 0.0, q0, float(model.polarizations()["P"]), True
    model.set_program(program)
    model.start_program(turn_rf_on=True)
    end = float(model._event_times_absolute[-1])
    dt = float(model.params.dt)
    n_steps = max(1, int(math.ceil((end - model.t) / dt)))
    model.step(n_steps, rf_on=True, dnp_on=bool(dnp_on))
    pol = model.polarizations()
    q1 = float(pol["Q"])
    return q1 - q0, q1, float(pol["P"]), False


def make_ideal_model(polarization=DEFAULT_P0, *, n_bins=DEFAULT_N_BINS, r_min=DEFAULT_R_MIN, r_max=DEFAULT_R_MAX):
    params = IdealBinParams(
        n_bins=int(n_bins),
        r_min=float(r_min),
        r_max=float(r_max),
        p0=float(polarization),
        q0=None,
        line_gamma=DEFAULT_LINE_GAMMA,
        line_asym=DEFAULT_LINE_ASYM,
        rf_enabled=False,
        dnp_enabled=False,
        diffusion_enabled=True,
        relax_enabled=False,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
    )
    return IdealBinModel(params)


class RFProfileEnv:
    """One-step environment: lineshape in, PulseProgram out, reward = ΔQ."""

    def __init__(
        self,
        *,
        n_bins=DEFAULT_N_BINS,
        r_min=DEFAULT_R_MIN,
        r_max=DEFAULT_R_MAX,
        u_max=U_MAX,
        t_min=T_MIN,
        t_max=T_MAX,
        candidate_mode="union",
        polarization=DEFAULT_P0,
    ):
        self.n_bins = int(n_bins)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.u_max = float(u_max)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.candidate_mode = candidate_mode
        self._default_polarization = float(polarization)
        self.model = None
        self.mask = None
        self.last_program = None
        self.reset(polarization)

    @property
    def observation_dim(self):
        return 2 * self.n_bins + 2

    @property
    def action_dim(self):
        return self.n_bins + 1

    def reset(self, polarization: Optional[float] = None):
        p0 = self._default_polarization if polarization is None else float(polarization)
        self.model = make_ideal_model(
            p0, n_bins=self.n_bins, r_min=self.r_min, r_max=self.r_max
        )
        self.mask = candidate_mask(self.model, mode=self.candidate_mode)
        self.last_program = None
        return self.observation()

    def observation(self):
        return observation_from_model(self.model)

    def action_to_program(self, action, *, name="RL profile"):
        rates, duration = decode_unit_action(
            action, u_max=self.u_max, t_min=self.t_min, t_max=self.t_max
        )
        if rates.size != self.n_bins:
            raise ValueError(f"expected {self.n_bins} rates, got {rates.size}")
        return rates_and_duration_to_program(
            rates,
            duration,
            r_min=self.r_min,
            r_max=self.r_max,
            mask=self.mask,
            name=name,
        )

    def step(self, action):
        program = self.action_to_program(action)
        self.last_program = program
        delta_q, q_final, p_final, empty = play_program(self.model, program)
        reward = 0.0 if empty else float(delta_q)
        info = {
            "empty": empty,
            "Q": q_final,
            "P": p_final,
            "delta_Q": 0.0 if empty else float(delta_q),
            "n_active": int(np.count_nonzero(self.mask)),
            "duration": float(program.profiles[0].pulses[0].duration) if program.profiles and program.profiles[0].pulses else 0.0,
        }
        return self.observation(), reward, True, info


class ProfileBuildEnv:
    """Sequential discrete (bin, U) and T choices that assemble one simultaneous PulseProgram.

    Each assignment is scored by replaying the full profile from the frozen Boltzmann
    state through ``IdealBinModel``. Reward is Δ population Q (ssRF-beta ``polarizations``).
    """

    def __init__(
        self,
        *,
        n_bins=DEFAULT_N_BINS,
        r_min=DEFAULT_R_MIN,
        r_max=DEFAULT_R_MAX,
        u_max=U_MAX,
        n_u_bins=11,
        t_min=T_MIN,
        t_max=T_MAX,
        n_t_bins=7,
        duration=1.0,
        max_steps=200,
        candidate_mode="union",
        polarization=DEFAULT_P0,
        enforce_full_spectrum=False,
    ):
        self.n_bins = int(n_bins)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.u_max = float(u_max)
        self.n_u_bins = max(1, int(n_u_bins))
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.n_t_bins = max(1, int(n_t_bins))
        self.default_duration = float(np.clip(duration, self.t_min, self.t_max))
        self.max_steps = int(max_steps)
        self.candidate_mode = candidate_mode
        self.enforce_full_spectrum = bool(enforce_full_spectrum)
        self._default_polarization = float(polarization)
        if self.n_u_bins == 1:
            self.u_values = np.array([0.0], dtype=float)
        else:
            self.u_values = np.linspace(0.0, self.u_max, self.n_u_bins)
        if self.n_t_bins == 1:
            self.duration_values = np.array([self.default_duration], dtype=float)
        else:
            self.duration_values = np.geomspace(self.t_min, self.t_max, self.n_t_bins)
        self.model = None
        self.mask = None
        self.rates = None
        self.duration = self.default_duration
        self.last_program = None
        self._n0 = None
        self._used = None
        self._step = 0
        self._scan_idx = 0
        self._p0 = 0.0
        self._q0 = 0.0
        self._q = 0.0
        self._p = 0.0
        self.reset(polarization)

    @property
    def f(self):
        return np.asarray(self.model.Rplus)

    @property
    def n_duration_actions(self):
        return int(self.duration_values.size)

    @property
    def n_rate_actions(self):
        if self.enforce_full_spectrum:
            return int(self.u_values.size)
        return int(self.n_bins * self.u_values.size)

    @property
    def n_actions(self):
        return self.n_duration_actions + self.n_rate_actions

    @property
    def observation_dim(self):
        return 3 * self.n_bins + 4

    @property
    def current_q(self):
        return float(self._q)

    @property
    def initial_q(self):
        return float(self._q0)

    @property
    def current_p(self):
        return float(self._p)

    def reset(self, polarization: Optional[float] = None):
        p0 = self._default_polarization if polarization is None else float(polarization)
        self.model = make_ideal_model(
            p0, n_bins=self.n_bins, r_min=self.r_min, r_max=self.r_max
        )
        self._n0 = np.asarray(self.model.n).copy()
        self.mask = candidate_mask(self.model, mode=self.candidate_mode)
        self.rates = np.zeros(self.n_bins, dtype=float)
        self.duration = float(self.default_duration)
        self.last_program = None
        self._used = set()
        self._step = 0
        self._scan_idx = 0
        pol = self.model.polarizations()
        self._p0 = float(pol["P"])
        self._q0 = float(pol["Q"])
        self._p = self._p0
        self._q = self._q0
        return self.observation()

    def observation(self):
        ip, im, _ = self.model.physical_intensities()
        u_norm = self.rates / max(self.u_max, 1e-30)
        t_norm = (self.duration - self.t_min) / max(self.t_max - self.t_min, 1e-30)
        step_frac = self._step / max(self.max_steps, 1)
        return np.concatenate(
            [
                np.asarray(ip, dtype=np.float32),
                np.asarray(im, dtype=np.float32),
                u_norm.astype(np.float32),
                np.array([self._p, self._q, t_norm, step_frac], dtype=np.float32),
            ]
        )

    def decode_action(self, action):
        action = int(action)
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"action {action} out of range for n_actions={self.n_actions}")
        if action < self.n_duration_actions:
            return {"kind": "duration", "t_idx": action, "duration": float(self.duration_values[action])}
        rate_action = action - self.n_duration_actions
        if self.enforce_full_spectrum:
            bin_idx = int(self._scan_idx)
            u_idx = rate_action
        else:
            n_u = int(self.u_values.size)
            bin_idx = rate_action // n_u
            u_idx = rate_action % n_u
        bin_idx = int(np.clip(bin_idx, 0, self.n_bins - 1))
        u_idx = int(np.clip(u_idx, 0, self.u_values.size - 1))
        return {
            "kind": "rate",
            "bin_idx": bin_idx,
            "u_idx": u_idx,
            "rate": float(self.u_values[u_idx]),
            "R": float(self.f[bin_idx]),
        }

    def valid_action_mask(self):
        mask = np.zeros(self.n_actions, dtype=bool)
        mask[: self.n_duration_actions] = True
        if self._step >= self.max_steps:
            return mask
        if self.enforce_full_spectrum:
            if self._scan_idx >= self.n_bins:
                return mask
            start = self.n_duration_actions
            mask[start] = True
            if self.mask[self._scan_idx]:
                mask[start : start + self.u_values.size] = True
            return mask
        n_u = int(self.u_values.size)
        for bin_idx in range(self.n_bins):
            if bin_idx in self._used:
                continue
            start = self.n_duration_actions + bin_idx * n_u
            mask[start] = True
            if self.mask[bin_idx]:
                mask[start : start + n_u] = True
        return mask

    def _restore_initial(self):
        self.model.n = self._n0.copy()
        self.model.t = 0.0
        self.model._epoch = None
        self.model._event_times_absolute = None
        self.model._command_override = None
        self.model._stopped = False
        self.model.set_rf_enabled(False)

    def _evaluate_profile(self):
        self._restore_initial()
        program = rates_and_duration_to_program(
            self.rates,
            self.duration,
            r_min=self.r_min,
            r_max=self.r_max,
            mask=self.mask,
            name="DQN profile",
        )
        self.last_program = program
        delta_q, q_final, p_final, empty = play_program(self.model, program)
        if empty:
            q_final = float(self._q0)
            p_final = float(self._p0)
            delta_q = 0.0
        return float(delta_q), float(q_final), float(p_final), bool(empty), program

    def step(self, action):
        parsed = self.decode_action(action)
        q_before = float(self._q)
        info = {
            "kind": parsed["kind"],
            "duration": float(self.duration),
            "q": self._q,
            "q_gain": self._q - self._q0,
            "P": self._p,
            "skipped": False,
            "empty": False,
            "bin_idx": -1,
            "f": 0.0,
            "rate": 0.0,
        }
        if parsed["kind"] == "duration":
            self.duration = float(parsed["duration"])
            _, q_final, p_final, empty, _ = self._evaluate_profile()
            reward = float(q_final - q_before)
            self._q = q_final
            self._p = p_final
            self._step += 1
            done = self._step >= self.max_steps
            if self.enforce_full_spectrum:
                done = done or self._scan_idx >= self.n_bins
            info.update(
                {
                    "duration": self.duration,
                    "q": self._q,
                    "q_gain": self._q - self._q0,
                    "P": self._p,
                    "empty": empty,
                    "t_idx": parsed["t_idx"],
                }
            )
            return self.observation(), reward, done, info

        bin_idx = int(parsed["bin_idx"])
        rate = float(parsed["rate"])
        info.update({"bin_idx": bin_idx, "f": parsed["R"], "rate": rate})
        if not self.enforce_full_spectrum and bin_idx in self._used:
            self._step += 1
            done = self._step >= self.max_steps
            return self.observation(), -1e-6, done, {**info, "repeated_bin": True}
        if not self.enforce_full_spectrum:
            self._used.add(bin_idx)
        else:
            self._scan_idx += 1
        if rate <= 0.0 or not self.mask[bin_idx]:
            self._step += 1
            done = self._step >= self.max_steps or (
                self.enforce_full_spectrum and self._scan_idx >= self.n_bins
            )
            info["skipped"] = True
            return self.observation(), 0.0, done, info
        self.rates[bin_idx] = rate
        _, q_final, p_final, empty, _ = self._evaluate_profile()
        reward = float(q_final - q_before)
        self._q = q_final
        self._p = p_final
        self._step += 1
        done = self._step >= self.max_steps or (
            self.enforce_full_spectrum and self._scan_idx >= self.n_bins
        )
        info.update(
            {
                "q": self._q,
                "q_gain": self._q - self._q0,
                "P": self._p,
                "empty": empty,
                "duration": self.duration,
            }
        )
        return self.observation(), reward, done, info
