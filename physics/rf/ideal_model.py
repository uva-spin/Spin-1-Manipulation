"""Ideal per-physical-bin RF control on the unchanged working population model.

Only the RF input and event-aware time stepping are extended. The Pake state,
readout, capacity weights, spin-diffusion currents, DNP and T1 are inherited
without changes from model.py. No Voigt function is evaluated on this path.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Tuple
import math
import numpy as np

from .model import Spin1Model, Spin1Params
from .pulse_program import BinPulse, RFProfile, PulseProgram


@dataclass
class IdealBinParams(Spin1Params):
    # Retained base fields are zero to make the legacy comparator unambiguous.
    rf_gaussian_fwhm_R: float = 0.0
    rf_lorentzian_fwhm_R: float = 0.0


class IdealBinModel(Spin1Model):
    """One command per physical bin, shared by the two overlapping branches.

    Commands are effective base equalization rates (not calibrated watts).
    The inherited optional capacity weight multiplies these rates exactly once,
    as in the working baseline. Set capacity_rate_power=0 for uniform per-spin
    coupling; it also changes the inherited DNP weighting, as before.
    """
    def __init__(self, params: Optional[Spin1Params] = None, program: Optional[PulseProgram] = None):
        super().__init__(params or IdealBinParams())
        if program is None:
            k = int(np.argmin(abs(self.Rplus-self.params.rf_burn_R)))
            program = PulseProgram(len(self.Rplus), float(self.Rplus[0]), float(self.Rplus[-1]),
                                   profiles=[RFProfile("Single-bin example", pulses=[BinPulse(k, 2.0, 0.0, 1.0)])])
        self.set_program(program)

    def _uses_legacy_discrete_rf(self):
        return False

    def _uses_physical_voigt_rf(self):
        # Parent derivative must use ``_rf_population_term`` so overridden
        # ``rf_rate_fields`` (pulse-program U_j) are applied. Voigt widths stay 0.
        return True

    def reset(self) -> None:
        saved = getattr(self, "program", None)
        super().reset()
        if not np.allclose(self.Rplus, -self.Rplus[::-1], rtol=0, atol=1e-12):
            raise ValueError("Ideal profiles require a symmetric physical-R grid for exact mirror indexing")
        self._epoch = None
        self._event_times_absolute = None
        self._command_override = None
        self._stopped = False
        self.delivered_exposure = np.zeros(len(self.Rplus))
        self.total_delivered_exposure = np.zeros(len(self.Rplus))
        self.last_substeps = 0
        if saved is not None:
            saved.validate_grid(self.Rplus)
            self._compiled = saved.compile()

    def set_program(self, program: PulseProgram) -> None:
        """Install a validated copy atomically. Stop RF; never reset populations."""
        candidate = program.clone()
        candidate.validate_grid(self.Rplus)
        compiled = candidate.compile()
        self.program = candidate
        self._compiled = compiled
        self._epoch = None
        self._event_times_absolute = None
        self._command_override = None
        self._stopped = False
        self.delivered_exposure = np.zeros(len(self.Rplus))
        self.set_rf_enabled(False)

    def set_program_gain(self, gain: float) -> None:
        gain = float(gain)
        if not math.isfinite(gain) or gain < 0:
            raise ValueError("RF gain must be finite and nonnegative")
        if not np.all(np.isfinite(gain*self._compiled.envelope)):
            raise ValueError("RF gain overflows the scheduled rates")
        self.program.gain = gain

    def start_program(self, turn_rf_on: bool = True) -> None:
        """Set program t=0 to the current simulation time, without changing n."""
        if self._compiled.end_time <= 0:
            raise ValueError("No enabled, positive-rate, positive-duration pulses are defined")
        times = self.t + self._compiled.times
        if len(times)>1 and not np.all(np.diff(times)>0):
            raise ValueError("Pulse boundaries are below floating-point time resolution; reset simulation time first")
        self._epoch = float(self.t)
        self._event_times_absolute = times
        self._stopped = False
        self.delivered_exposure.fill(0)
        if turn_rf_on:
            self.set_rf_enabled(True)

    def stop_program(self) -> None:
        self._epoch = None
        self._event_times_absolute = None
        self._stopped = True
        self.set_rf_enabled(False)

    @property
    def program_elapsed(self) -> float:
        return 0.0 if self._epoch is None else max(0.0, float(self.t-self._epoch))

    @property
    def program_state(self) -> str:
        if self._epoch is None:
            return "stopped" if self._stopped else "ready"
        if self.t >= self._event_times_absolute[-1]:
            return "finished"
        return "running" if np.any(self.commanded_rf_field()) else "waiting"

    def _interval_index(self, time: Optional[float] = None) -> int:
        if self._event_times_absolute is None:
            return -1
        return int(np.searchsorted(self._event_times_absolute, self.t if time is None else time, side="right"))-1

    def commanded_rf_field(self) -> np.ndarray:
        """Scheduled base U_j, before the master gate and packet coupling weight."""
        if self._command_override is not None:
            return self._command_override.copy()
        if self._epoch is None:
            return np.zeros(len(self.Rplus))
        return self.program.gain*self._compiled.field_for_index(self._interval_index()).copy()

    def applied_rf_field(self) -> np.ndarray:
        return self.commanded_rf_field() if self.params.rf_enabled else np.zeros(len(self.Rplus))

    def preview_rf_field(self) -> np.ndarray:
        return self.program.gain*self._compiled.envelope.copy()

    def rf_profile_physical(self, center_R=None) -> Tuple[np.ndarray, np.ndarray]:
        # The monitor location is not a source location for a whole profile.
        return self.Rplus.copy(), self.commanded_rf_field()

    def rf_profile_arrays(self, center_R=None) -> Tuple[np.ndarray, np.ndarray]:
        u = self.commanded_rf_field()
        return u, u[::-1].copy()

    def rf_rate_fields(self, center_R=None, gamma_rf=None) -> Tuple[np.ndarray, np.ndarray]:
        u, um = self.rf_profile_arrays()
        w = self.capacity_rate_weights()
        # gamma_rf in the baseline derivative is intentionally unused: rates
        # are supplied by the per-bin program, with program.gain as the sole
        # additional global multiplier. No automatic profile normalization.
        return w*u, w*um

    def rf_profile_summary(self, center_R=None):
        u = self.commanded_rf_field()
        return dict(profile_peak=float(np.max(u)), active_bins=int(np.count_nonzero(u)),
                    backend="ideal-bin pulse program", normalization="none")

    def effective_local_rates(self, R=None):
        if R is None: R = self.params.rf_burn_R
        kp, km = self.branch_indices(R)
        gp, gm = self.rf_rate_fields()
        u = self.commanded_rf_field()
        cap = self.local_capacity_factors(R)
        def at(v, k): return np.nan if k is None else float(v[k])
        return {**cap, "command_at_R": at(u, kp),
                "gamma_rf_Iplus_R": at(gp,kp), "gamma_rf_Iminus_R": at(gm,km),
                "gamma_rf_opposite_on_Iplus_packet": at(gm,kp),
                "gamma_rf_opposite_on_Iminus_packet": at(gp,km),
                "dnp_Iplus_R": float(self.params.dnp_rate*cap['w_Iplus_R']),
                "dnp_Iminus_R": float(self.params.dnp_rate*cap['w_Iminus_R'])}

    def rf_balance_estimate(self, R=None):
        # Avoid presenting the baseline single-carrier hold estimate for a
        # multibin schedule. This version deliberately has no optimizer.
        raise NotImplementedError("No automatic hold-rate estimation for ideal pulse programs")

    def _max_outgoing_bound(self, rf_on: bool, dnp_on: bool) -> float:
        """Conservative nonnegative-rate bound for safe Euler subdivision.

        This changes only the numerical step size when necessary, never the
        command amplitude or duration. It prevents high multibin commands from
        being silently corrected by a positivity clamp.
        """
        w = self.capacity_rate_weights()
        out = np.zeros(len(self.Rplus))
        if rf_on:
            gp, gm = self.rf_rate_fields()
            out += gp+gm
        if dnp_on:
            out += float(self.params.dnp_rate)*w
        out += float(self.params.t1_rate)
        if self.params.diffusion_enabled and self.params.diffusion_scale != 0:
            c = self.diffusion_connectivity(dnp_on=dnp_on)
            out += c['same_plus0']+c['same_0minus']+c['cross_plus']+c['cross_minus']
            out += c['double_quantum']
        if not np.all(np.isfinite(out)) or np.any(out<0):
            raise ValueError("All active rates must be finite and nonnegative")
        return float(out.max())

    def step(self, n_steps: int = 1, rf_on: Optional[bool] = None, dnp_on: Optional[bool] = None) -> None:
        """Advance in simulation time, splitting exactly at every pulse edge.

        QTimer controls only redraw opportunities, not pulse durations. RF OFF
        mutes the drive while program/simulation time continues. Pause is done
        by not calling step. A completed program applies zero RF while recovery
        continues. Kinetics use the baseline forward-Euler method, with safety
        subdivision for large outgoing rates and a hard failure on nonphysical
        populations rather than clipping a significant error.
        """
        dt = float(self.params.dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if isinstance(n_steps,bool) or int(n_steps)!=n_steps or n_steps<1:
            raise ValueError("n_steps must be a positive integer")
        rf_on = bool(self.params.rf_enabled) if rf_on is None else bool(rf_on)
        dnp_on = bool(self.params.dnp_enabled) if dnp_on is None else bool(dnp_on)
        self.last_substeps = 0
        for _ in range(int(n_steps)):
            target = self.t + dt
            if target <= self.t:
                raise ValueError("dt is below floating-point resolution at the current time")
            while self.t < target:
                index = self._interval_index()
                boundary = target
                if self._event_times_absolute is not None and index+1 < len(self._event_times_absolute):
                    boundary = min(boundary, float(self._event_times_absolute[index+1]))
                self._command_override = (np.zeros(len(self.Rplus)) if self._epoch is None else
                                          self.program.gain*self._compiled.field_for_index(index).copy())
                try:
                    bound = self._max_outgoing_bound(rf_on,dnp_on)
                    max_h = math.inf if bound == 0 else 0.2/bound
                    h = min(boundary-self.t,max_h)
                    if h <= 0 or self.t+h <= self.t:
                        raise ValueError("Unresolvable numerical substep at a pulse boundary")
                    dn = self.derivative(rf_on=rf_on,dnp_on=dnp_on)
                    candidate = self.n + h*dn
                    if not np.all(np.isfinite(candidate)) or np.any(candidate < -1e-13*self.mu[:,None]):
                        raise FloatingPointError("Nonphysical population step; reduce dt or active rates")
                    # Retain the baseline roundoff floor and per-packet normalization.
                    self.n = np.maximum(candidate,1e-30)
                    self.n *= self.mu[:,None]/np.maximum(self.n.sum(axis=1,keepdims=True),1e-30)
                    if rf_on:
                        exposure = h*self._command_override
                        self.delivered_exposure += exposure
                        self.total_delivered_exposure += exposure
                    self.t = boundary if h == boundary-self.t else self.t+h
                    self.last_substeps += 1
                    if self.last_substeps > 200000:
                        raise RuntimeError("Too many integration substeps; lower rates or steps/tick")
                finally:
                    self._command_override = None
