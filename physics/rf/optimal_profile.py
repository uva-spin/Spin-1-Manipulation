"""ssRF-beta synchronized tensor profile: candidate mask, RF-only seed, T scan, L-BFGS-B.

The search is a bounded local optimizer, not a global certificate. Forward
kinetics regroup the unchanged IdealBinModel ODE; each delivered program is
replayed through IdealBinModel.step. No ssRF-beta runtime import.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.optimize import minimize

from .ideal_model import IdealBinModel
from .profile_control import (
    T_MAX,
    T_MIN,
    U_MAX,
    candidate_mask,
    make_ideal_model,
)
from .pulse_program import BinPulse, PulseProgram, RFProfile


@dataclass
class OptimizerSettings:
    max_power: float = U_MAX
    min_duration: float = T_MIN
    max_duration: float = T_MAX
    duration_samples: int = 7
    duration_refinements: int = 3
    max_iterations: int = 30
    starts: int = 2
    search_dt: float = 0.004
    max_steps: int = 20000
    max_wall_seconds: float = 300.0
    q_absolute_tolerance: float = 1e-6
    q_relative_tolerance: float = 0.002
    candidate_mode: str = "union"
    candidate_threshold: float = 1e-10
    seed: int = 42

    def validate(self):
        if self.max_power <= 0:
            raise ValueError("max_power must be positive")
        if not 0 < self.min_duration <= self.max_duration:
            raise ValueError("Require 0 < min_duration <= max_duration")
        if not 2 <= self.duration_samples <= 40:
            raise ValueError("duration_samples must be 2..40")
        if not 0 <= self.duration_refinements <= 20:
            raise ValueError("duration_refinements must be 0..20")
        if not 1 <= self.max_iterations <= 1000 or not 1 <= self.starts <= 10:
            raise ValueError("Invalid iterations/starts")
        if not 2 <= self.max_steps <= 100000:
            raise ValueError("max_steps must be 2..100000")
        if self.candidate_mode not in ("union", "tensor_deficit", "rf_gain", "all"):
            raise ValueError("Unknown candidate mode")
        return self


# Shorter search used when generating DAE events (same objective and seed family).
DATA_GEN_SETTINGS = OptimizerSettings(
    duration_samples=5,
    duration_refinements=1,
    max_iterations=8,
    starts=1,
    max_wall_seconds=60.0,
)

QUICK_SETTINGS = OptimizerSettings(
    duration_samples=3,
    duration_refinements=0,
    max_iterations=3,
    starts=1,
    max_wall_seconds=15.0,
)


def polarizations(n):
    return {
        "P": float(np.sum(n[:, 0] - n[:, 2])),
        "Q": float(np.sum(n[:, 0] - 2.0 * n[:, 1] + n[:, 2])),
    }


class FrozenDynamics:
    """Unchanged population ODE, regrouped for many-trial evaluations."""

    def __init__(self, model: IdealBinModel):
        self.model = model
        self.p = model.params
        self.mu = model.mu.copy()
        self.n0 = model.n.copy()
        self.N = len(self.mu)
        self.grid = model.Rplus.copy()
        self.w = model.capacity_rate_weights().copy()
        self.scale = (
            self.p.diffusion_scale
            * (self.p.microwave_diffusion_factor if self.p.dnp_enabled else 1.0)
            if self.p.diffusion_enabled
            else 0.0
        )
        self.kx = self.scale * self.p.cross_branch_ratio
        self.kdq = self.scale * self.p.double_quantum_ratio
        self.decay = np.full(self.N, self.p.t1_rate)
        self.source = self.p.t1_rate * self.model.equilibrium_reference(self.p.t1_p_eq)
        if self.p.dnp_enabled:
            d = self.p.dnp_rate * self.w
            self.decay = self.decay + d
            self.source = self.source + d[:, None] * self.model.equilibrium_reference(self.p.p_dnp_sat)
        self._fft = None
        self.H = self.X = self.C = None
        if self.scale:
            r_grid = self.grid
            cut_bins = self.p.kernel_cutoff_widths * self.p.zq_width_R / self.model.dR
            cutoff_ambiguous = (
                self.p.kernel_cutoff_widths > 0 and abs(cut_bins - round(cut_bins)) < 1e-9
            )
            if self.p.orientation_corr_fraction == 0 and not cutoff_ambiguous:
                col = self.model._spectral_overlap(np.arange(self.N) * self.model.dR)
                self.L = next_fast_len(2 * self.N - 1)
                embed = np.zeros(self.L)
                embed[: self.N] = col
                embed[-self.N + 1 :] = col[1:][::-1]
                self._fft = rfft(embed)[:, None]
                self.cross_diag = 0.5 * self.model._spectral_overlap(2 * r_grid)
            else:
                theta = self.model._effective_theta()
                f = self.p.orientation_corr_fraction
                sig = np.deg2rad(self.p.orientation_corr_width_deg)
                c = (1 - f) + f * np.exp(-0.5 * ((theta[:, None] - theta[None, :]) / sig) ** 2)
                self.H = c * self.model._spectral_overlap(r_grid[:, None] - r_grid[None, :])
                self.X = c * self.model._spectral_overlap(r_grid[:, None] + r_grid[None, :])
                self.X[np.diag_indices(self.N)] *= 0.5
                self.C = c.copy()
                np.fill_diagonal(self.C, 0)

    def h(self, y):
        if self._fft is not None:
            return irfft(self._fft * rfft(y, n=self.L, axis=0), n=self.L, axis=0)[: self.N]
        return self.H @ y

    def x(self, y):
        if self._fft is not None:
            return self.h(y[::-1]) - self.cross_diag[:, None] * y
        return self.X @ y

    def c(self, y):
        return np.sum(y, axis=0)[None, :] if self.C is None else self.C @ y

    def rhs(self, n, u):
        gp = self.w * u
        gm = self.w * u[::-1]
        jp = gp * (n[:, 0] - n[:, 1])
        jm = gm * (n[:, 1] - n[:, 2])
        out = -self.decay[:, None] * n + self.source
        out[:, 0] -= jp
        out[:, 1] += jp - jm
        out[:, 2] += jm
        if self.scale:
            hn = self.h(n)
            a = self.scale * (n[:, 1] * hn[:, 0] - n[:, 0] * hn[:, 1])
            b = self.scale * (n[:, 2] * hn[:, 1] - n[:, 1] * hn[:, 2])
            out[:, 0] += a
            out[:, 1] += -a + b
            out[:, 2] -= b
            if self.kx:
                xn = self.x(n)
                c = self.kx * (n[:, 1] * xn[:, 1] - n[:, 0] * xn[:, 2])
                d = self.kx * (n[:, 1] * xn[:, 1] - n[:, 2] * xn[:, 0])
                out[:, 0] += c
                out[:, 1] -= c + d
                out[:, 2] += d
            if self.kdq:
                cn = self.c(n)
                z = self.kdq * (n[:, 2] * cn[:, 0] - n[:, 0] * cn[:, 2])
                out[:, 0] += z
                out[:, 2] -= z
        return out

    def vjp(self, n, u, adj):
        va = adj[:, 0] - adj[:, 1]
        vb = adj[:, 1] - adj[:, 2]
        vd = -vb
        vz = adj[:, 0] - adj[:, 2]
        gp = self.w * u
        gm = self.w * u[::-1]
        g = -self.decay[:, None] * adj
        g[:, 0] -= gp * va
        g[:, 1] += gp * va - gm * vb
        g[:, 2] += gm * vb
        gu = -self.w * (n[:, 0] - n[:, 1]) * va - (self.w * (n[:, 1] - n[:, 2]) * vb)[::-1]
        if self.scale:
            s = self.scale
            hn = self.h(n)
            g[:, 0] -= s * va * hn[:, 1]
            g[:, 1] += s * (va * hn[:, 0] - vb * hn[:, 2])
            g[:, 2] += s * vb * hn[:, 1]
            b = np.zeros_like(n)
            b[:, 0] = s * va * n[:, 1]
            b[:, 1] = s * (-va * n[:, 0] + vb * n[:, 2])
            b[:, 2] = -s * vb * n[:, 1]
            g = g + self.h(b)
            if self.kx:
                s = self.kx
                xn = self.x(n)
                g[:, 0] -= s * va * xn[:, 2]
                g[:, 1] += s * (va + vd) * xn[:, 1]
                g[:, 2] -= s * vd * xn[:, 0]
                b[:, 0] = -s * vd * n[:, 2]
                b[:, 1] = s * (va + vd) * n[:, 1]
                b[:, 2] = -s * va * n[:, 0]
                g = g + self.x(b)
            if self.kdq:
                s = self.kdq
                cn = self.c(n)
                g[:, 0] -= s * vz * cn[:, 2]
                g[:, 2] += s * vz * cn[:, 0]
                b.fill(0)
                b[:, 0] = s * vz * n[:, 2]
                b[:, 2] = -s * vz * n[:, 0]
                g = g + self.c(b)
        return g, gu

    def outgoing_bound(self, upper):
        out = self.w * (upper + upper[::-1]) + self.decay
        if self.scale:
            mu = self.mu[:, None]
            out = out + 2 * self.scale * self.h(mu)[:, 0]
            if self.kx:
                out = out + 2 * self.kx * self.x(mu)[:, 0]
            if self.kdq:
                out = out + self.kdq * np.broadcast_to(self.c(mu), (self.N, 1))[:, 0]
        return float(out.max())

    def evaluate(self, u, T, dt, max_steps=20000, gradient=False, check=None):
        steps = max(1, int(np.ceil(T / dt)))
        if steps > max_steps:
            raise ValueError(
                f"{steps} prediction steps exceed max_steps={max_steps}; "
                "reduce duration/power or increase the limit"
            )
        h = T / steps
        n = self.n0.copy()
        history = [] if gradient else None
        for it in range(steps):
            if check and it % 32 == 0:
                check()
            if gradient:
                history.append(n)
            n = n + h * self.rhs(n, u)
        if not np.isfinite(n).all() or np.any(n < -1e-11 * self.mu[:, None]):
            raise FloatingPointError("Prediction left the population simplex; reduce search_dt")
        q = polarizations(n)["Q"]
        if not gradient:
            return q, n
        adj = np.broadcast_to(np.array([1.0, -2.0, 1.0]), n.shape).copy()
        gu = np.zeros(self.N)
        for it in range(steps - 1, -1, -1):
            if check and it % 32 == 0:
                check()
            gn, gc = self.vjp(history[it], u, adj)
            gu = gu + h * gc
            adj = adj + h * gn
        return q, n, gu


def rf_only_seed(dyn, T, upper):
    """Isolated-bin exposure maxima used as the ssRF-beta starting guess."""
    n = dyn.n0
    a = n[:, 0] - n[:, 1]
    b = (n[:, 1] - n[:, 2])[::-1]
    cp = dyn.w
    cm = dyn.w[::-1]
    emax = upper * T
    e = np.stack((np.zeros_like(emax), emax, 0.5 * emax))
    with np.errstate(divide="ignore", invalid="ignore"):
        stationary = np.log((cm * b) / (cp * a)) / (2 * (cm - cp))
    valid = np.isfinite(stationary) & (stationary > 0) & (stationary < emax) & (cp * a > 0) & (cm * b > 0)
    e = np.vstack((e, np.where(valid, stationary, 0)))
    gains = 1.5 * (
        b[None, :] * (-np.expm1(-2 * cm[None, :] * e))
        - a[None, :] * (-np.expm1(-2 * cp[None, :] * e))
    )
    pick = np.argmax(gains, axis=0)
    return e[pick, np.arange(dyn.N)] / T


def synchronized_program(grid, powers, T):
    program = PulseProgram(
        n_bins=len(grid),
        r_min=float(grid[0]),
        r_max=float(grid[-1]),
        gain=1.0,
    )
    for title, mask in (("Negative-R region", grid < 0), ("Nonnegative-R region", grid >= 0)):
        rows = [
            BinPulse(int(j), float(powers[j]), 0.0, float(T))
            for j in np.flatnonzero(mask & (powers > 0))
        ]
        if rows:
            program.profiles.append(RFProfile(title, pulses=rows))
    if not program.profiles:
        program.profiles = [RFProfile("No beneficial RF selected", pulses=[BinPulse(0, 0.0, 0.0, 0.0, False)])]
    program.validate()
    return program


@dataclass
class ProfileDesign:
    program: PulseProgram
    duration: float
    rates: np.ndarray
    mask: np.ndarray
    q_initial: float
    q_final: float
    p_initial: float
    p_final: float
    status: str


def design_optimal_profile(model: IdealBinModel, settings: Optional[OptimizerSettings] = None) -> ProfileDesign:
    """Bounded T scan + L-BFGS-B on candidate bins; replay through IdealBinModel."""
    settings = (settings or OptimizerSettings()).validate()
    start_clock = time.monotonic()
    deadline = start_clock + settings.max_wall_seconds
    dyn = FrozenDynamics(model)
    mask = candidate_mask(
        model, mode=settings.candidate_mode, threshold=settings.candidate_threshold
    )
    upper = np.full(dyn.N, settings.max_power)
    upper[~mask] = 0.0
    mask = mask & (upper > 0)
    ix = np.flatnonzero(mask)
    bound = dyn.outgoing_bound(upper)
    dt = min(settings.search_dt, 0.18 / bound if bound > 0 else settings.search_dt)
    initial = polarizations(dyn.n0)
    choices = []
    warm = None
    rng = np.random.default_rng(settings.seed)

    def search_check():
        if time.monotonic() > deadline:
            raise TimeoutError("Search time budget reached")

    def at_duration(T):
        nonlocal warm
        search_check()
        qzero, _ = dyn.evaluate(np.zeros(dyn.N), T, dt, settings.max_steps, check=search_check)
        seed = rf_only_seed(dyn, T, upper)
        seed = np.clip(seed, 0.0, upper)
        seed[~mask] = 0.0
        starts = [seed]
        if warm is not None:
            starts.append(np.clip(warm, 0.0, upper))
        while len(starts) < settings.starts:
            starts.append(upper * (0.25 + 0.75 * rng.random(dyn.N)))
        best_q = qzero
        best_u = np.zeros(dyn.N)
        if len(ix):
            for trial in starts[: settings.starts]:

                def objective(z, trial_ix=ix):
                    u = np.zeros(dyn.N)
                    u[trial_ix] = upper[trial_ix] * z
                    q, _, grad = dyn.evaluate(
                        u, T, dt, settings.max_steps, True, search_check
                    )
                    nonlocal best_q, best_u
                    if q > best_q:
                        best_q = q
                        best_u = u.copy()
                    return -1000 * q, -1000 * grad[trial_ix] * upper[trial_ix]

                z0 = np.clip(trial[ix] / np.maximum(upper[ix], 1e-30), 0.0, 1.0)
                minimize(
                    objective,
                    z0,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=[(0.0, 1.0)] * len(ix),
                    options={
                        "maxiter": settings.max_iterations,
                        "gtol": 1e-6,
                        "ftol": 1e-11,
                        "maxls": 15,
                    },
                )
        else:
            best_u = np.zeros(dyn.N)
            best_q = qzero
        warm = best_u.copy()
        choices.append((float(T), best_u.copy(), float(best_q)))
        return choices[-1]

    durations = np.unique(
        np.geomspace(settings.min_duration, settings.max_duration, settings.duration_samples)
    )
    budget_hit = False
    try:
        for t in durations:
            at_duration(float(t))
        for _ in range(settings.duration_refinements):
            best = max(c[2] for c in choices)
            tol = max(
                settings.q_absolute_tolerance,
                settings.q_relative_tolerance * max(0.0, best - initial["Q"]),
            )
            eligible = sorted([c for c in choices if c[2] >= best - tol], key=lambda c: c[0])
            first = eligible[0]
            below = sorted([c[0] for c in choices if c[0] < first[0]])
            if not below:
                break
            at_duration(math.sqrt(below[-1] * first[0]))
    except TimeoutError:
        budget_hit = True
    if not choices:
        raise RuntimeError("No horizon completed before the time budget.")
    dt_fine = min(float(model.params.dt), dt)
    fine = []
    for t, u, _q in choices:
        qf, nf = dyn.evaluate(u, t, dt_fine, settings.max_steps)
        fine.append((t, u, qf, nf))
    best = max(c[2] for c in fine)
    tol = max(
        settings.q_absolute_tolerance,
        settings.q_relative_tolerance * max(0.0, best - initial["Q"]),
    )
    if best <= initial["Q"] + settings.q_absolute_tolerance:
        t = 0.0
        u = np.zeros(dyn.N)
        status = "no_improvement"
        endpoint = dyn.n0.copy()
    else:
        selected = min((c for c in fine if c[2] >= best - tol), key=lambda c: c[0])
        t, u, _q, endpoint = selected
        status = "completed"
        if not np.any(u):
            status = "no_RF_benefit"
    if budget_hit and status == "completed":
        status = "budget_reached"
    program = synchronized_program(dyn.grid, u, t)
    # Independent replay through the unchanged scheduler.
    replay_model = IdealBinModel(model.params.replace(), program)
    replay_model.n = dyn.n0.copy()
    replay_model.n_ref = model.n_ref.copy()
    replay_model.t = 0.0
    if t > 0.0 and np.any(u):
        replay_model.set_program(program)
        replay_model.start_program(True)
        n_steps = max(1, int(math.ceil(t / float(replay_model.params.dt))))
        replay_model.step(n_steps)
        endpoint = replay_model.n.copy()
    final = polarizations(endpoint)
    return ProfileDesign(
        program=program,
        duration=float(t),
        rates=u.copy(),
        mask=mask,
        q_initial=float(initial["Q"]),
        q_final=float(final["Q"]),
        p_initial=float(initial["P"]),
        p_final=float(final["P"]),
        status=status,
    )


def replay_program_spectra(model: IdealBinModel, program: PulseProgram, *, n_record: int):
    """Play ``program`` from the current state, capturing I±, P, Q at each Euler step.

    Frame 0 is the pre-RF state. ``n_record`` is the number of subsequent
    ``model.step(1)`` calls (so arrays have length ``n_record + 1``).
    """
    n_record = int(n_record)
    if n_record < 0:
        raise ValueError("n_record must be nonnegative")
    t_len = n_record + 1
    n_bins = len(model.Rplus)
    iplus_full = np.empty((t_len, n_bins))
    iminus_full = np.empty((t_len, n_bins))
    p_full = np.empty(t_len)
    q_full = np.empty(t_len)

    def _record(k):
        ip, im, _ = model.physical_intensities()
        pol = model.polarizations()
        iplus_full[k] = ip
        iminus_full[k] = im
        p_full[k] = pol["P"]
        q_full[k] = pol["Q"]

    _record(0)
    compiled = program.compile()
    if n_record == 0 or compiled.end_time <= 0.0:
        return iplus_full, iminus_full, p_full, q_full
    model.set_program(program)
    model.start_program(True)
    for k in range(1, t_len):
        model.step(1)
        _record(k)
    return iplus_full, iminus_full, p_full, q_full


def run_optimal_profile_polarization(
    polarization,
    *,
    n_steps,
    n_bins=701,
    r_min=-3.0,
    r_max=3.0,
    dt=None,
    settings=None,
):
    """Design the ssRF-beta profile at ``polarization`` and capture per-step spectra.

    ``n_steps`` is the number of Euler frames after equilibrium (same convention
    as ssRF burns). The program endpoint is always included when it falls past
    ``n_steps``.
    """
    p0 = float(polarization)
    model = make_ideal_model(p0, n_bins=int(n_bins), r_min=float(r_min), r_max=float(r_max))
    if dt is not None:
        model.params.dt = float(dt)
    design = design_optimal_profile(model, settings=settings)
    dt_used = float(model.params.dt)
    n_euler = 0 if design.duration <= 0.0 else max(1, int(math.ceil(design.duration / dt_used)))
    n_record = max(int(n_steps), n_euler)
    play = make_ideal_model(p0, n_bins=int(n_bins), r_min=float(r_min), r_max=float(r_max))
    play.params.dt = dt_used
    iplus_full, iminus_full, p_full, q_full = replay_program_spectra(
        play, design.program, n_record=n_record
    )
    f = play.Rplus.copy()
    return {
        "polarization": p0,
        "skipped": False,
        "n_steps": n_record + 1,
        "burn_steps": n_euler,
        "gamma_rf": float(np.max(design.rates)) if design.rates.size else 0.0,
        "duration": float(design.duration),
        "rates": np.asarray(design.rates, dtype=float),
        "mask": np.asarray(design.mask, dtype=bool),
        "status": design.status,
        "iplus_full": iplus_full,
        "iminus_full": iminus_full,
        "ps_full": iplus_full + iminus_full,
        "p_full": p_full,
        "q_full": q_full,
        "p_initial": float(design.p_initial),
        "q_initial": float(design.q_initial),
        "p_final": float(design.p_final),
        "q_final": float(design.q_final),
        "frequency": f,
        "center_bin": -1,
        "n_burns": 1 if design.duration > 0.0 else 0,
        "dt": dt_used,
        "program_end_step": n_euler,
    }
