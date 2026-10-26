"""LabVIEW Python Node API for analytic Pake / optional Dulya + ssRF / AFP dynamics.

Call ``InitSim`` once per LabVIEW Python session, then ``main(cmd, ...)`` in a
loop. Simulation state lives in module globals for the lifetime of the process.
Empty ``fit_params_path`` (default) uses the analytic Pake equilibrium, matching
``create_data.py`` without ``--fit-params``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
_REPO = _DIR.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from Data_Creation.spectrum_gen.afp_traj import qneg_sweep_indices
from Data_Creation.spectrum_gen.bridge import (
    afp_window_indices,
    build_equilibrium_spin1_model,
    configure_afp_recovery,
    configure_ssrf_burn,
    euler_n_sub,
    full_spectrum_intensities,
    level_pq,
)
from Data_Creation.spectrum_gen.constants import (
    AFP_CENTER_MARGIN,
    AFP_WINDOW,
    DT,
    F_MAX,
    F_MIN,
    NUM_BINS,
    RF_MODE_PHYSICAL_VOIGT,
    RF_MODE_SINGLE_BIN,
)
from Data_Creation.spectrum_gen.dulya import use_fit_params
from physics.rf.ideal_model import IdealBinModel, IdealBinParams
from physics.rf.optimal_profile import OptimizerSettings, design_optimal_profile
from physics.rf.profile_control import play_program

# Command codes for LabVIEW
CMD_GET_SPECTRUM = 0
CMD_STEP = 1
CMD_SSRF = 2
CMD_AFP = 3
CMD_SSRF_PROFILE = 4
CMD_AFP_PROFILE = 5
CMD_RESET = 6

AFP_MODE_QNEG = 0
AFP_MODE_SELECTIVE = 1

RF_MODE_VOIGT = 0
RF_MODE_SINGLE = 1

# Session state (LabVIEW Python process lifetime)
model = None
sim_time = 0.0
_init_kwargs = None
_last_freq = None
_last_intensity = None
_last_p = 0.0
_last_q = 0.0
_optimizer_settings = None


def _default_optimizer_settings():
    """Interactive-friendly defaults (tests may override ``_optimizer_settings``)."""
    return OptimizerSettings(
        duration_samples=5,
        duration_refinements=1,
        max_iterations=15,
        starts=1,
        max_wall_seconds=60.0,
        search_dt=0.01,
    )


def _empty_snapshot(status):
    empty = np.asarray([], dtype=float)
    return (empty, empty, 0.0, 0.0, float(sim_time), str(status))


def _snapshot(status='ok'):
    """Return LabVIEW-facing spectrum tuple from the live model (or last cache)."""
    global _last_freq, _last_intensity, _last_p, _last_q, sim_time
    if model is None:
        if _last_freq is None:
            return _empty_snapshot(status)
        return (
            np.asarray(_last_freq, dtype=float),
            np.asarray(_last_intensity, dtype=float),
            float(_last_p),
            float(_last_q),
            float(sim_time),
            str(status),
        )
    freq, iplus, iminus, total = model.spectrum()
    p, q = level_pq(model)
    _last_freq = np.asarray(freq, dtype=float)
    _last_intensity = np.asarray(total, dtype=float)
    _last_p = float(p)
    _last_q = float(q)
    sim_time = float(getattr(model, 't', sim_time))
    return (
        _last_freq.copy(),
        _last_intensity.copy(),
        _last_p,
        _last_q,
        float(sim_time),
        str(status),
    )


def _rf_mode_name(rf_mode):
    if int(rf_mode) == RF_MODE_SINGLE:
        return RF_MODE_SINGLE_BIN
    return RF_MODE_PHYSICAL_VOIGT


def _burn_bin_from_R(burn_R):
    r = np.asarray(model.Rplus, dtype=float)
    return int(np.argmin(np.abs(r - float(burn_R))))


def _normalize_fit_path(fit_params_path):
    """Return a fit JSON path, or None for analytic Pake (create_data default)."""
    if fit_params_path is None:
        return None
    path = str(fit_params_path).strip()
    return path or None


def _build_model(*, fit_params_path, P0, Q0, n_bins, dt):
    use_fit_params(_normalize_fit_path(fit_params_path))
    q0 = None if float(Q0) < 0.0 else float(Q0)
    m = build_equilibrium_spin1_model(
        polarization=float(P0),
        num_bins=int(n_bins),
        dt=float(dt),
        rf_enabled=False,
        relax_enabled=True,
        r_min=F_MIN,
        r_max=F_MAX,
        legacy_spectral_recovery=True,
    )
    if q0 is not None:
        m.params.q0 = q0
    return m


def InitSim(fit_params_path='', P0=0.45, Q0=-1.0, n_bins=500, dt=None):
    """Build equilibrium model into module globals (call once per LabVIEW session).

    ``fit_params_path`` is optional. Leave empty (default) to use the analytic
    Pake lineshape, same as ``create_data.py`` without ``--fit-params``. Pass a
    Dulya fit JSON path to seat that fitted equilibrium instead. ``Q0 < 0``
    keeps Boltzmann tensor Q.
    """
    global model, sim_time, _init_kwargs, _last_freq, _last_intensity, _last_p, _last_q
    dt_used = float(DT if dt is None else dt)
    n = int(n_bins) if n_bins else NUM_BINS
    _init_kwargs = {
        'fit_params_path': _normalize_fit_path(fit_params_path),
        'P0': float(P0),
        'Q0': float(Q0),
        'n_bins': n,
        'dt': dt_used,
    }
    model = _build_model(**_init_kwargs)
    sim_time = float(model.t)
    return _snapshot('ok')


def _cmd_step(dt, n_steps):
    n = max(1, int(n_steps))
    step_dt = float(dt) if float(dt) > 0.0 else float(model.params.dt)
    model.params.rf_enabled = False
    model._active_idx = None
    for _ in range(n):
        model.step_once(dt=step_dt, rf_on=False, dnp_on=False, copy=False)
    return _snapshot('ok')


def _cmd_ssrf(burn_R, gamma_rf, n_steps, rf_mode):
    burn_idx = _burn_bin_from_R(burn_R)
    g = float(gamma_rf)
    n = max(1, int(n_steps))
    configure_ssrf_burn(
        model,
        burn_idx,
        g,
        rf_mode=_rf_mode_name(rf_mode),
        legacy_spectral_recovery=True,
    )
    n_sub, dt_sub = euler_n_sub(g, float(model.params.dt))
    for _ in range(n):
        for _ in range(n_sub):
            model.step_once(dt=dt_sub, rf_on=True, dnp_on=False, copy=False)
    model.params.rf_enabled = False
    model._active_idx = None
    return _snapshot('ok')


def _apply_afp_subset(subset, efficiency):
    if not subset:
        return _snapshot('afp: empty subset')
    model.params.afp_enabled = True
    model.params.afp_efficiency = float(efficiency)
    model.params.afp_center_margin = AFP_CENTER_MARGIN
    model.params.afp_preserve_intensity_area = False
    model.params.afp_subset_indices = [int(i) for i in subset]
    model.afp_sweep()
    model.params.afp_enabled = False
    configure_afp_recovery(model)
    return _snapshot('ok')


def _cmd_afp(burn_R, afp_mode, efficiency):
    ip, im, _ = full_spectrum_intensities(model)
    q_eq = np.asarray(ip) - np.asarray(im)
    n_bins = int(q_eq.size)
    if int(afp_mode) == AFP_MODE_SELECTIVE:
        burn_idx = _burn_bin_from_R(burn_R)
        subset = afp_window_indices(burn_idx, n_bins, window=AFP_WINDOW)
    else:
        subset = qneg_sweep_indices(q_eq)
    return _apply_afp_subset(subset, efficiency)


def _cmd_afp_profile(efficiency):
    ip, im, _ = full_spectrum_intensities(model)
    q_eq = np.asarray(ip) - np.asarray(im)
    subset = qneg_sweep_indices(q_eq)
    return _apply_afp_subset(subset, efficiency)


def _ideal_from_session():
    """Copy current Spin1 state onto an IdealBinModel for profile design/play."""
    p0 = float(_init_kwargs['P0'])
    n_bins = int(len(model.Rplus))
    params = IdealBinParams(
        n_bins=n_bins,
        r_min=float(model.Rplus[0]),
        r_max=float(model.Rplus[-1]),
        p0=p0,
        q0=None,
        dt=float(model.params.dt),
        rf_enabled=False,
        dnp_enabled=False,
        diffusion_enabled=True,
        relax_enabled=False,
        d_same_plus0=0.0,
        d_same_0minus=0.0,
        d_spec_plus0=0.0,
        d_spec_0minus=0.0,
        capacity_rate_power=float(model.params.capacity_rate_power),
        diffusion_scale=float(model.params.diffusion_scale),
        zq_width_R=float(model.params.zq_width_R),
    )
    ideal = IdealBinModel(params)
    ideal.display_cal = float(model.display_cal)
    ideal.mu = np.asarray(model.mu, dtype=float).copy()
    ideal.base_density = ideal.mu / ideal.dR
    ideal.n = np.asarray(model.n, dtype=float).copy()
    ideal.n_ref = np.asarray(model.n_ref, dtype=float).copy()
    ideal.t = float(model.t)
    ideal._sync_level_populations(capture_initial=False)
    return ideal


def _cmd_ssrf_profile(n_steps):
    settings = _optimizer_settings or _default_optimizer_settings()
    ideal = _ideal_from_session()
    design = design_optimal_profile(ideal, settings=settings)
    play = _ideal_from_session()
    if design.duration > 0.0 and design.program is not None:
        play_program(play, design.program, dnp_on=False)
        # Optional extra recovery steps after the program ends
        extra = max(0, int(n_steps) - max(1, int(math.ceil(design.duration / float(play.params.dt)))))
        if extra > 0:
            play.params.rf_enabled = False
            play.step(extra, rf_on=False, dnp_on=False)
    ip, im, _ = play.physical_intensities()
    model.load_from_physical_intensities(ip, im)
    model.t = float(play.t)
    status = f'ok:{design.status}'
    return _snapshot(status)


def _cmd_reset():
    global model, sim_time
    if _init_kwargs is None:
        return _empty_snapshot('error: InitSim not called')
    model = _build_model(**_init_kwargs)
    sim_time = float(model.t)
    return _snapshot('ok')


def main(
    cmd,
    dt=0.0,
    burn_R=0.0,
    gamma_rf=2.0,
    n_steps=1,
    afp_mode=0,
    efficiency=1.0,
    rf_mode=0,
):
    """Dispatch a simulation command; always return ``(freq, intensity, P, Q, t, status)``."""
    try:
        c = int(cmd)
        if c == CMD_RESET:
            return _cmd_reset()
        if model is None:
            return _empty_snapshot('error: call InitSim first')
        if c == CMD_GET_SPECTRUM:
            return _snapshot('ok')
        if c == CMD_STEP:
            return _cmd_step(dt, n_steps)
        if c == CMD_SSRF:
            return _cmd_ssrf(burn_R, gamma_rf, n_steps, rf_mode)
        if c == CMD_AFP:
            return _cmd_afp(burn_R, afp_mode, efficiency)
        if c == CMD_SSRF_PROFILE:
            return _cmd_ssrf_profile(n_steps)
        if c == CMD_AFP_PROFILE:
            return _cmd_afp_profile(efficiency)
        return _snapshot(f'error: unknown cmd {c}')
    except Exception as exc:
        return _snapshot(f'error: {exc}')
