"""Equilibrium lineshape helpers for center selection."""
from __future__ import annotations

import numpy as np

from physics.rf.model import Spin1Model, Spin1Params

from .constants import F_MAX, F_MIN, NUM_BINS, PAKE_ASYM, PAKE_GAMMA
from .dulya import active_shape, apply_active_fit


def polarization_grid(p_min, p_max, p_step):
    g = np.arange(p_min, p_max + 1e-12, p_step)
    return g[np.abs(g) >= 1e-12]


def equilibrium_lineshape(P, f):
    f = np.asarray(f, dtype=float)
    if active_shape() is not None:
        fitted = Spin1Model(
            Spin1Params(
                n_bins=int(f.size),
                r_min=float(f[0]),
                r_max=float(f[-1]),
                p0=float(P),
                q0=None,
                line_gamma=PAKE_GAMMA,
                line_asym=PAKE_ASYM,
                rf_enabled=False,
            )
        )
        apply_active_fit(fitted)
        ip, im, total = fitted.physical_intensities()
        return np.asarray(total), np.asarray(ip), np.asarray(im)
    model = Spin1Model(
        Spin1Params(
            n_bins=int(f.size),
            r_min=float(f[0]),
            r_max=float(f[-1]),
            p0=float(P),
            q0=None,
            line_gamma=PAKE_GAMMA,
            line_asym=PAKE_ASYM,
            rf_enabled=False,
        )
    )
    ip, im, total = model.physical_intensities()
    return np.asarray(total), np.asarray(ip), np.asarray(im)


def equilibrium_q_profile(polarization, *, num_bins=NUM_BINS, r_min=None, r_max=None):
    r_lo = F_MIN if r_min is None else r_min
    r_hi = F_MAX if r_max is None else r_max
    f = np.linspace(r_lo, r_hi, num_bins)
    _, ip, im = equilibrium_lineshape(polarization, f)
    return ip - im
