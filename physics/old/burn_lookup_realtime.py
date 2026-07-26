"""
Realtime burn trajectories and lookup-index builders for ss-RF calibration.

Each trajectory applies repeated single-bin RF burns via
``ssrf_realtime.rate_equations_realtime.solve_rate_equations`` until Ps at the
burn bin is near zero, or the configured ``max_steps`` is reached. Every
trajectory emits exactly ``max_steps + 1`` rows (steps ``0 .. max_steps``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from physics.lineshape.Lineshape import GenerateVectorLineshape
from physics.ssrf_realtime import Spin1Params
from physics.ssrf_realtime.rate_equations_realtime import burn_preserves_ps_sign, solve_rate_equations

__all__ = [
    "BurnTrajectoryConfig",
    "initial_lineshape",
    "burn_trajectory_realtime",
    "build_burn_bin_index",
    "trajectory_row",
]


@dataclass
class BurnTrajectoryConfig:
    """Parameters for a single-bin realtime burn trajectory."""

    n_bins: int = 500
    f_min: float = -3.0
    f_max: float = 3.0
    gamma_rf: float = 2.0
    dt: float = 0.0015
    steps: int = 50
    max_steps: int = 500
    near_zero_frac: float = 1e-3
    stall_rtol: float = 1e-5
    store_dtype: type = np.float32
    burn_r_min: float = -2.0
    burn_r_max: float = 2.0

    def spin1_params(self, polarization: float) -> Spin1Params:
        return Spin1Params(
            n_bins=self.n_bins,
            r_min=self.f_min,
            r_max=self.f_max,
            p0=float(polarization),
            dnp_enabled=False,
            t1_rate=0.0,
            gamma_rf=self.gamma_rf,
            dt=self.dt,
            steps=self.steps,
        )

    @property
    def f(self) -> np.ndarray:
        return np.linspace(self.f_min, self.f_max, self.n_bins)

    def burn_bin_indices(self) -> np.ndarray:
        """Bin indices whose center frequency R lies in (burn_r_min, burn_r_max)."""
        f = self.f
        return np.flatnonzero((f > self.burn_r_min) & (f < self.burn_r_max))


def initial_lineshape(
    polarization: float,
    config: Optional[BurnTrajectoryConfig] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(f, Iplus, Iminus, Ps)`` normalized like ``lookup_table.py``."""

    cfg = config or BurnTrajectoryConfig()
    f = cfg.f.copy()
    ps, iplus, iminus = GenerateVectorLineshape(float(polarization), f)
    return f, np.asarray(iplus, dtype=float), np.asarray(iminus, dtype=float), np.asarray(ps, dtype=float)


def _burn_bin_values(
    iplus: np.ndarray,
    iminus: np.ndarray,
    burn_bin_idx: int,
    *,
    store_dtype: type = np.float32,
) -> tuple[Any, Any, Any, Any, float]:
    """Extract Ps, Qs, Iplus, and Iminus at the burn bin only."""
    burn_idx = int(burn_bin_idx)
    iplus_val = store_dtype(np.asarray(iplus, dtype=float)[burn_idx])
    iminus_val = store_dtype(np.asarray(iminus, dtype=float)[burn_idx])
    ps_val = store_dtype(float(iplus_val) + float(iminus_val))
    qs_val = store_dtype(float(iplus_val) - float(iminus_val))
    return ps_val, qs_val, iplus_val, iminus_val, float(ps_val)


def trajectory_row(
    polarization: float,
    burn_bin_idx: int,
    burn_step: int,
    burn_freq: float,
    iplus: np.ndarray,
    iminus: np.ndarray,
    *,
    store_dtype: type = np.float32,
) -> Dict[str, Any]:
    """Build one lookup-table row with burn-bin-only Ps/Qs/Iplus/Iminus values."""
    ps_val, qs_val, iplus_val, iminus_val, ps_at_burn = _burn_bin_values(
        iplus,
        iminus,
        burn_bin_idx,
        store_dtype=store_dtype,
    )
    return {
        "P": float(polarization),
        "burn_bin_idx": int(burn_bin_idx),
        "burn_step": int(burn_step),
        "burn_freq": float(burn_freq),
        "ps_at_burn_bin": ps_at_burn,
        "Ps": ps_val,
        "Qs": qs_val,
        "Iplus": iplus_val,
        "Iminus": iminus_val,
    }


def burn_trajectory_realtime(
    iplus0: np.ndarray,
    iminus0: np.ndarray,
    burn_idx: int,
    polarization: float,
    burn_freq: float,
    *,
    config: Optional[BurnTrajectoryConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Repeated single-bin RF steps for exactly ``max_steps`` burn iterations.

    Returns ``max_steps + 1`` rows with ``burn_step=0 .. max_steps``. When the
    burn can no longer progress, later rows repeat the last valid state.
    """
    cfg = config or BurnTrajectoryConfig()
    burn_idx = int(burn_idx)
    params = cfg.spin1_params(polarization)

    iplus = np.asarray(iplus0, dtype=float).copy()
    iminus = np.asarray(iminus0, dtype=float).copy()
    ps0_at_burn = float((iplus + iminus)[burn_idx])
    abs_ps0 = abs(ps0_at_burn)
    stall_scale = abs_ps0 if abs_ps0 > 0.0 else 1.0

    rows: List[Dict[str, Any]] = [
        trajectory_row(
            polarization,
            burn_idx,
            0,
            burn_freq,
            iplus,
            iminus,
            store_dtype=cfg.store_dtype,
        )
    ]

    can_burn = abs_ps0 > 0.0

    for step in range(1, cfg.max_steps + 1):
        if can_burn:
            ps_before = float(iplus[burn_idx] + iminus[burn_idx])
            iplus_new, iminus_new, _, _, _ = solve_rate_equations(
                iplus,
                iminus,
                cfg.dt,
                cfg.gamma_rf,
                burn_idx,
                params=params,
                rf_only=True,
            )
            iplus_new = np.asarray(iplus_new, dtype=float)
            iminus_new = np.asarray(iminus_new, dtype=float)
            ps_after = float(iplus_new[burn_idx] + iminus_new[burn_idx])

            if burn_preserves_ps_sign(iplus, iminus, iplus_new, iminus_new, burn_idx) and ps_after != ps_before:
                iplus = iplus_new
                iminus = iminus_new
                if abs(ps_after) <= cfg.near_zero_frac * abs_ps0:
                    can_burn = False
                elif abs(ps_after - ps_before) < cfg.stall_rtol * stall_scale:
                    can_burn = False
            else:
                can_burn = False

        rows.append(
            trajectory_row(
                polarization,
                burn_idx,
                step,
                burn_freq,
                iplus,
                iminus,
                store_dtype=cfg.store_dtype,
            )
        )

    return rows


def _column_at_burn_bin(series: pd.Series, burn_bin_idx: int) -> np.ndarray:
    """Vectorized burn-bin scalar extraction from scalar or legacy spectrum columns."""
    sample = np.asarray(series.iloc[0], dtype=float)
    if sample.ndim == 0 or sample.size == 1:
        return series.to_numpy(dtype=float)
    stacked = np.stack([np.asarray(value, dtype=float) for value in series.to_numpy()])
    if stacked.ndim == 1:
        return stacked
    return stacked[:, int(burn_bin_idx)]


def build_burn_bin_index(
    df: pd.DataFrame,
    burn_bin_idx: int,
    *,
    polarization: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    For a fixed burn location, sort rows by ``ps_at_burn_bin``.

    When ``polarization`` is given, restrict to that ``P`` trajectory.
    Otherwise build a Ps-only index from all polarizations at this bin.
    """
    burn_bin_idx = int(burn_bin_idx)
    mask = df["burn_bin_idx"].to_numpy() == burn_bin_idx
    if polarization is not None:
        mask &= np.isclose(df["P"].to_numpy(dtype=float), float(polarization))
    if not np.any(mask):
        raise ValueError(
            f"No rows for burn_bin_idx={burn_bin_idx}"
            + (f" and P={polarization}" if polarization is not None else "")
        )

    subset = df.loc[mask]
    ps_values = subset["ps_at_burn_bin"].to_numpy(dtype=float)
    iplus_values = _column_at_burn_bin(subset["Iplus"], burn_bin_idx)
    iminus_values = _column_at_burn_bin(subset["Iminus"], burn_bin_idx)
    order = np.argsort(ps_values)
    ps_values = ps_values[order]
    iplus_values = iplus_values[order]
    iminus_values = iminus_values[order]

    return {
        "burn_bin_idx": burn_bin_idx,
        "polarization": float(polarization) if polarization is not None else np.nan,
        "ps_values": ps_values,
        "iplus_values": iplus_values,
        "iminus_values": iminus_values,
    }
