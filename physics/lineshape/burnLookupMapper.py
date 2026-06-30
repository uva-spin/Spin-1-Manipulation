"""
Apply single-bin ss-RF burns using burn_lookup_table.pkl trajectories.

Each trajectory row stores burn-bin-only Ps/Qs/Iplus/Iminus values produced by
``Data_Creation/burn_lookup_table.py``. Mapping is done by sorting rows on
``ps_at_burn_bin`` at the burn location, pooled across all polarizations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physics.burn_lookup_realtime import build_burn_bin_index
from physics.lineshape.ssRFMapper import ssRFMapper


class BurnLookupMapper:
    """Map burned Ps at one bin to Iplus/Iminus using burn lookup trajectories."""

    def __init__(
        self,
        f: np.ndarray,
        polarization: float,
        burn_bin_idx: int,
        burn_lookup_df: pd.DataFrame,
        *,
        center_freq: float = 0.0,
    ):
        self.f = np.asarray(f, dtype=float)
        self.polarization = float(polarization)
        self.burn_bin_idx = int(burn_bin_idx)
        self.center_freq = float(center_freq)
        self.burn_lookup_df = burn_lookup_df
        self.burn_index: Optional[Dict[str, np.ndarray]] = None

    def compute_lookup_index(self) -> None:
        """Build a Ps -> (Iplus, Iminus) index at the burn bin from all trajectories."""
        self.burn_index = build_burn_bin_index(
            self.burn_lookup_df,
            self.burn_bin_idx,
        )

    @staticmethod
    def _map_ps(ps_values: np.ndarray, target_ps: float) -> int:
        idx = int(np.searchsorted(ps_values, target_ps, side="left"))
        if idx <= 0:
            return 0
        if idx >= len(ps_values):
            return len(ps_values) - 1
        if abs(target_ps - ps_values[idx - 1]) <= abs(target_ps - ps_values[idx]):
            return idx - 1
        return idx

    def map_burn_bin(self, ps_at_burn: float) -> Tuple[float, float]:
        """Return (Iplus, Iminus) at the burn bin for a target Ps value."""
        if self.burn_index is None:
            raise RuntimeError("Call compute_lookup_index() before map_burn_bin().")

        ps_values = self.burn_index["ps_values"]
        nearest_idx = self._map_ps(ps_values, float(ps_at_burn))
        return (
            float(self.burn_index["iplus_values"][nearest_idx]),
            float(self.burn_index["iminus_values"][nearest_idx]),
        )

    def mirror_burn_over_center(
        self, burn_effect: np.ndarray, burn_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        mirrored_burn = np.zeros_like(self.f)
        mirrored_indices = []

        for i, burn_idx in enumerate(burn_indices):
            freq = self.f[burn_idx]
            mirrored_freq = 2 * self.center_freq - freq
            freq_diff = np.abs(self.f - mirrored_freq)
            mirrored_idx = int(np.argmin(freq_diff))

            if freq_diff[mirrored_idx] < (self.f[1] - self.f[0]) / 2:
                mirrored_burn[mirrored_idx] = burn_effect[i]
                mirrored_indices.append(mirrored_idx)

        return mirrored_burn, np.asarray(mirrored_indices, dtype=int)

    def apply_bin_burn(
        self,
        ps: np.ndarray,
        iplus: np.ndarray,
        iminus: np.ndarray,
        bin_idx: int,
        amp: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply a single-bin burn and map Iplus/Iminus at the burn bin from the lookup.

        Input arrays are copied; callers keep their originals unchanged.
        """
        if self.burn_index is None:
            raise RuntimeError("Call compute_lookup_index() before apply_bin_burn().")

        ps = np.asarray(ps, dtype=float).copy()
        iplus = np.asarray(iplus, dtype=float).copy()
        iminus = np.asarray(iminus, dtype=float).copy()
        initial_ps = ps.copy()

        burn_indices = np.array([int(bin_idx)], dtype=int)
        burn_val = float(amp)
        if initial_ps[burn_indices[0]] < 0:
            burn_val = abs(burn_val)
        else:
            burn_val = -abs(burn_val)

        ps_at_burn = ps[burn_indices]
        clipped_burn_values = ssRFMapper._constrain_burn_no_zero_crossing(
            ps_at_burn, np.array([burn_val])
        )

        if np.all(clipped_burn_values == 0):
            return ps, iplus, iminus

        original_iplus_at_burn = iplus[burn_indices].copy()
        original_iminus_at_burn = iminus[burn_indices].copy()

        ps[burn_indices] += clipped_burn_values
        iplus_burn, iminus_burn = self.map_burn_bin(float(ps[burn_indices[0]]))

        iplus_delta = abs(original_iplus_at_burn[0] - iplus_burn)
        iminus_delta = abs(original_iminus_at_burn[0] - iminus_burn)

        iplus[burn_indices] = iplus_burn
        iminus[burn_indices] = iminus_burn

        mirrored_burn, mirrored_indices = self.mirror_burn_over_center(
            np.array([burn_val]), burn_indices
        )
        mirrored_burn /= 2.0

        if np.any(mirrored_indices):
            mirrored_burn[mirrored_indices] *= -1.0
            burn_at_mirrored = mirrored_burn[mirrored_indices].copy()
            ps_at_mirror = ps[mirrored_indices].copy()
            burn_at_mirrored = ssRFMapper._constrain_burn_no_zero_crossing(
                ps_at_mirror, burn_at_mirrored
            )

            iplus_burn_delta = iplus[burn_indices] - original_iplus_at_burn
            iminus_burn_delta = iminus[burn_indices] - original_iminus_at_burn

            iminus[mirrored_indices] -= 0.5 * iplus_burn_delta
            iplus[mirrored_indices] -= 0.5 * iminus_burn_delta

        return ps, iplus, iminus
