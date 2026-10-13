"""Ideal, bin-addressed RF schedules, independent of Qt and of the spin model.

Each pulse supplies a nonnegative effective base rate in ONE physical-R bin.
Intervals are half-open [start, start + duration). Times are simulation-time
units relative to pressing Start program. Profiles add where they overlap;
there is no interpolation, spectral leakage, normalization, or optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import csv
import json
import math

import numpy as np

SCHEMA = "ssrf-ideal-bin-program-1"


def _number(value, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        v = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(v) or v < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return v


def _boolean(value, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be true or false")
    return value


@dataclass
class BinPulse:
    bin_index: int
    rate: float = 2.0
    start: float = 0.0
    duration: float = 1.0
    enabled: bool = True

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def validate(self, n_bins: int) -> None:
        if isinstance(self.bin_index, bool) or not isinstance(self.bin_index, (int, np.integer)):
            raise ValueError("bin_index must be an integer (0-based)")
        if not 0 <= self.bin_index < n_bins:
            raise ValueError(f"bin_index {self.bin_index} is outside 0..{n_bins-1}")
        self.bin_index = int(self.bin_index)
        self.rate = _number(self.rate, "RF rate")
        self.start = _number(self.start, "start time")
        self.duration = _number(self.duration, "duration")
        self.enabled = _boolean(self.enabled, "pulse enabled")
        if not math.isfinite(self.stop):
            raise ValueError("start + duration must be finite")
        if self.duration > 0 and self.stop <= self.start:
            raise ValueError("Duration is below floating-point resolution at this start time")


@dataclass
class RFProfile:
    name: str = "Profile"
    enabled: bool = True
    pulses: List[BinPulse] = field(default_factory=list)

    def validate(self, n_bins: int) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Every profile needs a nonempty name")
        self.enabled = _boolean(self.enabled, "profile enabled")
        for pulse in self.pulses:
            pulse.validate(n_bins)


@dataclass
class PulseProgram:
    n_bins: int = 701
    r_min: float = -3.0
    r_max: float = 3.0
    gain: float = 1.0
    profiles: List[RFProfile] = field(default_factory=list)

    @property
    def grid(self) -> np.ndarray:
        return np.linspace(self.r_min, self.r_max, self.n_bins)

    def validate(self) -> None:
        if isinstance(self.n_bins, bool) or not isinstance(self.n_bins, (int, np.integer)) or self.n_bins < 5:
            raise ValueError("n_bins must be an integer >= 5")
        self.r_min = float(self.r_min)
        self.r_max = float(self.r_max)
        if not all(math.isfinite(v) for v in (self.r_min, self.r_max)) or self.r_min >= self.r_max:
            raise ValueError("A finite increasing R grid is required")
        # Exact mirror addressing must not silently interpolate an ideal command.
        if not math.isclose(self.r_min, -self.r_max, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("The ideal-bin program requires a symmetric R grid")
        self.gain = _number(self.gain, "global RF gain")
        for profile in self.profiles:
            profile.validate(self.n_bins)

    def validate_grid(self, grid: np.ndarray) -> None:
        self.validate()
        grid = np.asarray(grid, dtype=float)
        if grid.shape != (self.n_bins,) or not np.allclose(grid, self.grid, rtol=0, atol=1e-12):
            raise ValueError("Program grid does not match model grid; no automatic bin remapping is performed")

    def clone(self) -> "PulseProgram":
        return self.from_dict(self.to_dict())

    def to_dict(self) -> dict:
        self.validate()
        return {"schema": SCHEMA, "units": "simulation_time_and_base_RF_rate", "overlap": "add", **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "PulseProgram":
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ValueError(f"Expected JSON schema {SCHEMA}")
        if data.get("overlap", "add") != "add":
            raise ValueError("Only additive overlap is supported")
        try:
            profiles = [RFProfile(name=p["name"], enabled=p.get("enabled", True),
                                  pulses=[BinPulse(**row) for row in p.get("pulses", [])])
                        for p in data.get("profiles", [])]
            program = cls(n_bins=data["n_bins"], r_min=data["r_min"], r_max=data["r_max"],
                          gain=data.get("gain", 1.0), profiles=profiles)
            program.validate()
            return program
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Malformed RF program: {exc}") from exc

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path) -> "PulseProgram":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_csv(self, path) -> None:
        """Long-form editable table. Grid and gain are repeated for safe round trips."""
        self.validate()
        columns = ["profile_id", "profile", "profile_enabled", "enabled", "bin_index", "R",
                   "rate", "start", "duration", "n_bins", "r_min", "r_max", "gain"]
        grid = self.grid
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for i, profile in enumerate(self.profiles):
                # Empty profiles have no exposure and are not represented in CSV.
                for p in profile.pulses:
                    writer.writerow(dict(profile_id=i, profile=profile.name,
                        profile_enabled=int(profile.enabled), enabled=int(p.enabled), bin_index=p.bin_index,
                        R=repr(float(grid[p.bin_index])), rate=repr(p.rate), start=repr(p.start),
                        duration=repr(p.duration), n_bins=self.n_bins, r_min=self.r_min,
                        r_max=self.r_max, gain=self.gain))

    @classmethod
    def load_csv(cls, path) -> "PulseProgram":
        with Path(path).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("CSV has no pulse rows; use JSON for an empty program")
        def flag(text):
            if str(text).lower() in ("1", "true"): return True
            if str(text).lower() in ("0", "false"): return False
            raise ValueError("CSV enable flags must be 0/1 or true/false")
        try:
            first = rows[0]
            p = cls(n_bins=int(first["n_bins"]), r_min=float(first["r_min"]),
                    r_max=float(first["r_max"]), gain=float(first["gain"]))
            p.validate()
            groups = {}
            for row in rows:
                signature = (int(row["n_bins"]), float(row["r_min"]), float(row["r_max"]), float(row["gain"]))
                if signature != (p.n_bins, p.r_min, p.r_max, p.gain):
                    raise ValueError("Grid/gain columns must be consistent throughout the CSV")
                pid = row["profile_id"]
                name, enabled = row["profile"], flag(row["profile_enabled"])
                if pid not in groups:
                    groups[pid] = RFProfile(name=name, enabled=enabled)
                    p.profiles.append(groups[pid])
                if groups[pid].name != name or groups[pid].enabled != enabled:
                    raise ValueError("Profile name/enabled must be consistent within a profile_id")
                pulse = BinPulse(int(row["bin_index"]), float(row["rate"]), float(row["start"]),
                                 float(row["duration"]), flag(row["enabled"]))
                pulse.validate(p.n_bins)
                if abs(float(row["R"]) - p.grid[pulse.bin_index]) > 1e-10:
                    raise ValueError("CSV R does not match bin_index on the saved grid")
                groups[pid].pulses.append(pulse)
            p.validate()
            return p
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing or invalid CSV column: {exc}") from exc

    def compile(self) -> "CompiledProgram":
        self.validate()
        return CompiledProgram(self)


class CompiledProgram:
    """Sparse event list; only recompute the bin vector at a pulse boundary.

    No dense time-by-frequency array is allocated. Array values are reconstructed
    from active pulses at each event rather than repeatedly added/subtracted,
    avoiding residual RF after the last pulse ends.
    """
    def __init__(self, program: PulseProgram):
        active = [p for profile in program.profiles if profile.enabled for p in profile.pulses
                  if p.enabled and p.rate > 0 and p.duration > 0]
        self.n_bins = program.n_bins
        self.bins = np.array([p.bin_index for p in active], dtype=np.int64)
        self.rates = np.array([p.rate for p in active], dtype=float)
        self.starts = np.array([p.start for p in active], dtype=float)
        self.stops = np.array([p.stop for p in active], dtype=float)
        self.times = np.unique(np.concatenate(([0.0], self.starts, self.stops)))
        self.end_time = float(self.times[-1])
        self.envelope = np.zeros(self.n_bins)
        self.exposure = np.bincount(self.bins, weights=self.rates*(self.stops-self.starts), minlength=self.n_bins)
        self._last_index = -999
        self._last_field = np.zeros(self.n_bins)
        # True per-bin maximum commanded rate over time (not a sum of disjoint pulses).
        for i in range(len(self.times)):
            np.maximum(self.envelope, self.field_for_index(i), out=self.envelope)
        if not np.all(np.isfinite(self.envelope)) or not np.all(np.isfinite(self.exposure)):
            raise ValueError("Program rates or integrated exposures overflow floating point")
        self._last_index = -999

    def index_at(self, time: float) -> int:
        return int(np.searchsorted(self.times, time, side="right")) - 1

    def field_for_index(self, index: int) -> np.ndarray:
        if self._last_index == index:
            return self._last_field
        if index < 0 or index >= len(self.times)-1:
            out = np.zeros(self.n_bins)
        else:
            t = self.times[index]
            mask = (self.starts <= t) & (t < self.stops)
            out = np.bincount(self.bins[mask], weights=self.rates[mask], minlength=self.n_bins)
        self._last_index = index
        self._last_field = out
        return out

    def field_at(self, time: float) -> np.ndarray:
        return self.field_for_index(self.index_at(time)).copy()


def make_profile(grid, name: str, r_left: float, r_right: float,
                 shape: str = "flat", rate: float = 2.0, rate_right: Optional[float] = None,
                 start: float = 0.0, duration: float = 1.0, start_step: float = 0.0,
                 duration_right: Optional[float] = None) -> RFProfile:
    """Generate editable per-bin pulses; no shape normalization during execution.

    Bounds select the nearest grid centers inclusively. The generated rate is
    the peak for triangle/Gaussian, the left endpoint for a ramp. A Gaussian
    here is merely an editable command shape with exactly zero commands outside
    the selected bins; it is NOT a Voigt/material response.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 5 or not np.all(np.diff(grid)>0):
        raise ValueError("grid must be a strictly increasing 1-D grid")
    if not all(math.isfinite(float(x)) for x in (r_left, r_right)) or r_left > r_right:
        raise ValueError("Left R must be finite and <= right R")
    if r_left < grid[0] or r_right > grid[-1]:
        raise ValueError("Region is outside the simulation grid")
    lo, hi = (int(np.argmin(abs(grid-r))) for r in (r_left, r_right))
    bins = np.arange(lo, hi+1)
    x = np.linspace(0, 1, len(bins)) if len(bins)>1 else np.array([0.5])
    rate = _number(rate, "rate")
    end_rate = rate if rate_right is None else _number(rate_right, "right rate")
    if shape == "flat": values = np.full(len(bins), rate)
    elif shape == "linear": values = rate + (end_rate-rate)*x if len(bins)>1 else np.array([rate])
    elif shape in ("triangle", "gaussian"):
        y = (1-abs(2*x-1)) if shape == "triangle" else np.exp(-0.5*((x-0.5)/(1/6))**2)
        if float(y.max()) <= 0: y = np.ones_like(x)
        values = rate*(y/y.max())
    else: raise ValueError("shape must be flat, linear, triangle, or gaussian")
    start = _number(start, "start")
    start_step = _number(start_step, "start step")
    duration = _number(duration, "duration")
    dend = duration if duration_right is None else _number(duration_right, "right duration")
    durations = duration + (dend-duration)*x if len(bins)>1 else np.array([duration])
    profile = RFProfile(name=name, pulses=[BinPulse(int(b), float(v), start+i*start_step, float(d))
                                         for i,(b,v,d) in enumerate(zip(bins,values,durations))])
    profile.validate(len(grid))
    return profile


def example_program(grid) -> PulseProgram:
    """Two manually chosen profiles, not optimized or experimentally calibrated."""
    grid = np.asarray(grid)
    program = PulseProgram(len(grid), float(grid[0]), float(grid[-1]))
    program.profiles = [
        make_profile(grid, "Horn: short triangular pulse", -0.98, -0.72,
                     "triangle", 2.0, start=0.5, duration=0.5),
        make_profile(grid, "Pedestal: longer ramp", 1.15, 1.65,
                     "linear", 1.0, 3.0, start=0.8, duration=1.0, duration_right=1.8),
    ]
    program.validate()
    return program
