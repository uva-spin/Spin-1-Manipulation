"""Flexible RF / Voigt knobs: scalar, (lo, hi) range, or list of values."""
from __future__ import annotations

from itertools import product
from numbers import Real
from typing import Any, Mapping

import numpy as np

# Sentinel: sample this range once per burn (not expanded as a grid).
_RANGE = object()


def is_range_spec(spec: Any) -> bool:
    """True for a length-2 tuple/list of numbers meaning sample in [lo, hi].

    Lists are discrete grids; only ``tuple`` of length 2 is a sample range.
    """
    return isinstance(spec, tuple) and len(spec) == 2 and all(isinstance(x, Real) for x in spec)


def is_list_spec(spec: Any) -> bool:
    """True for an explicit discrete list (not a range tuple)."""
    return isinstance(spec, list)


def describe_param(spec: Any) -> str:
    """Short human-readable description for logs."""
    if is_range_spec(spec):
        lo, hi = spec
        return f'range[{lo}, {hi}]'
    if is_list_spec(spec):
        return f'grid{list(spec)}'
    return f'fixed={spec}'


def iteration_values(spec: Any) -> list[Any] | object:
    """Scalar → ``[v]``; list → that list; range tuple → ``_RANGE`` sentinel."""
    if is_range_spec(spec):
        return _RANGE
    if is_list_spec(spec):
        if not spec:
            raise ValueError('parameter list must be non-empty')
        return list(spec)
    if isinstance(spec, (np.ndarray,)):
        raise TypeError('use a Python list for discrete grids, or a 2-tuple for a sample range')
    return [spec]


def resolve_for_polarization(spec: Any, rng: np.random.Generator, *, as_int: bool = False) -> float | int:
    """Return one concrete draw. Caller samples once per burn and holds it for every step.

    - scalar: that value
    - 2-tuple range: uniform sample in ``[lo, hi]`` (ints if ``as_int``)
    - list: not valid here — expand via ``expand_param_product`` first
    """
    if is_list_spec(spec):
        raise TypeError('list specs must be expanded with expand_param_product before per-p resolve')
    if is_range_spec(spec):
        lo, hi = float(spec[0]), float(spec[1])
        if lo > hi:
            lo, hi = hi, lo
        if as_int:
            lo_i, hi_i = int(round(lo)), int(round(hi))
            if lo_i > hi_i:
                lo_i, hi_i = hi_i, lo_i
            return int(rng.integers(lo_i, hi_i + 1))
        return float(rng.uniform(lo, hi))
    if as_int:
        return int(spec)
    return float(spec)


def expand_param_product(specs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product over list-valued knobs; leave scalars and ranges intact.

    Parameters
    ----------
    specs :
        Mapping of name → scalar | (lo, hi) | list.

    Returns
    -------
    list of dicts
        Each dict has the same keys. List knobs are replaced by one concrete
        value per product cell; range tuples and scalars are copied as-is.
    """
    names = list(specs.keys())
    axes: list[list[Any]] = []
    for name in names:
        vals = iteration_values(specs[name])
        if vals is _RANGE:
            axes.append([specs[name]])  # keep the range tuple itself
        else:
            axes.append(vals)
    return [dict(zip(names, combo)) for combo in product(*axes)]


def ensure_rng(seed: int | None | np.random.Generator = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
