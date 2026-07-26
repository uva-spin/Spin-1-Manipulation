"""Ensure this package directory is importable when scripts are run as files."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
_pkg = str(_PKG_ROOT)
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)
