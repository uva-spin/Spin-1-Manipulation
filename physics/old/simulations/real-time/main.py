#!/usr/bin/env python3
"""
Real-time deuterium signal analysis GUI.

Interactive plotting with Voigt-profile ss-RF burning, using shared physics
modules from the repository (lineshape, ssRFMapper, lookup table).

Usage:
    python main.py
"""

import sys

from paths import REPO_ROOT, REALTIME_DIR

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REALTIME_DIR))

from gui import main

if __name__ == "__main__":
    main()
