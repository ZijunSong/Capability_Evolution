"""Pytest path setup: tests import the installed ``rsi`` package only."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the tests directory is importable for helpers; the package itself must
# come from the installed distribution (test_standalone_install).
sys.path.insert(0, str(Path(__file__).resolve().parent))
