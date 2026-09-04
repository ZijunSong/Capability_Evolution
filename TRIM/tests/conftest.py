from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

_THIS_DIR = Path(__file__).resolve().parent
_pkg = ModuleType("tests")
_pkg.__path__ = [str(_THIS_DIR)]
sys.modules["tests"] = _pkg
