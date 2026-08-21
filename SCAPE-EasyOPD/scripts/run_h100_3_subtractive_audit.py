from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

_SCAPE_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCAPE")
_MODULE_PATH = _SCAPE_ROOT / "scripts" / "run_h100_3_subtractive_audit.py"

if _MODULE_PATH.exists():
    spec = importlib.util.spec_from_file_location("scape_run_h100_3_subtractive_audit", _MODULE_PATH)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, module)
        spec.loader.exec_module(module)
        globals().update({k: getattr(module, k) for k in dir(module) if not k.startswith("_")})
