"""Deep-copy harness YAML dict and patch module flags."""

from __future__ import annotations

import copy
from typing import Any


def patch_module(
    harness_cfg: dict[str, Any],
    module_id: str,
    *,
    enabled: bool | None = None,
    flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(harness_cfg)
    section = cfg.setdefault(module_id, {})
    if enabled is not None:
        section["enabled"] = enabled
    if flags:
        section.update(flags)
    return cfg
