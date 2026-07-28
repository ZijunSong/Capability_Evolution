"""Load SCOPE YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_scope_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    base_path = root / "configs" / "scope" / "base.yaml"
    cfg: dict[str, Any] = {}
    if yaml is None:
        return {"scope": {"enabled": True}}
    if base_path.exists():
        with base_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        with p.open("r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)
    return cfg


def scope_section(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("scope") or {})
