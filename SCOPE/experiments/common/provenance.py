"""Provenance helpers: refuse silent fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MissingAssetError(RuntimeError):
    """Raised when a required model/checkpoint/manifest/metric asset is missing."""


def require_path(path: str | Path, *, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise MissingAssetError(f"{label} missing: {p} (no silent fallback)")
    return p


def require_checkpoint(path: str | Path | None, *, allow_base: bool = False) -> Path | None:
    if path is None or path == "" or path == "base":
        if allow_base:
            return None
        raise MissingAssetError("checkpoint missing and allow_base=False (no silent Base fallback)")
    return require_path(path, label="checkpoint")


def require_keys(d: dict[str, Any], keys: list[str], *, context: str) -> None:
    missing = [k for k in keys if k not in d or d[k] is None]
    if missing:
        raise MissingAssetError(f"{context}: missing keys {missing} (no silent default)")


def record_explicit_fallback(
    telemetry: list[dict[str, Any]],
    *,
    kind: str,
    reason: str,
    from_value: Any,
    to_value: Any,
) -> None:
    """Only allowed fallback path: must be config-declared and telemetried."""
    telemetry.append(
        {
            "event": "explicit_fallback",
            "kind": kind,
            "reason": reason,
            "from": from_value,
            "to": to_value,
        }
    )
