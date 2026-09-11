"""Policy version binding via adapter and template content hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def adapter_digest(adapter_dir: Path | str | None) -> str | None:
    """Content hash for adapter weights + config. None when no adapter on disk."""
    if not adapter_dir:
        return None
    root = Path(adapter_dir)
    weight = root / "adapter_model.safetensors"
    cfg = root / "adapter_config.json"
    if not weight.is_file():
        return None
    parts = [_file_sha256(weight)]
    if cfg.is_file():
        parts.append(_file_sha256(cfg))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def policy_digest_record(
    *,
    adapter_dir: Path | str | None,
    base_model: str,
    policy_version: str,
    template_hint: str = "",
) -> dict[str, Any]:
    ad = adapter_digest(adapter_dir)
    base_key = hashlib.sha256(str(base_model).encode()).hexdigest()[:12]
    tpl = hashlib.sha256(template_hint.encode()).hexdigest()[:8] if template_hint else ""
    digest = hashlib.sha256(f"{base_key}:{ad or 'base'}:{tpl}".encode()).hexdigest()[:16]
    return {
        "policy_version": policy_version,
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "adapter_digest": ad,
        "base_model_key": base_key,
        "policy_digest": digest,
    }


def assert_policy_digest(expected: dict[str, Any] | None, actual: dict[str, Any], *, what: str) -> None:
    if not expected:
        return
    exp = str(expected.get("policy_digest") or "")
    got = str(actual.get("policy_digest") or "")
    if exp and got and exp != got:
        raise RuntimeError(
            f"{what}: policy_digest mismatch expected={exp} got={got} "
            f"(adapter={actual.get('adapter_dir')})"
        )
