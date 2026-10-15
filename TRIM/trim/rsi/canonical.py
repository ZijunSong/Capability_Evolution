"""Canonical JSON and content hashes shared by bank and RSI artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value, key=repr)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(obj: Any) -> str:
    return canonical_bytes(obj).decode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_bytes(obj))


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))
