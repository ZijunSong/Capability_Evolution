"""Shared primitive types, canonical serialization and digest helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping, Sequence, Union

JSONValue = Union[
    None, bool, int, float, str, Sequence["JSONValue"], Mapping[str, "JSONValue"]
]

DIGEST_HEX_LEN = 64


def canonical_json(obj: Any) -> str:
    """Deterministic JSON with sorted keys and no NaN/Inf floats.

    Used for every digest computed in RSI. Raises on non-JSON values so that
    silently lossy encodings never enter an identity hash.
    """

    def _default(o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        raise TypeError(f"cannot canonically encode {type(o).__name__}")

    def _scrub(x: Any) -> Any:
        if isinstance(x, float) and not math.isfinite(x):
            raise ValueError("non-finite float in canonical json")
        if isinstance(x, Mapping):
            return {str(k): _scrub(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [_scrub(v) for v in x]
        return x

    return json.dumps(_scrub(obj), sort_keys=True, separators=(",", ":"), default=_default, ensure_ascii=False)


def digest_of(obj: Any) -> str:
    """sha256 hex digest of the canonical JSON encoding of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def digest_of_text(text: str) -> str:
    """sha256 hex digest of raw text (source files, prompts)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def short(digest: str, n: int = 12) -> str:
    return digest[:n]
