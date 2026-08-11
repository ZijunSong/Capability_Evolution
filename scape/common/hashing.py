"""Stable hashing utilities for manifests and splits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_rank(query_id: str, *, seed: int | str) -> str:
    """Deterministic rank key: sha256(f'{seed}:{qid}')."""
    return sha256_text(f"{seed}:{query_id}")


def stable_split(
    query_ids: Iterable[str],
    *,
    seed: int | str,
    n_take: int | None = None,
    fractions: tuple[float, ...] | None = None,
) -> list[list[str]]:
    """Sort by stable_rank then slice.

    If n_take is set, returns [taken, remainder].
    If fractions is set (sum<=1), returns consecutive slices.
    """
    ranked = sorted(query_ids, key=lambda q: stable_rank(q, seed=seed))
    if n_take is not None:
        return [ranked[:n_take], ranked[n_take:]]
    if not fractions:
        return [ranked]
    out: list[list[str]] = []
    start = 0
    n = len(ranked)
    for i, frac in enumerate(fractions):
        if i == len(fractions) - 1:
            end = n
        else:
            end = start + int(round(frac * n))
        out.append(ranked[start:end])
        start = end
    return out
