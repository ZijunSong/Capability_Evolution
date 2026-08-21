from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any


def query_disjoint_splits(query_ids: list[str], *, seed: int = 0) -> dict[str, set[str]]:
    ranked = sorted({str(q) for q in query_ids}, key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    n = len(ranked)
    return {"train": set(ranked[: int(0.7 * n)]), "dev": set(ranked[int(0.7 * n): int(0.85 * n)]), "test": set(ranked[int(0.85 * n):])}


def assert_query_disjoint(splits: dict[str, set[str]]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = splits[a] & splits[b]
            if overlap:
                raise AssertionError(f"query split overlap {a}/{b}: {sorted(overlap)[:5]}")


def shuffled_targets_preserve_marginal(rows: list[dict[str, Any]], *, key: str = "projected_action") -> list[dict[str, Any]]:
    targets = [r.get(key) for r in rows]
    if not targets:
        return []
    rotated = targets[1:] + targets[:1]
    out = []
    for row, target in zip(rows, rotated):
        new_row = dict(row)
        new_row[key] = target
        new_row["control"] = "shuffled_target"
        out.append(new_row)
    if Counter(map(str, targets)) != Counter(map(str, [r.get(key) for r in out])):
        raise AssertionError("shuffled target changed marginal distribution")
    return out
