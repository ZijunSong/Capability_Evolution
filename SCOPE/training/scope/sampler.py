"""Sample selection utilities for SCOPE v3 (Round 1: uniform)."""

from __future__ import annotations

import random
from typing import Sequence

from training.scope.schema import DecisionSupervisionSampleV3, Route


def filter_trainable(
    samples: Sequence[DecisionSupervisionSampleV3],
) -> list[DecisionSupervisionSampleV3]:
    return [
        s
        for s in samples
        if s.train_mask and s.route in {Route.ENDORSE, Route.CORRECT} and s.target_action
    ]


def balanced_capability_sample(
    samples: Sequence[DecisionSupervisionSampleV3],
    *,
    n: int,
    seed: int = 0,
) -> list[DecisionSupervisionSampleV3]:
    """Round-robin across capabilities (simple balance)."""
    by_cap: dict[str, list[DecisionSupervisionSampleV3]] = {}
    for s in filter_trainable(samples):
        by_cap.setdefault(s.capability_id, []).append(s)
    rng = random.Random(seed)
    for v in by_cap.values():
        rng.shuffle(v)
    keys = sorted(by_cap.keys())
    out: list[DecisionSupervisionSampleV3] = []
    idx = {k: 0 for k in keys}
    while len(out) < n and keys:
        progress = False
        for k in list(keys):
            i = idx[k]
            if i >= len(by_cap[k]):
                keys.remove(k)
                continue
            out.append(by_cap[k][i])
            idx[k] = i + 1
            progress = True
            if len(out) >= n:
                break
        if not progress:
            break
    return out
