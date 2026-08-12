"""A7: DecisionState field ablation + collision / entropy analysis."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any

DUP_FIELD_MASKS = {
    "a7_dup_no_candidate_text": ["candidate_text"],
    "a7_dup_no_candidate_id": ["candidate_id"],
    "a7_dup_no_curated_history": ["curated_history"],
    "a7_dup_no_dup_statistics": ["dup_statistics"],
    "a7_dup_no_query_context": ["query", "query_context"],
    "a7_dup_minimal_sufficient_state": [
        "candidate_text",
        "curated_history",
        "dup_statistics",
        "query_context",
    ],
}

RB_FIELD_MASKS = {
    "a7_rb_no_checkpoint_registry": ["checkpoint_registry"],
    "a7_rb_no_checkpoint_metadata": ["checkpoint_metadata"],
    "a7_rb_no_recovery_budget": ["recovery_budget"],
    "a7_rb_no_failure_history": ["failure_history"],
    "a7_rb_no_previous_operation": ["previous_operation"],
    "a7_rb_no_candidate_checkpoint_id": ["candidate_checkpoint_id"],
}


def drop_fields(state: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    out = dict(state)
    for f in fields:
        out.pop(f, None)
    return out


def effective_input_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def analyze_field_ablation(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    label_key: str = "label",
    max_tokens_fn=None,
) -> dict[str, Any]:
    masks = {**DUP_FIELD_MASKS, **RB_FIELD_MASKS}
    if variant not in masks:
        raise ValueError(f"unknown A7 variant: {variant}")
    fields = masks[variant]
    groups: dict[str, list[str]] = defaultdict(list)
    supports: Counter[str] = Counter()
    token_lens: list[int] = []
    trunc = 0
    for r in rows:
        st = drop_fields(dict(r.get("state") or r), fields)
        h = effective_input_hash(st)
        lab = str(r.get(label_key) or r.get("operation") or "")
        groups[h].append(lab)
        supports[lab] += 1
        if max_tokens_fn is not None:
            ntok = int(max_tokens_fn(st))
        else:
            ntok = len(json.dumps(st, ensure_ascii=False).split())
        token_lens.append(ntok)
        if r.get("truncated"):
            trunc += 1

    exact_collisions = sum(1 for labs in groups.values() if len(labs) > 1)
    conflicting = 0
    entropies = []
    for labs in groups.values():
        c = Counter(labs)
        if len(c) > 1:
            conflicting += 1
        n = sum(c.values())
        ent = -sum((v / n) * math.log(v / n + 1e-12) for v in c.values())
        entropies.append(ent)

    unidentifiable = conflicting > 0
    return {
        "variant": variant,
        "dropped_fields": fields,
        "n_rows": len(rows),
        "exact_effective_input_collision": exact_collisions,
        "conflicting_label_collision": conflicting,
        "unidentifiable": unidentifiable,
        "mean_conditional_label_entropy": sum(entropies) / max(len(entropies), 1),
        "class_support": dict(supports),
        "max_token_length": max(token_lens) if token_lens else 0,
        "mean_token_length": sum(token_lens) / max(len(token_lens), 1),
        "truncation_rate": trunc / max(len(rows), 1),
        "note": (
            "不可识别：字段移除导致 conflicting-label collision，"
            "低性能不得解释为优化失败"
            if unidentifiable
            else "identifiable"
        ),
    }
