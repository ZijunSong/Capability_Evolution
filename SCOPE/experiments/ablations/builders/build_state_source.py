"""A2: On-policy state collection source builders + diagnostics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def state_distribution_diagnostics(states: list[dict[str, Any]]) -> dict[str, Any]:
    if not states:
        raise ValueError("empty states")
    actions = Counter(str(s.get("action") or s.get("label") or "UNK") for s in states)
    turns = Counter(int(s.get("turn") or 0) for s in states)
    cand_counts = [int(s.get("candidate_count") or len(s.get("candidates") or []) or 1) for s in states]
    labels = Counter(str(s.get("label") or s.get("operation") or "UNK") for s in states)
    hashes = [_hash(s) for s in states]
    unique_hashes = set(hashes)
    collisions = len(hashes) - len(unique_hashes)
    queries = {str(s.get("query_id")) for s in states}
    return {
        "n_states": len(states),
        "action_distribution": dict(actions),
        "turn_distribution": {str(k): v for k, v in sorted(turns.items())},
        "mean_candidate_count": sum(cand_counts) / len(cand_counts),
        "label_support": dict(labels),
        "exact_collision": collisions,
        "unique_state_hashes": len(unique_hashes),
        "unique_query_count": len(queries),
    }


def build_states(variant: str, *, n: int = 16, seed: int = 42) -> list[dict[str, Any]]:
    """Synthetic / path-agnostic state source for smoke; production wraps Round3/6 collectors."""
    source_map = {
        "a2_current_student_on_policy": "current_student",
        "a2_base_model_states": "base_model",
        "a2_full_harness_states": "full_harness",
        "a2_stale_checkpoint_states": "stale_checkpoint",
        "a2_mixed_replay_states": "mixed_replay",
    }
    if variant not in source_map:
        raise ValueError(f"unknown A2 variant: {variant}")
    src = source_map[variant]
    rows = []
    for i in range(n):
        rows.append(
            {
                "query_id": f"q{(i + seed) % max(n // 2, 1)}",
                "turn": i % 8,
                "action": "KEEP_EVIDENCE" if i % 3 else "SKIP_DUPLICATE",
                "label": "KEEP_EVIDENCE" if i % 3 else "SKIP_DUPLICATE",
                "candidate_count": 1 + (i % 5),
                "candidate_text": f"{src} doc {i}",
                "state_source": src,
            }
        )
    if variant == "a2_mixed_replay_states":
        for i, r in enumerate(rows):
            r["state_source"] = ["current_student", "base_model", "stale_checkpoint"][i % 3]
    return rows


def build_and_report(variant: str, output_dir: Path, *, n: int = 16, seed: int = 42) -> dict[str, Any]:
    states = build_states(variant, n=n, seed=seed)
    diag = state_distribution_diagnostics(states)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "states.jsonl").open("w", encoding="utf-8") as f:
        for r in states:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (output_dir / "state_diagnostics.json").write_text(
        json.dumps(diag, indent=2) + "\n", encoding="utf-8"
    )
    return diag
