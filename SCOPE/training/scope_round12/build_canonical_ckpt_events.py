#!/usr/bin/env python3
"""A1 — freeze unified rollback-positive event set for Stage2 provenance."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.checkpoint_candidates import (
    assign_local_checkpoint_ids,
    order_checkpoint_candidates,
    summarize_candidate,
)
from training.scope_round11.stage1_views import build_stage2_prompt

BASE_LIVE = _REPO / "artifacts" / "datasets" / "scope_round10" / "frozen_replay" / "base_live.jsonl"
HOLDOUT = _REPO / "artifacts" / "datasets" / "scope_round9" / "hier_sdi" / "frozen_live_holdout.jsonl"
OUT_JSONL = _REPO / "artifacts" / "datasets" / "scope_round12" / "ckpt_canonical_base_live.jsonl"
OUT_COPY = _REPO / "outputs" / "scope_round12" / "phase_a_ckpt_provenance" / "CANONICAL_CKPT_EVENTS.jsonl"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm_cands(raw: list[dict]) -> list[dict]:
    out = []
    for c in raw:
        cid = c.get("checkpoint_id")
        if not cid:
            continue
        out.append(
            {
                "checkpoint_id": str(cid),
                "turn_id": int(c.get("turn_id", c.get("relative_turn", 0))),
                "relative_turn": int(c.get("relative_turn", c.get("turn_id", 0))),
                "n_curated": int(c.get("n_curated", c.get("evidence_count", 0))),
                "n_pool": int(c.get("n_pool", 0)),
                "n_verified": int(c.get("n_verified", c.get("verified_count", 0))),
                "evidence_count": int(c.get("evidence_count", c.get("n_curated", 0))),
                "verified_count": int(c.get("verified_count", c.get("n_verified", 0))),
                "remaining_recovery_budget": c.get("remaining_recovery_budget", "?"),
                "state_hash": c.get("state_hash"),
            }
        )
    enriched, _ = assign_local_checkpoint_ids(out)
    return enriched


def main() -> None:
    # Prefer frozen_replay rows (already have candidate_list + gold); enrich with stage2 text.
    events = []
    with BASE_LIVE.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("gold_operation") != "ROLLBACK_TO":
                continue
            cands = _norm_cands(list(row.get("candidate_list") or []))
            # Enforce deterministic order equal to order_checkpoint_candidates
            assert [c["checkpoint_id"] for c in cands] == [
                c["checkpoint_id"] for c in order_checkpoint_candidates(cands)
            ]
            gold = row.get("gold_checkpoint_global_id")
            gold_idx = next((i for i, c in enumerate(cands) if c["checkpoint_id"] == gold), None)
            renderer_parts = [summarize_candidate(c) for c in cands]
            renderer_text = "\n".join(renderer_parts)
            sample = {
                **row,
                "candidate_list": cands,
                "available_checkpoints": cands,
                "decision_state": {
                    "available_checkpoints": cands,
                    "rendered_context": row.get("effective_input_text", ""),
                },
            }
            stage2 = build_stage2_prompt(sample)
            event = {
                "event_id": row.get("event_id") or f"{row.get('query_id')}:{row.get('turn')}:{idx}",
                "query_id": row.get("query_id"),
                "turn": row.get("turn"),
                "split": "base_live",
                "gold_operation": "ROLLBACK_TO",
                "decision_state_hash": _sha(json.dumps({"turn": row.get("turn"), "qid": row.get("query_id")}, sort_keys=True)),
                "candidate_ids": [c["checkpoint_id"] for c in cands],
                "candidate_state_hashes": [c.get("state_hash") for c in cands],
                "candidate_count": len(cands),
                "gold_checkpoint_id": gold,
                "gold_checkpoint_index": gold_idx,
                "gold_checkpoint_local_id": None if gold_idx is None else f"C{gold_idx}",
                "candidate_renderer_text": renderer_text,
                "renderer_hash": _sha(renderer_text),
                "stage2_text": stage2,
                "candidate_list": cands,
                "effective_input_text": row.get("effective_input_text"),
                "prompt_sha256": row.get("prompt_sha256"),
            }
            events.append(event)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_COPY.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    OUT_COPY.write_bytes(OUT_JSONL.read_bytes())

    cov = sum(1 for e in events if e["gold_checkpoint_index"] is not None) / max(len(events), 1)
    summary = {
        "n_total_gold_rollback": len(events),
        "coverage": cov,
        "mean_candidate_count": sum(e["candidate_count"] for e in events) / max(len(events), 1),
        "path": str(OUT_JSONL),
    }
    print(json.dumps(summary, indent=2))
    assert len(events) == 750, f"expected 750 gold rollback events, got {len(events)}"
    assert cov == 1.0, f"coverage not 1.0: {cov}"


if __name__ == "__main__":
    main()
