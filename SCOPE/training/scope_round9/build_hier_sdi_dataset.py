#!/usr/bin/env python3
"""Build hierarchical rollback SDI dataset with local checkpoint IDs.

Train/valid from Round 8 offline SDI (query split).
Frozen live holdout from Round 9 base_live reconstructed states (no tuning).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.checkpoint_candidates import assign_local_checkpoint_ids, global_to_local_id

R8_TRAIN = _REPO / "artifacts/datasets/scope_round8/rollback_sdi/train.jsonl"
R8_VALID = _REPO / "artifacts/datasets/scope_round8/rollback_sdi/valid.jsonl"
FROZEN_BASE_LIVE = _REPO / "artifacts/datasets/scope_round9/frozen_replay/base_live.jsonl"
OUT = _REPO / "artifacts/datasets/scope_round9/hier_sdi"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def enrich_sample(row: dict, *, include_summary: bool = True) -> dict | None:
    ds = row.get("decision_state") or {}
    raw = list(ds.get("available_checkpoints") or [])
    # Frozen replay rows already have candidate_list with local IDs.
    if not raw and row.get("candidate_list"):
        raw = [
            {
                "checkpoint_id": c.get("checkpoint_id"),
                "turn_id": c.get("relative_turn", c.get("turn_id", 0)),
                "n_curated": c.get("evidence_count", 0),
                "n_pool": c.get("n_pool", 0),
                "n_verified": c.get("verified_count", 0),
                "remaining_recovery_budget": c.get("remaining_recovery_budget", "?"),
                "local_checkpoint_id": c.get("local_checkpoint_id"),
            }
            for c in row["candidate_list"]
        ]
    ordered, local_map = assign_local_checkpoint_ids(raw)
    ta = row.get("target_action") or {}
    gold_op = str(
        ta.get("operation")
        or row.get("gold_operation")
        or row.get("operation")
        or "CONTINUE"
    )
    gold_ck = ta.get("checkpoint_id") or row.get("gold_checkpoint_global_id")
    gold_local = global_to_local_id(str(gold_ck) if gold_ck else None, local_map)
    gold_in = gold_op != "ROLLBACK_TO" or gold_local is not None
    if gold_op == "ROLLBACK_TO" and not gold_in:
        return None
    ds = dict(ds)
    ds["available_checkpoints"] = ordered
    ds["checkpoint_local_map"] = local_map
    if not ds.get("rendered_context") and row.get("effective_input_text"):
        # Keep a compact student-visible state; trainer rebuilds effective input.
        ds["rendered_context"] = row.get("student_state_text") or row["effective_input_text"]
    return {
        **{k: v for k, v in row.items() if k not in ("hf_logits", "vllm_logits")},
        "decision_state": ds,
        "student_state_text": row.get("student_state_text") or ds.get("rendered_context") or "",
        "target_action": {"operation": gold_op, "checkpoint_id": gold_ck},
        "operation": gold_op,
        "gold_operation": gold_op,
        "gold_checkpoint_local_id": gold_local,
        "gold_checkpoint_global_id": gold_ck,
        "gold_in_candidates": gold_in,
        "include_candidate_summary": include_summary,
        "capability_id": "rollback_decision",
    }


def split_train_valid(rows: list[dict], rng: random.Random) -> tuple[list, list]:
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[str(r.get("query_id", ""))].append(r)
    train, valid = [], []
    for qid, items in by_q.items():
        (valid if rng.random() < 0.15 else train).extend(items)
    return train, valid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    raw = load_jsonl(R8_TRAIN) + load_jsonl(R8_VALID)

    enriched = []
    dropped = 0
    for row in raw:
        sample = enrich_sample(row, include_summary=True)
        if sample:
            enriched.append(sample)
        else:
            dropped += 1

    train, valid = split_train_valid(enriched, rng)

    holdout_raw = load_jsonl(FROZEN_BASE_LIVE)
    holdout = []
    holdout_dropped = 0
    for row in holdout_raw:
        sample = enrich_sample(row, include_summary=True)
        if sample:
            holdout.append(sample)
        else:
            holdout_dropped += 1

    rb = [r for r in enriched if r.get("gold_operation") == "ROLLBACK_TO"]
    coverage = sum(1 for r in rb if r.get("gold_in_candidates")) / max(len(rb), 1)
    holdout_rb = [r for r in holdout if r.get("gold_operation") == "ROLLBACK_TO"]
    holdout_cov = (
        sum(1 for r in holdout_rb if r.get("gold_in_candidates")) / max(len(holdout_rb), 1)
        if holdout_rb
        else 1.0
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in ("train", train), ("valid", valid), ("frozen_live_holdout", holdout):
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    gate = {
        "n_train": len(train),
        "n_valid": len(valid),
        "n_holdout": len(holdout),
        "dropped_offline": dropped,
        "dropped_holdout": holdout_dropped,
        "candidate_coverage": coverage,
        "holdout_candidate_coverage": holdout_cov,
        "gate_pass": coverage >= 0.99 and holdout_cov >= 0.99,
    }
    (args.output_dir / "DATASET_GATE.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2))
    if not gate["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
