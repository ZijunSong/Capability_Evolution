#!/usr/bin/env python3
"""Re-apply decide_rollback_operation from stored HF/vLLM logits (no GPU)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.checkpoint_candidates import global_to_local_id
from training.scope.decide_rollback_operation import decide_rollback_operation
from training.scope.rollback_operation_runtime import pick_rollback_checkpoint


def _logits(row: dict) -> dict[str, float] | None:
    return row.get("hf_logits") or row.get("vllm_logits")


def redecide_row(row: dict) -> dict:
    scores = _logits(row)
    if not scores:
        return row
    candidates = row.get("candidate_list") or []
    ck_meta = [
        {
            "checkpoint_id": c.get("checkpoint_id"),
            "turn_id": c.get("relative_turn", c.get("turn_id", 0)),
            "n_curated": c.get("evidence_count", 0),
            "n_pool": c.get("n_pool", 0),
        }
        for c in candidates
    ]
    ck_pick = pick_rollback_checkpoint(ck_meta, int(row.get("turn", 0)))
    decision = decide_rollback_operation(
        score_continue=float(scores.get("CONTINUE", -1e9)),
        score_replan=float(scores.get("REPLAN", -1e9)),
        score_rollback=float(scores.get("ROLLBACK_TO", -1e9)),
        threshold=0.0,
        candidate_checkpoint_id=ck_pick,
        disable_replan=True,
    )
    local_to_global = {
        c.get("local_checkpoint_id"): c.get("checkpoint_id") for c in candidates
    }
    pred_local = global_to_local_id(decision.checkpoint_id, local_to_global)
    valid_ids = set(local_to_global.values())
    fallback_reason = None
    if decision.predicted_operation.value == "ROLLBACK_TO" and (
        not valid_ids or decision.checkpoint_id not in valid_ids
    ):
        fallback_reason = "invalid_checkpoint_prediction"
    out = dict(row)
    out["pred_operation"] = decision.predicted_operation.value
    out["pred_checkpoint_local_id"] = pred_local
    out["pred_checkpoint_global_id"] = decision.checkpoint_id
    out["fallback_reason"] = fallback_reason
    return out


def redecide_file(path: Path) -> int:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(redecide_row(json.loads(line)))
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", type=Path, required=True)
    args = p.parse_args()
    n = 0
    for split in ("offline_valid", "base_live", "self_live"):
        for name in ("hf_replay.jsonl", "vllm_replay.jsonl"):
            path = args.variant_dir / split / name
            if path.exists():
                n += redecide_file(path)
                print(f"redecided {path} ({path.stat().st_size} bytes)")
    print(f"total rows touched: {n}")


if __name__ == "__main__":
    main()
