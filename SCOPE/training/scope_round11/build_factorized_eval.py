#!/usr/bin/env python3
"""Build factorized eval rows (stage1_text + stage2_text) for a Stage1 view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transformers import AutoTokenizer

from training.scope_round11.stage1_views import build_stage1_view, build_stage2_prompt

BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
R8_VALID = _REPO / "artifacts/datasets/scope_round8/rollback_sdi/valid.jsonl"
HOLD = _REPO / "artifacts/datasets/scope_round9/hier_sdi/frozen_live_holdout.jsonl"
OUT = _REPO / "artifacts/datasets/scope_round11/factorized_eval"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--view", required=True)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--max-length", type=int, default=1536)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    sources = {"offline_valid": R8_VALID, "base_live": HOLD}
    for split, src in sources.items():
        rows_out = []
        for sample in load_jsonl(src):
            ds = sample.get("decision_state") or {}
            qid = str(sample.get("query_id") or ds.get("task_id") or "")
            turn = int(ds.get("turn_id", sample.get("turn", 0)) or 0)
            eid = sample.get("event_id") or f"{qid}:{turn}"
            s1 = build_stage1_view(sample, tok, args.view, max_length=args.max_length)
            s2 = build_stage2_prompt(sample)
            rows_out.append(
                {
                    "query_id": qid,
                    "turn": turn,
                    "event_id": eid,
                    "state_source": split,
                    "view": args.view,
                    "effective_input_text": s1.effective_input_text,  # Stage1 for operation
                    "stage1_text": s1.effective_input_text,
                    "stage2_text": s2,
                    "prompt_sha256": s1.prompt_sha256,
                    "candidate_list": s1.candidate_list,
                    "gold_operation": s1.gold_operation,
                    "gold_checkpoint_local_id": s1.gold_checkpoint_local_id,
                    "gold_checkpoint_global_id": s1.gold_checkpoint_global_id,
                    "gold_in_candidates": s1.gold_in_candidates,
                    "truncated": s1.truncated,
                    "token_length_before": s1.token_length_before,
                    "token_length_after": s1.token_length_after,
                }
            )
        outp = args.out / args.view / f"{split}.jsonl"
        write_jsonl(outp, rows_out)
        print(f"wrote {outp} n={len(rows_out)}")


if __name__ == "__main__":
    main()
