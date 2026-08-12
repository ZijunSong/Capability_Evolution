#!/usr/bin/env python3
"""Merged-HF deterministic replay on frozen effective inputs (no re-wrap)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.capability.rollback_operation import RollbackOperation
from training.scope.checkpoint_candidates import global_to_local_id
from training.scope.decide_rollback_operation import decide_rollback_operation
from training.scope.operation_objectives import ScoreNorm
from training.scope.rollback_operation_objectives import score_rollback_prompt
from training.scope.rollback_operation_runtime import pick_rollback_checkpoint


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def replay_rows(
    model,
    tokenizer,
    rows: list[dict],
    *,
    device: torch.device,
) -> list[dict]:
    out = []
    total = len(rows)
    for idx, row in enumerate(rows):
        if idx % 25 == 0 or idx + 1 == total:
            print(f"[hf-replay] {idx}/{total}", flush=True)
        text = row["effective_input_text"]
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
        turn_id = int(row.get("turn", 0))
        s_cont, s_replan, s_roll = score_rollback_prompt(
            model,
            tokenizer,
            text,
            device=device,
            norm=ScoreNorm.MEAN,
        )
        ck_pick = pick_rollback_checkpoint(ck_meta, turn_id)
        decision = decide_rollback_operation(
            score_continue=float(s_cont.detach().item()),
            score_replan=float(s_replan.detach().item()),
            score_rollback=float(s_roll.detach().item()),
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
        if (
            decision.predicted_operation == RollbackOperation.ROLLBACK_TO
            and (not valid_ids or decision.checkpoint_id not in valid_ids)
        ):
            fallback_reason = "invalid_checkpoint_prediction"
        out.append(
            {
                **row,
                "event_id": row.get("event_id") or f"{row.get('query_id')}:{row.get('turn')}:{idx}",
                "hf_logits": {
                    "CONTINUE": float(s_cont.detach().item()),
                    "REPLAN": float(s_replan.detach().item()),
                    "ROLLBACK_TO": float(s_roll.detach().item()),
                },
                "pred_operation": decision.predicted_operation.value,
                "pred_checkpoint_local_id": pred_local,
                "pred_checkpoint_global_id": decision.checkpoint_id,
                "fallback_reason": fallback_reason,
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Model weight dtype. float32 is slower but closer to vLLM on near-ties.",
    )
    args = p.parse_args()

    rows = load_jsonl(args.input)
    # Default: bf16 weights + fp32 log_softmax. Optional full float32 for parity repair.
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if not torch.cuda.is_available():
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=dtype, trust_remote_code=True
    )
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        replayed = replay_rows(model, tokenizer, rows, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in replayed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"HF replay: {len(replayed)} rows -> {args.output}")


if __name__ == "__main__":
    main()
