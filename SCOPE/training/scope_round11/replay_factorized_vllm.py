#!/usr/bin/env python3
"""Canonical vLLM replay with optional learned Stage2 checkpoint ranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from openai import OpenAI

from training.scope.canonical_rollback_scorer import CanonicalRollbackOperationScorer
from training.scope.vllm_rollback_scorer import VllmRollbackScorer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_text_completion(scorer: VllmRollbackScorer, prompt: str, completion: str) -> float:
    tok = scorer._get_tokenizer()
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    comp_ids = tok.encode(completion, add_special_tokens=False)
    if not comp_ids:
        return -1e9
    input_ids = prompt_ids + comp_ids
    resp = scorer.client.completions.create(
        model=scorer.model,
        prompt=input_ids,
        max_tokens=0,
        echo=True,
        logprobs=1,
    )
    choice = resp.choices[0]
    tok_lps = scorer._completion_logprobs_from_token_ids(choice, len(prompt_ids), len(comp_ids))
    return sum(tok_lps) / max(len(tok_lps), 1) if tok_lps else -1e9


def rank_checkpoint(scorer: VllmRollbackScorer, row: dict) -> tuple[str | None, str | None, list[dict]]:
    stage2 = row.get("stage2_text") or ""
    candidates = row.get("candidate_list") or []
    if not stage2 or not candidates:
        return None, None, []
    scored = []
    for c in candidates:
        local = str(c.get("local_checkpoint_id") or "")
        global_id = c.get("checkpoint_id")
        s = score_text_completion(scorer, stage2, f" {local}")
        scored.append(
            {
                "local_checkpoint_id": local,
                "checkpoint_id": global_id,
                "score": s,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    if not scored:
        return None, None, scored
    best = scored[0]
    return best.get("checkpoint_id"), best.get("local_checkpoint_id"), scored


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--use-stage2-ranker", action="store_true")
    p.add_argument("--operation-from-oracle", action="store_true")
    p.add_argument("--checkpoint-from-oracle", action="store_true")
    args = p.parse_args()

    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{args.port}/v1")
    scorer = VllmRollbackScorer(
        client=client,
        model=Path(args.model_path).name,
        model_path=str(args.model_path),
    )
    canonical = CanonicalRollbackOperationScorer(scorer, threshold=0.0, disable_replan=True)
    rows = load_jsonl(args.input)
    out = []
    total = len(rows)
    for idx, row in enumerate(rows):
        if idx % 25 == 0 or idx + 1 == total:
            print(f"[factorized-replay] {idx}/{total}", flush=True)
        # Operation via Stage1 text (effective_input_text)
        bundle = canonical.decide_row(row, prompt_is_final=True)
        pred_op = bundle.pred_operation
        pred_ck_g = bundle.pred_checkpoint_global_id
        pred_ck_l = bundle.pred_checkpoint_local_id
        ck_rank_scores = []
        if args.operation_from_oracle:
            pred_op = row.get("gold_operation")
        if args.use_stage2_ranker and pred_op == "ROLLBACK_TO":
            pred_ck_g, pred_ck_l, ck_rank_scores = rank_checkpoint(scorer, row)
        if args.checkpoint_from_oracle and pred_op == "ROLLBACK_TO":
            pred_ck_g = row.get("gold_checkpoint_global_id")
            pred_ck_l = row.get("gold_checkpoint_local_id")
        out.append(
            {
                **row,
                "vllm_logits": dict(bundle.scores),
                "canonical_logits": dict(bundle.scores),
                "pred_operation": pred_op,
                "pred_checkpoint_local_id": pred_ck_l,
                "pred_checkpoint_global_id": pred_ck_g,
                "checkpoint_rank_scores": ck_rank_scores,
                "fallback_reason": bundle.fallback_reason,
                "scorer_backend": "vllm_canonical_factorized",
                "decision_threshold": 0.0,
                "disable_replan": True,
                "used_stage2_ranker": bool(args.use_stage2_ranker),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"factorized vLLM replay: {len(out)} -> {args.output}")


if __name__ == "__main__":
    main()
