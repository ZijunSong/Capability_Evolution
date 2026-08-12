#!/usr/bin/env python3
"""HF float32/bf16 rescore of mismatch ledger events (parity forensics)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.decide_rollback_operation import decide_rollback_operation
from training.scope.rollback_operation_objectives import ScoreNorm, score_rollback_prompt


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="mismatch_events.jsonl or full ledger")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32")
    p.add_argument("--only-disagreement", action="store_true")
    args = p.parse_args()

    rows = load_jsonl(args.input)
    if args.only_disagreement:
        rows = [r for r in rows if not r.get("operation_agreement")]
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(args.device)
    model.eval()

    out_rows = []
    for i, row in enumerate(rows):
        if i % 10 == 0:
            print(f"[hf-rescore-{args.dtype}] {i}/{len(rows)}", flush=True)
        prompt = row["rendered_prompt_text"]
        s_cont, s_replan, s_roll = score_rollback_prompt(
            model, tok, prompt, device=args.device, norm=ScoreNorm.MEAN
        )
        d = decide_rollback_operation(
            score_continue=float(s_cont.detach().item()),
            score_replan=float(s_replan.detach().item()),
            score_rollback=float(s_roll.detach().item()),
            threshold=0.0,
            disable_replan=True,
        )
        out = dict(row)
        out[f"hf_{args.dtype}_score_continue"] = d.score_continue
        out[f"hf_{args.dtype}_score_rollback"] = d.score_rollback
        out[f"hf_{args.dtype}_margin"] = d.margin
        out[f"hf_{args.dtype}_operation"] = d.predicted_operation.value
        out[f"agree_vllm_after_{args.dtype}"] = (
            d.predicted_operation.value == row.get("vllm_operation")
        )
        out_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_agree = sum(1 for r in out_rows if r.get(f"agree_vllm_after_{args.dtype}"))
    print(
        json.dumps(
            {
                "n": len(out_rows),
                "agree_vllm": n_agree,
                "agreement": n_agree / max(len(out_rows), 1),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
