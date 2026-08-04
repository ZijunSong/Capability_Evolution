#!/usr/bin/env python3
"""Exact HF replay of live decision traces (Round 7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.capability.dup_operation import DupOperation
from training.scope.decide_dup_operation import decide_dup_operation
from training.scope.operation_scorer import score_rendered_prompt
from training.scope_round7.common import OUT, load_jsonl, write_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    trace_path = args.trace_dir / "live_dup_decision_trace.jsonl"
    traces = load_jsonl(trace_path)
    out = args.output_dir or (OUT / "contract_trace/replay_hf" / args.trace_dir.name)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=dtype, trust_remote_code=True
    )
    model.eval().to(device)

    rows: list[dict] = []
    for tr in traces:
        prompt = tr.get("rendered_prompt") or ""
        if not prompt:
            sidecar = args.trace_dir / "prompt_sidecar" / f"{tr['rendered_prompt_sha256']}.txt"
            if sidecar.exists():
                prompt = sidecar.read_text(encoding="utf-8")
        result = score_rendered_prompt(model, tokenizer, prompt, device=device)
        sk = result.scores[DupOperation.KEEP_EVIDENCE.value]
        ss = result.scores[DupOperation.SKIP_DUPLICATE.value]
        decision = decide_dup_operation(
            score_keep=sk, score_skip=ss, threshold=float(tr.get("threshold", 0.0))
        )
        rows.append({
            "event_id": tr["event_id"],
            "score_keep_hf": sk,
            "score_skip_hf": ss,
            "margin_hf": decision.margin,
            "operation_hf": decision.predicted_operation.value,
            "score_keep_live": tr.get("score_keep"),
            "score_skip_live": tr.get("score_skip"),
            "margin_live": tr.get("margin"),
            "operation_live": tr.get("predicted_operation_pre_realizer"),
        })

    write_json(out / "hf_replay.json", {"n": len(rows), "rows": rows})
    print(f"HF replay: {len(rows)} events -> {out}")


if __name__ == "__main__":
    main()
