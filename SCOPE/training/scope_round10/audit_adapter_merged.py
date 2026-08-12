#!/usr/bin/env python3
"""Adapter vs merged HF operation parity audit (R10-P7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.decide_rollback_operation import decide_rollback_operation
from training.scope.rollback_operation_objectives import ScoreNorm, score_rollback_prompt

BASE = "/data/ppnm/models/Qwen2.5-7B-Instruct"
P0 = _REPO / "outputs/scope_round9/wave_b_p0"
OUT = _REPO / "outputs/scope_round10/phase_a/audits"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    variant = f"rollback_hier_o7_seed{args.seed}"
    merged = P0 / variant / "merged"
    lora = P0 / variant / "lora"
    if not (lora / "adapter_config.json").exists() and (lora / "lora" / "adapter_config.json").exists():
        lora = lora / "lora"

    rows = []
    with (_REPO / "artifacts/datasets/scope_round9/frozen_replay/offline_valid.jsonl").open() as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            if line.strip():
                rows.append(json.loads(line))

    tok = AutoTokenizer.from_pretrained(merged, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m_merged = AutoModelForCausalLM.from_pretrained(
        merged, torch_dtype=torch.float32, trust_remote_code=True
    ).to(args.device).eval()
    m_base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, trust_remote_code=True
    )
    m_ad = PeftModel.from_pretrained(m_base, str(lora)).to(args.device).eval()

    agree = 0
    out_rows = []
    for i, r in enumerate(rows):
        prompt = r["effective_input_text"]
        c1, r1, b1 = score_rollback_prompt(
            m_merged, tok, prompt, device=args.device, norm=ScoreNorm.MEAN
        )
        c2, r2, b2 = score_rollback_prompt(
            m_ad, tok, prompt, device=args.device, norm=ScoreNorm.MEAN
        )
        d1 = decide_rollback_operation(
            score_continue=float(c1.detach().item()),
            score_replan=float(r1.detach().item()),
            score_rollback=float(b1.detach().item()),
            threshold=0.0,
            disable_replan=True,
        )
        d2 = decide_rollback_operation(
            score_continue=float(c2.detach().item()),
            score_replan=float(r2.detach().item()),
            score_rollback=float(b2.detach().item()),
            threshold=0.0,
            disable_replan=True,
        )
        ok = d1.predicted_operation == d2.predicted_operation
        agree += int(ok)
        out_rows.append(
            {
                "event_id": r.get("event_id"),
                "merged": d1.predicted_operation.value,
                "adapter": d2.predicted_operation.value,
                "agree": ok,
            }
        )
        if i % 20 == 0:
            print(i, flush=True)

    summary = {"n": len(rows), "agree": agree, "agreement": agree / max(len(rows), 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"adapter_merged_seed{args.seed}.json"
    path.write_text(json.dumps({"summary": summary, "rows": out_rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
