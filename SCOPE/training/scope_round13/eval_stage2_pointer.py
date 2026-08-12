#!/usr/bin/env python3
"""Evaluate Round13 candidate-independent pointer scorer on VALID/TEST targeted splits."""

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

from training.scope_round13.run_stage2_pointer_train import candidate_prompt, load_jsonl

DATA = _REPO / "artifacts/datasets/scope_round13/checkpoint_targeted"
BASE = "/data/ppnm/models/Qwen2.5-7B-Instruct"


@torch.no_grad()
def score_candidate(model, tok, prompt: str, device, max_length: int = 1536) -> float:
    def one(verb: str) -> float:
        ids = tok.encode(prompt + verb, add_special_tokens=False)
        if len(ids) > max_length:
            ids = ids[-max_length:]
        inp = torch.tensor([ids], device=device)
        return float((-model(inp, labels=inp).loss).item())

    return one(" TARGET") - one(" NOT_TARGET")


def eval_split(model, tok, rows: list[dict], device) -> dict:
    hits = 0
    mrr = 0.0
    n = 0
    n_cov = 0
    for sample in rows:
        ds = sample.get("decision_state") or {}
        cands = list(ds.get("available_checkpoints") or [])
        gold = str(
            (sample.get("target_action") or {}).get("checkpoint_id")
            or sample.get("gold_checkpoint_id")
            or ""
        )
        if not cands or not gold:
            continue
        n += 1
        if any(str(c.get("checkpoint_id")) == gold for c in cands):
            n_cov += 1
        scored = []
        for c in cands:
            sc = score_candidate(model, tok, candidate_prompt(sample, c), device)
            scored.append((sc, str(c.get("checkpoint_id"))))
        scored.sort(key=lambda x: x[0], reverse=True)
        rank = next((i + 1 for i, (_, cid) in enumerate(scored) if cid == gold), None)
        if rank is None:
            continue
        if rank == 1:
            hits += 1
        mrr += 1.0 / rank
    return {
        "n": n,
        "top1": hits / max(n, 1),
        "MRR": mrr / max(n, 1),
        "coverage": n_cov / max(n, 1),
        "invalid_checkpoint": 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", type=Path, required=True)
    p.add_argument("--split", choices=["valid", "test"], default="valid")
    p.add_argument("--gpu", default="cuda:0")
    args = p.parse_args()

    merged = args.variant_dir / "merged"
    lora = args.variant_dir / "lora"
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if (merged / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(
            merged, torch_dtype=dtype, trust_remote_code=True
        ).to(device)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=dtype, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, str(lora)).to(device)
    model.eval()

    rows = load_jsonl(DATA / f"{args.split}.jsonl")
    metrics = eval_split(model, tok, rows, device)
    out_dir = args.variant_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"split": args.split, "variant_dir": str(args.variant_dir), "metrics": metrics}
    (out_dir / "METRICS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
