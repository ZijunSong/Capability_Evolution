#!/usr/bin/env python3
"""Phase C: candidate-independent pointer scorer (TARGET vs NOT_TARGET)."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.run_wave_b_train import merge_lora

DATA = _REPO / "artifacts/datasets/scope_round13/checkpoint_targeted"
BASE = "/data/ppnm/models/Qwen2.5-7B-Instruct"
OUT_DEFAULT = _REPO / "outputs/scope_round13/stage2_targeted/training"

VARIANTS = {
    "r13_ckpt_pointer_seed42": 42,
    "r13_ckpt_pointer_seed43": 43,
    "r13_ckpt_pointer_seed44": 44,
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def state_text(sample: dict) -> str:
    ds = sample.get("decision_state") or {}
    return str(sample.get("student_state_text") or ds.get("rendered_context") or "")


def candidate_prompt(sample: dict, cand: dict) -> str:
    st = state_text(sample)
    meta = {
        "checkpoint_id": cand.get("checkpoint_id"),
        "turn_id": cand.get("turn_id"),
        "n_curated": cand.get("n_curated", cand.get("evidence_count", 0)),
        "n_verified": cand.get("n_verified", cand.get("verified_count", 0)),
        "n_pool": cand.get("n_pool", 0),
    }
    return (
        "Rollback checkpoint targeting.\n"
        f"Decision state:\n{st}\n\n"
        f"Candidate checkpoint:\n{json.dumps(meta, ensure_ascii=False)}\n\n"
        "Is this the TARGET checkpoint to roll back to?"
    )


class PointerTrainer:
    def __init__(self, out: Path, seed: int, device: str) -> None:
        self.out = out
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dtype, trust_remote_code=True)
        model = get_peft_model(
            model,
            LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.train()
        self.max_length = 1536
        self.grad_accum = 16

    def _score(self, prompt: str, verbalizer: str) -> torch.Tensor:
        ids = self.tok.encode(prompt + verbalizer, add_special_tokens=False)
        if len(ids) > self.max_length:
            ids = ids[-self.max_length :]
        inp = torch.tensor([ids], device=self.device)
        return -self.model(inp, labels=inp).loss

    def _pair_score(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_t = self._score(prompt, " TARGET")
        s_n = self._score(prompt, " NOT_TARGET")
        return s_t, s_n, s_t - s_n

    def train(self, rows: list[dict]) -> dict[str, Any]:
        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        steps = max(1, len(rows) * 3 // self.grad_accum)
        sched = get_linear_schedule_with_warmup(opt, max(1, steps // 10), steps)
        step = 0
        for _epoch in range(3):
            random.shuffle(rows)
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

                pos_idx = next(
                    (
                        i
                        for i, c in enumerate(cands)
                        if str(c.get("checkpoint_id")) == gold
                    ),
                    None,
                )
                neg_idxs = [
                    i
                    for i, c in enumerate(cands)
                    if str(c.get("checkpoint_id")) != gold
                ]
                if pos_idx is None or not neg_idxs:
                    continue

                # Score only a capped negative pool to find an approximate hardest neg.
                pool = list(neg_idxs)
                random.shuffle(pool)
                pool = pool[:6]
                det_scores: dict[int, float] = {}
                with torch.inference_mode():
                    for i in pool:
                        prompt = candidate_prompt(sample, cands[i])
                        _, _, score = self._pair_score(prompt)
                        det_scores[i] = float(score)
                hard = max(pool, key=lambda i: det_scores[i])

                # Train subset: gold + hardest + up to 2 other pool negs.
                train_idxs = {pos_idx, hard}
                rest = [i for i in pool if i != hard]
                random.shuffle(rest)
                for i in rest[:2]:
                    train_idxs.add(i)
                train_list = sorted(train_idxs)
                n = max(len(train_list), 1)
                scale = 1.0 / (n * self.grad_accum)
                for i in train_list:
                    prompt = candidate_prompt(sample, cands[i])
                    s_t, s_n, _ = self._pair_score(prompt)
                    logits = torch.stack([s_n, s_t])
                    is_pos = i == pos_idx
                    target = torch.tensor(1 if is_pos else 0, device=self.device)
                    ce_i = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                    (ce_i * scale).backward()

                # Pairwise margin gold > hardest (scores already have grad from CE pass
                # only if recomputed — recompute the pair).
                p_pos = candidate_prompt(sample, cands[pos_idx])
                p_neg = candidate_prompt(sample, cands[hard])
                _, _, s_pos = self._pair_score(p_pos)
                _, _, s_neg = self._pair_score(p_neg)
                margin_loss = F.relu(0.1 - (s_pos - s_neg))
                (0.5 * margin_loss / self.grad_accum).backward()

                if (step + 1) % self.grad_accum == 0:
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"[stage2-train] steps={step} n_train={len(rows)}", flush=True)

        lora = self.out / "lora"
        lora.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(lora)
        self.tok.save_pretrained(lora)
        return {"n_train": len(rows), "steps": step}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--out-root", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()
    out = args.out_root / args.variant
    out.mkdir(parents=True, exist_ok=True)
    if (out / "merged" / "config.json").exists():
        print(f"[skip] {args.variant}")
        return
    train = load_jsonl(DATA / "train.jsonl")
    t0 = time.time()
    report = PointerTrainer(out, VARIANTS[args.variant], args.gpu).train(train)
    merged = merge_lora(out)
    full = {
        "variant": args.variant,
        "report": report,
        "merged": str(merged),
        "wall_s": time.time() - t0,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip(),
    }
    (out / "train_only_report.json").write_text(json.dumps(full, indent=2) + "\n")
    (out / "DONE").write_text("ok\n")
    print(json.dumps(full, indent=2))


if __name__ == "__main__":
    main()
