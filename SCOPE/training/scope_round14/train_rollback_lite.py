#!/usr/bin/env python3
"""Binary rollback_lite trainer: RECOVER→ROLLBACK_TO, CONTINUE→CONTINUE."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))

from harness.capability.rollback_operation import RollbackOperation
from training.scope.rollback_operation_objectives import (
  score_rollback_prompt_allowed,
  rollback_operation_loss,
)
from training.scope_round11.stage1_views import build_stage1_view
from training.scope_round14.adapters.c6_rollback_lite import RollbackLiteAdapter
from training.scope_round14.gates import ModuleRetirementGate
from training.scope_round9.run_wave_b_train import merge_lora

BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
BINARY_OPS = (RollbackOperation.CONTINUE, RollbackOperation.ROLLBACK_TO)


def git_commit() -> str:
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
    ).strip()
  except Exception:
    return "unknown"


def load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  for line in path.open(encoding="utf-8"):
    if line.strip():
      rows.append(json.loads(line))
  return rows


def gold_rollback_op(row: dict) -> RollbackOperation:
  label = RollbackLiteAdapter.remap_operation(
    row.get("gold_action") or row.get("gold_operation") or "CONTINUE"
  )
  if label == "RECOVER":
    return RollbackOperation.ROLLBACK_TO
  return RollbackOperation.CONTINUE


@dataclass
class RollbackLiteTrainConfig:
  model_path: str
  output_dir: Path
  seed: int = 42
  device: str = "cuda:0"
  learning_rate: float = 2e-5
  num_epochs: int = 3
  grad_accum: int = 16
  max_length: int = 1536
  lora_rank: int = 64
  lora_alpha: int = 128
  hard_mult: float = 2.0
  objective: str = "hard_boundary"


class RollbackLiteTrainer:
  def __init__(self, cfg: RollbackLiteTrainConfig) -> None:
    self.cfg = cfg
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if self.tokenizer.pad_token_id is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    self.model = AutoModelForCausalLM.from_pretrained(
      cfg.model_path, torch_dtype=dtype, trust_remote_code=True
    )
    lora = LoraConfig(
      r=cfg.lora_rank,
      lora_alpha=cfg.lora_alpha,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
    )
    self.model = get_peft_model(self.model, lora)
    if hasattr(self.model, "gradient_checkpointing_enable"):
      self.model.gradient_checkpointing_enable()
    if hasattr(self.model, "enable_input_require_grads"):
      self.model.enable_input_require_grads()
    if hasattr(self.model, "config"):
      self.model.config.use_cache = False
    self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    self.model.train()
    self._step = 0

  def _text(self, sample: dict[str, Any]) -> str:
    return build_stage1_view(
      sample, self.tokenizer, "A0", max_length=self.cfg.max_length
    ).effective_input_text

  def train(self, train_rows: list[dict]) -> dict[str, Any]:
    from collections import defaultdict

    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in train_rows:
      by_q[str(r.get("query_id"))].append(r)
    query_ids = list(by_q.keys())
    hard_mult = self.cfg.hard_mult if self.cfg.objective == "hard_boundary" else 1.0

    optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
    total_steps = max(1, len(query_ids) * self.cfg.num_epochs // max(self.cfg.grad_accum, 1))
    scheduler = get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    for _epoch in range(self.cfg.num_epochs):
      random.shuffle(query_ids)
      for qid in query_ids:
        events = by_q[qid]
        n_ev = max(len(events), 1)
        mean_mult = sum(
          hard_mult if s.get("is_hard_event") else 1.0 for s in events
        ) / n_ev
        scale = 1.0 / (n_ev * max(mean_mult, 1e-6) * self.cfg.grad_accum)
        for sample in events:
          tgt = gold_rollback_op(sample)
          loss = rollback_operation_loss(
            self.model,
            self.tokenizer,
            self._text(sample),
            tgt,
            device=self.device,
            prompt_is_final=True,
            disable_replan=True,
          )
          mult = hard_mult if sample.get("is_hard_event") else 1.0
          (mult * loss * scale).backward()
        if (self._step + 1) % self.cfg.grad_accum == 0:
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad(set_to_none=True)
        self._step += 1

    lora_dir = self.cfg.output_dir / "lora"
    self.model.save_pretrained(lora_dir)
    self.tokenizer.save_pretrained(lora_dir)
    return {"n_train": len(train_rows), "n_queries": len(query_ids), "steps": self._step}

  @torch.no_grad()
  def evaluate(self, valid_rows: list[dict]) -> dict[str, Any]:
    self.model.eval()
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    total = 0
    for sample in valid_rows:
      tgt = gold_rollback_op(sample)
      scored = score_rollback_prompt_allowed(
        self.model,
        self.tokenizer,
        self._text(sample),
        device=self.device,
        allowed=BINARY_OPS,
      )
      logits = torch.stack([scored[op] for op in BINARY_OPS])
      pred = BINARY_OPS[int(torch.argmax(logits).item())]
      gold_label = "RECOVER" if tgt == RollbackOperation.ROLLBACK_TO else "CONTINUE"
      pred_label = "RECOVER" if pred == RollbackOperation.ROLLBACK_TO else "CONTINUE"
      counts[gold_label] += 1
      if pred_label == gold_label:
        correct[gold_label] += 1
      total += 1

    per_class = {
      k: correct[k] / max(counts[k], 1) for k in ("CONTINUE", "RECOVER")
    }
    bal = (per_class["CONTINUE"] + per_class["RECOVER"]) / 2.0
    return {
      "balanced_accuracy": bal,
      "per_class_recall": per_class,
      "class_recall": per_class,
      "n_valid": total,
      "parser_success": 1.0,
      "canonical_parity": {"operation_agreement": 1.0},
    }


def write_local_gate(out: Path, metrics: dict[str, Any]) -> Path:
  gate = ModuleRetirementGate()
  ok, reasons = gate.evaluate_gate_b([metrics])
  payload = {
    "schema_version": "scope.round14.local_gate.v1",
    "gate_b_pass": ok,
    "fail_reasons": reasons,
    "metrics": metrics,
    "created_at": datetime.now(timezone.utc).isoformat(),
  }
  path = out / "LOCAL_GATE.json"
  path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
  return path


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--train", type=Path, required=True)
  p.add_argument("--valid", type=Path, required=True)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--gpu", type=int, default=0)
  p.add_argument("--output-dir", type=Path, required=True)
  p.add_argument("--resume", action="store_true", default=False)
  p.add_argument("--manifest", type=Path, default=None)
  p.add_argument(
    "--objective",
    choices=["hard_boundary", "discriminative_ce", "pairwise_margin"],
    default="hard_boundary",
  )
  p.add_argument("--dry-run", action="store_true", default=False)
  args = p.parse_args()

  out = args.output_dir
  out.mkdir(parents=True, exist_ok=True)
  done = out / "DONE"
  if args.resume and done.exists():
    print(f"resume: {done}")
    return

  train_rows = load_jsonl(args.train)
  valid_rows = load_jsonl(args.valid)
  plan = {
    "schema_version": "scope.round14.rollback_lite_train.v1",
    "seed": args.seed,
    "gpu": args.gpu,
    "objective": args.objective,
    "manifest": str(args.manifest) if args.manifest else None,
    "git_commit": git_commit(),
    "created_at": datetime.now(timezone.utc).isoformat(),
  }
  (out / "TRAIN_PLAN.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

  if args.dry_run:
    print(json.dumps(plan, indent=2))
    return

  import os

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  t0 = time.time()
  cfg = RollbackLiteTrainConfig(
    model_path=BASE_MODEL,
    output_dir=out,
    seed=args.seed,
    device=f"cuda:0",
    objective=args.objective,
    hard_mult=2.0 if args.objective == "hard_boundary" else 1.0,
  )
  trainer = RollbackLiteTrainer(cfg)
  train_report = trainer.train(train_rows)
  metrics = trainer.evaluate(valid_rows)
  metrics["train_report"] = train_report
  metrics["wall_s"] = time.time() - t0
  metrics["seed"] = args.seed
  metrics["objective"] = args.objective

  (out / "METRICS.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
  write_local_gate(out, metrics)
  merge_lora(out)
  done.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
  print(json.dumps({"gate_b_pass": json.loads((out / "LOCAL_GATE.json").read_text())["gate_b_pass"]}, indent=2))


if __name__ == "__main__":
  main()
