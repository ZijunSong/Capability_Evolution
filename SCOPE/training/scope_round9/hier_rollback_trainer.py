#!/usr/bin/env python3
"""Hierarchical rollback trainer: Stage1 operation + Stage2 checkpoint ranker."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from harness.capability.rollback_operation import RollbackOperation
from training.scope.rollback_effective_input import build_rollback_effective_input
from training.scope.rollback_operation_objectives import rollback_operation_loss
from training.scope.sdi_trainer import SDISampleDataset, SDITrainConfig


@dataclass
class HierTrainConfig(SDITrainConfig):
    lambda_ckpt: float = 1.0
    operation_only: bool = False
    checkpoint_only: bool = False
    include_candidate_summary: bool = True
    hint_distill: bool = False
    max_listwise_candidates: int = 4
    disable_replan: bool = True
    use_class_weight: bool = True
    continue_target_frac: float | None = None  # informational; resampling is offline


class HierRollbackTrainer:
    def __init__(self, cfg: HierTrainConfig) -> None:
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
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
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._step = 0

    def _hint(self) -> str:
        if not self.cfg.hint_distill:
            return ""
        return (
            "Hint: if recent queries repeat or evidence stalls, prefer ROLLBACK_TO "
            "a prior checkpoint instead of continuing the failing branch."
        )

    def _effective(self, sample: dict[str, Any]):
        return build_rollback_effective_input(
            sample,
            self.tokenizer,
            hint=self._hint(),
            max_length=self.cfg.max_length,
            include_candidate_summaries=self.cfg.include_candidate_summary,
        )

    def _select_listwise_candidates(self, candidates: list[dict], gold_local: str) -> list[dict]:
        """Keep gold + up to max_listwise_candidates-1 others (memory cap)."""
        if len(candidates) <= self.cfg.max_listwise_candidates:
            return list(candidates)
        gold = [c for c in candidates if c.get("local_checkpoint_id") == gold_local]
        rest = [c for c in candidates if c.get("local_checkpoint_id") != gold_local]
        keep = gold + rest[: max(self.cfg.max_listwise_candidates - len(gold), 0)]
        return keep

    def _checkpoint_listwise_loss(
        self, sample: dict[str, Any], eff_text: str
    ) -> torch.Tensor | None:
        gold_local = sample.get("gold_checkpoint_local_id")
        candidates = sample.get("decision_state", {}).get("available_checkpoints") or []
        if not gold_local or not candidates:
            return None
        candidates = self._select_listwise_candidates(candidates, str(gold_local))
        target_idx = next(
            (i for i, ck in enumerate(candidates) if ck.get("local_checkpoint_id") == gold_local),
            None,
        )
        if target_idx is None:
            return None
        # One random negative pairwise hinge — keeps only 2 graphs in memory.
        neg_idxs = [i for i in range(len(candidates)) if i != target_idx]
        if not neg_idxs:
            return torch.tensor(0.0, device=self.device)
        import random

        neg_idx = random.choice(neg_idxs)
        gold_ck = candidates[target_idx]
        neg_ck = candidates[neg_idx]
        gold_ids = self.tokenizer.encode(
            eff_text + f" {gold_ck.get('local_checkpoint_id', '')}",
            add_special_tokens=False,
        )
        neg_ids = self.tokenizer.encode(
            eff_text + f" {neg_ck.get('local_checkpoint_id', '')}",
            add_special_tokens=False,
        )
        s_gold = -self.model(
            torch.tensor([gold_ids], device=self.device),
            labels=torch.tensor([gold_ids], device=self.device),
        ).loss
        s_neg = -self.model(
            torch.tensor([neg_ids], device=self.device),
            labels=torch.tensor([neg_ids], device=self.device),
        ).loss
        return F.relu(0.1 - (s_gold - s_neg))

    def _class_weight(self, rows: list[dict]) -> dict[str, float]:
        keys = (
            ("CONTINUE", "ROLLBACK_TO")
            if self.cfg.disable_replan
            else ("CONTINUE", "REPLAN", "ROLLBACK_TO")
        )
        if not self.cfg.use_class_weight:
            return {k: 1.0 for k in keys}
        counts = {k: 0 for k in keys}
        for r in rows:
            op = str((r.get("target_action") or {}).get("operation") or r.get("operation", "CONTINUE"))
            if op in counts:
                counts[op] += 1
        total = sum(counts.values()) or 1
        n_cls = max(len(counts), 1)
        # Mild CONTINUE boost on top of inverse-freq (P0 prior correction).
        weights = {k: total / (n_cls * max(v, 1)) for k, v in counts.items()}
        if self.cfg.disable_replan and "CONTINUE" in weights:
            weights["CONTINUE"] *= 1.25
        return weights

    def train(self, train_path: Path, valid_path: Path | None = None) -> dict[str, Any]:
        train_rows = SDISampleDataset(train_path).rows
        valid_rows = SDISampleDataset(valid_path).rows if valid_path and valid_path.exists() else []
        weights = self._class_weight(train_rows)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        total_steps = max(1, len(train_rows) * self.cfg.num_epochs // max(self.cfg.grad_accum, 1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
        )

        for epoch in range(self.cfg.num_epochs):
            for sample in train_rows:
                eff = self._effective(sample)
                loss = torch.tensor(0.0, device=self.device)
                tgt_op = RollbackOperation(
                    str((sample.get("target_action") or {}).get("operation") or "CONTINUE")
                )
                did_backward = False
                if not self.cfg.checkpoint_only:
                    op_loss = rollback_operation_loss(
                        self.model,
                        self.tokenizer,
                        eff.effective_input_text,
                        tgt_op,
                        device=self.device,
                        prompt_is_final=True,
                        disable_replan=self.cfg.disable_replan,
                    )
                    op_loss = weights.get(tgt_op.value, 1.0) * op_loss
                    (op_loss / self.cfg.grad_accum).backward()
                    did_backward = True
                    loss = loss + op_loss.detach()
                if (
                    not self.cfg.operation_only
                    and tgt_op == RollbackOperation.ROLLBACK_TO
                ):
                    ck_loss = self._checkpoint_listwise_loss(sample, eff.effective_input_text)
                    if ck_loss is not None:
                        (self.cfg.lambda_ckpt * ck_loss / self.cfg.grad_accum).backward()
                        did_backward = True
                        loss = loss + self.cfg.lambda_ckpt * ck_loss.detach()
                if not did_backward:
                    continue
                if (getattr(self, "_step", 0) + 1) % self.cfg.grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                self._step = getattr(self, "_step", 0) + 1

        # Save before validate — offline Wave B eval is the real gate; do not lose a long train on OOM.
        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        try:
            valid_metrics = self.evaluate(valid_rows) if valid_rows else {}
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            valid_metrics = {"skipped_due_to_oom": 1.0}
        report = {
            "n_train": len(train_rows),
            "n_valid": len(valid_rows),
            "class_weights": weights,
            "valid_metrics": valid_metrics,
        }
        return report

    def evaluate(self, rows: list[dict]) -> dict[str, float]:
        if not rows:
            return {}
        op_correct = 0
        ck_correct = 0
        ck_total = 0
        was_training = self.model.training
        self.model.eval()
        from training.scope.rollback_operation_objectives import score_rollback_prompt
        from training.scope.decide_rollback_operation import decide_rollback_operation

        with torch.inference_mode():
            for sample in rows:
                eff = self._effective(sample)
                tgt = str((sample.get("target_action") or {}).get("operation") or "CONTINUE")
                s0, s1, s2 = score_rollback_prompt(
                    self.model,
                    self.tokenizer,
                    eff.effective_input_text,
                    device=self.device,
                )
                decision = decide_rollback_operation(
                    score_continue=float(s0),
                    score_replan=float(s1),
                    score_rollback=float(s2),
                    disable_replan=self.cfg.disable_replan,
                )
                pred = decision.predicted_operation.value
                op_correct += int(pred == tgt)
                if self.cfg.operation_only:
                    continue
                if tgt == "ROLLBACK_TO":
                    ck_total += 1
                    gold_local = sample.get("gold_checkpoint_local_id")
                    candidates = sample.get("decision_state", {}).get("available_checkpoints") or []
                    if candidates and gold_local:
                        candidates = self._select_listwise_candidates(candidates, str(gold_local))
                        scores = []
                        for ck in candidates:
                            local = ck.get("local_checkpoint_id")
                            ids = self.tokenizer.encode(
                                eff.effective_input_text + f" {local}",
                                add_special_tokens=False,
                            )
                            if len(ids) > self.cfg.max_length:
                                ids = ids[-self.cfg.max_length :]
                            inp = torch.tensor([ids], device=self.device)
                            out = self.model(inp, labels=inp)
                            scores.append(-float(out.loss))
                        best = int(max(range(len(scores)), key=lambda i: scores[i]))
                        if candidates[best].get("local_checkpoint_id") == gold_local:
                            ck_correct += 1
        if was_training:
            self.model.train()
        return {
            "operation_accuracy": op_correct / len(rows),
            "checkpoint_top1": ck_correct / max(ck_total, 1),
        }
