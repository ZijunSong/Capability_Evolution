#!/usr/bin/env python3
"""Factorized Stage1/Stage2 hierarchical trainer for Round11."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from harness.capability.rollback_operation import RollbackOperation
from training.scope.rollback_operation_objectives import rollback_operation_loss
from training.scope.sdi_trainer import SDISampleDataset, SDITrainConfig
from training.scope_round11.stage1_views import build_stage1_view, build_stage2_prompt

Stage1ViewName = Literal["A0", "A1", "A2", "A3", "A4"]
CkptLoss = Literal["pairwise", "listwise"]


@dataclass
class FactorizedTrainConfig(SDITrainConfig):
    lambda_ckpt: float = 1.0
    operation_only: bool = False
    checkpoint_only: bool = False
    stage1_view: Stage1ViewName = "A3"
    checkpoint_loss: CkptLoss = "pairwise"
    max_listwise_candidates: int = 8
    disable_replan: bool = True
    use_class_weight: bool = False
    pairwise_margin: float = 0.1


class FactorizedRollbackTrainer:
    def __init__(self, cfg: FactorizedTrainConfig) -> None:
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

    def _stage1_text(self, sample: dict[str, Any]) -> str:
        return build_stage1_view(
            sample,
            self.tokenizer,
            self.cfg.stage1_view,
            max_length=self.cfg.max_length,
        ).effective_input_text

    def _stage2_text(self, sample: dict[str, Any]) -> str:
        return build_stage2_prompt(sample)

    def _score_completion(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.cfg.max_length:
            ids = ids[-self.cfg.max_length :]
        inp = torch.tensor([ids], device=self.device)
        return -self.model(inp, labels=inp).loss

    def _select_candidates(self, sample: dict[str, Any]) -> tuple[list[dict], str | None]:
        gold_local = sample.get("gold_checkpoint_local_id")
        candidates = list(sample.get("decision_state", {}).get("available_checkpoints") or [])
        if not gold_local or not candidates:
            return [], gold_local
        # Ensure local ids exist via view builder path.
        from training.scope.checkpoint_candidates import assign_local_checkpoint_ids

        ordered, _ = assign_local_checkpoint_ids(candidates)
        if len(ordered) <= self.cfg.max_listwise_candidates:
            return ordered, str(gold_local)
        gold = [c for c in ordered if c.get("local_checkpoint_id") == gold_local]
        rest = [c for c in ordered if c.get("local_checkpoint_id") != gold_local]
        keep = gold + rest[: max(self.cfg.max_listwise_candidates - len(gold), 0)]
        return keep, str(gold_local)

    def _checkpoint_loss(self, sample: dict[str, Any]) -> torch.Tensor | None:
        candidates, gold_local = self._select_candidates(sample)
        if not candidates or not gold_local:
            return None
        target_idx = next(
            (i for i, ck in enumerate(candidates) if ck.get("local_checkpoint_id") == gold_local),
            None,
        )
        if target_idx is None:
            return None
        stage2 = self._stage2_text(sample)
        if self.cfg.checkpoint_loss == "listwise":
            scores = []
            for ck in candidates:
                local = ck.get("local_checkpoint_id", "")
                scores.append(self._score_completion(stage2 + f" {local}"))
            logits = torch.stack(scores)
            log_probs = F.log_softmax(logits, dim=0)
            return -log_probs[target_idx]

        # pairwise: gold > each non-gold (mean over sampled negs, memory-safe)
        neg_idxs = [i for i in range(len(candidates)) if i != target_idx]
        if not neg_idxs:
            return torch.tensor(0.0, device=self.device)
        # Sample up to 2 negatives per step to control memory.
        random.shuffle(neg_idxs)
        neg_idxs = neg_idxs[:2]
        gold_ck = candidates[target_idx]
        s_gold = self._score_completion(stage2 + f" {gold_ck.get('local_checkpoint_id', '')}")
        losses = []
        for ni in neg_idxs:
            neg_ck = candidates[ni]
            s_neg = self._score_completion(stage2 + f" {neg_ck.get('local_checkpoint_id', '')}")
            losses.append(F.relu(self.cfg.pairwise_margin - (s_gold - s_neg)))
        return torch.stack(losses).mean()

    def _class_weight(self, rows: list[dict]) -> dict[str, float]:
        keys = ("CONTINUE", "ROLLBACK_TO")
        if not self.cfg.use_class_weight:
            return {k: 1.0 for k in keys}
        counts = {k: 0 for k in keys}
        for r in rows:
            op = str((r.get("target_action") or {}).get("operation") or r.get("operation", "CONTINUE"))
            if op in counts:
                counts[op] += 1
        total = sum(counts.values()) or 1
        return {k: total / (len(counts) * max(v, 1)) for k, v in counts.items()}

    def train(self, train_path: Path, valid_path: Path | None = None) -> dict[str, Any]:
        train_rows = SDISampleDataset(train_path).rows
        if self.cfg.checkpoint_only:
            train_rows = [
                r
                for r in train_rows
                if str((r.get("target_action") or {}).get("operation") or r.get("gold_operation"))
                == "ROLLBACK_TO"
            ]
        valid_rows = SDISampleDataset(valid_path).rows if valid_path and valid_path.exists() else []
        weights = self._class_weight(train_rows)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        total_steps = max(1, len(train_rows) * self.cfg.num_epochs // max(self.cfg.grad_accum, 1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
        )

        for _epoch in range(self.cfg.num_epochs):
            for sample in train_rows:
                loss_acc = torch.tensor(0.0, device=self.device)
                tgt_op = RollbackOperation(
                    str((sample.get("target_action") or {}).get("operation") or "CONTINUE")
                )
                did_backward = False
                if not self.cfg.checkpoint_only:
                    s1 = self._stage1_text(sample)
                    op_loss = rollback_operation_loss(
                        self.model,
                        self.tokenizer,
                        s1,
                        tgt_op,
                        device=self.device,
                        prompt_is_final=True,
                        disable_replan=self.cfg.disable_replan,
                    )
                    op_loss = weights.get(tgt_op.value, 1.0) * op_loss
                    (op_loss / self.cfg.grad_accum).backward()
                    did_backward = True
                    loss_acc = loss_acc + op_loss.detach()
                if (
                    not self.cfg.operation_only
                    and tgt_op == RollbackOperation.ROLLBACK_TO
                ):
                    ck_loss = self._checkpoint_loss(sample)
                    if ck_loss is not None:
                        (self.cfg.lambda_ckpt * ck_loss / self.cfg.grad_accum).backward()
                        did_backward = True
                        loss_acc = loss_acc + self.cfg.lambda_ckpt * ck_loss.detach()
                if not did_backward:
                    continue
                if (self._step + 1) % self.cfg.grad_accum == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                self._step += 1

        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        try:
            valid_metrics = self.evaluate(valid_rows) if valid_rows else {}
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            valid_metrics = {"skipped_due_to_oom": 1.0}
        return {
            "n_train": len(train_rows),
            "n_valid": len(valid_rows),
            "class_weights": weights,
            "stage1_view": self.cfg.stage1_view,
            "checkpoint_loss": self.cfg.checkpoint_loss,
            "operation_only": self.cfg.operation_only,
            "checkpoint_only": self.cfg.checkpoint_only,
            "valid_metrics": valid_metrics,
        }

    def evaluate(self, rows: list[dict]) -> dict[str, float]:
        if not rows:
            return {}
        op_correct = 0
        ck_correct = 0
        ck_total = 0
        was_training = self.model.training
        self.model.eval()
        from training.scope.decide_rollback_operation import decide_rollback_operation
        from training.scope.rollback_operation_objectives import score_rollback_prompt

        with torch.inference_mode():
            for sample in rows:
                if not self.cfg.checkpoint_only:
                    s1 = self._stage1_text(sample)
                    tgt = str((sample.get("target_action") or {}).get("operation") or "CONTINUE")
                    s0, s1s, s2 = score_rollback_prompt(
                        self.model, self.tokenizer, s1, device=self.device
                    )
                    decision = decide_rollback_operation(
                        score_continue=float(s0),
                        score_replan=float(s1s),
                        score_rollback=float(s2),
                        disable_replan=self.cfg.disable_replan,
                    )
                    op_correct += int(decision.predicted_operation.value == tgt)
                if self.cfg.operation_only:
                    continue
                tgt = str((sample.get("target_action") or {}).get("operation") or "CONTINUE")
                if tgt != "ROLLBACK_TO":
                    continue
                ck_total += 1
                candidates, gold_local = self._select_candidates(sample)
                if not candidates or not gold_local:
                    continue
                stage2 = self._stage2_text(sample)
                scores = [
                    float(self._score_completion(stage2 + f" {ck.get('local_checkpoint_id', '')}"))
                    for ck in candidates
                ]
                best = int(max(range(len(scores)), key=lambda i: scores[i]))
                if candidates[best].get("local_checkpoint_id") == gold_local:
                    ck_correct += 1
        if was_training:
            self.model.train()
        out = {
            "checkpoint_top1": ck_correct / max(ck_total, 1),
            "n_checkpoint_eval": float(ck_total),
        }
        if not self.cfg.checkpoint_only:
            out["operation_accuracy"] = op_correct / max(len(rows), 1)
        return out
