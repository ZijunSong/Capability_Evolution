"""SDI trainer for rollback_decision typed operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from harness.capability.rollback_operation import RollbackOperation
from training.scope.rollback_operation_objectives import (
    rollback_operation_loss,
    score_rollback_operations,
)
from training.scope.sdi_trainer import SDISampleDataset, SDITrainConfig


@dataclass
class RollbackTrainConfig(SDITrainConfig):
    hint_distill: bool = False
    trajectory_imitation: bool = False
    soft_replan_only: bool = False


class RollbackSDITrainer:
    def __init__(self, cfg: RollbackTrainConfig) -> None:
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

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    def _state_text(self, sample: dict[str, Any]) -> str:
        ds = sample.get("decision_state") or {}
        raw = str(
            sample.get("student_state_text") or ds.get("rendered_context") or json.dumps(ds)[:4000]
        )
        reserve = 96
        return self._truncate_text(raw, max(64, self.cfg.max_length - reserve))

    def _checkpoints(self, sample: dict[str, Any]) -> list[dict]:
        ds = sample.get("decision_state") or {}
        return list(ds.get("available_checkpoints") or [])

    def _target_operation(self, sample: dict[str, Any]) -> RollbackOperation | None:
        ta = sample.get("target_action") or {}
        op = str(ta.get("operation") or sample.get("operation") or "").upper()
        if not op:
            return None
        try:
            return RollbackOperation(op)
        except ValueError:
            return None

    def _hint(self) -> str:
        if not self.cfg.hint_distill:
            return ""
        return (
            "Hint: if recent queries repeat or evidence stalls, prefer ROLLBACK_TO "
            "a prior checkpoint instead of continuing the failing branch."
        )

    def _predict_operation(self, sample: dict[str, Any]) -> RollbackOperation:
        text = self._state_text(sample)
        ck = self._checkpoints(sample)
        s0, s1, s2 = score_rollback_operations(
            self.model,
            self.tokenizer,
            text,
            device=self.device,
            available_checkpoints=ck,
            hint=self._hint(),
        )
        logits = torch.stack([s0, s1, s2])
        idx = int(torch.argmax(logits).item())
        return [
            RollbackOperation.CONTINUE,
            RollbackOperation.REPLAN,
            RollbackOperation.ROLLBACK_TO,
        ][idx]

    def _train_accum_step(
        self,
        batch_examples: list[dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        n_active = 0
        for ex in batch_examples:
            tgt = self._target_operation(ex)
            if tgt is None:
                continue
            if self.cfg.soft_replan_only and tgt == RollbackOperation.ROLLBACK_TO:
                continue
            loss = rollback_operation_loss(
                self.model,
                self.tokenizer,
                self._state_text(ex),
                tgt,
                device=self.device,
                available_checkpoints=self._checkpoints(ex),
                hint=self._hint(),
            )
            (loss / max(len(batch_examples), 1)).backward()
            losses.append(float(loss.detach().item()))
            n_active += 1
        if not n_active:
            return {"loss": 0.0}
        optimizer.step()
        scheduler.step()
        return {"loss": sum(losses) / len(losses)}

    def train(self, train_path: Path, valid_path: Path | None = None) -> dict[str, Any]:
        train_ds = SDISampleDataset(train_path)
        rows = train_ds.rows
        if self.cfg.route_filter:
            want = self.cfg.route_filter.upper()
            rows = [r for r in rows if str(r.get("route", "")).upper() == want]
        if self.cfg.soft_replan_only:
            rows = [
                r
                for r in rows
                if str((r.get("target_action") or {}).get("operation", r.get("operation", ""))).upper()
                != RollbackOperation.ROLLBACK_TO.value
            ]
        train_ds.rows = rows

        loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=lambda xs: xs,
        )
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.learning_rate,
        )
        total_steps = max(
            1,
            (len(loader) * self.cfg.num_epochs + self.cfg.grad_accum - 1) // self.cfg.grad_accum,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(total_steps * self.cfg.warmup_ratio),
            total_steps,
        )
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.cfg.num_epochs):
            pending: list[dict[str, Any]] = []
            for batch in loader:
                pending.extend(batch)
                if len(pending) < self.cfg.grad_accum:
                    continue
                self._train_accum_step(pending, optimizer, scheduler)
                pending = []
            if pending:
                self._train_accum_step(pending, optimizer, scheduler)

        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        summary: dict[str, Any] = {
            "output_dir": str(self.cfg.output_dir),
            "n_train": len(train_ds),
        }
        if valid_path and valid_path.exists():
            summary["valid_metrics"] = self.evaluate(valid_path)
        return summary

    def evaluate(self, valid_path: Path) -> dict[str, Any]:
        rows = SDISampleDataset(valid_path).rows
        self.model.eval()
        correct = 0
        n = 0
        with torch.no_grad():
            for row in rows:
                tgt = self._target_operation(row)
                if tgt is None:
                    continue
                pred = self._predict_operation(row)
                n += 1
                if pred == tgt:
                    correct += 1
        return {
            "n": n,
            "operation_accuracy": correct / max(n, 1),
            "balanced_accuracy": correct / max(n, 1),
        }
