"""Dup-only SDI trainer: action-level CE + stabilization KL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from training.scope.collator import collate_sdi_batch
from training.scope.losses import LossMode, SDILossConfig, compute_sdi_loss, operation_balance_weights
from training.scope.compact_target import compact_target_from_sample
from harness.capability.dup_operation import DupOperation
from training.scope.operation_objectives import (
    ObjectiveId,
    operation_loss,
    resolve_typed_tokens,
)
from training.scope.operation_scorer import operation_ce_loss, score_operations


class SDISampleDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


@dataclass
class SDITrainConfig:
    model_path: str
    output_dir: Path
    adapter_path: str | None = None
    eval_only: bool = False
    kl_coef: float = 0.01
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    max_length: int = 4096
    lora_rank: int = 16
    lora_alpha: int = 32
    warmup_ratio: float = 0.03
    device: str = "cuda"
    loss_mode: str = "sample_normalized_action_ce"
    route_balancing: bool = False
    class_balancing: bool = False  # KEEP/SKIP balance for operation_ce
    compact_target: bool = False
    route_filter: str | None = None  # "ENDORSE" | "CORRECT" | None
    seed: int = 42


class DupSDITrainer:
    def __init__(self, cfg: SDITrainConfig) -> None:
        self.cfg = cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if cfg.adapter_path:
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self.model = PeftModel.from_pretrained(base, cfg.adapter_path)
            self.ref_model = None
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
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
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
        self._typed_tokens = None
        if cfg.loss_mode == LossMode.SINGLE_TOKEN.value:
            self._typed_tokens = resolve_typed_tokens(self.tokenizer)

    def _operation_objective_id(self) -> str:
        mode = self.cfg.loss_mode
        mapping = {
            LossMode.OPERATION_CE.value: ObjectiveId.O0.value,
            LossMode.DISCRIMINATIVE_CE.value: ObjectiveId.O1.value,
            LossMode.PAIRWISE_MARGIN.value: ObjectiveId.O2.value,
            LossMode.SINGLE_TOKEN.value: ObjectiveId.O3.value,
            LossMode.DISCRIMINATIVE_CE_SUM.value: ObjectiveId.O5.value,
            LossMode.DISCRIMINATIVE_CE_MEAN.value: ObjectiveId.O6.value,
        }
        return mapping.get(mode, mode)

    def _sample_target_operation(self, sample: dict[str, Any]) -> DupOperation | None:
        compact = compact_target_from_sample(sample)
        if compact is not None:
            return compact.operation
        target = sample.get("target_action") or {}
        op = str(target.get("operation", "")).upper()
        try:
            return DupOperation(op)
        except ValueError:
            return None

    def _state_text(self, sample: dict[str, Any]) -> str:
        return str(
            sample.get("student_state_text")
            or (sample.get("decision_state") or {}).get("rendered_context")
            or ""
        )

    _OPERATION_LOSS_MODES = frozenset({
        LossMode.OPERATION_CE.value,
        LossMode.DISCRIMINATIVE_CE.value,
        LossMode.PAIRWISE_MARGIN.value,
        LossMode.SINGLE_TOKEN.value,
        LossMode.DISCRIMINATIVE_CE_SUM.value,
        LossMode.DISCRIMINATIVE_CE_MEAN.value,
    })

    def _operation_context(self, sample: dict[str, Any]) -> tuple[str | None, list[str]]:
        compact = compact_target_from_sample(sample)
        ds = sample.get("decision_state") or {}
        curated = list(ds.get("curated_document_ids") or ds.get("curated_evidence_ids") or [])
        cid = compact.candidate_id if compact else None
        return cid, curated

    def _forward_loss_operation_ce(
        self, batch_examples: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses: list[torch.Tensor] = []
        ops: list[str] = []
        objective = self._operation_objective_id()
        use_legacy = self.cfg.loss_mode == LossMode.OPERATION_CE.value
        for ex in batch_examples:
            tgt = self._sample_target_operation(ex)
            if tgt is None:
                continue
            ops.append(tgt.value)
            cid, curated = self._operation_context(ex)
            if use_legacy:
                losses.append(
                    operation_ce_loss(
                        self.model,
                        self.tokenizer,
                        self._state_text(ex),
                        tgt,
                        device=self.device,
                        candidate_id=cid,
                        curated_document_ids=curated,
                    )
                )
            else:
                losses.append(
                    operation_loss(
                        self.model,
                        self.tokenizer,
                        self._state_text(ex),
                        tgt,
                        objective=objective,
                        device=self.device,
                        typed_tokens=self._typed_tokens,
                        candidate_id=cid,
                        curated_document_ids=curated,
                    )
                )
        if not losses:
            z = torch.zeros((), device=self.device, requires_grad=True)
            return z, {"loss": 0.0, "sdi_loss": 0.0, "kl_loss": 0.0, "n_active": 0.0}
        w = operation_balance_weights(ops, enabled=self.cfg.class_balancing or self.cfg.route_balancing)
        w = w.to(self.device)
        stacked = torch.stack(losses)
        loss = (stacked * w).sum() / w.sum().clamp(min=1e-8)
        return loss, {
            "loss": float(loss.detach().item()),
            "sdi_loss": float(loss.detach().item()),
            "kl_loss": 0.0,
            "n_active": float(len(losses)),
        }

    def _forward_loss(self, batch_examples: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, float]]:
        if self.cfg.loss_mode in self._OPERATION_LOSS_MODES:
            return self._forward_loss_operation_ce(batch_examples)
        batch = collate_sdi_batch(
            batch_examples,
            self.tokenizer,
            max_length=self.cfg.max_length,
        )
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        labels = batch.labels.to(self.device)
        weights = batch.sample_weights.to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        ref_logits = None
        if self.ref_model is not None:
            with torch.no_grad():
                ref_logits = self.ref_model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
        out = compute_sdi_loss(
            logits,
            labels,
            sample_weights=weights,
            routes=[m.get("route", "") for m in batch.meta],
            ref_logits=ref_logits,
            config=SDILossConfig(
                kl_coef=self.cfg.kl_coef,
                loss_mode=LossMode(self.cfg.loss_mode),
                route_balancing=self.cfg.route_balancing,
            ),
        )
        metrics = {
            "loss": float(out.loss.detach().item()),
            "sdi_loss": float(out.metrics.get("sdi_loss", 0.0)),
            "kl_loss": float(out.metrics.get("kl_loss", 0.0)),
            "n_active": float(out.n_active),
        }
        return out.loss, metrics

    def train(self, train_path: Path, valid_path: Path | None = None) -> dict[str, Any]:
        train_ds = SDISampleDataset(train_path)
        rows = train_ds.rows
        if self.cfg.compact_target:
            from training.scope.compact_target import apply_compact_target_to_sample

            rows = [apply_compact_target_to_sample(r) for r in rows]
            train_ds.rows = rows
        if self.cfg.route_filter:
            want = self.cfg.route_filter.upper()
            rows = [r for r in rows if str(r.get("route", "")).upper() == want]
            train_ds.rows = rows
        train_loader = DataLoader(
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
            (len(train_loader) * self.cfg.num_epochs + self.cfg.grad_accum - 1)
            // self.cfg.grad_accum,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(total_steps * self.cfg.warmup_ratio),
            total_steps,
        )

        history: list[dict[str, float]] = []
        global_step = 0
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.cfg.num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            pending: list[dict[str, Any]] = []
            for batch in train_loader:
                pending.extend(batch)
                if len(pending) < self.cfg.grad_accum:
                    continue
                metrics = self._train_accum_step(pending, optimizer, scheduler)
                pending = []
                global_step += 1
                epoch_loss += metrics["loss"]
                n_batches += 1
                if global_step % 10 == 0:
                    print(
                        f"[sdi] epoch={epoch+1} step={global_step} "
                        f"loss={metrics['loss']:.4f} kl={metrics['kl_loss']:.4f}",
                        flush=True,
                    )
            if pending:
                metrics = self._train_accum_step(pending, optimizer, scheduler)
                epoch_loss += metrics["loss"]
                n_batches += 1
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": epoch_loss / max(n_batches, 1),
                }
            )

        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        summary = {
            "history": history,
            "output_dir": str(self.cfg.output_dir),
            "n_train": len(train_ds),
        }
        if valid_path and valid_path.exists():
            summary["valid_metrics"] = self.evaluate(valid_path)
        (self.cfg.output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary

    def _train_accum_step(
        self,
        examples: list[dict[str, Any]],
        optimizer: torch.optim.Optimizer,
        scheduler,
    ) -> dict[str, float]:
        metrics_acc = {"loss": 0.0, "kl_loss": 0.0, "sdi_loss": 0.0, "n_active": 0.0}
        optimizer.zero_grad(set_to_none=True)
        for ex in examples:
            loss, m = self._forward_loss([ex])
            (loss / len(examples)).backward()
            for k in metrics_acc:
                metrics_acc[k] += m.get(k, 0.0)
        optimizer.step()
        scheduler.step()
        return {k: v / len(examples) for k, v in metrics_acc.items()}

    def _action_dict(self, action: dict[str, Any] | None) -> dict[str, Any] | None:
        if not action:
            return None
        from harness.capability.action_space import CapabilityAction

        return CapabilityAction.from_dict(action).to_dict()

    def _teacher_forced_token_acc(
        self, sample: dict[str, Any]
    ) -> tuple[float, int]:
        batch = collate_sdi_batch(
            [sample], self.tokenizer, max_length=self.cfg.max_length
        )
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        labels = batch.labels.to(self.device)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        active = shift_labels != -100
        n_active = int(active.sum().item())
        if n_active == 0:
            return 0.0, 0
        preds = shift_logits.argmax(dim=-1)
        correct = ((preds == shift_labels) & active).sum().item()
        return correct / n_active, n_active

    @torch.no_grad()
    def evaluate(self, valid_path: Path) -> dict[str, float]:
        ds = SDISampleDataset(valid_path)
        self.model.eval()
        total = 0.0
        n = 0
        correct = 0
        endorse_ok = endorse_n = 0
        correct_ok = correct_n = 0
        parse_ok = 0
        tf_tok_correct = 0
        tf_tok_total = 0
        for i in range(len(ds)):
            single = ds[i]
            _, m = self._forward_loss([single])
            total += m["loss"]
            n += 1
            tf_acc, tf_n = self._teacher_forced_token_acc(single)
            tf_tok_correct += tf_acc * tf_n
            tf_tok_total += tf_n
            pred = self._greedy_action(single)
            if pred is not None:
                parse_ok += 1
            tgt = self._action_dict(single.get("target_action"))
            pred_norm = self._action_dict(pred) if pred else None
            if pred_norm == tgt:
                correct += 1
            route = str(single.get("route", "")).upper()
            if route == "ENDORSE":
                endorse_n += 1
                if pred_norm == self._action_dict(single.get("student_action")):
                    endorse_ok += 1
            elif route == "CORRECT":
                correct_n += 1
                if pred_norm == tgt:
                    correct_ok += 1
        self.model.train()
        return {
            "loss": total / max(n, 1),
            "teacher_forced_token_acc": tf_tok_correct / max(tf_tok_total, 1),
            "greedy_parse_rate": parse_ok / max(n, 1),
            "action_match_rate": correct / max(n, 1),
            "endorse_accuracy": endorse_ok / max(endorse_n, 1),
            "correct_accuracy": correct_ok / max(correct_n, 1),
            "n": float(n),
        }

    @staticmethod
    def _extract_action_prefix(text: str) -> str:
        line = text.split("\n", 1)[0].strip()
        if not line.startswith("{"):
            return line
        depth = 0
        for i, ch in enumerate(line):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return line[: i + 1]
        return line

    def _greedy_action(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        from harness.capability.adapters import parse_policy_action
        from training.scope.compact_target import infer_operation_from_action
        from training.scope.prompting import format_compact_prompt, format_sdi_prompt

        state_text = self._state_text(sample)
        if self.cfg.loss_mode in self._OPERATION_LOSS_MODES:
            self.model.eval()
            if self.cfg.loss_mode == LossMode.SINGLE_TOKEN.value:
                from training.scope.prompting import format_operation_prompt

                tt = self._typed_tokens or resolve_typed_tokens(self.tokenizer)
                cid, curated = self._operation_context(sample)
                prompt = format_operation_prompt(
                    state_text, candidate_id=cid, curated_document_ids=curated,
                )
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                input_ids = torch.tensor(
                    [prompt_ids], dtype=torch.long, device=self.device
                )
                logits = self.model(input_ids=input_ids).logits
                last_logits = logits[0, -1, :]
                pred_id = (
                    tt.keep_id
                    if last_logits[tt.keep_id] >= last_logits[tt.skip_id]
                    else tt.skip_id
                )
                op = (
                    DupOperation.KEEP_EVIDENCE
                    if pred_id == tt.keep_id
                    else DupOperation.SKIP_DUPLICATE
                )
                return {"operation": op.value}
            cid, curated = self._operation_context(sample)
            result = score_operations(
                self.model, self.tokenizer, state_text, device=self.device,
                candidate_id=cid,
                curated_document_ids=curated,
            )
            return {"operation": result.predicted.value}

        target = sample.get("target_action") or {}
        if infer_operation_from_action(target) is not None or (
            (sample.get("metadata") or {}).get("target_format") == "compact_operation"
        ):
            prompt = format_compact_prompt(state_text)
        else:
            prompt = format_sdi_prompt(state_text)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = self.tokenizer.decode(
            out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        text = self._extract_action_prefix(gen.strip())
        if text.startswith("{") and "operation" in text:
            try:
                import json

                return json.loads(text)
            except json.JSONDecodeError:
                pass
        cap = parse_policy_action(text)
        return cap.to_dict() if cap else None
