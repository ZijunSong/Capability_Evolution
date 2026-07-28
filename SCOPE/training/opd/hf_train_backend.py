"""HF Transformers training backend: log-prob scoring and OPD backward only.

Rollout/generation must go through vLLM (see vllm_rollout_backend.py).
Designed to be wrapped by FSDP2 in production (see enable_fsdp stub).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.opd._policy_backend import OPDTransition, TrainBackend


class HFTrainBackend(TrainBackend):
    """HF causal LM for score_tokens + train_step (no generate())."""

    def __init__(
        self,
        model_path: str,
        *,
        device_map: str | dict[str, int] = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        trainable: bool = True,
        learning_rate: float = 1e-5,
        freeze: bool = False,
    ) -> None:
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.trainable = trainable and not freeze
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        if freeze or not self.trainable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        self.optimizer: torch.optim.Optimizer | None = None
        self._fsdp_wrapped = False

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def score_tokens(
        self, prefix_ids: list[int], target_ids: list[int]
    ) -> list[float]:
        if not target_ids:
            return []
        full = prefix_ids + target_ids
        inp = torch.tensor([full], device=self._device())
        with torch.no_grad():
            logits = self.model(inp).logits[0]
        start = len(prefix_ids) - 1
        logps: list[float] = []
        for i, token_id in enumerate(target_ids):
            pos = start + i
            logps.append(F.log_softmax(logits[pos], dim=-1)[token_id].item())
        return logps

    def train_step(
        self, batch: list[OPDTransition], loss_config: dict[str, Any]
    ) -> dict[str, float]:
        if not self.trainable:
            raise RuntimeError("train_step called on frozen TrainBackend")
        if not batch:
            return {"loss": 0.0, "batch_size": 0.0}

        self.model.train()
        if self.optimizer is None:
            lr = float(loss_config.get("lr", self.learning_rate))
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(params, lr=lr)

        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        for transition in batch:
            token_ids = transition.student_input_ids + transition.action_ids
            if not token_ids:
                continue
            inp = torch.tensor([token_ids], device=self._device())
            labels = torch.full_like(inp, -100)
            action_start = len(transition.student_input_ids)
            for i, token_id in enumerate(transition.action_ids):
                labels[0, action_start + i] = token_id
            outputs = self.model(inp, labels=labels)
            if outputs.loss is None:
                continue
            outputs.loss.backward()
            total_loss += float(outputs.loss.item())

        if total_loss > 0:
            self.optimizer.step()
        self.model.eval()
        return {
            "loss": total_loss / max(len(batch), 1),
            "batch_size": float(len(batch)),
        }

    def enable_fsdp(self) -> None:
        """Optional FSDP2 wrap — call before first train_step in distributed runs."""
        if self._fsdp_wrapped:
            return
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError as exc:
            raise RuntimeError("FSDP unavailable in this PyTorch build") from exc
        self.model = FSDP(self.model)
        self._fsdp_wrapped = True


def build_frozen_ref(model_path: str, **kwargs: Any) -> HFTrainBackend:
    """Frozen reference model for teacher privileged-context scoring."""
    return HFTrainBackend(model_path, trainable=False, freeze=True, **kwargs)


# Deprecated alias
HFPolicyBackend = HFTrainBackend
