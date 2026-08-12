"""Binary KEEP/SKIP classification head for A6 ablation.

Uses the LM's last-layer hidden state at the final prompt token, then a linear
head to 2 classes. This is orthogonal to verbalizer / token-CE objectives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from harness.capability.dup_operation import DupOperation
from training.scope.prompting import format_operation_prompt

CLASS_TO_OP = (DupOperation.KEEP_EVIDENCE, DupOperation.SKIP_DUPLICATE)
OP_TO_CLASS = {
    DupOperation.KEEP_EVIDENCE: 0,
    DupOperation.SKIP_DUPLICATE: 1,
}
HEAD_FILENAME = "classification_head.pt"
HEAD_META_FILENAME = "classification_head.json"


@dataclass
class ClassificationHeadConfig:
    hidden_size: int
    num_classes: int = 2
    dropout: float = 0.0


class DupOperationClassificationHead(nn.Module):
    def __init__(self, cfg: ClassificationHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        if cfg.dropout > 0:
            layers.append(nn.Dropout(cfg.dropout))
        layers.append(nn.Linear(cfg.hidden_size, cfg.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled: [B, H] -> logits [B, C]."""
        return self.classifier(pooled)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / HEAD_FILENAME)
        meta = {
            "hidden_size": self.cfg.hidden_size,
            "num_classes": self.cfg.num_classes,
            "dropout": self.cfg.dropout,
            "class_order": [op.value for op in CLASS_TO_OP],
        }
        (directory / HEAD_META_FILENAME).write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, device: torch.device | str | None = None) -> "DupOperationClassificationHead":
        meta = json.loads((directory / HEAD_META_FILENAME).read_text(encoding="utf-8"))
        head = cls(
            ClassificationHeadConfig(
                hidden_size=int(meta["hidden_size"]),
                num_classes=int(meta.get("num_classes", 2)),
                dropout=float(meta.get("dropout", 0.0)),
            )
        )
        state = torch.load(directory / HEAD_FILENAME, map_location=device or "cpu")
        head.load_state_dict(state)
        if device is not None:
            head.to(device)
        return head


def resolve_hidden_size(model: PreTrainedModel) -> int:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("model has no config; cannot resolve hidden_size")
    for key in ("hidden_size", "d_model", "n_embd"):
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    # PEFT wraps base
    base = getattr(model, "base_model", None) or getattr(model, "model", None)
    if base is not None:
        return resolve_hidden_size(base)
    raise ValueError("unable to resolve hidden_size from model config")


def encode_prompt_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    device: torch.device,
    max_length: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    return input_ids, attention_mask


def pool_last_prompt_hidden(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return [B, H] pooled at last non-pad token."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if out.hidden_states is None:
        raise RuntimeError("model did not return hidden_states")
    hidden = out.hidden_states[-1]  # [B, T, H]
    # index of last attended token
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    bsz = hidden.size(0)
    pooled = hidden[torch.arange(bsz, device=hidden.device), lengths]
    return pooled


def build_operation_prompt(
    decision_state_text: str,
    *,
    candidate_id: str | None = None,
    curated_document_ids: list[str] | tuple[str, ...] | None = None,
) -> str:
    return format_operation_prompt(
        decision_state_text,
        candidate_id=candidate_id,
        curated_document_ids=curated_document_ids,
    )


def classification_logits(
    model: PreTrainedModel,
    head: DupOperationClassificationHead,
    tokenizer: PreTrainedTokenizerBase,
    decision_state_text: str,
    *,
    device: torch.device,
    candidate_id: str | None = None,
    curated_document_ids: list[str] | tuple[str, ...] | None = None,
    max_length: int = 4096,
) -> torch.Tensor:
    prompt = build_operation_prompt(
        decision_state_text,
        candidate_id=candidate_id,
        curated_document_ids=curated_document_ids,
    )
    input_ids, attention_mask = encode_prompt_ids(
        tokenizer, prompt, device=device, max_length=max_length
    )
    pooled = pool_last_prompt_hidden(model, input_ids, attention_mask)
    return head(pooled)  # [1, 2]


def classification_head_loss(
    model: PreTrainedModel,
    head: DupOperationClassificationHead,
    tokenizer: PreTrainedTokenizerBase,
    decision_state_text: str,
    target: DupOperation,
    *,
    device: torch.device,
    candidate_id: str | None = None,
    curated_document_ids: list[str] | tuple[str, ...] | None = None,
    max_length: int = 4096,
) -> torch.Tensor:
    if target not in OP_TO_CLASS:
        raise ValueError(f"unsupported target for classification_head: {target}")
    logits = classification_logits(
        model,
        head,
        tokenizer,
        decision_state_text,
        device=device,
        candidate_id=candidate_id,
        curated_document_ids=curated_document_ids,
        max_length=max_length,
    )
    label = torch.tensor([OP_TO_CLASS[target]], device=device, dtype=torch.long)
    return F.cross_entropy(logits, label)


def predict_operation_from_head(
    model: PreTrainedModel,
    head: DupOperationClassificationHead,
    tokenizer: PreTrainedTokenizerBase,
    decision_state_text: str,
    *,
    device: torch.device,
    candidate_id: str | None = None,
    curated_document_ids: list[str] | tuple[str, ...] | None = None,
    max_length: int = 4096,
) -> DupOperation:
    logits = classification_logits(
        model,
        head,
        tokenizer,
        decision_state_text,
        device=device,
        candidate_id=candidate_id,
        curated_document_ids=curated_document_ids,
        max_length=max_length,
    )
    idx = int(logits.argmax(dim=-1).item())
    return CLASS_TO_OP[idx]


def trainable_parameter_count(modules: list[nn.Module]) -> dict[str, Any]:
    total = 0
    trainable = 0
    for mod in modules:
        for p in mod.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
    return {"total_parameters": total, "trainable_parameters": trainable}
