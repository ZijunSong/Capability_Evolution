"""Unit tests for A6 classification_head and sequence_ce_plus_operation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.capability.dup_operation import DupOperation
from training.scope.classification_head import (
    ClassificationHeadConfig,
    DupOperationClassificationHead,
    OP_TO_CLASS,
    trainable_parameter_count,
)
from training.scope.losses import LossMode
from training.scope.operation_objectives import ObjectiveId, objective_math_description


class _TinyLM(nn.Module):
    """Minimal causal LM stub exposing hidden_states for head tests."""

    def __init__(self, hidden: int = 16, vocab: int = 32):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden})()
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        if return_dict:
            return type(
                "O",
                (),
                {
                    "logits": logits,
                    "hidden_states": (h,) if output_hidden_states else None,
                },
            )()
        return logits


class _Tok:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=64, add_special_tokens=True):
        ids = [1, 2, 3, 4, 5]
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def test_loss_mode_enum_has_a6_extensions():
    assert LossMode.CLASSIFICATION_HEAD.value == "classification_head"
    assert LossMode.SEQUENCE_CE_PLUS_OPERATION.value == "sequence_ce_plus_operation"


def test_objective_math_docs():
    d = objective_math_description(ObjectiveId.CLASSIFICATION_HEAD.value)
    assert "h_last_prompt" in d["form"]
    d2 = objective_math_description(ObjectiveId.SEQUENCE_CE_PLUS_OPERATION.value)
    assert "sample_normalized" in d2["form"]


def test_head_forward_and_ce_gradient():
    head = DupOperationClassificationHead(ClassificationHeadConfig(hidden_size=8))
    pooled = torch.randn(2, 8, requires_grad=True)
    logits = head(pooled)
    assert logits.shape == (2, 2)
    labels = torch.tensor([0, 1])
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert pooled.grad is not None
    assert any(p.grad is not None for p in head.parameters())


def test_classification_head_loss_with_tiny_lm():
    from training.scope.classification_head import classification_head_loss, predict_operation_from_head

    model = _TinyLM()
    head = DupOperationClassificationHead(ClassificationHeadConfig(hidden_size=16))
    tok = _Tok()
    device = torch.device("cpu")
    loss = classification_head_loss(
        model,
        head,
        tok,  # type: ignore[arg-type]
        "state text",
        DupOperation.KEEP_EVIDENCE,
        device=device,
    )
    assert loss.ndim == 0
    assert loss.item() >= 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())

    pred = predict_operation_from_head(
        model, head, tok, "state text", device=device  # type: ignore[arg-type]
    )
    assert pred in (DupOperation.KEEP_EVIDENCE, DupOperation.SKIP_DUPLICATE)


def test_head_save_load(tmp_path: Path):
    head = DupOperationClassificationHead(ClassificationHeadConfig(hidden_size=8, dropout=0.1))
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.5)
    head.save(tmp_path)
    loaded = DupOperationClassificationHead.load(tmp_path, device="cpu")
    for a, b in zip(head.parameters(), loaded.parameters()):
        assert torch.allclose(a, b)


def test_op_to_class_bijection():
    assert OP_TO_CLASS[DupOperation.KEEP_EVIDENCE] == 0
    assert OP_TO_CLASS[DupOperation.SKIP_DUPLICATE] == 1


def test_trainable_parameter_count():
    head = DupOperationClassificationHead(ClassificationHeadConfig(hidden_size=4))
    stats = trainable_parameter_count([head])
    assert stats["trainable_parameters"] == 4 * 2 + 2  # Linear(4,2)


def test_sequence_plus_operation_coefs_in_trainer_config():
    from training.scope.sdi_trainer import SDITrainConfig

    cfg = SDITrainConfig(
        model_path="x",
        output_dir=Path("/tmp/unused"),
        loss_mode="sequence_ce_plus_operation",
        sequence_ce_coef=0.3,
        operation_ce_coef=0.7,
    )
    assert cfg.sequence_ce_coef == pytest.approx(0.3)
    assert cfg.operation_ce_coef == pytest.approx(0.7)
