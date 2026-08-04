"""Restricted verbalizer operation scorer — shared by training and inference.

Scores KEEP_EVIDENCE vs SKIP_DUPLICATE with length-normalized log-probability:

    s(c) = (1/|v_c|) * sum_j log p(v_{c,j} | d_t, v_{c,<j})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from harness.capability.dup_operation import DupOperation
from training.scope.prompting import format_operation_prompt, format_operation_prompt_from_sample

VERBALIZERS: tuple[DupOperation, ...] = (
    DupOperation.KEEP_EVIDENCE,
    DupOperation.SKIP_DUPLICATE,
)


@dataclass(frozen=True)
class OperationScoreResult:
    scores: dict[str, float]
    predicted: DupOperation
    log_probs: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "predicted": self.predicted.value,
            "log_probs": dict(self.log_probs),
        }


def _completion_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    *,
    device: torch.device,
) -> tuple[float, int]:
    """Length-normalized mean log-prob of completion tokens given prompt."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    comp_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not comp_ids:
        return 0.0, 0
    input_ids = torch.tensor(
        [prompt_ids + comp_ids], dtype=torch.long, device=device
    )
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    # Causal shift: predict token t from position t-1
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    start = len(prompt_ids) - 1
    if start < 0:
        start = 0
    target_labels = shift_labels[:, start:]
    target_logits = shift_logits[:, start:, :]
    if target_labels.numel() == 0:
        return 0.0, 0
    log_probs = F.log_softmax(target_logits, dim=-1)
    tok_lp = log_probs.gather(2, target_labels.unsqueeze(-1)).squeeze(-1)
    n_tok = target_labels.shape[1]
    mean_lp = float(tok_lp.sum().item()) / max(n_tok, 1)
    return mean_lp, n_tok


def score_rendered_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    rendered_prompt: str,
    *,
    device: torch.device | None = None,
    verbalizers: tuple[DupOperation, ...] = VERBALIZERS,
) -> OperationScoreResult:
    """Score verbalizers on an already-rendered operation prompt (exact replay)."""
    dev = device or next(model.parameters()).device
    scores: dict[str, float] = {}
    log_probs: dict[str, float] = {}
    for op in verbalizers:
        lp, n_tok = _completion_logprob(
            model, tokenizer, rendered_prompt, op.value, device=dev
        )
        scores[op.value] = lp
        log_probs[op.value] = lp * max(n_tok, 1)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return OperationScoreResult(
        scores=scores,
        predicted=DupOperation(best),
        log_probs=log_probs,
    )


def score_operations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    decision_state_text: str,
    *,
    device: torch.device | None = None,
    verbalizers: tuple[DupOperation, ...] = VERBALIZERS,
    candidate_id: str | None = None,
    curated_document_ids: list[str] | tuple[str, ...] | None = None,
) -> OperationScoreResult:
    """Score each verbalizer; return length-normalized scores and argmax."""
    dev = device or next(model.parameters()).device
    prompt = format_operation_prompt(
        decision_state_text,
        candidate_id=candidate_id,
        curated_document_ids=curated_document_ids,
    )
    scores: dict[str, float] = {}
    log_probs: dict[str, float] = {}
    for op in verbalizers:
        lp, n_tok = _completion_logprob(
            model, tokenizer, prompt, op.value, device=dev
        )
        scores[op.value] = lp  # already length-normalized (mean per token)
        log_probs[op.value] = lp * max(n_tok, 1)
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return OperationScoreResult(
        scores=scores,
        predicted=DupOperation(best),
        log_probs=log_probs,
    )


def operation_ce_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    decision_state_text: str,
    target: DupOperation,
    *,
    device: torch.device | None = None,
    candidate_id: str | None = None,
    curated_document_ids: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """-log softmax(s)[target] with differentiable completion scoring."""
    dev = device or next(model.parameters()).device
    prompt = format_operation_prompt(
        decision_state_text,
        candidate_id=candidate_id,
        curated_document_ids=curated_document_ids,
    )
    score_tensors: list[torch.Tensor] = []
    for op in VERBALIZERS:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        comp_ids = tokenizer.encode(op.value, add_special_tokens=False)
        input_ids = torch.tensor(
            [prompt_ids + comp_ids], dtype=torch.long, device=dev
        )
        logits = model(input_ids=input_ids).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        start = max(len(prompt_ids) - 1, 0)
        target_labels = shift_labels[:, start:]
        target_logits = shift_logits[:, start:, :]
        if target_labels.numel() == 0:
            score_tensors.append(torch.zeros((), device=dev))
            continue
        log_probs = F.log_softmax(target_logits, dim=-1)
        tok_lp = log_probs.gather(2, target_labels.unsqueeze(-1)).squeeze(-1)
        mean_lp = tok_lp.mean()
        score_tensors.append(mean_lp)
    stacked = torch.stack(score_tensors)
    log_probs = F.log_softmax(stacked, dim=0)
    idx = list(VERBALIZERS).index(target)
    return -log_probs[idx]
