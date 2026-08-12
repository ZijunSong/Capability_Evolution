"""Discriminative CE over rollback typed operations."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from harness.capability.rollback_operation import RollbackOperation
from training.scope.operation_objectives import ScoreNorm, _completion_score, _encode_ids


def format_rollback_operation_prompt(
    student_state_text: str,
    *,
    available_checkpoints: list[dict] | None = None,
    hint: str = "",
) -> str:
    parts: list[str] = []
    if hint:
        parts.append(hint.strip())
    ctx = (student_state_text or "").strip()
    if ctx:
        parts.append(ctx)
    if available_checkpoints:
        ids = [str(c.get("checkpoint_id", "")) for c in available_checkpoints[:12]]
        parts.append("Available checkpoint IDs: " + ", ".join(ids))
    parts.append("Recovery operation:")
    return "\n".join(parts)


def score_rollback_prompt(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    device: torch.device,
    norm: ScoreNorm = ScoreNorm.MEAN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score a finalized effective prompt without re-wrapping."""
    scores = []
    for op in (
        RollbackOperation.CONTINUE,
        RollbackOperation.REPLAN,
        RollbackOperation.ROLLBACK_TO,
    ):
        scores.append(
            _completion_score(
                model,
                tokenizer,
                prompt,
                op.value,
                device=device,
                norm=norm,
                differentiable=True,
            )
        )
    return scores[0], scores[1], scores[2]


def score_rollback_prompt_allowed(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    device: torch.device,
    allowed: tuple[RollbackOperation, ...],
    norm: ScoreNorm = ScoreNorm.MEAN,
) -> dict[RollbackOperation, torch.Tensor]:
    """Score only the allowed verbalizers (for binary CONTINUE/ROLLBACK CE)."""
    out: dict[RollbackOperation, torch.Tensor] = {}
    for op in allowed:
        out[op] = _completion_score(
            model,
            tokenizer,
            prompt,
            op.value,
            device=device,
            norm=norm,
            differentiable=True,
        )
    return out


def score_rollback_operations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    decision_state_text: str,
    *,
    device: torch.device,
    available_checkpoints: list[dict] | None = None,
    hint: str = "",
    norm: ScoreNorm = ScoreNorm.MEAN,
    prompt_is_final: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prompt_is_final:
        prompt = decision_state_text
    else:
        prompt = format_rollback_operation_prompt(
            decision_state_text,
            available_checkpoints=available_checkpoints,
            hint=hint,
        )
    return score_rollback_prompt(
        model, tokenizer, prompt, device=device, norm=norm
    )


def rollback_operation_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    decision_state_text: str,
    target: RollbackOperation,
    *,
    device: torch.device,
    available_checkpoints: list[dict] | None = None,
    hint: str = "",
    prompt_is_final: bool = False,
    disable_replan: bool = False,
) -> torch.Tensor:
    if prompt_is_final:
        prompt = decision_state_text
    else:
        prompt = format_rollback_operation_prompt(
            decision_state_text,
            available_checkpoints=available_checkpoints,
            hint=hint,
        )
    if disable_replan:
        allowed = (RollbackOperation.CONTINUE, RollbackOperation.ROLLBACK_TO)
        if target not in allowed:
            # Unsupported gold label under P0 binary policy — ignore gently.
            return torch.tensor(0.0, device=device, requires_grad=True)
        scored = score_rollback_prompt_allowed(
            model, tokenizer, prompt, device=device, allowed=allowed
        )
        logits = torch.stack([scored[op] for op in allowed])
        log_probs = F.log_softmax(logits, dim=0)
        idx = {op: i for i, op in enumerate(allowed)}[target]
        return -log_probs[idx]

    s_cont, s_replan, s_roll = score_rollback_operations(
        model,
        tokenizer,
        decision_state_text,
        device=device,
        available_checkpoints=available_checkpoints,
        hint=hint,
        prompt_is_final=prompt_is_final,
    )
    logits = torch.stack([s_cont, s_replan, s_roll])
    log_probs = F.log_softmax(logits, dim=0)
    idx = {
        RollbackOperation.CONTINUE: 0,
        RollbackOperation.REPLAN: 1,
        RollbackOperation.ROLLBACK_TO: 2,
    }[target]
    return -log_probs[idx]
