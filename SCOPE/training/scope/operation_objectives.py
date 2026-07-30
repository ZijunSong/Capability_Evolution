"""Round 5 operation-level objectives O0–O7 (differentiable scoring paths)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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


class ObjectiveId(str, Enum):
    O0 = "operation_ce"
    O1 = "discriminative_ce"
    O2 = "pairwise_margin"
    O3 = "single_token"
    O4 = "compact_json_sample_norm"
    O5 = "discriminative_ce_sum"
    O6 = "discriminative_ce_mean"
    O7 = "discriminative_ce_r64"


class ScoreNorm(str, Enum):
    MEAN = "mean"
    SUM = "sum"


@dataclass(frozen=True)
class TypedTokenIds:
    keep_id: int
    skip_id: int
    keep_token: str
    skip_token: str


def _encode_ids(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _completion_score(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str,
    *,
    device: torch.device,
    norm: ScoreNorm = ScoreNorm.MEAN,
    differentiable: bool = True,
) -> torch.Tensor:
    prompt_ids = _encode_ids(tokenizer, prompt)
    comp_ids = _encode_ids(tokenizer, completion)
    if not comp_ids:
        return torch.zeros((), device=device)
    input_ids = torch.tensor(
        [prompt_ids + comp_ids], dtype=torch.long, device=device
    )
    logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    start = max(len(prompt_ids) - 1, 0)
    target_labels = shift_labels[:, start:]
    target_logits = shift_logits[:, start:, :]
    if target_labels.numel() == 0:
        return torch.zeros((), device=device)
    log_probs = F.log_softmax(target_logits, dim=-1)
    tok_lp = log_probs.gather(2, target_labels.unsqueeze(-1)).squeeze(-1)
    if norm == ScoreNorm.SUM:
        return tok_lp.sum()
    return tok_lp.mean()


def score_both_operations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    decision_state_text: str,
    *,
    device: torch.device,
    norm: ScoreNorm = ScoreNorm.MEAN,
    candidate_id: str | None = None,
    curated_document_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = format_operation_prompt(
        decision_state_text,
        candidate_id=candidate_id,
        curated_document_ids=curated_document_ids,
    )
    s_keep = _completion_score(
        model, tokenizer, prompt, DupOperation.KEEP_EVIDENCE.value,
        device=device, norm=norm, differentiable=True,
    )
    s_skip = _completion_score(
        model, tokenizer, prompt, DupOperation.SKIP_DUPLICATE.value,
        device=device, norm=norm, differentiable=True,
    )
    return s_keep, s_skip


def resolve_typed_tokens(tokenizer: PreTrainedTokenizerBase) -> TypedTokenIds:
    """Pick two single-token symbols for O3 (verified at runtime)."""
    for keep_tok, skip_tok in (("A", "B"), ("K", "S"), ("+", "-")):
        keep_ids = _encode_ids(tokenizer, keep_tok)
        skip_ids = _encode_ids(tokenizer, skip_tok)
        if len(keep_ids) == 1 and len(skip_ids) == 1 and keep_ids[0] != skip_ids[0]:
            return TypedTokenIds(
                keep_id=keep_ids[0],
                skip_id=skip_ids[0],
                keep_token=keep_tok,
                skip_token=skip_tok,
            )
    raise ValueError("Could not find two distinct single-token symbols for O3")


def operation_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    decision_state_text: str,
    target: DupOperation,
    *,
    objective: str,
    device: torch.device,
    typed_tokens: TypedTokenIds | None = None,
    candidate_id: str | None = None,
    curated_document_ids: list[str] | tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Unified entry for O0–O3, O5–O6."""
    obj = ObjectiveId(objective)

    if obj in (ObjectiveId.O0, ObjectiveId.O1, ObjectiveId.O6, ObjectiveId.O7):
        s_keep, s_skip = score_both_operations(
            model, tokenizer, decision_state_text, device=device, norm=ScoreNorm.MEAN,
            candidate_id=candidate_id, curated_document_ids=curated_document_ids,
        )
    elif obj == ObjectiveId.O5:
        s_keep, s_skip = score_both_operations(
            model, tokenizer, decision_state_text, device=device, norm=ScoreNorm.SUM,
            candidate_id=candidate_id, curated_document_ids=curated_document_ids,
        )
    elif obj == ObjectiveId.O2:
        s_keep, s_skip = score_both_operations(
            model, tokenizer, decision_state_text, device=device, norm=ScoreNorm.MEAN,
            candidate_id=candidate_id, curated_document_ids=curated_document_ids,
        )
        margin = s_skip - s_keep
        if target == DupOperation.KEEP_EVIDENCE:
            return F.softplus(margin)
        return F.softplus(-margin)
    elif obj == ObjectiveId.O3:
        tt = typed_tokens or resolve_typed_tokens(tokenizer)
        prompt = format_operation_prompt(
            decision_state_text,
            candidate_id=candidate_id,
            curated_document_ids=curated_document_ids,
        )
        prompt_ids = _encode_ids(tokenizer, prompt)
        target_id = tt.keep_id if target == DupOperation.KEEP_EVIDENCE else tt.skip_id
        input_ids = torch.tensor([prompt_ids + [target_id]], dtype=torch.long, device=device)
        logits = model(input_ids=input_ids).logits
        last_logits = logits[0, len(prompt_ids) - 1, :]
        pair_logits = torch.stack([last_logits[tt.keep_id], last_logits[tt.skip_id]])
        label = 0 if target == DupOperation.KEEP_EVIDENCE else 1
        return F.cross_entropy(pair_logits.unsqueeze(0), torch.tensor([label], device=device))
    else:
        raise ValueError(f"Unsupported objective for operation_loss: {objective}")

    if obj in (ObjectiveId.O0, ObjectiveId.O1, ObjectiveId.O5, ObjectiveId.O6, ObjectiveId.O7):
        logits = torch.stack([s_keep, s_skip])
        log_probs = F.log_softmax(logits, dim=0)
        idx = 0 if target == DupOperation.KEEP_EVIDENCE else 1
        return -log_probs[idx]

    raise RuntimeError(f"Unhandled objective branch: {objective}")


def objective_math_description(objective: str) -> dict[str, Any]:
    """Document the mathematical form for B2.1."""
    forms = {
        ObjectiveId.O0.value: {
            "form": "CE([s_keep, s_skip], target) where s_c = mean_t log p(v_{c,t}|prompt,v_{c,<t})",
            "note": "Same as O1/O6; legacy name operation_ce",
        },
        ObjectiveId.O1.value: {
            "form": "CE([score_KEEP, score_SKIP], target) with full verbalizer sequence scores",
            "note": "Explicit discriminative two-operation CE",
        },
        ObjectiveId.O2.value: {
            "form": "KEEP: softplus(s_skip - s_keep); SKIP: softplus(s_keep - s_skip)",
            "note": "Pairwise margin logistic",
        },
        ObjectiveId.O3.value: {
            "form": "CE([logit_A, logit_B], target) on single-token typed IDs",
            "note": "Bypasses multi-token verbalizer credit assignment",
        },
        ObjectiveId.O5.value: {
            "form": "CE with SUM logprob (no length normalization)",
            "note": "O1 variant — sum over verbalizer tokens",
        },
        ObjectiveId.O6.value: {
            "form": "CE with MEAN logprob",
            "note": "O1 variant — mean over verbalizer tokens",
        },
    }
    return forms.get(objective, {"form": "unknown", "note": ""})
