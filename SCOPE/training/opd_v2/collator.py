"""Collate OPDTransitionV2 batches for scoring backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from harness.artifacts.schema import GuidanceMode
from training.opd_v2.transitions import OPDTransitionV2


TokenizeFn = Callable[[str], list[int]]


@dataclass
class CollatedOPDBatch:
    endorse: list[dict[str, Any]]
    correct: list[dict[str, Any]]


def _default_tokenize(text: str) -> list[int]:
    # Deterministic stub tokenizer for offline tests / dry-run
    return [min(255, ord(c)) for c in (text or "")[:128]]


def collate_transitions(
    transitions: list[OPDTransitionV2],
    *,
    tokenize: TokenizeFn | None = None,
    score_fn: Callable[[str, str], list[float]] | None = None,
) -> CollatedOPDBatch:
    """Build endorse/correct scoring payloads.

    score_fn(prefix_text, action_text) -> token logprobs if provided;
    otherwise uses zeros sized by tokenized action length.
    """
    tokenize = tokenize or _default_tokenize
    endorse: list[dict[str, Any]] = []
    correct: list[dict[str, Any]] = []

    for tr in transitions:
        if not tr.validity_mask:
            continue
        action_ids = list(tr.student_action_token_ids) or tokenize(tr.student_action_text)
        if tr.mode == GuidanceMode.ENDORSE:
            if score_fn is not None:
                s_logps = score_fn(tr.student_state_text, tr.student_action_text)
                t_prefix = tr.teacher_state_text or tr.student_state_text
                t_logps = score_fn(t_prefix, tr.student_action_text)
            else:
                s_logps = [0.0] * max(1, len(action_ids))
                t_logps = [0.1] * max(1, len(action_ids))
            endorse.append(
                {
                    "transition_id": tr.transition_id,
                    "module_id": tr.module_id,
                    "student_logps": s_logps,
                    "teacher_logps": t_logps,
                    "validity_mask": tr.validity_mask,
                    "module_weight": tr.module_weight,
                }
            )
        elif tr.mode == GuidanceMode.CORRECT and tr.recommended_action_text:
            rec_ids = (
                list(tr.recommended_action_token_ids)
                if tr.recommended_action_token_ids is not None
                else tokenize(tr.recommended_action_text)
            )
            if score_fn is not None:
                s_orig = score_fn(tr.student_state_text, tr.student_action_text)
                s_rec = score_fn(tr.student_state_text, tr.recommended_action_text)
            else:
                s_orig = [-1.0] * max(1, len(action_ids))
                s_rec = [-0.5] * max(1, len(rec_ids))
            correct.append(
                {
                    "transition_id": tr.transition_id,
                    "module_id": tr.module_id,
                    "student_logps_original": s_orig,
                    "student_logps_recommended": s_rec,
                    "validity_mask": tr.validity_mask,
                    "module_weight": tr.module_weight,
                }
            )
    return CollatedOPDBatch(endorse=endorse, correct=correct)
