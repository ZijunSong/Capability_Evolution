"""Convert ProjectedTrainingStep → Tinker-style CE datum.

Teacher-only artifacts must never appear in model_input.
Prompt tokens always have weight 0. Supervised target weights are
normalized so sum(weights) == lambda_opd for one optimizer substep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from scape.training.action_codec import parse_action
from scape.training.opd_dataset import ProjectedTrainingStep, prompt_has_teacher_leak
from scape.training.rl_opd_types import OPD_WEIGHT_NORMALIZATION


EncodeFn = Callable[[str], list[int]]


@dataclass
class TinkerOPDDatum:
    model_input: str
    prompt_token_ids: list[int]
    target_tokens: list[int]
    weights: list[float]
    policy_version: str
    n_supervised_tokens: int
    projection_confidence: float = 1.0
    target_action: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    teacher_prompt_token_ids: list[int] = field(default_factory=list)
    opd_loss: str = "sr_opd_ce"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_input": self.model_input,
            "target_tokens": list(self.target_tokens),
            "weights": list(self.weights),
            "policy_version": self.policy_version,
            "n_supervised_tokens": self.n_supervised_tokens,
            "opd_weight_normalization": OPD_WEIGHT_NORMALIZATION,
        }


def default_encode(text: str) -> list[int]:
    """Deterministic byte-level stand-in when no tokenizer is bound."""
    return list(text.encode("utf-8")) or [0]


def _n_supervised(step: ProjectedTrainingStep, encode: EncodeFn) -> int:
    ids = encode(step.target_text)
    if step.token_mask is None:
        return max(1, len(ids))
    mask = list(step.token_mask)
    if len(mask) != len(ids):
        return max(1, len(ids))
    return max(1, sum(1 for bit in mask if bit))


def build_tinker_opd_datums(
    steps: Sequence[ProjectedTrainingStep],
    *,
    lambda_opd: float,
    encode_fn: EncodeFn | None = None,
    policy_version: str,
    opd_loss: str = "sr_opd_ce",
) -> list[TinkerOPDDatum]:
    """One datum per materialized ALIGN/DIRECT Student tool call."""
    encode = encode_fn or default_encode
    if float(lambda_opd) <= 0.0 or not steps:
        return []

    denom = 0.0
    counts: list[int] = []
    for step in steps:
        if prompt_has_teacher_leak(step.prompt_reduced):
            raise ValueError("teacher-only observation leaked into Student prefix")
        n_tok = _n_supervised(step, encode)
        conf = float(step.projection_confidence if step.projection_confidence else step.weight or 1.0)
        counts.append(n_tok)
        denom += conf * n_tok
    if denom <= 0:
        return []

    datums: list[TinkerOPDDatum] = []
    for step, n_tok in zip(steps, counts):
        prompt_ids = encode(step.prompt_reduced)
        target_ids = encode(step.target_text)
        teacher_prompt = str((step.metadata or {}).get("prompt_full") or "")
        teacher_ids = encode(teacher_prompt) if teacher_prompt else []
        conf = float(step.projection_confidence if step.projection_confidence else step.weight or 1.0)
        token_w = float(lambda_opd) * conf / denom
        mask = list(step.token_mask) if step.token_mask is not None else [True] * len(target_ids)
        if len(mask) != len(target_ids):
            mask = [True] * len(target_ids)
        target_weights = [token_w if bit else 0.0 for bit in mask]
        # Full causal sequence: prompt ignored, only projected action trained.
        full_targets = [0] * len(prompt_ids) + list(target_ids)
        full_weights = [0.0] * len(prompt_ids) + target_weights
        try:
            parsed = parse_action(step.target_text)
        except Exception:
            parsed = dict(step.target_action or {})
        datums.append(
            TinkerOPDDatum(
                model_input=step.prompt_reduced,
                prompt_token_ids=prompt_ids,
                target_tokens=full_targets,
                weights=full_weights,
                policy_version=policy_version,
                n_supervised_tokens=n_tok,
                projection_confidence=conf,
                target_action=parsed,
                teacher_prompt_token_ids=teacher_ids,
                opd_loss=str(opd_loss),
                metadata={
                    "projection_kind": step.projection_kind,
                    "source_event_ids": list(step.source_event_ids),
                    "opd_weight_normalization": OPD_WEIGHT_NORMALIZATION,
                    **dict(step.metadata or {}),
                },
            )
        )
    return datums


def supervised_weight_sum(datums: Sequence[TinkerOPDDatum]) -> float:
    return float(sum(w for d in datums for w in d.weights))
