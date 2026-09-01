"""Convert training steps → Tinker-style OPD datums.

CE path: ProjectedTrainingStep with weights summing to lambda_opd.
SEED sampled-gap path: on-policy action tokens; weights are a 0/1 mask
and lambda_opd is applied at FB time as token-mean.

Teacher-only artifacts must never appear in model_input.
Prompt tokens always have weight 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from scape.training.action_codec import parse_action
from scape.training.opd_dataset import (
    ProjectedTrainingStep,
    prompt_has_teacher_leak,
    render_student_prompt,
    render_teacher_prompt,
)
from scape.training.rl_opd_types import (
    OPD_LOSS_SAMPLED_GAP,
    OPD_WEIGHT_NORMALIZATION,
    SCAPE_RL_OPD_GATE_BETA,
    StudentDecisionPoint,
)


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


def _student_prefix_ids(
    point: StudentDecisionPoint,
    encode: EncodeFn,
    *,
    component_id: str,
) -> tuple[list[int], str]:
    pids = list(getattr(point, "student_prompt_token_ids", None) or [])
    text = point.student_model_input if isinstance(point.student_model_input, str) else ""
    if pids:
        return pids, text
    if not text:
        text = render_student_prompt(point.pre_action_snapshot, component_id=component_id)
    if prompt_has_teacher_leak(text):
        raise ValueError("teacher-only observation leaked into Student prefix")
    return encode(text), text


def build_sampled_opd_datums(
    points: Sequence[StudentDecisionPoint],
    *,
    lambda_opd: float,
    encode_fn: EncodeFn | None = None,
    policy_version: str,
    component_id: str = "",
    gate_beta: float = SCAPE_RL_OPD_GATE_BETA,
    opd_loss: str = OPD_LOSS_SAMPLED_GAP,
) -> list[TinkerOPDDatum]:
    """SEED OPD rows: CISPO sampled action tokens, DualView teacher prefix.

    Token weights are a 0/1 mask. ``lambda_opd`` is applied at FB time as
    ``λ × token-mean(g · (sg[ℓ^T] − ℓ^S))``, not baked into the weights.
    """
    encode = encode_fn or default_encode
    if float(lambda_opd) <= 0.0 or not points:
        return []
    lam = float(lambda_opd)
    beta = float(gate_beta)
    datums: list[TinkerOPDDatum] = []
    for point in points:
        action_ids = [int(x) for x in (point.student_action_tokens or [])]
        if not action_ids:
            continue
        prompt_ids, student_text = _student_prefix_ids(
            point, encode, component_id=component_id
        )
        teacher_text = render_teacher_prompt(
            point.pre_action_snapshot, component_id=component_id
        )
        teacher_ids = encode(teacher_text) if teacher_text else []
        if not teacher_ids:
            continue
        n_tok = len(action_ids)
        full_targets = [0] * len(prompt_ids) + list(action_ids)
        full_weights = [0.0] * len(prompt_ids) + [1.0] * n_tok
        datums.append(
            TinkerOPDDatum(
                model_input=student_text or "",
                prompt_token_ids=list(prompt_ids),
                target_tokens=full_targets,
                weights=full_weights,
                policy_version=policy_version,
                n_supervised_tokens=n_tok,
                projection_confidence=1.0,
                target_action={},
                teacher_prompt_token_ids=list(teacher_ids),
                opd_loss=str(opd_loss),
                metadata={
                    "projection_kind": "sampled_on_policy",
                    "source_event_ids": [],
                    "sampled_action": True,
                    "projector_used": False,
                    "lambda_opd": lam,
                    "gate_beta": beta,
                    "opd_weight_normalization": "seed_token_mean",
                },
            )
        )
    return datums


def supervised_weight_sum(datums: Sequence[TinkerOPDDatum]) -> float:
    return float(sum(w for d in datums for w in d.weights))
