"""Convert training steps → Tinker-style OPD datums.

CE path: ProjectedTrainingStep with weights summing to lambda_opd.
SEED sampled-gap path: on-policy action tokens; weights are a 0/1 mask
and lambda_opd is applied at FB time as token-mean.
Projected-seed path: same SEED token-mean on projector outputs a*.

Teacher-only artifacts must never appear in model_input.
Prompt tokens always have weight 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from trim.training.action_encoding import (
    assert_action_visible,
    encode_supervised_action,
    prompt_ids_hash,
    resolve_effective_weight,
    resolve_student_prompt_ids,
    visible_doc_ids_for_decision,
)
from trim.training.opd_dataset import (
    ProjectedTrainingStep,
    prompt_has_teacher_leak,
    render_student_prompt,
)
from trim.training.rl_opd_types import (
    OPD_LOSS_PROJECTED_GAP,
    OPD_LOSS_SAMPLED_GAP,
    OPD_WEIGHT_NORMALIZATION,
    PROJECTED_GAP_OBJECTIVE_VERSION,
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
        n_p = len(self.prompt_token_ids)
        meta = dict(self.metadata or {})
        return {
            "schema_version": str(meta.get("schema_version") or "tinker_opd_datum_v2"),
            "model_input": self.model_input,
            "prompt_token_ids": list(self.prompt_token_ids),
            "prompt_ids": list(self.prompt_token_ids),
            "target_tokens": list(self.target_tokens),
            "target_ids": list(self.target_tokens[n_p:] if len(self.target_tokens) >= n_p else self.target_tokens),
            "weights": list(self.weights),
            "policy_version": self.policy_version,
            "n_supervised_tokens": self.n_supervised_tokens,
            "projection_confidence": self.projection_confidence,
            "target_action": dict(self.target_action or {}),
            "metadata": meta,
            "teacher_prompt_token_ids": list(self.teacher_prompt_token_ids or []),
            "teacher_prompt_ids": list(self.teacher_prompt_token_ids or []),
            "opd_loss": self.opd_loss,
            "loss_id": self.opd_loss,
            "lambda_opd": meta.get("lambda_opd"),
            "gate_beta": meta.get("gate_beta"),
            "opd_weight_normalization": meta.get("opd_weight_normalization", OPD_WEIGHT_NORMALIZATION),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TinkerOPDDatum":
        loss = str(raw.get("opd_loss") or raw.get("loss_id") or "")
        if not loss:
            raise ValueError("TinkerOPDDatum.from_dict missing loss_id / opd_loss; refuse CE fallback")
        prompt = [int(x) for x in list(raw.get("prompt_token_ids") or raw.get("prompt_ids") or [])]
        targets = [int(x) for x in list(raw.get("target_tokens") or [])]
        if not targets:
            resp = [int(x) for x in list(raw.get("target_ids") or [])]
            targets = [0] * len(prompt) + resp
        weights = [float(x) for x in list(raw.get("weights") or [])]
        meta = dict(raw.get("metadata") or {})
        if raw.get("lambda_opd") is not None:
            meta.setdefault("lambda_opd", float(raw["lambda_opd"]))
        if raw.get("gate_beta") is not None:
            meta.setdefault("gate_beta", float(raw["gate_beta"]))
        teacher = [int(x) for x in list(raw.get("teacher_prompt_token_ids") or raw.get("teacher_prompt_ids") or [])]
        return cls(
            model_input=str(raw.get("model_input") or ""),
            prompt_token_ids=prompt,
            target_tokens=targets,
            weights=weights,
            policy_version=str(raw.get("policy_version") or ""),
            n_supervised_tokens=int(raw.get("n_supervised_tokens") or 0),
            projection_confidence=float(raw.get("projection_confidence") or 1.0),
            target_action=dict(raw.get("target_action") or {}),
            metadata=meta,
            teacher_prompt_token_ids=teacher,
            opd_loss=loss,
        )


def default_encode(text: str) -> list[int]:
    """Deterministic byte-level stand-in when no tokenizer is bound."""
    return list(text.encode("utf-8")) or [0]


def _supervised_mask(step: ProjectedTrainingStep, target_ids: Sequence[int]) -> list[bool]:
    if step.token_mask is None:
        return [True] * len(target_ids)
    mask = [bool(bit) for bit in list(step.token_mask)]
    if len(mask) != len(target_ids):
        raise ValueError(
            f"CE target/mask length mismatch: mask={len(mask)} target={len(target_ids)}"
        )
    return mask


def _n_supervised_from_mask(mask: Sequence[bool]) -> int:
    return int(sum(1 for bit in mask if bit))


def _step_snapshot(step: ProjectedTrainingStep) -> Any:
    raw = getattr(step, "student_snapshot", None)
    if not raw:
        return None
    if not isinstance(raw, dict):
        return raw
    from trim.state.snapshot import EnvironmentSnapshot

    try:
        return EnvironmentSnapshot.from_dict(raw)
    except Exception:
        return None


def teacher_context_hash(
    *,
    teacher_ids: Sequence[int],
    target_ids: Sequence[int],
    policy_version: str,
    branch_id: str = "",
    source_type: str = "",
    template: str = "",
) -> str:
    """Cache/version key: prompt IDs + target + branch, not snapshot hash alone."""
    import hashlib
    import json

    payload = {
        "teacher_ids": [int(x) for x in teacher_ids],
        "target_ids": [int(x) for x in target_ids],
        "policy_version": str(policy_version),
        "branch_id": str(branch_id),
        "source_type": str(source_type),
        "template": str(template),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def recover_teacher_prompt_ids(
    *,
    teacher_ids: Sequence[int] | None,
    metadata: dict[str, Any] | None,
    snapshot: Any = None,
    encode: EncodeFn,
    model_enc: Any | None = None,
    allow_offline: bool = False,
) -> list[int]:
    """Use collected teacher IDs or recoverable decision-time refs. No student copy.

    Explicit token IDs win. Online training ignores ``prompt_full`` debug text and
    keeps recovering from the decision-time snapshot. Pass ``allow_offline=True``
    only for archived datums that never stored token IDs; that path may encode
    ``prompt_full``. Do not copy the student prefix into the teacher prefix.
    """
    if teacher_ids:
        return [int(x) for x in list(teacher_ids)]
    meta = dict(metadata or {})
    if meta.get("teacher_prompt_token_ids"):
        return [int(x) for x in list(meta["teacher_prompt_token_ids"])]
    prompt_full = str(meta.get("prompt_full") or "")
    if prompt_full and allow_offline:
        return list(encode(prompt_full))
    snap = snapshot
    if snap is None:
        return []
    if isinstance(snap, dict):
        from trim.state.snapshot import EnvironmentSnapshot

        try:
            snap = EnvironmentSnapshot.from_dict(snap)
        except Exception:
            return []
    from trim.training.opd_dataset import snapshot_query_text
    from trim.training.opd_prompt_encoding import (
        encode_teacher_rollout_style_prompt,
        snapshot_teacher_wm_text,
    )
    from trim.training.opd_prompt_encoding import _snapshot_acts_and_wm

    acts, _student_wm = _snapshot_acts_and_wm(snap)
    teacher_wm = snapshot_teacher_wm_text(snap)
    if not acts and not teacher_wm:
        return []
    if model_enc is None:
        return []
    ids, _ = encode_teacher_rollout_style_prompt(
        model_enc,
        snapshot_query_text(snap),
        acts=acts,
        wm_text=teacher_wm,
    )
    return list(ids)


def explain_teacher_recovery_failure(
    *,
    metadata: dict[str, Any] | None,
    snapshot: Any = None,
    model_enc: Any | None = None,
    allow_offline: bool = False,
) -> str:
    """Why ``recover_teacher_prompt_ids`` returned no IDs. Does not invent context."""
    meta = dict(metadata or {})
    prompt_full = str(meta.get("prompt_full") or "")
    if prompt_full and allow_offline:
        return "offline_prompt_full_empty"
    snap = snapshot
    if snap is None:
        if prompt_full:
            return "online_debug_prompt_full_without_snapshot"
        return "missing_snapshot"
    if isinstance(snap, dict):
        from trim.state.snapshot import EnvironmentSnapshot

        try:
            snap = EnvironmentSnapshot.from_dict(snap)
        except Exception:
            return "snapshot_decode_failed"
    from trim.training.opd_prompt_encoding import (
        _snapshot_acts_and_wm,
        snapshot_teacher_wm_text,
    )

    acts, _student_wm = _snapshot_acts_and_wm(snap)
    teacher_wm = snapshot_teacher_wm_text(snap)
    if not acts and not teacher_wm:
        if prompt_full and not allow_offline:
            return "online_debug_prompt_full_snapshot_has_no_teacher_context"
        return "snapshot_missing_teacher_context"
    if model_enc is None:
        return "missing_model_encoder"
    return "snapshot_encode_empty"


def build_tinker_opd_datums(
    steps: Sequence[ProjectedTrainingStep],
    *,
    lambda_opd: float,
    encode_fn: EncodeFn | None = None,
    policy_version: str,
    opd_loss: str = "sr_opd_ce",
    model_enc: Any | None = None,
    allow_offline: bool = False,
) -> list[TinkerOPDDatum]:
    """One datum per materialized ALIGN/DIRECT Student tool call.

    On-policy Student prompt IDs are the only authority for the CE prefix.
    ``lambda_opd`` is baked into token weights once; effective_weight already
    includes confidence and same-state candidate sharing.
    """
    encode = encode_fn or default_encode
    if float(lambda_opd) <= 0.0 or not steps:
        return []

    prepared: list[tuple[ProjectedTrainingStep, list[int], list[int], list[bool], float, int, dict[str, Any]]] = []
    denom = 0.0
    for step in steps:
        if prompt_has_teacher_leak(step.prompt_reduced):
            raise ValueError("teacher-only observation leaked into Student prefix")
        meta = dict(step.metadata or {})
        prompt_ids = resolve_student_prompt_ids(
            metadata=meta,
            prompt_reduced=step.prompt_reduced,
            encode=encode,
            allow_offline=allow_offline,
        )
        encoded = encode_supervised_action(
            target_action=step.target_action,
            target_text=step.target_text,
            encode=encode,
            model_enc=model_enc,
        )
        target_ids = list(encoded.target_ids)
        target_text = encoded.target_text
        if not target_ids:
            continue
        step.target_text = target_text
        mask = _supervised_mask(step, target_ids) if step.token_mask is not None else list(encoded.supervision_mask)
        n_tok = _n_supervised_from_mask(mask)
        if n_tok <= 0:
            continue
        weight = resolve_effective_weight(
            weight=step.weight,
            projection_confidence=step.projection_confidence,
            metadata=meta,
        )
        if weight == 0.0:
            continue
        visible = visible_doc_ids_for_decision(
            snapshot=_step_snapshot(step),
            prompt_ids=prompt_ids,
            enc=model_enc,
            metadata=meta,
        )
        if visible is not None and step.target_action:
            assert_action_visible(step.target_action, visible)
        prepared.append((step, prompt_ids, target_ids, mask, weight, n_tok, encoded.canonical_action))
        denom += weight * n_tok
    if denom <= 0 or not prepared:
        return []

    datums: list[TinkerOPDDatum] = []
    for step, prompt_ids, target_ids, mask, weight, n_tok, canon in prepared:
        meta = dict(step.metadata or {})
        teacher_ids = recover_teacher_prompt_ids(
            teacher_ids=meta.get("teacher_prompt_token_ids"),
            metadata=meta,
            snapshot=_step_snapshot(step),
            encode=encode,
            model_enc=model_enc,
            allow_offline=allow_offline,
        )
        token_w = float(lambda_opd) * weight / denom
        target_weights = [token_w if bit else 0.0 for bit in mask]
        meta.update(
            {
                "projection_kind": step.projection_kind,
                "source_event_ids": list(step.source_event_ids),
                "opd_weight_normalization": OPD_WEIGHT_NORMALIZATION,
                "effective_weight": weight,
                "projected_action_token_ids": list(target_ids),
                "prompt_hash": prompt_ids_hash(prompt_ids),
                "supervised_mask": [1 if bit else 0 for bit in mask],
                "lambda_opd": float(lambda_opd),
                "termination_kind": "eos" if model_enc is not None else "none",
            }
        )
        datums.append(
            TinkerOPDDatum(
                model_input=step.prompt_reduced,
                prompt_token_ids=list(prompt_ids),
                target_tokens=[0] * len(prompt_ids) + list(target_ids),
                weights=[0.0] * len(prompt_ids) + target_weights,
                policy_version=policy_version,
                n_supervised_tokens=n_tok,
                projection_confidence=weight,
                target_action=dict(canon or step.target_action or {}),
                teacher_prompt_token_ids=teacher_ids,
                opd_loss=str(opd_loss),
                metadata=meta,
            )
        )
    return datums


def build_projected_seed_datums(
    steps: Sequence[ProjectedTrainingStep],
    *,
    lambda_opd: float,
    encode_fn: EncodeFn | None = None,
    policy_version: str,
    gate_beta: float = SCAPE_RL_OPD_GATE_BETA,
    opd_loss: str = OPD_LOSS_PROJECTED_GAP,
    model_enc: Any | None = None,
    allow_offline: bool = False,
) -> tuple[list[TinkerOPDDatum], dict[str, Any]]:
    """SEED-scale OPD on projected student-legal actions.

    Target tokens are the projected action ``a*``, not the CISPO sample.
    Token weights are ``effective_weight × mask`` and do **not** include
    lambda or the global Z_gap; ``λ / Z_gap × Σ w g (sg[ℓ^T] − ℓ^S)`` is
    applied at FB time. Student prefix is the on-policy reduced IDs;
    teacher prefix is DualView ``H_full`` from decision-time refs.
    """
    encode = encode_fn or default_encode
    stats = {
        "n_skip_missing_teacher": 0,
        "n_skip_missing_student": 0,
        "n_skip_zero_mask": 0,
        "n_skip_empty_target": 0,
        "n_kept": 0,
        "objective_version": PROJECTED_GAP_OBJECTIVE_VERSION,
        "lambda_opd": float(lambda_opd),
        "gate_beta": float(gate_beta),
    }
    if float(lambda_opd) <= 0.0 or not steps:
        return [], stats
    lam = float(lambda_opd)
    beta = float(gate_beta)
    datums: list[TinkerOPDDatum] = []

    for step in steps:
        meta = dict(step.metadata or {})
        if prompt_has_teacher_leak(step.prompt_reduced):
            raise ValueError("teacher-only observation leaked into Student prefix")
        if meta.get("student_prompt_token_ids"):
            prompt_ids = [int(x) for x in list(meta["student_prompt_token_ids"])]
        else:
            stats["n_skip_missing_student"] += 1
            continue
        encoded = encode_supervised_action(
            target_action=step.target_action,
            target_text=step.target_text,
            encode=encode,
            model_enc=model_enc,
        )
        target_ids = list(encoded.target_ids)
        if not target_ids:
            stats["n_skip_empty_target"] += 1
            continue
        teacher_ids = recover_teacher_prompt_ids(
            teacher_ids=meta.get("teacher_prompt_token_ids"),
            metadata=meta,
            snapshot=_step_snapshot(step),
            encode=encode,
            model_enc=model_enc,
            allow_offline=allow_offline,
        )
        if not teacher_ids:
            stats["n_skip_missing_teacher"] += 1
            reason = explain_teacher_recovery_failure(
                metadata=meta,
                snapshot=_step_snapshot(step),
                model_enc=model_enc,
                allow_offline=allow_offline,
            )
            reasons = stats.setdefault("missing_teacher_reasons", {})
            reasons[reason] = int(reasons.get(reason) or 0) + 1
            continue
        mask = list(step.token_mask) if step.token_mask is not None else list(encoded.supervision_mask)
        if len(mask) != len(target_ids):
            raise ValueError("projected target/mask length mismatch")
        n_tok = _n_supervised_from_mask(mask)
        weight = resolve_effective_weight(
            weight=step.weight,
            projection_confidence=step.projection_confidence,
            metadata=meta,
        )
        if weight == 0.0 or n_tok <= 0:
            stats["n_skip_zero_mask"] += 1
            continue
        visible = visible_doc_ids_for_decision(
            snapshot=_step_snapshot(step),
            prompt_ids=prompt_ids,
            enc=model_enc,
            metadata=meta,
        )
        if visible is not None and step.target_action:
            assert_action_visible(step.target_action, visible)
        branch_id = str(meta.get("branch_id") or meta.get("component_id") or "")
        ctx_hash = teacher_context_hash(
            teacher_ids=teacher_ids,
            target_ids=target_ids,
            policy_version=policy_version,
            branch_id=branch_id,
            source_type=str(meta.get("teacher_branch_source_type") or ""),
        )
        meta.update(
            {
                "projection_kind": step.projection_kind,
                "source_event_ids": list(step.source_event_ids),
                "sampled_action": False,
                "projector_used": True,
                "target_source": "projected",
                "lambda_opd": lam,
                "gate_beta": beta,
                "opd_weight_normalization": "seed_token_mean",
                "objective_version": PROJECTED_GAP_OBJECTIVE_VERSION,
                "schema_version": "tinker_opd_datum_v2",
                "effective_weight": weight,
                "projected_action_token_ids": list(target_ids),
                "supervised_mask": [1 if bit else 0 for bit in mask],
                "context_hash": ctx_hash,
                "loss_id": str(opd_loss),
            }
        )
        datums.append(
            TinkerOPDDatum(
                model_input=step.prompt_reduced,
                prompt_token_ids=prompt_ids,
                target_tokens=[0] * len(prompt_ids) + list(target_ids),
                weights=[0.0] * len(prompt_ids) + [float(weight) if bit else 0.0 for bit in mask],
                policy_version=policy_version,
                n_supervised_tokens=n_tok,
                projection_confidence=weight,
                target_action=dict(encoded.canonical_action or step.target_action or {}),
                teacher_prompt_token_ids=teacher_ids,
                opd_loss=str(opd_loss),
                metadata=meta,
            )
        )
    stats["n_kept"] = len(datums)
    return datums, stats


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
    model_enc: Any | None = None,
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
        if point.teacher_prompt_token_ids:
            teacher_ids = list(point.teacher_prompt_token_ids)
        else:
            teacher_ids = recover_teacher_prompt_ids(
                teacher_ids=None,
                metadata={},
                snapshot=point.pre_action_snapshot,
                encode=encode,
                model_enc=model_enc,
                allow_offline=False,
            )
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
                    "effective_weight": 1.0,
                },
            )
        )
    return datums


def supervised_weight_sum(datums: Sequence[TinkerOPDDatum]) -> float:
    return float(sum(w for d in datums for w in d.weights))
