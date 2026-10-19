"""Full decision-point serializer for Student on-policy SFT."""

from __future__ import annotations

from typing import Any

from trim.training.sft_ablation.contracts import fingerprint


def encode_decision_point(point: Any, *, adapter_digest: str) -> dict[str, Any]:
    from copy import deepcopy
    from dataclasses import fields

    if not adapter_digest:
        raise ValueError("missing rollout adapter digest")
    payload: dict[str, Any] = {}
    for field in fields(point):
        value = getattr(point, field.name)
        if field.name in {"pre_action_snapshot", "post_action_snapshot"}:
            payload[field.name] = None if value is None else deepcopy(value.to_dict())
        else:
            payload[field.name] = deepcopy(value)
    payload["student_model_input"] = _json_safe(payload.get("student_model_input"))
    return {
        "schema_version": "sft_decision_point_v1",
        "adapter_digest": adapter_digest,
        "point": payload,
        "point_sha256": fingerprint(payload),
    }


def decode_decision_point(record: dict[str, Any], *, adapter_digest: str, policy_version: str) -> Any:
    from copy import deepcopy

    from trim.state.snapshot import EnvironmentSnapshot
    from trim.training.rl_opd_types import StudentDecisionPoint

    if record.get("schema_version") != "sft_decision_point_v1":
        raise ValueError("unsupported decision point schema")
    if not adapter_digest or record.get("adapter_digest") != adapter_digest:
        raise ValueError("rollout adapter digest mismatch")
    payload = deepcopy(record["point"])
    if fingerprint(payload) != record.get("point_sha256"):
        raise ValueError("decision point checksum mismatch")
    if payload.get("policy_version") != policy_version:
        raise ValueError("stale decision point")
    for key in ("pre_action_snapshot", "post_action_snapshot"):
        if payload.get(key) is not None:
            payload[key] = EnvironmentSnapshot.from_dict(payload[key])
    point = StudentDecisionPoint(**payload)
    if point.pre_action_snapshot.content_hash() != point.pre_action_snapshot_hash:
        raise ValueError("student pre-state hash mismatch")
    return point


def student_collection_kwargs(
    *,
    student_zero_mask: dict[str, bool],
    policy_version: str,
    group_size: int,
    max_turns: int,
    max_new_tokens: int,
    seed: int,
    enc: Any,
    retrieval: Any = None,
    train_env: str,
    session: Any = None,
) -> dict[str, Any]:
    return {
        "component_id": "all",
        "teacher_mode": False,
        "harness_mask": student_zero_mask,
        "collection_mode": "rl+opd",
        "opd_loss": "sr_opd_ce",
        "enable_rl": False,
        "policy_version": policy_version,
        "group_size": int(group_size),
        "max_turns": int(max_turns),
        "max_new": int(max_new_tokens),
        "seed": int(seed),
        "sample": True,
        "enc": enc,
        "searcher": retrieval,
        "train_env": train_env,
        "train_session": session,
    }


def _json_safe(value: Any) -> Any:
    from trim.training.sft_ablation.contracts import fingerprint

    try:
        fingerprint(value)
    except (TypeError, ValueError):
        return {"_omitted_type": type(value).__name__}
    return value
