"""TRIM branch targets for Student on-policy SFT. CE only."""

from __future__ import annotations

from typing import Any, Sequence

from trim.training.sft_ablation.contracts import SFTBatchRow, fingerprint


class PolicyIdentityError(RuntimeError):
    """Policy, checksum, or snapshot identity failed. Do not skip the row."""


class RowReject(ValueError):
    """One row is unusable. Count it and keep the rest of the round."""


def projected_step_to_ce_row(
    step: Any,
    *,
    enc: Any,
    policy_version: str,
    query_id: str,
    episode_id: str,
    max_length: int,
    row_id: str,
) -> SFTBatchRow:
    from trim.state.snapshot import EnvironmentSnapshot
    from trim.training.action_encoding import (
        assert_action_visible,
        encode_supervised_action,
        resolve_effective_weight,
        visible_doc_ids_for_decision,
    )

    meta = dict(step.metadata)
    if meta.get("source_policy_version") != policy_version:
        raise PolicyIdentityError("stale on-policy target")
    prompt_ids = list(meta.get("student_prompt_token_ids") or [])
    if not prompt_ids:
        raise RowReject("missing sampled student prefix")
    snapshot = EnvironmentSnapshot.from_dict(step.student_snapshot)
    visible = visible_doc_ids_for_decision(
        snapshot=snapshot,
        prompt_ids=prompt_ids,
        enc=enc,
        metadata=meta,
    )
    if visible is None:
        raise RowReject("unknown prompt document visibility")
    try:
        assert_action_visible(step.target_action, visible)
        encoded = encode_supervised_action(
            target_action=step.target_action,
            target_text=step.target_text,
            encode=enc.encode,
            model_enc=enc,
        )
        weight = resolve_effective_weight(
            weight=step.weight,
            projection_confidence=step.projection_confidence,
            metadata=meta,
        )
    except ValueError as exc:
        raise RowReject(str(exc)) from exc
    row = SFTBatchRow(
        row_id=row_id,
        experiment="student_onpolicy_sft",
        query_id=query_id,
        episode_id=episode_id,
        decision_id=str(meta["decision_point_id"]),
        prompt_ids=prompt_ids,
        response_ids=list(encoded.target_ids),
        response_mask=list(encoded.supervision_mask),
        weight=weight,
        policy_version=policy_version,
        prompt_owner="student_zero",
        target_source="trim_component_branches_v1",
        metadata={
            "source_event_ids": list(step.source_event_ids),
            "student_pre_snapshot_hash": meta.get("student_pre_snapshot_hash"),
            "n_same_state_candidates": meta.get("n_same_state_candidates"),
            "selection_rule": meta.get("selection_rule"),
        },
    )
    try:
        row.validate(max_length)
    except ValueError as exc:
        raise RowReject(str(exc)) from exc
    return row


def build_legal_ce_targets(
    points: Sequence[Any],
    *,
    policy_version: str,
    adapter_digest: str,
    enc: Any,
    max_length: int,
    per_trajectory: int,
    seed: int,
) -> tuple[list[SFTBatchRow], dict[str, Any]]:
    from trim.training.four_cell_runtime import teacher_for
    from trim.training.tinker_rl_opd_trainer import project_on_policy_decisions, sample_decision_points

    for point in points:
        if point.policy_version != policy_version:
            raise PolicyIdentityError("mixed or stale rollout policy")
    before = fingerprint([_point_digest(point) for point in points])
    chosen = sample_decision_points(
        points,
        per_trajectory=int(per_trajectory),
        seed=int(seed),
        include_valid_failures=True,
        include_format_errors=False,
    )
    teacher = teacher_for("all")
    if teacher is None:
        raise PolicyIdentityError("missing TRIM teacher branch")
    steps, audit, extras = project_on_policy_decisions(
        chosen,
        teacher_event_fn=teacher,
        component_id="all",
    )
    after = fingerprint([_point_digest(point) for point in points])
    if before != after:
        raise PolicyIdentityError("teacher branch modified the student decision point")
    by_decision = {str(point.decision_point_id): point for point in chosen}
    rows: list[SFTBatchRow] = []
    rejected: list[dict[str, str]] = []
    for index, step in enumerate(steps):
        meta = dict(step.metadata)
        point = by_decision.get(str(meta.get("decision_point_id")))
        query_id = str(getattr(point, "query_id", "") or "")
        episode_id = str(getattr(point, "episode_id", "") or meta.get("decision_point_id") or index)
        try:
            rows.append(
                projected_step_to_ce_row(
                    step,
                    enc=enc,
                    policy_version=policy_version,
                    query_id=query_id,
                    episode_id=episode_id,
                    max_length=max_length,
                    row_id=f"student_onpolicy_sft:{adapter_digest[:12]}:{meta.get('decision_point_id')}:{index}",
                )
            )
        except PolicyIdentityError:
            raise
        except RowReject as exc:
            rejected.append({"row": str(index), "reason": str(exc)})
    return rows, {
        "n_points": len(points),
        "n_chosen": len(chosen),
        "n_steps": len(steps),
        "n_rows": len(rows),
        "rejected": rejected,
        "projection": extras,
        "audit_skip": getattr(audit, "n_skip", None),
        "collection_semantics": "student_states_for_ce",
        "reward_used_for_optimization": False,
    }


def _point_digest(point: Any) -> dict[str, Any]:
    return {
        "decision_point_id": point.decision_point_id,
        "policy_version": point.policy_version,
        "pre_action_snapshot_hash": point.pre_action_snapshot_hash,
        "student_prompt_token_ids": list(point.student_prompt_token_ids or []),
        "student_action_tokens": list(point.student_action_tokens or []),
        "reward": point.reward,
    }
