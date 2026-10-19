"""Teacher SFT rows: Full native actions, no projection."""

from __future__ import annotations

from typing import Any, Mapping

from trim.training.action_codec import canonicalize_action
from trim.training.sft_ablation.contracts import SFTBatchRow
from trim.training.sft_ablation.encoding import encode_action, fits_budget
from trim.training.sft_ablation.teacher_archive import ARCHIVE_SCHEMA
from trim.training.tool_mask import ACTION_ARG_SCHEMAS, CURATE_PRIVILEGED_KEYS, legal_tool_names
from trim.upstream_harness1.v8d_flags import all_enabled_mask


def full_action_allowed(action: Mapping[str, Any]) -> bool:
    try:
        canon = canonicalize_action(action)
    except ValueError:
        return False
    name = canon["name"]
    if name in {"", "truncated", "unknown"}:
        return False
    legal = set(legal_tool_names(harness_mask=all_enabled_mask("Harness-1")))
    if name not in legal:
        return False
    allowed = set(ACTION_ARG_SCHEMAS.get(name, ()))
    if name == "curate":
        allowed |= set(CURATE_PRIVILEGED_KEYS)
    return set(canon["arguments"]).issubset(allowed)


def build_teacher_sft_rows(
    archive: Mapping[str, Any],
    encoder: Any,
    contract: Mapping[str, Any],
) -> tuple[list[SFTBatchRow], list[dict[str, str]]]:
    if archive.get("schema_version") != ARCHIVE_SCHEMA:
        raise ValueError("archive is not a Full teacher trajectory package")
    if archive.get("mask") != "all" or archive.get("teacher_mode") is not False:
        raise ValueError("archive mask/teacher_mode does not match Teacher SFT")
    if str(archive.get("policy_version") or "") == "":
        raise ValueError("archive missing teacher policy version")
    max_length = int(contract["max_model_len"])
    max_new = int(contract["max_new_tokens"])
    rows: list[SFTBatchRow] = []
    rejected: list[dict[str, str]] = []
    for trajectory in archive.get("trajectories") or []:
        for decision in trajectory.get("decisions") or []:
            reason = _reject_reason(decision)
            action = decision.get("action") or {}
            if reason:
                rejected.append({"decision_id": str(decision.get("decision_id")), "reason": reason})
                continue
            prompt_ids = [int(token) for token in decision.get("prompt_ids") or []]
            try:
                canon = canonicalize_action(action)
                encoded = encode_action(canon, enc=encoder)
                fits_budget(
                    prompt_ids,
                    encoded.target_ids,
                    max_model_len=max_length,
                    max_new_tokens=max_new,
                )
                row = SFTBatchRow(
                    row_id=f"teacher_sft:{trajectory['trajectory_id']}:{decision['decision_id']}",
                    experiment="teacher_sft",
                    query_id=str(trajectory["query_id"]),
                    episode_id=str(trajectory["trajectory_id"]),
                    decision_id=str(decision["decision_id"]),
                    prompt_ids=prompt_ids,
                    response_ids=list(encoded.target_ids),
                    response_mask=list(encoded.supervision_mask),
                    weight=1.0,
                    policy_version=str(trajectory["policy_version"]),
                    prompt_owner="teacher_full",
                    target_source="full_model_trajectory_v1",
                    metadata={
                        "native_action": canon,
                        "pre_state_hash": decision.get("pre_state_hash"),
                    },
                )
                row.validate(max_length)
            except ValueError as exc:
                rejected.append({"decision_id": str(decision.get("decision_id")), "reason": str(exc)})
                continue
            rows.append(row)
    return rows, rejected


def _reject_reason(decision: Mapping[str, Any]) -> str:
    if not decision.get("structurally_valid", True):
        return "truncated_or_invalid"
    if not decision.get("executed_ok", False):
        return "execution_failed"
    action = decision.get("action") or {}
    if not full_action_allowed(action):
        return "illegal_full_action"
    if not decision.get("prompt_ids"):
        return "missing_pre_action_prompt"
    return ""
