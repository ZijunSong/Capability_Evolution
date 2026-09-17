"""Convert TRIM rollout groups into per-decision training records."""

from __future__ import annotations

from typing import Any, Sequence

from trim.integrations.verl.trajectory import TransitionRecord
from trim.training.action_encoding import prompt_ids_hash


def groups_to_rl_rows(groups: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        payload = getattr(group, "trajectory_group", None) or {}
        for rec in list(payload.get("rl_rows") or []):
            item = dict(rec)
            item.setdefault("query_id", getattr(group, "query_id", ""))
            item.setdefault("policy_version", getattr(group, "policy_version", ""))
            rows.append(item)
    return rows


def rl_row_to_transition(
    rec: dict[str, Any],
    *,
    run_id: str,
    attempt_id: str,
    rollout_batch_id: int,
) -> TransitionRecord:
    prompt_ids = [int(x) for x in (rec.get("effective_prompt_ids") or rec.get("prompt_ids") or [])]
    response_ids = [int(x) for x in (rec.get("action_ids") or rec.get("response_ids") or [])]
    logprobs = [float(x) for x in (rec.get("token_logprobs") or rec.get("behavior_logprobs") or [])]
    mask = [int(x) for x in (rec.get("action_mask") or rec.get("response_loss_mask") or [1] * len(response_ids))]
    return TransitionRecord(
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        rollout_batch_id=int(rollout_batch_id),
        query_id=str(rec.get("query_id") or ""),
        episode_id=str(rec.get("episode_id") or ""),
        turn_id=int(rec.get("turn_id") or 0),
        policy_version=str(rec.get("policy_version") or ""),
        effective_prompt_ids=tuple(prompt_ids),
        prompt_hash=str(rec.get("prompt_hash") or prompt_ids_hash(prompt_ids)),
        response_ids=tuple(response_ids),
        behavior_logprobs=tuple(logprobs),
        response_loss_mask=tuple(mask),
        sampling_params=dict(rec.get("sampling_params") or {}),
        finish_reason=str(rec.get("finish_reason") or ""),
        parse_valid=bool(rec.get("valid") or rec.get("parse_valid")),
        exec_ok=bool(rec.get("exec_ok") or rec.get("executed_ok")),
        visible_doc_ids=tuple(str(x) for x in (rec.get("visible_doc_ids") or [])),
        terminal_reward=None if rec.get("reward") is None else float(rec.get("reward")),
        reward_parts=dict(rec.get("reward_parts") or {}),
        episode_advantage=None if rec.get("advantage") is None else float(rec.get("advantage")),
    )


def require_verl() -> None:
    """Kept for the T3 pin check; FSDP2 actor does not import vendored verl 0.5."""
    return
