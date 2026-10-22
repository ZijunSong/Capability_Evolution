"""Sequential Zero replay of a fixed Full teacher archive."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from trim.state.snapshot import EnvironmentSnapshot
from trim.training.opd_realizability import curated_ids_of
from trim.training.sft_ablation.contracts import SFTBatchRow
from trim.training.sft_ablation.encoding import encode_action, fits_budget, require_visible_action


class UnusableSupervision(ValueError):
    """Target cannot be trained under this exact student prompt."""


class NoopCurate(ValueError):
    """An exactly verified no-op; consume the event without stepping env."""


class RetryableReplayError(RuntimeError):
    """Infrastructure failure (timeout/429/5xx/service down).

    The attempt is not committed: the whole trajectory is replayed from a fresh
    state. Never confuse this with a semantic stop that keeps its legal prefix.
    """


@dataclass
class ReplayDecision:
    snapshot: EnvironmentSnapshot
    prompt_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecuteResult:
    ok: bool
    reason: str = ""
    post_state_hash: str = ""
    snapshot: EnvironmentSnapshot | None = None
    prompt_ids: list[int] = field(default_factory=list)
    ended: bool = False
    failure_class: str = ""  # "infra" | "semantic" | ""
    retryable: bool = False


def replay_projected_trace(
    events: Sequence[Any],
    env: Any,
    choose: Callable[[ReplayDecision, Sequence[Any]], tuple[Any, Sequence[Any]]],
    make_row: Callable[[ReplayDecision, Any, Any], SFTBatchRow],
    *,
    scan: int = 8,
    max_steps: int = 40,
) -> tuple[list[SFTBatchRow], list[dict[str, Any]]]:
    if scan < 1 or max_steps < 1:
        raise ValueError("scan/max_steps must be positive")
    all_ids = [event.event_id for event in events]
    if any(not item for item in all_ids) or len(all_ids) != len(set(all_ids)):
        raise ValueError("event IDs must be unique and nonempty")
    rows: list[SFTBatchRow] = []
    audit: list[dict[str, Any]] = []
    cursor = 0
    executed = 0
    while cursor < len(events) and not env.ended and executed < max_steps:
        before = env.decision()
        window = events[cursor : cursor + scan]
        projection, steps = choose(before, window)
        if projection.kind.value != "direct":
            audit.append(
                {
                    "cursor": cursor,
                    "status": "skip_window",
                    "reason": projection.reject_reason,
                }
            )
            cursor += len(window)
            continue
        ids = [event.event_id for event in window]
        if projection.anchor_event_id not in ids:
            raise ValueError("anchor outside bounded window")
        consumed = ids.index(projection.anchor_event_id) + 1
        if not steps:
            audit.append({"cursor": cursor, "status": "reject_anchor", "reason": "materialize_failed"})
            cursor += consumed
            continue
        if len(steps) != 1:
            raise ValueError("v1 requires one action per anchor")
        step = steps[0]
        try:
            row = make_row(before, step, projection)
        except NoopCurate:
            audit.append({"cursor": cursor, "status": "skip_noop_curate"})
            cursor += consumed
            continue
        except UnusableSupervision as exc:
            audit.append({"cursor": cursor, "status": "stop_unusable", "reason": str(exc)})
            break
        result = env.execute(step.target_action)
        if not result.ok:
            if getattr(result, "retryable", False):
                raise RetryableReplayError(f"infra failure at cursor {cursor}: {result.reason}")
            audit.append(
                {
                    "cursor": cursor,
                    "status": "stop_execution",
                    "reason": result.reason,
                    "failure_class": getattr(result, "failure_class", "") or "semantic",
                }
            )
            break
        row.metadata["student_post_state_hash"] = result.post_state_hash
        row.metadata["teacher_event_cursor"] = cursor
        row.metadata["student_action_step"] = executed
        rows.append(row)
        audit.append({"cursor": cursor, "status": "aligned", "anchor": projection.anchor_event_id})
        cursor += consumed
        executed += 1
    return rows, audit


def make_projector_adapter(projector: Any, audit: Any) -> Callable[..., tuple[Any, list[Any]]]:
    from trim.training.opd_dataset import project_and_materialize

    def choose(before: ReplayDecision, events: Sequence[Any]) -> tuple[Any, list[Any]]:
        return project_and_materialize(
            student_snapshot=before.snapshot,
            teacher_events=list(events),
            student_mask=before.snapshot.harness_mask,
            component_id="all",
            projector=projector,
            audit=audit,
        )

    return choose


def row_from_projected_step(
    before: ReplayDecision,
    step: Any,
    projection: Any,
    *,
    enc: Any,
    query_id: str,
    episode_id: str,
    policy_version: str,
    max_model_len: int,
    max_new_tokens: int,
    row_id: str,
) -> SFTBatchRow:
    action = dict(step.target_action)
    before_ids = set(curated_ids_of(before.snapshot))
    if action.get("name") == "curate":
        added = {str(item) for item in (action.get("arguments") or {}).get("add_ids") or []}
        removed = {str(item) for item in (action.get("arguments") or {}).get("remove_ids") or []}
        if (before_ids - removed) | added == before_ids:
            raise NoopCurate("curated set unchanged")
    try:
        visible = require_visible_action(
            action,
            prompt_ids=before.prompt_ids,
            enc=enc,
            metadata=dict(before.metadata),
            snapshot=before.snapshot,
        )
        encoded = encode_action(action, enc=enc, target_text=str(getattr(step, "target_text", "") or ""))
        fits_budget(before.prompt_ids, encoded.target_ids, max_model_len=max_model_len, max_new_tokens=max_new_tokens)
    except ValueError as exc:
        raise UnusableSupervision(str(exc)) from exc
    weight = float(getattr(step, "weight", None) or getattr(step, "projection_confidence", None) or 1.0)
    row = SFTBatchRow(
        row_id=row_id,
        experiment="projected_teacher_sft",
        query_id=query_id,
        episode_id=episode_id,
        decision_id=str(projection.anchor_event_id),
        prompt_ids=list(before.prompt_ids),
        response_ids=list(encoded.target_ids),
        response_mask=list(encoded.supervision_mask),
        weight=weight,
        policy_version=policy_version,
        prompt_owner="student_zero",
        target_source="full_model_trajectory_v1",
        metadata={
            "visible_doc_ids": visible,
            "anchor_event_id": projection.anchor_event_id,
            "native_projected_action": action,
        },
    )
    try:
        row.validate(max_model_len)
    except ValueError as exc:
        raise UnusableSupervision(str(exc)) from exc
    return row
