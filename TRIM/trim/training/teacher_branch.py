"""Unified teacher-branch result, skip reasons, and implementation tables.

Teacher events remain the projector input. This module records whether a
branch actually triggered a capability, produced a synthetic heuristic, or
was skipped. Mask-flip teacher prompts are not capability replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from trim.training.opd_events import HarnessEvent, obs_transform

SOURCE_CAPABILITY = "capability_effect"
SOURCE_SYNTHETIC = "synthetic_heuristic"
SOURCE_SKIP_UNTRIGGERED = "skip_untriggered"
SOURCE_SKIP_UNREGISTERED = "skip_unregistered"

HARNESS1_IMPLEMENTED_TEACHERS = frozenset(
    {
        "auto_populate_first_search",
        "sentence_compress",
        "token_budget_marker",
        "adaptive_rerank_instruction",
        "verify_tool",
    }
)
HARNESS1_UNIMPLEMENTED_TEACHERS = frozenset(
    {
        "subtractive_curation",
        "importance_tagging",
        "evidence_graph",
        "chunk_neighbors",
        "content_dedup",
    }
)


@dataclass
class TeacherBranchResult:
    decision_state_id: str
    component_id: str
    triggered: bool
    source_type: str
    events: list[HarnessEvent] = field(default_factory=list)
    teacher_condition: dict[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    worth_supervising: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events"] = [e.to_dict() if hasattr(e, "to_dict") else e for e in self.events]
        return payload


def coerce_teacher_branch(
    raw: Any,
    *,
    point: Any,
    component_id: str,
) -> TeacherBranchResult:
    """Accept TeacherBranchResult or a legacy event list from teacher_for()."""
    if isinstance(raw, TeacherBranchResult):
        return raw
    events = list(raw or [])
    skip = teacher_skip_kind(events)
    triggered = skip is None and bool(events)
    source = skip or SOURCE_CAPABILITY
    condition: dict[str, Any] = {}
    for event in events:
        meta = getattr(event, "metadata", None) or {}
        if isinstance(meta, dict) and meta.get("teacher_condition"):
            condition = dict(meta["teacher_condition"])
            break
    return TeacherBranchResult(
        decision_state_id=str(getattr(point, "decision_point_id", "") or ""),
        component_id=str(component_id),
        triggered=bool(triggered and skip is None),
        source_type=source,
        events=events,
        teacher_condition=condition,
        skip_reason=skip,
        worth_supervising=skip is None,
    )


def skip_teacher_events(
    component_id: str,
    *,
    turn_id: int = 0,
    reason: str,
    teacher_kind: str = SOURCE_SKIP_UNTRIGGERED,
) -> list[HarnessEvent]:
    return [
        obs_transform(
            component_id,
            turn_id=turn_id,
            observation={"owner": "teacher_full", "skip_reason": reason},
            visible_to_student=False,
            metadata={
                "teacher_kind": teacher_kind,
                "source_type": teacher_kind,
                "skip_reason": reason,
                "not_a_continuous_teacher_rollout": True,
            },
        )
    ]


def tag_source(
    events: Sequence[HarnessEvent],
    source_type: str,
    extra: Mapping[str, Any] | None = None,
) -> list[HarnessEvent]:
    out = list(events)
    for event in out:
        meta = getattr(event, "metadata", None)
        if isinstance(meta, dict):
            meta.setdefault("source_type", source_type)
            meta.setdefault("teacher_kind", source_type)
            if extra:
                meta.update(dict(extra))
    return out


def teacher_skip_kind(events: Sequence[Any] | None) -> str | None:
    for event in events or []:
        meta = getattr(event, "metadata", None) or {}
        if not isinstance(meta, dict):
            continue
        kind = str(meta.get("teacher_kind") or meta.get("source_type") or "")
        if kind in {
            SOURCE_SKIP_UNTRIGGERED,
            SOURCE_SKIP_UNREGISTERED,
            SOURCE_SYNTHETIC,
        }:
            return kind
        reason = str(meta.get("skip_reason") or "")
        if reason:
            if "unregistered" in reason:
                return SOURCE_SKIP_UNREGISTERED
            return SOURCE_SKIP_UNTRIGGERED
    return None


def is_worth_supervising(
    wm: Mapping[str, Any],
    action_name: str,
    arguments: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Legal student actions can still be no-ops; count them separately."""
    args = dict(arguments or {})
    hist = list(wm.get("tool_history") or [])
    if action_name in {"search_corpus", "grep_corpus", "fan_out_search"}:
        query = str(args.get("query") or "")
        for item in hist:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or "") not in {"search_corpus", "grep_corpus", "fan_out_search"}:
                continue
            prev = (item.get("arguments") or {}).get("query") if isinstance(item.get("arguments"), dict) else None
            if str(prev or "") == query and query:
                return False, "repeat_search"
        last = str(wm.get("last_tool_name") or "")
        last_q = str(wm.get("last_search_query") or "")
        if last in {"search_corpus", "grep_corpus", "fan_out_search"} and last_q == query and query:
            return False, "repeat_search"
    if action_name == "read_document":
        doc_id = str(args.get("doc_id") or "")
        for item in hist:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or "") != "read_document":
                continue
            prev = (item.get("arguments") or {}).get("doc_id") if isinstance(item.get("arguments"), dict) else None
            if str(prev or "") == doc_id and doc_id:
                return False, "repeat_read"
    if action_name == "curate":
        add_ids = [str(x) for x in (args.get("add_ids") or [])]
        remove_ids = [str(x) for x in (args.get("remove_ids") or [])]
        curated = {str(x) for x in (wm.get("curated_ids") or [])}
        new_add = [x for x in add_ids if x not in curated]
        new_remove = [x for x in remove_ids if x in curated]
        if not new_add and not new_remove:
            return False, "noop_curate"
    return True, "ok"


def filter_unworthy_events(
    wm: Mapping[str, Any],
    events: Sequence[HarnessEvent],
    *,
    component_id: str,
    turn_id: int,
) -> list[HarnessEvent]:
    actions = [e for e in events if getattr(e, "action_name", None)]
    if not actions:
        return list(events)
    kept: list[HarnessEvent] = []
    skipped = False
    reason = ""
    for event in events:
        name = getattr(event, "action_name", None)
        if not name:
            kept.append(event)
            continue
        ok, why = is_worth_supervising(wm, str(name), getattr(event, "arguments", None))
        if ok:
            kept.append(event)
        else:
            skipped = True
            reason = why
    if skipped and not any(getattr(e, "action_name", None) for e in kept):
        return skip_teacher_events(component_id, turn_id=turn_id, reason=reason)
    return kept


def component_implementation_table(
    requested_ids: Sequence[str],
    *,
    implemented: set[str] | frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    registered = set(implemented or HARNESS1_IMPLEMENTED_TEACHERS)
    table: dict[str, dict[str, Any]] = {}
    for cid in requested_ids:
        key = str(cid)
        is_impl = key in registered
        table[key] = {
            "requested": True,
            "implemented": is_impl,
            "status": "implemented" if is_impl else "unregistered",
            "unimplemented_known": key in HARNESS1_UNIMPLEMENTED_TEACHERS,
        }
    for cid in sorted(HARNESS1_UNIMPLEMENTED_TEACHERS):
        table.setdefault(
            cid,
            {
                "requested": cid in set(requested_ids),
                "implemented": False,
                "status": "unregistered",
                "unimplemented_known": True,
            },
        )
    return table
