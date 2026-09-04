"""Skip-to-anchor projection for Harness-G advanced components.

Student-legal tools: init, select, lookup, answer.
Teacher-only: answer_with (projects to select), plus harness-only obs transforms
(bridge ranking, SNC previews, synonym expansion, …) which are skipped until
a Student-native action appears.

This is the Harness-G analogue of EasyOPD skip-to-anchor for Harness-1 v8d.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from trim.adapters.harness_g_components import RUNTIME_TOOLS, TEACHER_ONLY_TOOLS
from trim.state.snapshot import EnvironmentSnapshot
from trim.training.opd_events import HarnessEvent
from trim.training.opd_projection import (
    REJECT_DOC_NOT_ACCESSIBLE,
    REJECT_ILLEGAL_TOOL,
    REJECT_NO_SEMANTIC_ANCHOR,
    REJECT_TEACHER_ONLY_INFORMATION,
    ProjectedAction,
    ProjectionKind,
    ProjectionResult,
)
from trim.training.opd_realizability import accessible_doc_ids_of

STUDENT_NATIVE = frozenset(RUNTIME_TOOLS)
HARNESS_ONLY_EVENT_TYPES = frozenset(
    {
        "bridge_entities_privileged_context",
        "entity_synonyms_privileged_context",
        "sentence_neighbors_privileged_context",
        "hybrid_init_privileged_context",
        "snc_frontier_privileged_context",
        "invalid_target_filter_runtime_check",
        "lookup_dedup_runtime_check",
    }
)
MAX_SCAN = 8


def _as_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if x is not None and str(x)]
    if isinstance(value, dict):
        return [str(x) for x in value.keys() if str(x)]
    return [str(value)]


def _event_dict(event: HarnessEvent | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(event, HarnessEvent):
        return {
            "event_id": event.event_id,
            "kind": event.kind.value if hasattr(event.kind, "value") else str(event.kind),
            "event_type": (event.metadata or {}).get("event_type"),
            "action_name": event.action_name,
            "arguments": dict(event.arguments or {}),
            "state_delta": dict(event.state_delta or {}),
            "payload": dict(event.metadata or {}),
            "projectable_target": (event.metadata or {}).get("projectable_target"),
            "visible_to_student": bool(event.visible_to_student),
            "component_id": event.component_id,
            "harness_only": bool((event.metadata or {}).get("harness_only")),
            "teacher_only": bool((event.metadata or {}).get("teacher_only")),
        }
    payload = dict(event)
    payload.setdefault("payload", dict(event.get("metadata") or {}))
    return payload


def _accessible(snapshot: EnvironmentSnapshot) -> set[str]:
    ids = set(accessible_doc_ids_of(snapshot))
    wm = snapshot.working_memory or {}
    for key in ("visible_sids", "selected_sids", "frontier_eids", "visited_eids", "accessible_sids"):
        ids.update(_as_ids(wm.get(key)))
    return ids


def student_action_of(event: Mapping[str, Any]) -> ProjectedAction | None:
    event_id = str(event.get("event_id") or "")
    target = event.get("projectable_target") or (event.get("payload") or {}).get("projectable_target")
    if isinstance(target, Mapping) and target.get("name"):
        name = str(target["name"]).lower()
        if name in STUDENT_NATIVE:
            return ProjectedAction(
                name=name,
                arguments=dict(target.get("arguments") or {}),
                source_event_ids=[event_id],
            )
    name = str(event.get("action_name") or event.get("tool_name") or "").lower()
    args = dict(event.get("arguments") or event.get("parameters") or {})
    if name == "answer_with":
        sids = _as_ids(args.get("sids")) or _as_ids(args.get("sid"))
        if not sids:
            return None
        return ProjectedAction(
            name="select",
            arguments={"sid": sids[0]},
            source_event_ids=[event_id],
        )
    if name in STUDENT_NATIVE:
        return ProjectedAction(name=name, arguments=args, source_event_ids=[event_id])
    delta = event.get("state_delta") or {}
    added = _as_ids(delta.get("added") or delta.get("after_selected"))
    before = set(_as_ids(delta.get("before_selected") or delta.get("selected_before")))
    after = set(_as_ids(delta.get("after_selected") or delta.get("selected_after")))
    added = added or sorted(after - before)
    if added:
        return ProjectedAction(
            name="select",
            arguments={"sid": added[0]},
            source_event_ids=[event_id],
        )
    return None


def is_harness_only(event: Mapping[str, Any]) -> bool:
    if student_action_of(event) is not None and str(event.get("action_name") or "").lower() != "answer_with":
        if str(event.get("action_name") or "").lower() in STUDENT_NATIVE:
            return False
    event_type = str(event.get("event_type") or "")
    if event_type in HARNESS_ONLY_EVENT_TYPES:
        return True
    if bool(event.get("harness_only")):
        return True
    kind = str(event.get("kind") or "")
    name = str(event.get("action_name") or event.get("tool_name") or "").lower()
    if name in TEACHER_ONLY_TOOLS:
        return False  # answer_with has a projection, handled as student_action_of
    if kind in {"obs_transform", "harness_mutation", "tool_observation"}:
        return student_action_of(event) is None
    return False


def action_realizable(
    action: ProjectedAction,
    *,
    accessible: Iterable[str],
    legal_tools: Iterable[str] | None = None,
) -> tuple[bool, str | None]:
    legal = set(legal_tools or STUDENT_NATIVE)
    if action.name not in legal or action.name in TEACHER_ONLY_TOOLS:
        return False, REJECT_ILLEGAL_TOOL
    access = {str(x) for x in accessible}
    refs: list[str] = []
    args = action.arguments or {}
    refs.extend(_as_ids(args.get("sids")))
    if args.get("sid"):
        refs.append(str(args["sid"]))
    if args.get("eid"):
        refs.append(str(args["eid"]))
    if action.name in {"init", "answer", "lookup"}:
        if action.name == "lookup" and args.get("eid") and str(args["eid"]) not in access:
            # LOOKUP of a teacher-only bridge entity is not Student-realizable.
            return False, REJECT_TEACHER_ONLY_INFORMATION
        return True, None
    missing = [rid for rid in refs if rid not in access]
    if missing:
        return False, REJECT_DOC_NOT_ACCESSIBLE
    return True, None


def project_harness_g_events(
    events: list[HarnessEvent | Mapping[str, Any]],
    *,
    student_snapshot: EnvironmentSnapshot,
    student_mask: Mapping[str, bool] | None = None,
    component_id: str = "",
) -> ProjectionResult:
    del student_mask
    payload = [_event_dict(e) for e in events]
    accessible = _accessible(student_snapshot)
    legal = list(STUDENT_NATIVE)
    skipped: list[str] = []
    start_id = str(payload[0].get("event_id") or "") if payload else ""
    for idx, event in enumerate(payload[:MAX_SCAN]):
        event_id = str(event.get("event_id") or f"evt_{idx}")
        action = student_action_of(event)
        if is_harness_only(event) and action is None:
            skipped.append(event_id)
            continue
        if action is None:
            skipped.append(event_id)
            continue
        ok, reason = action_realizable(action, accessible=accessible, legal_tools=legal)
        if not ok:
            skipped.append(event_id)
            if reason == REJECT_TEACHER_ONLY_INFORMATION:
                continue
            return ProjectionResult(
                kind=ProjectionKind.SKIP,
                actions=[],
                teacher_start_event=start_id,
                teacher_end_event=event_id,
                skipped_event_ids=skipped,
                reject_reason=reason,
                component_id=component_id or str(event.get("component_id") or ""),
            )
        return ProjectionResult(
            kind=ProjectionKind.DIRECT,
            actions=[action],
            teacher_start_event=start_id,
            teacher_end_event=event_id,
            skipped_event_ids=skipped,
            anchor_event_id=event_id,
            anchor_distance=idx,
            component_id=component_id or str(event.get("component_id") or ""),
        )
    last = str(payload[-1].get("event_id") or "") if payload else ""
    return ProjectionResult(
        kind=ProjectionKind.SKIP,
        actions=[],
        teacher_start_event=start_id,
        teacher_end_event=last,
        skipped_event_ids=skipped,
        reject_reason=REJECT_NO_SEMANTIC_ANCHOR,
        component_id=component_id,
    )
