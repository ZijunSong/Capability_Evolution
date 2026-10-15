"""Project one continuous bank branch onto a single student-legal action."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trim.harness_bank.recipes import load_recipe
from trim.harness_bank.types import PROJECTOR_VERSION, BankBranchTrace, BankSpec
from trim.state.snapshot import EnvironmentSnapshot
from trim.training.action_codec import canonicalize_action
from trim.training.opd_events import EventKind, HarnessEvent
from trim.training.opd_projection import (
    MAX_MACRO_ACTIONS,
    ProjectionKind,
    ProjectionResult,
)
from trim.training.opd_realizability import (
    accessible_doc_ids_of,
    check_action_realizability,
    curated_ids_of,
)
from trim.rsi.canonical import sha256_json

NO_OP = "NO_OP_ACTION"
FUTURE_STATE = "DEPENDS_ON_FUTURE_STATE"
UNREGISTERED = "UNREGISTERED_BANK_COMPONENT"
TEACHER_ONLY = "TEACHER_ONLY_INFORMATION"


def project_ordered_events(
    events: Sequence[Any],
    student_snapshot: EnvironmentSnapshot,
    student_mask: Mapping[str, bool],
    *,
    component_id: str = "bank",
    max_curated: int = 8,
    recipes: Mapping[str, Mapping[str, Any]] | None = None,
) -> ProjectionResult:
    """Project a joint event list without splitting it by component id."""
    recipe_map = dict(recipes or {})
    native: list[tuple[int, dict[str, Any], str]] = []
    for index, event in enumerate(events):
        cid = str(_field(event, "component_id") or component_id)
        recipe_map.setdefault(cid, {"route": "native_anchor", "target_tools": []})
        action, cid, reason = _native_action(event, recipe_map)
        if reason:
            return _skip(events, reason, component_id=cid or component_id)
        if action is None:
            continue
        native.append((index, action, cid))
        if len(native) > MAX_MACRO_ACTIONS:
            return _skip(events, "MACRO_TOO_LONG", component_id=cid)
    if not native:
        return _skip(events, "NO_NATIVE_ANCHOR", component_id=component_id)
    index, action, cid = native[0]
    if _depends_on_skipped_private_facts(events[:index], action):
        return _skip(events, FUTURE_STATE, component_id=cid)
    canon = canonicalize_action(action)
    if _is_noop(canon, student_snapshot):
        return _skip(events, NO_OP, component_id=cid)
    if canon["name"] == "curate":
        add_ids = [str(x) for x in (canon["arguments"].get("add_ids") or [])]
        accessible = set(accessible_doc_ids_of(student_snapshot))
        if any(did not in accessible for did in add_ids):
            return _skip(events, TEACHER_ONLY, component_id=cid)
        curated = set(curated_ids_of(student_snapshot))
        if len(curated | set(add_ids)) > int(max_curated):
            return _skip(events, "CURATED_CAPACITY_EXCEEDED", component_id=cid)
    report = check_action_realizability(
        action=canon,
        student_snapshot=student_snapshot,
        student_mask=student_mask,
        component_id=cid,
    )
    if not report.passed:
        return _skip(events, ",".join(report.reason_codes) or "UNREALIZABLE", component_id=cid)
    from trim.training.opd_projection import ProjectedAction

    projected = ProjectedAction(
        name=canon["name"],
        arguments=dict(canon["arguments"]),
        source_event_ids=[_event_id(events[index])],
        confidence=1.0,
    )
    return ProjectionResult(
        kind=ProjectionKind.DIRECT,
        actions=[projected],
        teacher_start_event=_event_id(events[0]),
        teacher_end_event=_event_id(events[index]),
        skipped_event_ids=[_event_id(event) for event in events[:index]],
        anchor_event_id=_event_id(events[index]),
        anchor_distance=index,
        realizability=report,
        component_id=cid,
        alignment_source="bank_continuous_native",
    )


def project_continuous_trace(
    *,
    bank: BankSpec,
    trace: BankBranchTrace,
    student_snapshot: EnvironmentSnapshot,
    student_mask: Mapping[str, bool],
    max_curated: int = 8,
) -> ProjectionResult:
    """Keep event order. Emit at most one action that the original state can run."""
    if not trace.continuous:
        raise ValueError("project_continuous_trace requires a joint bank branch")
    events = list(trace.events)
    if len(events) > int(bank.caps.get("max_shadow_events_per_decision") or 8):
        return _skip(events, "SHADOW_EVENT_CAP", component_id=trace.branch_id)
    recipes = {}
    for spec in bank.components:
        try:
            recipes[spec.id] = load_recipe(bank, spec)
        except Exception as exc:
            return _skip(events, f"{UNREGISTERED}:{exc}", component_id=spec.id)
    native: list[tuple[int, dict[str, Any], str]] = []
    for index, event in enumerate(events):
        action, cid, reason = _native_action(event, recipes)
        if reason:
            return _skip(events, reason, component_id=cid or trace.branch_id)
        if action is None:
            continue
        native.append((index, action, cid))
        if len(native) > MAX_MACRO_ACTIONS:
            break
    if not native:
        return _skip(events, "NO_NATIVE_ANCHOR", component_id=trace.branch_id)
    index, action, cid = native[0]
    if _depends_on_skipped_private_facts(events[:index], action):
        return _skip(events, FUTURE_STATE, component_id=cid)
    canon = canonicalize_action(action)
    if _is_noop(canon, student_snapshot):
        return _skip(events, NO_OP, component_id=cid)
    if canon["name"] == "curate":
        add_ids = [str(x) for x in (canon["arguments"].get("add_ids") or [])]
        accessible = set(accessible_doc_ids_of(student_snapshot))
        if any(did not in accessible for did in add_ids):
            return _skip(events, TEACHER_ONLY, component_id=cid)
        curated = set(curated_ids_of(student_snapshot))
        if len(curated | set(add_ids)) > int(max_curated):
            return _skip(events, "CURATED_CAPACITY_EXCEEDED", component_id=cid)
    report = check_action_realizability(
        action=canon,
        student_snapshot=student_snapshot,
        student_mask=student_mask,
        component_id=cid,
    )
    if not report.passed:
        return _skip(events, ",".join(report.reason_codes) or "UNREALIZABLE", component_id=cid)
    from trim.training.opd_projection import ProjectedAction

    projected = ProjectedAction(
        name=canon["name"],
        arguments=dict(canon["arguments"]),
        source_event_ids=[_event_id(events[index])],
        confidence=1.0,
    )
    start = _event_id(events[0]) if events else ""
    end = _event_id(events[index]) if events else ""
    return ProjectionResult(
        kind=ProjectionKind.DIRECT,
        actions=[projected],
        teacher_start_event=start,
        teacher_end_event=end,
        skipped_event_ids=[_event_id(event) for event in events[:index]],
        anchor_event_id=end,
        anchor_distance=index,
        realizability=report,
        reject_reason=None,
        component_id=cid,
        alignment_source="bank_continuous_native",
    )


def provenance_record(
    *,
    trace: BankBranchTrace,
    student_snapshot: EnvironmentSnapshot,
    target_action: Mapping[str, Any],
    teacher_prefix_hash: str,
) -> dict[str, Any]:
    target_hash = sha256_json(dict(target_action))
    return {
        "student_pre_state_hash": student_snapshot.content_hash(),
        "branch_id": trace.branch_id,
        "bank_sha256": trace.bank_sha256,
        "policy_digest": trace.source_policy_digest,
        "teacher_prefix_hash": teacher_prefix_hash,
        "target_hash": target_hash,
        "projector_version": PROJECTOR_VERSION,
        "continuous": True,
    }


def _native_action(
    event: Any, recipes: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any] | None, str, str | None]:
    cid = str(_field(event, "component_id") or "")
    meta = dict(_field(event, "metadata") or {})
    provenance = list(meta.get("information_provenance") or _field(event, "information_provenance") or [])
    if any(not bool(item.get("available_to_student", True)) for item in provenance if isinstance(item, dict)):
        name = str(_field(event, "action_name") or "")
        if name == "curate":
            return None, cid, TEACHER_ONLY
    recipe = recipes.get(cid) or {}
    route = str(recipe.get("route") or meta.get("distill_route") or "native_anchor")
    if route == "persistent_uninternalized":
        return None, cid, "PERSISTENT_SYSTEM_NOT_INTERNALIZED"
    action_name = _field(event, "action_name")
    if not action_name:
        return None, cid, None
    action = {"name": str(action_name), "arguments": dict(_field(event, "arguments") or {})}
    allowed = set(recipe.get("target_tools") or [])
    if allowed and action["name"] not in allowed and route in {"native_recipe", "native_anchor", "explicit_map"}:
        return None, cid, "TARGET_TOOL_NOT_IN_RECIPE"
    return action, cid, None


def _depends_on_skipped_private_facts(skipped: Sequence[Any], action: Mapping[str, Any]) -> bool:
    introduced: set[str] = set()
    for event in skipped:
        meta = dict(_field(event, "metadata") or {})
        if meta.get("introduces_unseen_docs"):
            introduced.update(str(x) for x in (meta.get("new_doc_ids") or []))
        obs = _field(event, "observation") or {}
        if isinstance(obs, dict) and obs.get("student_visible") is False:
            introduced.update(str(x) for x in (obs.get("doc_ids") or []))
    if not introduced:
        return False
    args = action.get("arguments") or {}
    referenced = [str(x) for x in (args.get("add_ids") or args.get("doc_ids") or [])]
    if action.get("name") == "curate":
        return any(did in introduced for did in referenced)
    return False


def _is_noop(action: Mapping[str, Any], snapshot: EnvironmentSnapshot) -> bool:
    if action.get("name") != "curate":
        return False
    add_ids = [str(x) for x in (action.get("arguments", {}).get("add_ids") or [])]
    remove_ids = [str(x) for x in (action.get("arguments", {}).get("remove_ids") or [])]
    curated = set(curated_ids_of(snapshot))
    if not add_ids and not remove_ids:
        return True
    changed = any(did not in curated for did in add_ids) or any(did in curated for did in remove_ids)
    return not changed


def _skip(events: Sequence[Any], reason: str, *, component_id: str) -> ProjectionResult:
    start = _event_id(events[0]) if events else ""
    end = _event_id(events[-1]) if events else ""
    return ProjectionResult(
        kind=ProjectionKind.SKIP,
        actions=[],
        teacher_start_event=start,
        teacher_end_event=end,
        skipped_event_ids=[_event_id(event) for event in events],
        reject_reason=reason,
        component_id=component_id,
        alignment_source="bank_continuous_native",
    )


def _event_id(event: Any) -> str:
    if isinstance(event, HarnessEvent):
        return event.event_id
    if isinstance(event, Mapping):
        return str(event.get("event_id") or "")
    return str(getattr(event, "event_id", "") or "")


def _field(event: Any, name: str) -> Any:
    if isinstance(event, Mapping):
        return event.get(name)
    return getattr(event, name, None)


def event_from_effect(
    *,
    component_id: str,
    action: Mapping[str, Any] | None,
    order: int,
    provenance: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HarnessEvent:
    meta = dict(metadata or {})
    if provenance is not None:
        meta["information_provenance"] = [dict(item) for item in provenance]
    return HarnessEvent(
        event_id=f"{component_id}:{order}",
        turn_id=int(order),
        kind=EventKind.MODEL_ACTION if action else EventKind.HARNESS_MUTATION,
        component_id=component_id,
        action_name=None if action is None else str(action.get("name") or ""),
        arguments={} if action is None else dict(action.get("arguments") or {}),
        visible_to_student=False,
        metadata=meta,
    )
