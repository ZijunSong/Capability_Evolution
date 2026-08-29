"""Skip-to-anchor projector for EasyOPD + Harness-1.

After projection, a Teacher event is only one of two things:

* ``align`` — a Student-native tool call (the OPD label)
* ``skip``  — ε: this Harness-only event produces no Student action;
  keep scanning for the next realizable Student-native anchor

Nothing else is emitted. Harness-only transforms are never distilled as
Teacher tokens. Recovery macros (invented ``review_docs`` / ``read_document``
chains) are not emitted: if the downstream action is not Student-realizable
from the current observation, it is skipped rather than reconstructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

ALIGN = "align"
SKIP = "skip"

STUDENT_NATIVE_TOOLS = frozenset(
    {
        "fan_out_search",
        "search_corpus",
        "grep_corpus",
        "read_document",
        "review_docs",
        "curate",
        "end_search",
    }
)
TEACHER_ONLY_TOOLS = frozenset({"verify", "importance_tagging"})
HARNESS_ONLY_KINDS = frozenset({"obs_transform", "harness_mutation", "component_event"})
HARNESS_ONLY_EVENT_TYPES = frozenset(
    {
        "evidence_graph_privileged_context",
        "sentence_compress_privileged_context",
        "token_budget_marker_visible",
        "adaptive_rerank_instruction_available",
        "chunk_neighbors_no_runtime_hook_detected",
        "verify_tool_action_available",
        "near_duplicate_pool_suppressed",
    }
)
MAX_ANCHOR_SCAN_EVENTS = 8


@dataclass
class ProjectedStudentAction:
    name: str
    arguments: dict[str, Any]
    source_event_id: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "source_event_id": self.source_event_id,
            "confidence": float(self.confidence),
        }


@dataclass
class SkipToAnchorResult:
    kind: str
    actions: list[ProjectedStudentAction] = field(default_factory=list)
    skipped_event_ids: list[str] = field(default_factory=list)
    teacher_start_event: str = ""
    teacher_end_event: str = ""
    anchor_event_id: str | None = None
    anchor_distance: int | None = None
    reason: str | None = None
    component_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["actions"] = [a.to_dict() if hasattr(a, "to_dict") else a for a in self.actions]
        return payload


def _as_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if x is not None and str(x)]
    if isinstance(value, dict):
        return [str(x) for x in value.keys() if str(x)]
    return [str(value)]


def curated_delta(event: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    target = event.get("projectable_target") or {}
    if isinstance(target, Mapping) and str(target.get("name") or "") == "curate":
        args = target.get("arguments") or {}
        return _as_ids(args.get("add_ids")), _as_ids(args.get("remove_ids"))
    delta = event.get("state_delta") or {}
    if "added" in delta or "removed" in delta:
        return _as_ids(delta.get("added")), _as_ids(delta.get("removed"))
    before = set(_as_ids(delta.get("before_curated") or delta.get("curated_before")))
    after = set(_as_ids(delta.get("after_curated") or delta.get("curated_after")))
    if before or after:
        return sorted(after - before), sorted(before - after)
    payload = event.get("payload") or event.get("arguments") or {}
    if isinstance(payload, Mapping):
        added = _as_ids(payload.get("delta_add") or payload.get("added"))
        removed = _as_ids(payload.get("delta_remove") or payload.get("removed"))
        if added or removed:
            return added, removed
    return [], []


def student_action_of(event: Mapping[str, Any]) -> ProjectedStudentAction | None:
    event_id = str(event.get("event_id") or "")
    added, removed = curated_delta(event)
    if added or removed:
        return ProjectedStudentAction(
            name="curate",
            arguments={"add_ids": sorted(added), "remove_ids": sorted(removed)},
            source_event_id=event_id,
        )
    target = event.get("projectable_target")
    if isinstance(target, Mapping) and target.get("name"):
        name = str(target["name"])
        if name in STUDENT_NATIVE_TOOLS:
            return ProjectedStudentAction(
                name=name,
                arguments=dict(target.get("arguments") or {}),
                source_event_id=event_id,
            )
    name = str(event.get("action_name") or event.get("tool_name") or "")
    if name in STUDENT_NATIVE_TOOLS:
        args = dict(event.get("arguments") or event.get("parameters") or {})
        args.pop("result_ids", None)
        args.pop("importance", None)
        return ProjectedStudentAction(name=name, arguments=args, source_event_id=event_id)
    return None


def is_harness_only(event: Mapping[str, Any]) -> bool:
    added, removed = curated_delta(event)
    if added or removed:
        return False
    event_type = str(event.get("event_type") or "")
    if event_type in HARNESS_ONLY_EVENT_TYPES:
        return True
    if bool(event.get("harness_only")):
        return True
    kind = str(event.get("kind") or "")
    name = str(event.get("action_name") or event.get("tool_name") or "")
    if name in TEACHER_ONLY_TOOLS:
        return True
    if kind == "tool_observation" and not bool(event.get("visible_to_student")):
        return True
    if kind in HARNESS_ONLY_KINDS:
        return student_action_of(event) is None
    return False


def action_realizable(
    action: ProjectedStudentAction,
    *,
    accessible_doc_ids: Iterable[str],
    legal_tools: Iterable[str] | None = None,
    student_search_results: Iterable[str] | None = None,
    teacher_needed_doc_ids: Iterable[str] | None = None,
) -> tuple[bool, str | None]:
    legal = set(legal_tools or STUDENT_NATIVE_TOOLS)
    if action.name not in legal or action.name in TEACHER_ONLY_TOOLS:
        return False, "ILLEGAL_TOOL"
    accessible = {str(x) for x in accessible_doc_ids}
    refs: list[str] = []
    for key in ("add_ids", "doc_ids", "evidence_ids"):
        refs.extend(_as_ids(action.arguments.get(key)))
    if action.arguments.get("doc_id"):
        refs.append(str(action.arguments["doc_id"]))
    acquisition = {"read_document", "fan_out_search", "search_corpus", "grep_corpus"}
    if action.name not in acquisition:
        missing = [doc for doc in refs if doc not in accessible]
        if missing:
            return False, "DOC_NOT_ACCESSIBLE"
    needed = {str(x) for x in (teacher_needed_doc_ids or [])}
    student_hits = {str(x) for x in (student_search_results or [])}
    if action.name in {"search_corpus", "fan_out_search", "grep_corpus"} and needed:
        if not needed.issubset(student_hits | accessible):
            return False, "TRANSITION_NOT_REPRODUCIBLE"
    return True, None


def classify_event(event: Mapping[str, Any]) -> str:
    if is_harness_only(event):
        return SKIP
    if student_action_of(event) is None:
        return SKIP
    return ALIGN


def project_events(
    events: list[Mapping[str, Any]],
    *,
    accessible_doc_ids: Iterable[str],
    legal_tools: Iterable[str] | None = None,
    student_search_results: Iterable[str] | None = None,
    teacher_needed_doc_ids: Iterable[str] | None = None,
    max_anchor_scan_events: int = MAX_ANCHOR_SCAN_EVENTS,
    component_id: str | None = None,
    first_only: bool = False,
) -> list[SkipToAnchorResult]:
    """Walk Teacher events. SKIP harness-only, ALIGN the next realizable Student tool call."""
    aligned: list[SkipToAnchorResult] = []
    skipped: list[str] = []
    start_id = str(events[0].get("event_id") or "") if events else ""
    scan_from = 0
    for idx, event in enumerate(events):
        event_id = str(event.get("event_id") or f"evt_{idx}")
        if idx >= scan_from + max_anchor_scan_events:
            skipped.append(event_id)
            continue
        if classify_event(event) == SKIP:
            skipped.append(event_id)
            continue
        action = student_action_of(event)
        assert action is not None
        accessible = event.get("accessible_doc_ids")
        ok, reason = action_realizable(
            action,
            accessible_doc_ids=accessible if accessible is not None else accessible_doc_ids,
            legal_tools=legal_tools,
            student_search_results=student_search_results,
            teacher_needed_doc_ids=teacher_needed_doc_ids,
        )
        if not ok:
            skipped.append(event_id)
            continue
        result = SkipToAnchorResult(
            kind=ALIGN,
            actions=[action],
            skipped_event_ids=list(skipped),
            teacher_start_event=start_id or event_id,
            teacher_end_event=event_id,
            anchor_event_id=event_id,
            anchor_distance=len(skipped),
            component_id=component_id or event.get("component_id"),
        )
        if result.kind not in {ALIGN, SKIP}:
            raise ValueError(f"projector must emit align or skip, got {result.kind}")
        aligned.append(result)
        if first_only:
            return aligned
        skipped = []
        start_id = ""
        scan_from = idx + 1
    if aligned:
        return aligned
    end_id = str(events[-1].get("event_id") or "") if events else ""
    return [
        SkipToAnchorResult(
            kind=SKIP,
            actions=[],
            skipped_event_ids=skipped,
            teacher_start_event=start_id,
            teacher_end_event=end_id,
            reason="NO_STUDENT_ANCHOR",
            component_id=component_id,
        )
    ]


def teacher_events_from_bridge_steps(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert Harness1Bridge.step outputs into a Teacher event list.

    A search that only triggered a Harness-only component is not itself an
    OPD label. The component event is SKIP; a later Student-native action
    (typically ``curate``) is the alignable anchor.
    """
    events: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        event = dict(step.get("event") or {})
        action = dict(step.get("student_action") or {})
        pre = step.get("pre_state") or {}
        post = step.get("post_state") or {}
        env = dict((post or pre).get("student_observable_env_state") or {})
        accessible = list(env.get("visible_doc_ids") or env.get("pool_ids") or [])
        component = str(event.get("component") or action.get("component") or "")
        if event.get("event_type") or event.get("event_active"):
            events.append(
                {
                    "event_id": f"evt_{index}",
                    "kind": "component_event",
                    "event_type": event.get("event_type"),
                    "action_name": None,
                    "arguments": {},
                    "state_delta": {
                        "added": (event.get("payload") or {}).get("delta_add")
                        or (event.get("payload") or {}).get("added"),
                        "removed": (event.get("payload") or {}).get("delta_remove")
                        or (event.get("payload") or {}).get("removed"),
                        "before_curated": (event.get("payload") or {}).get("curated_ids_pre"),
                        "after_curated": (event.get("payload") or {}).get("curated_ids_teacher_post"),
                    },
                    "payload": event.get("payload") or {},
                    "projectable_target": event.get("projectable_target"),
                    "visible_to_student": False,
                    "harness_only": bool(event.get("harness_only")),
                    "component_id": component,
                    "accessible_doc_ids": accessible,
                    "step_index": index,
                }
            )
            continue
        name = str(action.get("tool_name") or action.get("name") or "")
        if not name:
            continue
        events.append(
            {
                "event_id": f"act_{index}",
                "kind": "model_action",
                "event_type": None,
                "action_name": name,
                "arguments": dict(action.get("parameters") or action.get("arguments") or {}),
                "state_delta": {},
                "payload": {},
                "projectable_target": None,
                "visible_to_student": True,
                "component_id": component,
                "accessible_doc_ids": list(
                    (pre.get("student_observable_env_state") or {}).get("visible_doc_ids")
                    or (pre.get("student_observable_env_state") or {}).get("pool_ids")
                    or []
                ),
                "step_index": index,
            }
        )
    return events


def project_bridge_steps(
    steps: list[Mapping[str, Any]],
    *,
    component_id: str,
    legal_tools: Iterable[str] | None = None,
    max_anchor_scan_events: int = MAX_ANCHOR_SCAN_EVENTS,
) -> list[SkipToAnchorResult]:
    events = teacher_events_from_bridge_steps(steps)
    accessible: list[str] = []
    for event in events:
        for doc in event.get("accessible_doc_ids") or []:
            if doc not in accessible:
                accessible.append(str(doc))
    return project_events(
        events,
        accessible_doc_ids=accessible,
        legal_tools=legal_tools,
        max_anchor_scan_events=max_anchor_scan_events,
        component_id=component_id,
    )
