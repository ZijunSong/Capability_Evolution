"""Realizability gate and decision-relevant semantic state for SR-OPD."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from scape.state.snapshot import EnvironmentSnapshot, capture_snapshot
from scape.training.action_codec import TEACHER_ONLY_TOOLS, canonicalize_action
from scape.training.tool_mask import (
    legal_tool_names,
    validate_action_arguments,
)


@dataclass(frozen=True)
class DecisionStateSignature:
    curated_ids: tuple[str, ...]
    accessible_doc_ids: tuple[str, ...]
    terminated: bool


@dataclass
class RealizabilityReport:
    legal_tool: bool
    valid_arguments: bool
    referenced_objects_accessible: bool
    information_derivable: bool
    transition_reproducible: bool
    reason_codes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(
            [
                self.legal_tool,
                self.valid_arguments,
                self.referenced_objects_accessible,
                self.information_derivable,
                self.transition_reproducible,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "legal_tool": self.legal_tool,
            "valid_arguments": self.valid_arguments,
            "referenced_objects_accessible": self.referenced_objects_accessible,
            "information_derivable": self.information_derivable,
            "transition_reproducible": self.transition_reproducible,
            "reason_codes": list(self.reason_codes),
            "passed": self.passed,
        }


def _as_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if x is not None and str(x)]
    if isinstance(value, dict):
        return [str(x) for x in value.keys() if str(x)]
    return [str(value)]


def curated_ids_of(snapshot: EnvironmentSnapshot | Mapping[str, Any]) -> list[str]:
    wm = snapshot.working_memory if isinstance(snapshot, EnvironmentSnapshot) else dict(snapshot)
    if wm.get("curated_ids") is not None:
        return _as_id_list(wm.get("curated_ids"))
    docs = wm.get("curated_docs") or []
    out: list[str] = []
    for doc in docs:
        if isinstance(doc, Mapping):
            did = doc.get("id")
            if did is not None:
                out.append(str(did))
        else:
            out.append(str(doc))
    return out


def accessible_doc_ids_of(snapshot: EnvironmentSnapshot | Mapping[str, Any]) -> list[str]:
    wm = snapshot.working_memory if isinstance(snapshot, EnvironmentSnapshot) else dict(snapshot)
    seen: list[str] = []

    def _add(values: Any) -> None:
        for item in _as_id_list(values):
            if item not in seen:
                seen.append(item)

    _add(wm.get("accessible_doc_ids"))
    _add(wm.get("curated_ids"))
    _add(wm.get("reviewed_ids"))
    _add(wm.get("full_text_ids"))
    _add(wm.get("pool"))
    for key in ("documents", "curated_docs"):
        for doc in wm.get(key) or []:
            if isinstance(doc, Mapping):
                _add(doc.get("id"))
            else:
                _add(doc)
    return seen


def referenced_doc_ids(action: Mapping[str, Any] | Any) -> list[str]:
    canon = canonicalize_action(action)
    args = canon["arguments"]
    ids: list[str] = []
    for key in ("add_ids", "remove_ids", "doc_ids", "evidence_ids", "result_ids"):
        ids.extend(_as_id_list(args.get(key)))
    if args.get("doc_id"):
        ids.append(str(args["doc_id"]))
    # stable unique
    out: list[str] = []
    for item in ids:
        if item not in out:
            out.append(item)
    return out


def has_full_text(snapshot: EnvironmentSnapshot, doc_id: str) -> bool:
    wm = snapshot.working_memory
    reviewed = set(_as_id_list(wm.get("reviewed_ids"))) | set(_as_id_list(wm.get("full_text_ids")))
    if str(doc_id) in reviewed:
        return True
    min_chars = int(wm.get("full_text_min_chars") or 1)
    for key in ("documents", "curated_docs"):
        for doc in wm.get(key) or []:
            if not isinstance(doc, Mapping):
                continue
            if str(doc.get("id")) != str(doc_id):
                continue
            text = str(doc.get("text") or doc.get("content") or "")
            return len(text) >= min_chars
    return False


def decision_state_signature(snapshot: EnvironmentSnapshot) -> DecisionStateSignature:
    return DecisionStateSignature(
        curated_ids=tuple(sorted(curated_ids_of(snapshot))),
        accessible_doc_ids=tuple(sorted(accessible_doc_ids_of(snapshot))),
        terminated=bool(snapshot.working_memory.get("terminated")),
    )


def curated_set_delta(
    before: EnvironmentSnapshot | Mapping[str, Any],
    after: EnvironmentSnapshot | Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    before_ids = set(curated_ids_of(before))
    after_ids = set(curated_ids_of(after))
    return sorted(after_ids - before_ids), sorted(before_ids - after_ids)


def check_action_realizability(
    *,
    action: Any,
    student_snapshot: EnvironmentSnapshot,
    student_mask: Mapping[str, bool],
    student_tool_schemas: Mapping[str, Any] | None = None,
    component_id: str = "",
) -> RealizabilityReport:
    """R1–R5 gate. Does not inspect component-specific losses."""
    del student_tool_schemas, component_id  # schemas currently live in tool_mask
    canon = canonicalize_action(action)
    reasons: list[str] = []

    legal_tools = set(legal_tool_names(harness_mask=student_mask))
    legal_tool = canon["name"] in legal_tools and canon["name"] not in TEACHER_ONLY_TOOLS
    if not legal_tool:
        reasons.append("ILLEGAL_TOOL")

    arg_ok, arg_reason = validate_action_arguments(
        canon["name"], canon["arguments"], harness_mask=student_mask
    )
    if not arg_ok:
        reasons.append(arg_reason or "INVALID_ARGUMENT_SCHEMA")

    refs = referenced_doc_ids(canon)
    accessible = set(accessible_doc_ids_of(student_snapshot))
    # Acquisition tools obtain docs; they do not require prior access.
    acquisition_tools = {"read_document", "fan_out_search", "search_corpus", "grep_corpus"}
    refs_needed = [
        did
        for did in refs
        if did not in set(_as_id_list(canon["arguments"].get("remove_ids")))
    ]
    if canon["name"] in acquisition_tools:
        refs_ok = True
    else:
        refs_ok = all(did in accessible for did in refs_needed)
    if not refs_ok:
        reasons.append("DOC_NOT_ACCESSIBLE")

    info_ok = True
    if canon["name"] == "verify":
        info_ok = False
        reasons.append("TEACHER_ONLY_INFORMATION")
    if "importance" in canon["arguments"] and not student_mask.get("importance_tagging", False):
        info_ok = False
        reasons.append("TEACHER_ONLY_INFORMATION")

    transition_ok = True
    teacher_needed = student_snapshot.metadata.get("teacher_needed_doc_ids")
    student_search = student_snapshot.metadata.get("student_search_results")
    if (
        canon["name"] in {"search_corpus", "fan_out_search", "grep_corpus"}
        and teacher_needed
        and student_search is not None
    ):
        if not set(str(x) for x in teacher_needed).issubset(set(str(x) for x in student_search) | accessible):
            transition_ok = False
            reasons.append("TRANSITION_NOT_REPRODUCIBLE")

    return RealizabilityReport(
        legal_tool=legal_tool,
        valid_arguments=arg_ok,
        referenced_objects_accessible=refs_ok,
        information_derivable=info_ok,
        transition_reproducible=transition_ok,
        reason_codes=reasons,
    )


def fork_snapshot(snapshot: EnvironmentSnapshot) -> EnvironmentSnapshot:
    """Deep-copy a snapshot. Teacher and Student shadows must never merge."""
    return EnvironmentSnapshot.from_dict(deepcopy(snapshot.to_dict()))


def apply_student_action(
    snapshot: EnvironmentSnapshot,
    action: Mapping[str, Any] | Any,
) -> EnvironmentSnapshot:
    """Advance the Student shadow only. Never copies Teacher observations."""
    canon = canonicalize_action(action)
    wm = deepcopy(snapshot.working_memory)
    name = canon["name"]
    args = canon["arguments"]
    if name == "curate":
        curated = set(curated_ids_of(snapshot))
        curated |= set(_as_id_list(args.get("add_ids")))
        curated -= set(_as_id_list(args.get("remove_ids")))
        wm["curated_ids"] = sorted(curated)
        if "importance" in args:
            raise ValueError("student shadow must not store importance arguments")
    elif name in {"review_docs", "read_document"}:
        ids = _as_id_list(args.get("doc_ids"))
        if args.get("doc_id"):
            ids.append(str(args["doc_id"]))
        reviewed = _as_id_list(wm.get("reviewed_ids"))
        accessible = accessible_doc_ids_of(snapshot)
        for did in ids:
            if did not in reviewed:
                reviewed.append(did)
            if did not in accessible:
                accessible.append(did)
        wm["reviewed_ids"] = reviewed
        wm["full_text_ids"] = list(reviewed)
        wm["accessible_doc_ids"] = accessible
    elif name in {"fan_out_search", "search_corpus", "grep_corpus"}:
        results = _as_id_list(args.get("result_ids") or wm.get("last_search_results"))
        accessible = accessible_doc_ids_of(snapshot)
        for did in results:
            if did not in accessible:
                accessible.append(did)
        wm["accessible_doc_ids"] = accessible
        wm["last_search_results"] = results
    elif name == "end_search":
        wm["terminated"] = True
    elif name in TEACHER_ONLY_TOOLS:
        raise ValueError(f"refusing to execute teacher-only tool on student shadow: {name}")

    hist = list(snapshot.tool_history)
    hist.append({"step": snapshot.step, "action": canon})
    obs = list(snapshot.observations)
    obs.append({"step": snapshot.step + 1, "ok": True, "action": name, "student_only": True})
    next_snap = capture_snapshot(
        query_id=snapshot.query_id,
        step=snapshot.step + 1,
        harness_mask=snapshot.harness_mask,
        working_memory=wm,
        tool_history=hist,
        observations=obs,
        metadata=dict(snapshot.metadata),
    )
    next_snap.assert_no_future(max_known_step=next_snap.step)
    return next_snap
