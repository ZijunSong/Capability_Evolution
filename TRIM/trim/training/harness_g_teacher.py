"""Teacher side-branches for Harness-G advanced components.

Each helper emits Full-Harness events on a Student snapshot. Projection then
maps them onto the thin runtime tools (select / lookup / answer). Analogous
to Harness-1's auto_populate / verify / sentence_compress teachers.
"""

from __future__ import annotations

from typing import Any, Mapping

from trim.training.opd_events import HarnessEvent, model_action, obs_transform
from trim.training.rl_opd_types import StudentDecisionPoint


def _wm(point_or_wm: Mapping[str, Any] | StudentDecisionPoint) -> dict[str, Any]:
    if isinstance(point_or_wm, StudentDecisionPoint):
        return dict(point_or_wm.pre_action_snapshot.working_memory or {})
    return dict(point_or_wm or {})


def _visible_sids(wm: Mapping[str, Any]) -> list[str]:
    sids = [str(x) for x in (wm.get("visible_sids") or []) if str(x)]
    if sids:
        return sids
    for rec in wm.get("documents") or []:
        if isinstance(rec, Mapping) and rec.get("id"):
            sids.append(str(rec["id"]))
    return sids


def _selected_sids(wm: Mapping[str, Any]) -> list[str]:
    sids = [str(x) for x in (wm.get("selected_sids") or wm.get("curated_ids") or []) if str(x)]
    return sids


def _frontier_eids(wm: Mapping[str, Any]) -> list[str]:
    return [str(x) for x in (wm.get("frontier_eids") or []) if str(x)]


def _entities(wm: Mapping[str, Any]) -> dict[str, Any]:
    rec = wm.get("entities") or {}
    return dict(rec) if isinstance(rec, dict) else {}


def answer_with_events_from_wm(wm: Mapping[str, Any], *, turn_id: int = 0) -> list[HarnessEvent]:
    visible = [s for s in _visible_sids(wm) if s not in set(_selected_sids(wm))]
    if not visible:
        return [
            model_action(
                "answer",
                {"reason": "selected evidence is enough"},
                turn_id=turn_id,
                component_id="answer_with",
            )
        ]
    sid = visible[0]
    return [
        model_action(
            "answer_with",
            {"sid": sid, "sids": [sid]},
            turn_id=turn_id,
            component_id="answer_with",
            visible_to_student=False,
            metadata={"teacher_only": True, "projectable_target": {"name": "select", "arguments": {"sid": sid}}},
        )
    ]


def bridge_entities_events_from_wm(wm: Mapping[str, Any], *, turn_id: int = 0) -> list[HarnessEvent]:
    entities = _entities(wm)
    frontier = set(_frontier_eids(wm))
    selected = set(_selected_sids(wm))
    bridge_eid = None
    for eid, rec in entities.items():
        if eid in frontier:
            continue
        sids = [str(x) for x in (rec.get("sids") or [])]
        if selected and not (set(sids) & selected):
            bridge_eid = eid
            break
    if bridge_eid is None:
        visible = [s for s in _visible_sids(wm) if s not in selected]
        if visible:
            return [
                obs_transform(
                    "bridge_entities",
                    turn_id=turn_id,
                    observation={"bridge_candidates": []},
                    visible_to_student=False,
                    metadata={"event_type": "bridge_entities_privileged_context", "harness_only": True},
                ),
                model_action("select", {"sid": visible[0]}, turn_id=turn_id, component_id="bridge_entities"),
            ]
        return [model_action("answer", {}, turn_id=turn_id, component_id="bridge_entities")]
    return [
        obs_transform(
            "bridge_entities",
            turn_id=turn_id,
            observation={"bridge_eid": bridge_eid},
            visible_to_student=False,
            metadata={"event_type": "bridge_entities_privileged_context", "harness_only": True},
        ),
        model_action(
            "lookup",
            {"eid": bridge_eid},
            turn_id=turn_id,
            component_id="bridge_entities",
            metadata={"bridge_lookup": True},
        ),
    ]


def entity_synonyms_events_from_wm(wm: Mapping[str, Any], *, turn_id: int = 0) -> list[HarnessEvent]:
    entities = _entities(wm)
    eid = None
    extra_sids: list[str] = []
    for rec in entities.values():
        syns = rec.get("synonyms") or []
        if syns:
            eid = rec.get("eid")
            for syn in syns:
                extra_sids.extend((entities.get(syn) or {}).get("sids") or [])
            break
    events = [
        obs_transform(
            "entity_synonyms",
            turn_id=turn_id,
            observation={"expanded_sids": extra_sids[:6]},
            visible_to_student=False,
            metadata={"event_type": "entity_synonyms_privileged_context", "harness_only": True},
        )
    ]
    visible = _visible_sids(wm)
    if extra_sids:
        sid = extra_sids[0]
        events.append(
            model_action(
                "select",
                {"sid": sid},
                turn_id=turn_id,
                component_id="entity_synonyms",
                metadata={"projectable_target": {"name": "select", "arguments": {"sid": sid}}},
            )
        )
    elif eid:
        events.append(model_action("lookup", {"eid": eid}, turn_id=turn_id, component_id="entity_synonyms"))
    elif visible:
        events.append(model_action("select", {"sid": visible[0]}, turn_id=turn_id, component_id="entity_synonyms"))
    else:
        events.append(model_action("answer", {}, turn_id=turn_id, component_id="entity_synonyms"))
    return events


def sentence_neighbors_events_from_wm(wm: Mapping[str, Any], *, turn_id: int = 0) -> list[HarnessEvent]:
    sentences = wm.get("sentences") or {}
    selected = _selected_sids(wm)
    neighbor = None
    for sid in selected:
        rec = sentences.get(sid) or {}
        for nid in rec.get("neighbors") or []:
            if nid not in selected:
                neighbor = str(nid)
                break
        if neighbor:
            break
    events = [
        obs_transform(
            "sentence_neighbors",
            turn_id=turn_id,
            observation={"neighbor_sid": neighbor},
            visible_to_student=False,
            metadata={"event_type": "sentence_neighbors_privileged_context", "harness_only": True},
        )
    ]
    if neighbor:
        events.append(model_action("select", {"sid": neighbor}, turn_id=turn_id, component_id="sentence_neighbors"))
    else:
        visible = [s for s in _visible_sids(wm) if s not in set(selected)]
        if visible:
            events.append(model_action("select", {"sid": visible[0]}, turn_id=turn_id, component_id="sentence_neighbors"))
        else:
            events.append(model_action("answer", {}, turn_id=turn_id, component_id="sentence_neighbors"))
    return events


def hybrid_init_events_from_wm(wm: Mapping[str, Any], *, turn_id: int = 0) -> list[HarnessEvent]:
    visible = _visible_sids(wm)
    events = [
        obs_transform(
            "hybrid_init_retrieve",
            turn_id=turn_id,
            observation={"fused_ranking": visible[:6]},
            visible_to_student=False,
            metadata={"event_type": "hybrid_init_privileged_context", "harness_only": True},
        )
    ]
    if not wm.get("initialized") and not visible:
        events.append(model_action("init", {}, turn_id=turn_id, component_id="hybrid_init_retrieve"))
        return events
    if visible:
        events.append(model_action("select", {"sid": visible[0]}, turn_id=turn_id, component_id="hybrid_init_retrieve"))
    else:
        events.append(model_action("answer", {}, turn_id=turn_id, component_id="hybrid_init_retrieve"))
    return events


def snc_frontier_events_from_wm(wm: Mapping[str, Any], *, turn_id: int = 0) -> list[HarnessEvent]:
    menu = wm.get("action_map") or {}
    best = None
    best_score = -1.0
    for action in menu.values():
        score = float(action.get("snc_preview") or 0.0)
        if score > best_score and action.get("type") in {"SELECT", "LOOKUP"}:
            best_score = score
            best = action
    events = [
        obs_transform(
            "snc_frontier",
            turn_id=turn_id,
            observation={"best_snc_score": best_score, "best": best},
            visible_to_student=False,
            metadata={"event_type": "snc_frontier_privileged_context", "harness_only": True},
        )
    ]
    if best and best.get("type") == "SELECT":
        events.append(
            model_action("select", {"sid": best.get("sid")}, turn_id=turn_id, component_id="snc_frontier")
        )
    elif best and best.get("type") == "LOOKUP":
        events.append(
            model_action("lookup", {"eid": best.get("eid")}, turn_id=turn_id, component_id="snc_frontier")
        )
    else:
        visible = _visible_sids(wm)
        if visible:
            events.append(model_action("select", {"sid": visible[0]}, turn_id=turn_id, component_id="snc_frontier"))
        else:
            events.append(model_action("answer", {}, turn_id=turn_id, component_id="snc_frontier"))
    return events


def runtime_anchor_events_from_wm(
    wm: Mapping[str, Any],
    component_id: str,
    *,
    turn_id: int = 0,
) -> list[HarnessEvent]:
    """invalid_target_filter / lookup_dedup: skip-only runtime checks, then SELECT/ANSWER."""
    visible = [s for s in _visible_sids(wm) if s not in set(_selected_sids(wm))]
    events = [
        obs_transform(
            component_id,
            turn_id=turn_id,
            observation={"runtime_anchor": True},
            visible_to_student=False,
            metadata={"event_type": f"{component_id}_runtime_check", "harness_only": True},
        )
    ]
    if visible:
        events.append(model_action("select", {"sid": visible[0]}, turn_id=turn_id, component_id=component_id))
    else:
        events.append(model_action("answer", {}, turn_id=turn_id, component_id=component_id))
    return events


TEACHER_FROM_WM = {
    "answer_with": answer_with_events_from_wm,
    "bridge_entities": bridge_entities_events_from_wm,
    "entity_synonyms": entity_synonyms_events_from_wm,
    "sentence_neighbors": sentence_neighbors_events_from_wm,
    "hybrid_init_retrieve": hybrid_init_events_from_wm,
    "snc_frontier": snc_frontier_events_from_wm,
    "invalid_target_filter": lambda wm, turn_id=0: runtime_anchor_events_from_wm(
        wm, "invalid_target_filter", turn_id=turn_id
    ),
    "lookup_dedup": lambda wm, turn_id=0: runtime_anchor_events_from_wm(
        wm, "lookup_dedup", turn_id=turn_id
    ),
}


def teacher_events_from_wm(component_id: str, wm: Mapping[str, Any], *, turn_id: int = 0) -> list[HarnessEvent]:
    fn = TEACHER_FROM_WM.get(component_id)
    if fn is None:
        return []
    return fn(wm, turn_id=turn_id)


def teacher_events_from_point(point: StudentDecisionPoint) -> list[HarnessEvent]:
    cid = str(point.pre_action_snapshot.metadata.get("component_id") or "")
    first = cid.split(",")[0].strip() if cid else "answer_with"
    return teacher_events_from_wm(first, _wm(point), turn_id=int(point.turn_id))


def teacher_events_from_point_for(component_id: str, point: StudentDecisionPoint) -> list[HarnessEvent]:
    return teacher_events_from_wm(component_id, _wm(point), turn_id=int(point.turn_id))
