"""Teacher side-branch for auto_populate_first_search.

The full Harness automatically copies the first search pool into curated.  The
student cannot see that hidden mutation, so SR-OPD exposes only the equivalent
student-executable curated delta and never the full-only view.

This builder only fires at the first-search trigger: before any search, or
immediately after the first search. Later leftover uncurated docs are skip.
"""

from __future__ import annotations

from typing import Any, Mapping

from trim.eval.h1_component_runtime import AUTO_POPULATE_TOP_K
from trim.training.opd_events import HarnessEvent, harness_mutation, model_action, obs_transform
from trim.training.rl_opd_types import StudentDecisionPoint

COMPONENT_ID = "auto_populate_first_search"
SEARCH_TOOLS = frozenset({"search_corpus", "grep_corpus", "fan_out_search"})


def _records(wm: Mapping[str, Any]) -> dict[str, Any]:
    pool = wm.get("pool") or {}
    return dict(pool) if isinstance(pool, dict) else {}


def _search_count(wm: Mapping[str, Any]) -> int:
    return int(wm.get("search_count") or wm.get("n_search_calls") or 0)


def _last_tool_name(wm: Mapping[str, Any]) -> str:
    last = wm.get("last_tool_name")
    if last:
        return str(last)
    hist = wm.get("tool_history") or []
    if hist and isinstance(hist[-1], dict):
        return str(hist[-1].get("name") or "")
    return ""


def first_search_pending(wm: Mapping[str, Any]) -> bool:
    if "first_search_pending" in wm:
        return bool(wm.get("first_search_pending"))
    if wm.get("first_search_done") is True:
        return False
    return _search_count(wm) == 0


def is_first_search_trigger(wm: Mapping[str, Any]) -> str | None:
    """Return 'need_first_search', 'apply_auto_populate', or None (skip)."""
    pending = first_search_pending(wm)
    n_search = _search_count(wm)
    last = _last_tool_name(wm)
    if pending or n_search == 0:
        return "need_first_search"
    if (not pending) and n_search == 1 and last in SEARCH_TOOLS:
        return "apply_auto_populate"
    return None


def _skip(turn_id: int, reason: str) -> list[HarnessEvent]:
    return [
        obs_transform(
            COMPONENT_ID,
            turn_id=turn_id,
            observation={"owner": "teacher_full", "skip_reason": reason},
            visible_to_student=False,
            metadata={
                "teacher_kind": "skip_untriggered",
                "source_type": "skip_untriggered",
                "skip_reason": reason,
                "not_a_continuous_teacher_rollout": True,
            },
        )
    ]


def teacher_events_from_wm(
    wm: Mapping[str, Any], *, turn_id: int = 0, query: str | None = None
) -> list[HarnessEvent]:
    """Emit first-search search or the real auto-populate curated delta."""
    trigger = is_first_search_trigger(wm)
    q = str(query if query is not None else wm.get("query") or "")
    if trigger is None:
        return _skip(turn_id, "not_first_search_trigger")
    if trigger == "need_first_search":
        return [
            model_action(
                "search_corpus",
                {"query": q},
                turn_id=turn_id,
                component_id=COMPONENT_ID,
                metadata={
                    "owner": "teacher_full",
                    "auto_anchor": True,
                    "trigger": trigger,
                    "source_type": "capability_effect",
                    "teacher_kind": "capability_effect",
                },
            )
        ]
    pool = _records(wm)
    curated = [str(x) for x in (wm.get("curated_ids") or [])]
    ranked = sorted(
        pool.items(),
        key=lambda item: (-float((item[1] or {}).get("score") or 0.0), str(item[0])),
    )
    added = [str(did) for did, _ in ranked[: int(AUTO_POPULATE_TOP_K)] if str(did) not in set(curated)]
    after = list(dict.fromkeys(curated + added))
    if after == curated:
        return _skip(turn_id, "first_search_no_uncurated_delta")
    return [
        harness_mutation(
            COMPONENT_ID,
            {"before_curated": curated, "after_curated": after},
            turn_id=turn_id,
            metadata={
                "owner": "teacher_full",
                "hidden_auto_effect": True,
                "trigger": trigger,
                "first_search_bound": True,
                "source_type": "capability_effect",
                "teacher_kind": "capability_effect",
            },
        )
    ]


def teacher_events_from_point(point: StudentDecisionPoint) -> list[HarnessEvent]:
    wm = point.pre_action_snapshot.working_memory
    return teacher_events_from_wm(wm, turn_id=int(point.turn_id), query=wm.get("query"))
