"""Alignment: map Teacher branch events to Student-native OPD supervision.

Route priority per Student decision (task book §8):

1. ``explicit_map`` — a component recipe expands an advanced Teacher action
   into a basic action program with preconditions; the *next executable* step
   is the target (max 1 per decision by default).
2. ``native_anchor`` — skip internal events, align on a later Student-native
   action that passes the three-layer realizability + provenance checks.
3. ``skip`` — record the reason; manufacture no supervision.
4. ``runtime_only`` — the component keeps runtime value but yields no OPD
   sample (kept in the runtime archive).

Execution and alignment are decoupled: component authors may ship
``alignment_hints`` but cannot declare themselves ``distillable``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.events import Event
from rsi.contracts.observation import ToolSpec
from rsi.contracts.trace import AlignmentResult
from rsi.opd.anchors import align_to_anchor
from rsi.opd.provenance import build_provenance_report, collect_fact_refs_by_visibility
from rsi.opd.realizability import build_realizability_report
from rsi.opd.recipes import Recipe, project_next_action


@dataclass
class AlignmentContext:
    catalog: tuple[ToolSpec, ...]
    state_view: Mapping[str, Any]
    uncertain_realizability: str = "skip"
    max_targets_per_decision: int = 1
    routes_enabled: tuple[str, ...] = ("explicit_map", "native_anchor")


def is_native_action(action: Action, catalog: Sequence[ToolSpec]) -> bool:
    if isinstance(action, Finish):
        return True
    return any(t.name == action.name for t in catalog)


def align_decision(
    *,
    teacher_events: Sequence[Event],
    student_events: Sequence[Event],
    student_pre_state_digest: str,
    ctx: AlignmentContext,
    recipes: Sequence[Recipe] = (),
    recipe_step_index: int = 0,
    visible_facts: Mapping[str, str] | None = None,
) -> AlignmentResult:
    """Align one Student decision point against its Teacher branch events."""
    facts = dict(visible_facts or {})
    # Visible facts from the Student's own history take priority.
    student_refs, _ = collect_fact_refs_by_visibility(student_events)
    for r in student_refs:
        facts.setdefault(r, r)

    # --- A-route: explicit recipe mapping -----------------------------
    if "explicit_map" in ctx.routes_enabled:
        for recipe in recipes:
            target, _next_idx, unresolved = project_next_action(
                recipe,
                step_index=recipe_step_index,
                visible_facts=facts,
                max_targets_per_decision=ctx.max_targets_per_decision,
            )
            if target is None:
                if unresolved:
                    # Missing info: try acquire-then-supervise for the first ref.
                    legality = build_realizability_report(
                        ToolCall(name="read_file", arguments={"path": unresolved[0]})
                        if "/" in unresolved[0] or "." in unresolved[0]
                        else ToolCall(name="get", arguments={"key": unresolved[0]}),
                        catalog=ctx.catalog,
                        state_view=ctx.state_view,
                        student_fact_refs=student_refs,
                    )
                    if legality.syntax_legal and legality.acquire_first_action is not None:
                        prov = build_provenance_report(
                            legality.acquire_first_action,
                            student_events=student_events,
                            teacher_events=teacher_events,
                        )
                        if prov.verdict != "leak":
                            return AlignmentResult(
                                route="explicit_map",
                                target_action=legality.acquire_first_action,
                                source_event_ids=tuple(e.event_id for e in teacher_events[-3:]),
                                anchor_index=None,
                                student_pre_state_digest=student_pre_state_digest,
                                legality=legality,
                                provenance=prov,
                                target_kind="teacher_scored_projected_target",
                            )
                continue
            # Validate the projected step like any target.
            teacher_private_refs = _private_refs(teacher_events)
            legality = build_realizability_report(
                target,
                catalog=ctx.catalog,
                state_view=ctx.state_view,
                student_fact_refs=student_refs,
                teacher_private_fact_refs=teacher_private_refs,
            )
            prov = build_provenance_report(
                target, student_events=student_events, teacher_events=teacher_events
            )
            if legality.ok and prov.verdict == "clean":
                return AlignmentResult(
                    route="explicit_map",
                    target_action=target,
                    source_event_ids=tuple(e.event_id for e in teacher_events[-3:]),
                    anchor_index=None,
                    student_pre_state_digest=student_pre_state_digest,
                    legality=legality,
                    provenance=prov,
                    target_kind="teacher_scored_projected_target",
                )

    # --- B-route: skip-to-anchor -------------------------------------
    if "native_anchor" in ctx.routes_enabled:
        return align_to_anchor(
            teacher_events,
            student_events=student_events,
            student_pre_state_digest=student_pre_state_digest,
            catalog=ctx.catalog,
            state_view=ctx.state_view,
            is_native_action=lambda a: is_native_action(a, ctx.catalog),
            uncertain_realizability=ctx.uncertain_realizability,
        )

    return AlignmentResult(
        route="skip",
        target_action=None,
        source_event_ids=(),
        anchor_index=None,
        student_pre_state_digest=student_pre_state_digest,
        reject_reason="no_route_enabled",
    )


def _private_refs(events: Sequence[Event]) -> tuple[str, ...]:
    from rsi.contracts.events import Visibility

    refs: set[str] = set()
    for ev in events:
        if ev.visibility == Visibility.TEACHER_PRIVATE and ev.observation is not None:
            if ev.observation.provenance is not None:
                refs.add(ev.observation.provenance.source_ref)
    return tuple(sorted(refs))
