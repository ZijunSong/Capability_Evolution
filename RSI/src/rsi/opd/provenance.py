"""Information-provenance checks (task book §8.4).

Source marks are produced by the runtime/adapter, never trusted from candidate
declarations.  A first version cannot formally prove information flow for
arbitrary generated Python; the engineering checks are: controlled tool API,
external-observation source tracking, object-reference validation, isolated
replay and rejecting uncertainty.  Reports must present these as engineering
checks, not formal semantic guarantees.
"""

from __future__ import annotations

from typing import Sequence

from rsi.contracts.action import Action
from rsi.contracts.events import Event, Visibility
from rsi.contracts.trace import ProvenanceReport
from rsi.opd.realizability import extract_fact_refs, _looks_like_fact_ref


def collect_fact_refs_by_visibility(
    events: Sequence[Event],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split event-derived fact refs into (student-visible, teacher-private)."""
    student: set[str] = set()
    private: set[str] = set()
    for ev in events:
        refs: list[str] = []
        if ev.observation is not None:
            prov = ev.observation.provenance
            if prov is not None:
                refs.append(prov.source_ref)
            refs.extend(extract_fact_refs_from_text(ev.observation.content))
        if ev.action is not None:
            refs.extend(extract_fact_refs(ev.action))
        target = private if ev.visibility == Visibility.TEACHER_PRIVATE else student
        for r in refs:
            if _looks_like_fact_ref(r):
                target.add(r)
    return tuple(sorted(student)), tuple(sorted(private))


def extract_fact_refs_from_text(text: str) -> tuple[str, ...]:
    """Conservative scan of observation text for object references.

    Tokens with path-like or id-like shape; over-inclusive on purpose so that
    uncertain cases are *rejected* rather than silently allowed.
    """
    out: list[str] = []
    for tok in text.replace("\n", " ").split(" "):
        tok = tok.strip(" ,;:()[]{}\"'")
        if _looks_like_fact_ref(tok) and len(tok) >= 2:
            out.append(tok)
    return tuple(out)


def build_provenance_report(
    action: Action,
    *,
    student_events: Sequence[Event],
    teacher_events: Sequence[Event] = (),
    uncertain: str = "reject",
) -> ProvenanceReport:
    """Check where the target action's facts come from.

    ``uncertain='reject'`` marks uncertain cases ``verdict='uncertain'`` so the
    alignment layer skips them (task book default ``uncertain_realizability:
    skip``).
    """
    student_refs, _ = collect_fact_refs_by_visibility(student_events)
    _, private_refs = collect_fact_refs_by_visibility(teacher_events)
    needed = tuple(r for r in extract_fact_refs(action) if _looks_like_fact_ref(r))
    satisfied = tuple(r for r in needed if r in student_refs)
    leaked = tuple(r for r in needed if r in private_refs and r not in student_refs)
    missing = tuple(r for r in needed if r not in student_refs and r not in private_refs)
    if leaked:
        verdict = "leak"
    elif missing and uncertain == "reject":
        verdict = "uncertain"
    else:
        verdict = "clean"
    return ProvenanceReport(
        required_fact_refs=needed,
        satisfied_by_student_history=satisfied,
        leaked_refs=leaked,
        verdict=verdict,
    )
