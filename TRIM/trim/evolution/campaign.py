"""Harness search that is allowed to stop, and is not allowed to start TRIM."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from trim.evolution.operators import Proposal
from trim.evolution.search import evolve_bank
from trim.harness_bank.adaptation import project_ordered_events
from trim.harness_bank.loader import load_bank
from trim.harness_bank.runtime import EpisodeBankRuntime
from trim.state.snapshot import EnvironmentSnapshot
from trim.adapters.components import zero_mask
from trim.training.opd_events import EventKind, HarnessEvent
from trim.training.opd_projection import ProjectionKind


def distillability_probe(bank_dir: str) -> dict[str, Any]:
    bank = load_bank(bank_dir)
    positive = EpisodeBankRuntime(bank, role="all")
    rendered = positive.render(
        {"query": "visible evidence", "visible_search_history": [{"name": "search_corpus"}]},
        role="all",
    )
    negative = EpisodeBankRuntime(bank, role="all")
    quiet = negative.render({"query": "visible evidence", "visible_search_history": []}, role="all")
    triggered = bool(rendered.get("teacher_guidance")) and not quiet.get("teacher_guidance")
    snap = EnvironmentSnapshot(
        query_id="probe",
        step=0,
        harness_mask=zero_mask("Harness-1"),
        working_memory={"query": "visible evidence", "accessible_doc_ids": []},
        query_text="visible evidence",
    )
    event = HarnessEvent(
        event_id="probe",
        turn_id=0,
        kind=EventKind.MODEL_ACTION,
        component_id=bank.active_components[0] if bank.active_components else "evo.query_repair",
        action_name="search_corpus",
        arguments={"query": "visible evidence"},
        metadata={"information_provenance": [{"source": "visible_query", "available_to_student": True}]},
    )
    projected = project_ordered_events([event], snap, zero_mask("Harness-1"))
    return {
        "coverage": 1.0 if triggered else 0.0,
        "projection_complete": projected.kind == ProjectionKind.DIRECT,
        "coverage_source": "structural_trigger",
    }


def score_against_zero(
    proposal: Proposal,
    all_traces: Sequence[Mapping[str, Any]],
    zero_by_id: Mapping[str, float],
) -> dict[str, Any]:
    paired: list[float] = []
    solved: list[float] = []
    all_scores: list[float] = []
    for trace in all_traces:
        qid = str(trace.get("query_id") or "")
        score = float(trace.get("final_answer_recall") if trace.get("final_answer_recall") is not None else trace.get("recall") or 0.0)
        all_scores.append(score)
        if qid not in zero_by_id:
            continue
        base = float(zero_by_id[qid])
        paired.append(score - base)
        if base >= 0.999:
            solved.append(score - base)
    if not paired:
        return {
            "quality": 0.0,
            "delta_h": 0.0,
            "per_question": {},
            "cost": {"total": 0.0},
            "retention_ok": False,
            "protocol_ok": False,
            "projection_complete": False,
            "used_unavailable_info": False,
            "coverage": 0.0,
            "distillable": False,
            "active_components": 1,
            "trigger_stats": {},
        }
    probe = distillability_probe(proposal.bank_dir)
    quality = sum(all_scores) / len(all_scores)
    retention_ok = (not solved) or (sum(solved) / len(solved) >= -0.02)
    return {
        "quality": quality,
        "delta_h": sum(paired) / len(paired),
        "per_question": {"n": len(paired)},
        "cost": {"total": float(sum(float(trace.get("n_tool_calls") or 0.0) for trace in all_traces))},
        "retention_ok": retention_ok,
        "protocol_ok": True,
        "projection_complete": bool(probe["projection_complete"]),
        "used_unavailable_info": False,
        "coverage": float(probe["coverage"]),
        "distillable": bool(probe["projection_complete"]) and float(probe["coverage"]) >= 0.10,
        "active_components": 1,
        "trigger_stats": {"coverage_source": probe["coverage_source"]},
    }


def search_until_trim(
    *,
    proposer,
    score_fn: Callable[[Proposal], Mapping[str, Any]],
    baseline_quality: float,
    min_generations: int = 3,
    max_generations: int = 8,
    max_candidate_evaluations: int = 16,
    max_invalid_proposals: int = 12,
) -> dict[str, Any]:
    """Run harness search. A CONFIRM_TEACHER result still does not launch training."""
    result = evolve_bank(
        proposer=proposer,
        score_fn=score_fn,
        baseline_quality=baseline_quality,
        min_generations=min_generations,
        max_generations=max_generations,
        max_candidate_evaluations=max_candidate_evaluations,
        max_invalid_proposals=max_invalid_proposals,
    )
    decision = str(result.get("decision") or "")
    ready = decision == "CONFIRM_TEACHER" and bool(result.get("has_eligible_champion"))
    result.update(
        {
            "executed_training": False,
            "started_trim": False,
            "stop_after": "harness_search",
            "next_stage": "trim" if ready else None,
            "ready_for_trim": ready,
        }
    )
    return result
