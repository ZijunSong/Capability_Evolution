"""Lexicographic bank search. Novelty does not buy teacher admission."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from trim.evolution.archive import Archive, CandidateRecord
from trim.evolution.operators import Proposal
from trim.evolution.proposer import Proposer
from trim.evolution.validator import validate_proposal
from trim.rsi.switch_policy import EVOLVING, SearchMeasurement, decide


def select_champion(
    records: list[CandidateRecord],
    *,
    epsilon_select: float,
) -> CandidateRecord | None:
    pool = [
        record
        for record in records
        if record.distillable
        and record.protocol_ok
        and record.retention_ok
        and record.projection_complete
        and not record.used_unavailable_info
        and not record.reject_reason
    ]
    if not pool:
        return None
    best_quality = max(record.quality for record in pool)
    close = [record for record in pool if best_quality - record.quality <= epsilon_select]
    close.sort(
        key=lambda record: (
            float(record.cost.get("total") or 0.0),
            -float(record.coverage),
            int(record.active_components),
            record.content_hash,
        )
    )
    return close[0]


def evolve_bank(
    *,
    proposer: Proposer,
    score_fn: Callable[[Proposal], Mapping[str, Any]],
    baseline_quality: float,
    min_generations: int = 3,
    max_generations: int = 8,
    max_candidate_evaluations: int = 16,
    max_invalid_proposals: int = 12,
    max_edits: int = 1,
    max_active_components: int = 6,
    plateau_patience: int = 3,
    epsilon_plateau: float = 0.005,
    epsilon_select: float = 0.005,
    min_observed_gain: float = 0.02,
    delta_sufficient: float = 0.05,
    prune_every: int = 2,
    forbidden_ids: set[str] | None = None,
) -> dict[str, Any]:
    archive = Archive()
    quality_history = [float(baseline_quality)]
    cost_history = [0.0]
    evaluated = 0
    invalid = 0
    generation = 0
    recent_gains: list[float] = []
    decision_status = EVOLVING
    stop_reason = ""
    champion: CandidateRecord | None = None
    budget_exhausted = False
    complexity_blocked = False
    while generation < max_generations and decision_status == EVOLVING:
        packet = {
            "generation": generation,
            "want_prune": bool(prune_every and generation and generation % prune_every == 0),
        }
        try:
            proposals = list(proposer.propose(packet))
        except Exception as exc:
            invalid += 1
            if invalid >= max_invalid_proposals:
                budget_exhausted = True
                break
            stop_reason = f"proposer_error:{exc}"
            break
        if not proposals:
            invalid += 1
            if invalid >= max_invalid_proposals:
                budget_exhausted = True
                break
            continue
        generation_records: list[CandidateRecord] = []
        saw_valid = False
        for proposal in proposals:
            if evaluated >= max_candidate_evaluations:
                budget_exhausted = True
                break
            problem = validate_proposal(
                proposal,
                max_edits=max_edits,
                max_active_components=max_active_components,
                forbidden_ids=forbidden_ids,
            )
            if problem:
                invalid += 1
                archive.reject(
                    CandidateRecord(
                        content_hash=proposal.content_hash or proposal.proposal_id,
                        source=proposal.bank_dir,
                        lineage=list(proposal.lineage),
                        per_question={},
                        cost={},
                        trigger_stats={},
                        proposal_type=proposal.proposal_type,
                    ),
                    problem,
                )
                if invalid >= max_invalid_proposals:
                    budget_exhausted = True
                    break
                continue
            saw_valid = True
            payload = dict(score_fn(proposal))
            evaluated += 1
            if payload.get("complexity_blocked"):
                complexity_blocked = True
            record = CandidateRecord(
                content_hash=proposal.content_hash,
                source=proposal.bank_dir,
                lineage=list(proposal.lineage),
                per_question=dict(payload.get("per_question") or {}),
                cost=dict(payload.get("cost") or {}),
                trigger_stats=dict(payload.get("trigger_stats") or {}),
                distillable=bool(payload.get("distillable", True)),
                quality=float(payload.get("quality") or 0.0),
                proposal_type=proposal.proposal_type,
                active_components=int(payload.get("active_components") or len(proposal.touched_components)),
                coverage=float(payload.get("coverage") or 0.0),
                protocol_ok=bool(payload.get("protocol_ok", True)),
                retention_ok=bool(payload.get("retention_ok", True)),
                projection_complete=bool(payload.get("projection_complete", True)),
                used_unavailable_info=bool(payload.get("used_unavailable_info", False)),
            )
            if not record.protocol_ok:
                archive.reject(record, "protocol")
            elif not record.retention_ok:
                archive.reject(record, "retention")
            elif record.used_unavailable_info or not record.projection_complete:
                archive.reject(record, "projection")
            else:
                archive.add(record)
            generation_records.append(record)
        if budget_exhausted:
            break
        if not saw_valid:
            continue
        generation += 1
        champion = select_champion(list(archive.distillable.values()), epsilon_select=epsilon_select)
        quality = baseline_quality if champion is None else champion.quality
        cost = 0.0 if champion is None else float(champion.cost.get("total") or 0.0)
        quality_history.append(quality)
        cost_history.append(cost)
        gain = 0.0 if champion is None else float(champion.quality - baseline_quality)
        recent_gains.append(gain)
        measurement = SearchMeasurement(
            generation=generation,
            min_generations=min_generations,
            quality_history=quality_history,
            cost_history=cost_history,
            epsilon_plateau=epsilon_plateau,
            epsilon_select=epsilon_select,
            plateau_patience=plateau_patience,
            delta_h=gain,
            min_observed_gain=min_observed_gain,
            delta_sufficient=delta_sufficient,
            recent_gains=list(recent_gains),
            budget_exhausted=False,
            complexity_blocked=complexity_blocked,
            g0_pass=champion is not None,
            g2_pass=bool(champion and champion.coverage >= 0.10 and champion.projection_complete),
            g2_insufficient=False,
            g3_pass=not complexity_blocked,
            retention_ok=bool(champion and champion.retention_ok),
            has_champion=champion is not None,
            evidence_status="ok",
            empty_bank=champion is None,
        )
        decision = decide(measurement)
        decision_status = decision.status
        stop_reason = decision.reason
        if budget_exhausted:
            break
    if budget_exhausted and decision_status == EVOLVING:
        measurement = SearchMeasurement(
            generation=max(generation, min_generations),
            min_generations=min_generations,
            quality_history=quality_history,
            cost_history=cost_history,
            budget_exhausted=True,
            has_champion=champion is not None,
            delta_h=0.0 if champion is None else champion.quality - baseline_quality,
            g0_pass=champion is not None,
            g2_pass=bool(champion and champion.projection_complete),
            g3_pass=not complexity_blocked,
            retention_ok=bool(champion and champion.retention_ok),
            empty_bank=champion is None,
            evidence_status="ok",
        )
        decision = decide(measurement)
        decision_status = decision.status
        stop_reason = decision.reason
    eligible = decision_status == "CONFIRM_TEACHER"
    return {
        "has_eligible_champion": eligible,
        "stop_reason": decision_status if decision_status != EVOLVING else (stop_reason or "EVOLVING"),
        "decision": decision_status,
        "champion": None
        if champion is None or not eligible
        else {"bank_sha256": champion.content_hash, "source": champion.source, "quality": champion.quality},
        "generations": generation,
        "evaluated": evaluated,
        "invalid": invalid,
        "archive_runtime": len(archive.runtime),
        "archive_distillable": len(archive.distillable),
        "quality_history": quality_history,
    }
