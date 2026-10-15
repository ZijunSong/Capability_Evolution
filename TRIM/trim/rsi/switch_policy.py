"""Deterministic search-to-TRIM decision table. Feelings about a plateau are not an input."""

from __future__ import annotations

from dataclasses import dataclass, field

from trim.rsi.types import RoundDecision

EVOLVING = "EVOLVING"
STOP_NO_TEACHER_GAIN = "STOP_NO_TEACHER_GAIN"
STOP_UNDISTILLABLE = "STOP_UNDISTILLABLE"
STOP_INSUFFICIENT_EVIDENCE = "STOP_INSUFFICIENT_EVIDENCE"
CONFIRM_TEACHER = "CONFIRM_TEACHER"
TRAINING = "TRAINING"
STOP_CONFIRMATION_FAILED = "STOP_CONFIRMATION_FAILED"


@dataclass
class SearchMeasurement:
    generation: int
    min_generations: int
    quality_history: list[float]
    cost_history: list[float] = field(default_factory=list)
    epsilon_plateau: float = 0.005
    epsilon_select: float = 0.005
    plateau_patience: int = 3
    material_cost_reduction: float = 0.05
    delta_h: float = 0.0
    min_observed_gain: float = 0.02
    delta_sufficient: float = 0.05
    recent_gains: list[float] = field(default_factory=list)
    budget_exhausted: bool = False
    complexity_blocked: bool = False
    g0_pass: bool = False
    g2_pass: bool = False
    g2_insufficient: bool = False
    g3_pass: bool = False
    retention_ok: bool = False
    has_champion: bool = False
    evidence_status: str = "ok"
    lcb: float | None = None
    confirm_attempted: bool = False
    confirm_passed: bool = False
    empty_bank: bool = False


def generations_since_record(
    quality: list[float],
    cost: list[float] | None,
    *,
    epsilon: float,
    epsilon_select: float,
    material_cost_reduction: float,
) -> int:
    """Count generations since quality or real cost last moved the record point."""
    if not quality:
        return 0
    record_q = quality[0]
    record_c = cost[0] if cost else None
    since = 0
    for index, value in enumerate(quality):
        if index == 0:
            continue
        improved = value - record_q > epsilon
        cheaper = False
        if cost and record_c is not None and index < len(cost):
            dropped = record_c - cost[index]
            quality_held = value >= record_q - epsilon_select
            cheaper = quality_held and record_c > 0 and dropped >= material_cost_reduction * record_c
        if improved or cheaper:
            record_q = value
            if cost and index < len(cost):
                record_c = cost[index]
            since = 0
        else:
            since += 1
    return since


def stop_reasons(measurement: SearchMeasurement) -> list[str]:
    reasons: list[str] = []
    if measurement.generation < measurement.min_generations:
        return reasons
    since = generations_since_record(
        measurement.quality_history,
        measurement.cost_history,
        epsilon=measurement.epsilon_plateau,
        epsilon_select=measurement.epsilon_select,
        material_cost_reduction=measurement.material_cost_reduction,
    )
    if since >= measurement.plateau_patience:
        reasons.append("plateau")
    recent = list(measurement.recent_gains[-2:])
    if (
        measurement.has_champion
        and measurement.delta_h >= measurement.delta_sufficient
        and len(recent) >= 2
        and all(value >= measurement.delta_sufficient for value in recent)
    ):
        reasons.append("sufficient_gap")
    if measurement.budget_exhausted:
        reasons.append("budget_exhausted")
    if measurement.complexity_blocked:
        reasons.append("complexity_blocked")
    return reasons


def g1_pass(measurement: SearchMeasurement) -> bool:
    return (
        measurement.has_champion
        and not measurement.empty_bank
        and measurement.delta_h >= measurement.min_observed_gain
        and measurement.retention_ok
        and measurement.evidence_status == "ok"
    )


def eligible(measurement: SearchMeasurement) -> bool:
    return bool(
        measurement.g0_pass
        and g1_pass(measurement)
        and measurement.g2_pass
        and not measurement.g2_insufficient
        and measurement.g3_pass
        and measurement.evidence_status == "ok"
    )


def decide(measurement: SearchMeasurement) -> RoundDecision:
    if measurement.confirm_attempted:
        if measurement.confirm_passed and eligible(measurement) and (measurement.lcb or 0.0) > 0.0:
            return RoundDecision(
                status=TRAINING,
                reason="confirm_h_pass",
                stop_search=True,
                eligible=True,
                confirm_h_pass=True,
            )
        return RoundDecision(
            status=STOP_CONFIRMATION_FAILED,
            reason="teacher confirmation failed; the same panel cannot host a second candidate",
            stop_search=True,
            eligible=eligible(measurement),
            confirm_h_pass=False,
        )
    reasons = stop_reasons(measurement)
    if not reasons:
        return RoundDecision(status=EVOLVING, reason="search budget remains", stop_search=False)
    if not g1_pass(measurement):
        if measurement.evidence_status != "ok" or measurement.g2_insufficient:
            return RoundDecision(
                status=STOP_INSUFFICIENT_EVIDENCE,
                reason="stopped without enough paired support to claim a teacher gain",
                stop_search=True,
                details={"stop": reasons},
            )
        return RoundDecision(
            status=STOP_NO_TEACHER_GAIN,
            reason="search stopped without a pre-registered all-zero gain",
            stop_search=True,
            details={"stop": reasons, "delta_h": measurement.delta_h},
        )
    if measurement.g2_insufficient or measurement.evidence_status != "ok":
        return RoundDecision(
            status=STOP_INSUFFICIENT_EVIDENCE,
            reason="runtime gain is not statistically supported",
            stop_search=True,
            details={"stop": reasons},
        )
    if not measurement.g2_pass:
        return RoundDecision(
            status=STOP_UNDISTILLABLE,
            reason="the harness gains are not yet a legal student-side target",
            stop_search=True,
            details={"stop": reasons},
        )
    if not (measurement.g0_pass and measurement.g3_pass):
        return RoundDecision(
            status=STOP_NO_TEACHER_GAIN,
            reason="complexity or base-contract gate failed",
            stop_search=True,
            details={"stop": reasons},
        )
    return RoundDecision(
        status=CONFIRM_TEACHER,
        reason="development gates passed; freeze one champion and confirm it once",
        stop_search=True,
        eligible=True,
        details={"stop": reasons},
    )
