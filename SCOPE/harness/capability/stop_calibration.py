"""Stop-vs-Continue bilateral calibration signals for Premature Stop selector."""

from __future__ import annotations

from dataclasses import dataclass

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState

_STOPPING_ACTIONS = frozenset(
    {
        CapabilityActionType.STOP_AND_ANSWER,
        CapabilityActionType.ANSWER,
        CapabilityActionType.ABSTAIN,
    }
)

_CONTINUING_ACTIONS = frozenset(
    {
        CapabilityActionType.SEARCH,
        CapabilityActionType.GREP,
        CapabilityActionType.OPEN_DOCUMENT,
        CapabilityActionType.REVIEW_DOCS,
        CapabilityActionType.CONTINUE_SEARCH,
        CapabilityActionType.REWRITE_QUERY,
        CapabilityActionType.CURATE_DOCUMENT,
        CapabilityActionType.UPDATE_EVIDENCE,
        CapabilityActionType.VERIFY_CLAIM,
    }
)


def _has_conflict(state: DecisionState) -> bool:
    for rec in state.verification_records:
        vals = list(rec.judgments.values())
        if vals and (True in vals) and (False in vals):
            return True
    return False


def _positive_verification(state: DecisionState) -> bool:
    if not state.verification_records:
        return False
    for rec in state.verification_records:
        if any(rec.judgments.values()):
            return True
    return False


def _supported_claim_ratio(state: DecisionState) -> float:
    claims = state.evidence_claims
    if not claims:
        return 0.0
    supported = sum(
        1
        for c in claims
        if str(c.status).lower() in {"supported", "verified", "linked"}
        and c.supporting_document_ids
    )
    return supported / len(claims)


@dataclass(frozen=True)
class StopCalibrationConfig:
    high_coverage_threshold: float = 0.75
    min_curated_docs: int = 1
    stagnant_search_steps: int = 2
    low_remaining_turns: int = 5
    require_positive_verify: bool = True


@dataclass(frozen=True)
class StopCalibrationSignals:
    coverage_score: float
    supported_claim_ratio: float
    curated_count: int
    positive_verify: bool
    has_conflict: bool
    stagnant_search_steps: int
    low_budget: bool
    sufficient_for_stop: bool


def _recent_search_steps(state: DecisionState, window: int = 6) -> list[str]:
    steps: list[str] = []
    for rec in reversed(state.action_history):
        if rec.action_type in {
            CapabilityActionType.SEARCH.value,
            CapabilityActionType.GREP.value,
            CapabilityActionType.CONTINUE_SEARCH.value,
            CapabilityActionType.REWRITE_QUERY.value,
        }:
            steps.append(rec.action_type)
            if len(steps) >= window:
                break
    return list(reversed(steps))


def _stagnant_search_count(state: DecisionState, window: int = 6) -> int:
    """Count trailing search-like actions with no new curated growth."""
    if not state.action_history:
        return 0
    stagnant = 0
    for rec in reversed(state.action_history):
        at = rec.action_type
        if at in {
            CapabilityActionType.SEARCH.value,
            CapabilityActionType.GREP.value,
            CapabilityActionType.CONTINUE_SEARCH.value,
            CapabilityActionType.OPEN_DOCUMENT.value,
            CapabilityActionType.REVIEW_DOCS.value,
        }:
            stagnant += 1
            continue
        if at in {
            CapabilityActionType.CURATE_DOCUMENT.value,
            CapabilityActionType.UPDATE_EVIDENCE.value,
            CapabilityActionType.VERIFY_CLAIM.value,
        }:
            break
    return stagnant


def compute_coverage_score(state: DecisionState) -> float:
    curated_n = len(state.curated_document_ids)
    claim_ratio = _supported_claim_ratio(state)
    verify_bonus = 0.25 if _positive_verification(state) else 0.0
    curated_term = min(1.0, curated_n / 3.0) * 0.45
    claim_term = claim_ratio * 0.45
    return min(1.0, curated_term + claim_term + verify_bonus)


def compute_stop_calibration_signals(
    state: DecisionState,
    *,
    config: StopCalibrationConfig | None = None,
) -> StopCalibrationSignals:
    cfg = config or StopCalibrationConfig()
    state = state.with_derived_claims()
    coverage = compute_coverage_score(state)
    positive_verify = _positive_verification(state)
    conflict = _has_conflict(state)
    curated_count = len(state.curated_document_ids)
    claim_ratio = _supported_claim_ratio(state)
    stagnant = _stagnant_search_count(state)
    low_budget = state.remaining_turns <= cfg.low_remaining_turns

    sufficient = (
        curated_count >= cfg.min_curated_docs
        and claim_ratio >= cfg.high_coverage_threshold
        and positive_verify
        and not conflict
        and coverage >= cfg.high_coverage_threshold
    )
    if cfg.require_positive_verify and not positive_verify:
        sufficient = False

    return StopCalibrationSignals(
        coverage_score=coverage,
        supported_claim_ratio=claim_ratio,
        curated_count=curated_count,
        positive_verify=positive_verify,
        has_conflict=conflict,
        stagnant_search_steps=stagnant,
        low_budget=low_budget,
        sufficient_for_stop=sufficient,
    )


def should_trigger_stop_calibration(
    state: DecisionState,
    student_action: CapabilityAction,
    *,
    config: StopCalibrationConfig | None = None,
) -> tuple[bool, str]:
    """Trigger Shadow check on student-visited states (not only before stop)."""
    cfg = config or StopCalibrationConfig()
    if student_action.action_type in _STOPPING_ACTIONS:
        return False, ""

    if student_action.action_type not in _CONTINUING_ACTIONS:
        return False, ""

    signals = compute_stop_calibration_signals(state, config=cfg)
    if not signals.sufficient_for_stop:
        return False, ""

    if signals.stagnant_search_steps >= cfg.stagnant_search_steps:
        return True, "high_coverage_stagnant_search"

    if (
        student_action.action_type
        in {
            CapabilityActionType.SEARCH,
            CapabilityActionType.GREP,
            CapabilityActionType.CONTINUE_SEARCH,
            CapabilityActionType.REWRITE_QUERY,
        }
        and signals.coverage_score >= cfg.high_coverage_threshold
    ):
        return True, "high_coverage_continue_search"

    if signals.low_budget:
        return True, "low_budget_sufficient_evidence"

    return False, ""


def evidence_sufficient_for_stop(state: DecisionState) -> bool:
    return compute_stop_calibration_signals(state).sufficient_for_stop


_STOPPING_TYPES = frozenset(
    {
        CapabilityActionType.STOP_AND_ANSWER,
        CapabilityActionType.ANSWER,
        CapabilityActionType.ABSTAIN,
    }
)


def student_wants_stop(action: CapabilityAction) -> bool:
    return action.action_type in _STOPPING_TYPES


def shadow_wants_stop(recommended: CapabilityAction | None) -> bool:
    if recommended is None:
        return False
    return recommended.action_type in _STOPPING_TYPES


def classify_stop_quadrant(
    student_action: CapabilityAction,
    shadow_action: CapabilityAction | None,
) -> str:
    """Classify stop-vs-continue decision into four quadrants."""
    s_stop = student_wants_stop(student_action)
    sh_stop = shadow_wants_stop(shadow_action)
    if s_stop and sh_stop:
        return "STOP→STOP"
    if s_stop and not sh_stop:
        return "STOP→CONTINUE"
    if not s_stop and sh_stop:
        return "CONTINUE→STOP"
    return "CONTINUE→CONTINUE"


@dataclass
class StopQuadrantStats:
    n_decision_points: int = 0
    stop_to_stop: int = 0
    stop_to_continue: int = 0
    continue_to_stop: int = 0
    continue_to_continue: int = 0

    def record(self, quadrant: str) -> None:
        self.n_decision_points += 1
        if quadrant == "STOP→STOP":
            self.stop_to_stop += 1
        elif quadrant == "STOP→CONTINUE":
            self.stop_to_continue += 1
        elif quadrant == "CONTINUE→STOP":
            self.continue_to_stop += 1
        else:
            self.continue_to_continue += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "n_decision_points": self.n_decision_points,
            "STOP→STOP": self.stop_to_stop,
            "STOP→CONTINUE": self.stop_to_continue,
            "CONTINUE→STOP": self.continue_to_stop,
            "CONTINUE→CONTINUE": self.continue_to_continue,
        }

