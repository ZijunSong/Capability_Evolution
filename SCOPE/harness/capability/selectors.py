"""Critical state selectors for typed shadow modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.state import DecisionState


class CriticalStateSelector(Protocol):
    def select(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> list[str]:
        """Return module ids that should run on this state."""
        ...


@dataclass
class SelectorConfig:
    before_stop: bool = True
    after_curate: bool = True
    after_verify: bool = True
    after_review: bool = True
    after_pool_growth: bool = True
    repeated_query: bool = False
    low_remaining_turns: bool = False
    evidence_enabled: bool = True
    verification_enabled: bool = True
    budget_enabled: bool = False
    low_turn_threshold: int = 3


@dataclass
class TriggerEvent:
    module_id: str
    trigger: str
    student_action: str


class RuleBasedCriticalStateSelector:
    """Rule-based selector for verification / evidence / budget modules."""

    def __init__(self, config: SelectorConfig | None = None) -> None:
        self.config = config or SelectorConfig()
        self.last_triggers: list[TriggerEvent] = []

    def select(
        self,
        state: DecisionState,
        student_action: CapabilityAction,
    ) -> list[str]:
        self.last_triggers = []
        modules: list[str] = []

        if self.config.verification_enabled:
            modules.extend(self._select_verification(state, student_action))
        if self.config.evidence_enabled:
            modules.extend(self._select_evidence(state, student_action))
        if self.config.budget_enabled:
            modules.extend(self._select_budget(state, student_action))

        # Deduplicate preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for mid in modules:
            if mid not in seen:
                seen.add(mid)
                ordered.append(mid)
        return ordered

    def _emit(self, module_id: str, trigger: str, action: CapabilityAction) -> None:
        self.last_triggers.append(
            TriggerEvent(
                module_id=module_id,
                trigger=trigger,
                student_action=action.action_type.value,
            )
        )

    def _select_verification(
        self,
        state: DecisionState,
        action: CapabilityAction,
    ) -> list[str]:
        hits: list[str] = []
        # Primary triggers: stop_and_answer / answer / verify_claim
        if self.config.before_stop and action.action_type in {
            CapabilityActionType.STOP_AND_ANSWER,
            CapabilityActionType.ANSWER,
            CapabilityActionType.ABSTAIN,
        }:
            hits.append("verification")
            trigger = (
                "before_answer"
                if action.action_type == CapabilityActionType.ANSWER
                else "before_stop"
            )
            self._emit("verification", trigger, action)
        if self.config.after_verify and action.action_type == CapabilityActionType.VERIFY_CLAIM:
            hits.append("verification")
            self._emit("verification", "after_verify", action)
        # Student marks evidence supported
        if action.action_type == CapabilityActionType.UPDATE_EVIDENCE:
            status = str(action.arguments.get("status", "")).lower()
            if status in {"supported", "verified"}:
                hits.append("verification")
                self._emit("verification", "evidence_supported", action)
        # Unverified claims when preparing to stop/answer
        unresolved = [
            c
            for c in state.evidence_claims
            if c.status.lower() in {"unverified", "unsupported", "conflict"}
        ]
        if unresolved and action.action_type in {
            CapabilityActionType.STOP_AND_ANSWER,
            CapabilityActionType.ANSWER,
            CapabilityActionType.ABSTAIN,
        }:
            if "verification" not in hits:
                hits.append("verification")
                self._emit("verification", "unverified_claims", action)
        # Conflicts in verification records (even mid-episode)
        for rec in state.verification_records:
            vals = list(rec.judgments.values())
            if vals and (True in vals) and (False in vals):
                hits.append("verification")
                self._emit("verification", "conflict", action)
                break
        return hits

    def _select_evidence(
        self,
        state: DecisionState,
        action: CapabilityAction,
    ) -> list[str]:
        hits: list[str] = []
        if self.config.after_curate and action.action_type in {
            CapabilityActionType.CURATE_DOCUMENT,
            CapabilityActionType.UPDATE_EVIDENCE,
        }:
            hits.append("evidence_state")
            self._emit("evidence_state", "after_curate", action)
        if self.config.after_review and action.action_type == CapabilityActionType.REVIEW_DOCS:
            hits.append("evidence_state")
            self._emit("evidence_state", "after_review", action)
        if self.config.after_pool_growth and action.action_type in {
            CapabilityActionType.SEARCH,
            CapabilityActionType.GREP,
            CapabilityActionType.OPEN_DOCUMENT,
        }:
            hits.append("evidence_state")
            self._emit("evidence_state", "pool_growth", action)
        if any(c.status != getattr(c, "_prev_status", c.status) for c in state.evidence_claims):
            hits.append("evidence_state")
            self._emit("evidence_state", "claim_status_change", action)
        return hits

    def _select_budget(
        self,
        state: DecisionState,
        action: CapabilityAction,
    ) -> list[str]:
        hits: list[str] = []
        if self.config.repeated_query and (state.repeated_query_score or 0.0) >= 0.8:
            hits.append("budget_control")
            self._emit("budget_control", "repeated_query", action)
        if self.config.low_remaining_turns and state.remaining_turns <= self.config.low_turn_threshold:
            hits.append("budget_control")
            self._emit("budget_control", "low_remaining_turns", action)
        if action.action_type in {
            CapabilityActionType.STOP_AND_ANSWER,
            CapabilityActionType.ANSWER,
            CapabilityActionType.ABSTAIN,
        }:
            hits.append("budget_control")
            self._emit("budget_control", "prepare_stop", action)
        return hits
