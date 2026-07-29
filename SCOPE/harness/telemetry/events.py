"""SCOPE telemetry event types and helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCOPE_EVENT_TYPES = frozenset(
    {
        "episode_start",
        "decision_state_exported",
        "student_action_parsed",
        "shadow_trigger",
        "artifact_generated",
        "visibility_check",
        "guidance_routed",
        "candidate_generated",
        "candidate_validated",
        "supervision_sample_emitted",
        "opd_transition_created",
        "loss_computed",
        "episode_end",
    }
)


@dataclass
class ScopeEvent:
    event: str
    episode_id: str
    turn_id: int = 0
    module_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ScopeStats:
    """Aggregate counters required by SCOPE §18.2."""

    shadow_calls: dict[str, int] = field(default_factory=dict)
    endorse_count: int = 0
    correct_count: int = 0
    ignore_count: int = 0
    visibility_violations: int = 0
    candidate_pass: int = 0
    candidate_total: int = 0
    reason_code_counts: dict[str, int] = field(default_factory=dict)
    module_loss: dict[str, list[float]] = field(default_factory=dict)
    action_transitions: dict[str, int] = field(default_factory=dict)

    shadow_mutation_count: int = 0
    visibility_violation_count: int = 0
    invalid_action_count: int = 0
    verifier_reject_count: int = 0
    capability_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_guidance(self, mode: str, module_id: str, reason_code: str) -> None:
        self.shadow_calls[module_id] = self.shadow_calls.get(module_id, 0) + 1
        if mode == "endorse":
            self.endorse_count += 1
        elif mode == "correct":
            self.correct_count += 1
        else:
            self.ignore_count += 1
        self.reason_code_counts[reason_code] = (
            self.reason_code_counts.get(reason_code, 0) + 1
        )

    def record_capability_route(
        self,
        capability_id: str,
        route: str,
        *,
        visibility_violation: bool = False,
        shadow_mutation: bool = False,
        invalid_action: bool = False,
        verifier_reject: bool = False,
    ) -> None:
        bucket = self.capability_stats.setdefault(
            capability_id,
            {"calls": 0, "endorse": 0, "correct": 0, "ignore": 0},
        )
        bucket["calls"] += 1
        key = route.lower()
        if key in bucket:
            bucket[key] += 1
        else:
            bucket["ignore"] += 1
        if visibility_violation:
            self.visibility_violation_count += 1
        if shadow_mutation:
            self.shadow_mutation_count += 1
        if invalid_action:
            self.invalid_action_count += 1
        if verifier_reject:
            self.verifier_reject_count += 1

    def to_dict(self) -> dict[str, Any]:
        total_g = max(1, self.endorse_count + self.correct_count + self.ignore_count)
        n_shadow = max(1, sum(self.shadow_calls.values()))
        return {
            "shadow_calls": dict(self.shadow_calls),
            "endorse_ratio": self.endorse_count / total_g,
            "correct_ratio": self.correct_count / total_g,
            "ignore_ratio": self.ignore_count / total_g,
            "visibility_violations": self.visibility_violations,
            "visibility_violation_rate": self.visibility_violation_count / n_shadow,
            "shadow_mutation_rate": self.shadow_mutation_count / n_shadow,
            "invalid_action_rate": self.invalid_action_count / n_shadow,
            "verifier_reject_rate": self.verifier_reject_count / n_shadow,
            "candidate_pass_rate": self.candidate_pass / max(1, self.candidate_total),
            "reason_code_counts": dict(self.reason_code_counts),
            "capability_stats": dict(self.capability_stats),
            "module_loss_mean": {
                k: (sum(v) / len(v) if v else 0.0) for k, v in self.module_loss.items()
            },
            "action_transitions": dict(self.action_transitions),
        }
