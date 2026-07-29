"""E0 distillability data schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.capability.capability_id import CapabilityId
from training.scope.distillability.modes import DistillabilityMode


@dataclass
class ProcAuditStats:
    visibility_violation_rate: float = 0.0
    new_observation_from_proc: int = 0
    external_call_from_proc: int = 0
    hidden_field_access: int = 0
    state_mutation_rate: float = 0.0
    n_proc_interventions: int = 0
    n_shadow_calls: int = 0

    @property
    def information_safe(self) -> bool:
        return (
            self.visibility_violation_rate == 0.0
            and self.new_observation_from_proc == 0
            and self.external_call_from_proc == 0
            and self.hidden_field_access == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "visibility_violation_rate": self.visibility_violation_rate,
            "new_observation_from_proc": self.new_observation_from_proc,
            "external_call_from_proc": self.external_call_from_proc,
            "hidden_field_access": self.hidden_field_access,
            "state_mutation_rate": self.state_mutation_rate,
            "n_proc_interventions": self.n_proc_interventions,
            "n_shadow_calls": self.n_shadow_calls,
            "proc_information_safe": self.information_safe,
        }


@dataclass
class DistillabilityMetricResult:
    metric: str
    R_off: float
    R_proc: float
    R_full: float
    delta_proc: float
    delta_full: float
    P_raw: float | None
    P_clipped: float | None
    probe_valid: bool
    invalid_reason: str = ""
    ci95: dict[str, list[float]] = field(default_factory=dict)
    paired_wins: int = 0
    paired_losses: int = 0
    paired_ties: int = 0
    confidence: str = "HIGH"
    n_queries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "R_off": self.R_off,
            "R_proc": self.R_proc,
            "R_full": self.R_full,
            "delta_proc": self.delta_proc,
            "delta_full": self.delta_full,
            "P_raw": self.P_raw,
            "P_clipped": self.P_clipped,
            "probe_valid": self.probe_valid,
            "invalid_reason": self.invalid_reason,
            "ci95": self.ci95,
            "paired_wins": self.paired_wins,
            "paired_losses": self.paired_losses,
            "paired_ties": self.paired_ties,
            "confidence": self.confidence,
            "n_queries": self.n_queries,
        }


@dataclass
class CapabilityDistillabilityResult:
    capability_id: str
    probe_supported: bool
    metrics: dict[str, DistillabilityMetricResult] = field(default_factory=dict)
    capability_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    proc_audit: ProcAuditStats = field(default_factory=ProcAuditStats)
    decision: str = "INCONCLUSIVE"
    decision_notes: str = ""
    full_reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        primary = self.metrics.get("recall")
        out: dict[str, Any] = {
            "capability_id": self.capability_id,
            "probe_supported": self.probe_supported,
            "full_reused": self.full_reused,
            "decision": self.decision,
            "decision_notes": self.decision_notes,
            "proc_audit": self.proc_audit.to_dict(),
            "capability_metrics": self.capability_metrics,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }
        if primary is not None:
            out.update(
                {
                    "R_off": primary.R_off,
                    "R_proc": primary.R_proc,
                    "R_full": primary.R_full,
                    "delta_proc": primary.delta_proc,
                    "delta_full": primary.delta_full,
                    "P_raw": primary.P_raw,
                    "P_clipped": primary.P_clipped,
                    "ci95": primary.ci95,
                    "probe_valid": primary.probe_valid,
                    "confidence": primary.confidence,
                    "n_queries": primary.n_queries,
                }
            )
        return out


@dataclass
class E0RunManifest:
    capability_id: CapabilityId
    mode: DistillabilityMode
    model_path: str
    harness_config: str
    config_hash: str
    query_ids: list[str]
    seed: int
    bm25_index_path: str
    max_turns: int
    temperature: float
    parallel: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id.value,
            "mode": self.mode.value,
            "model_path": self.model_path,
            "harness_config": self.harness_config,
            "config_hash": self.config_hash,
            "n_queries": len(self.query_ids),
            "query_ids": self.query_ids,
            "seed": self.seed,
            "bm25_index_path": self.bm25_index_path,
            "max_turns": self.max_turns,
            "temperature": self.temperature,
            "parallel": self.parallel,
        }
