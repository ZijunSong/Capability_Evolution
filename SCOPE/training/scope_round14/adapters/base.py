"""Abstract capability adapter for Round14 module-retirement protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from training.scope_round14.typed_schema import CapabilitySchema, get_schema


class CapabilityAdapter(ABC):
    """Uniform interface for capability-local decision + retirement eval."""

    capability_id: str

    def __init__(self, capability_id: str) -> None:
        self.capability_id = capability_id
        self.schema: CapabilitySchema = get_schema(capability_id)

    @abstractmethod
    def build_decision_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Construct info-safe decision state from a trajectory row."""

    @abstractmethod
    def shadow_label(self, raw: dict[str, Any]) -> str:
        """Return gold typed action for this capability."""

    def candidate_actions(self) -> list[str]:
        return list(self.schema.actions)

    @abstractmethod
    def capability_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
        """Capability-specific behavior metric from closed-loop rollout."""

    @abstractmethod
    def side_effect_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
        """Non-trigger / task side-effect metrics."""

    def module_config(self, harness_cfg: dict[str, Any], *, on: bool) -> dict[str, Any]:
        """Return harness config with target module policy on/off."""
        return self.module_enable(harness_cfg) if on else self.module_disable(harness_cfg)

    @abstractmethod
    def module_enable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
        """Enable target harness module for this capability."""

    @abstractmethod
    def module_disable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
        """Disable target harness module (module-retirement test)."""

    def normalize_row(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Standard training row with typed fields."""
        ds = self.build_decision_state(raw)
        label = self.shadow_label(raw)
        return {
            "capability_id": self.capability_id,
            "event_id": raw.get("event_id"),
            "query_id": raw.get("query_id"),
            "decision_state": ds,
            "gold_action": label,
            "candidate_actions": self.candidate_actions(),
            "provenance": raw.get("provenance") or {},
            "collection_mode": raw.get("collection_mode", "natural"),
        }
