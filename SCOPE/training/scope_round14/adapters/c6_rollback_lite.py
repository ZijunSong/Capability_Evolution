"""C6 rollback_lite — RECOVER / CONTINUE from R13 operation labels."""

from __future__ import annotations

from typing import Any

from training.scope_round14.adapters._harness_patch import patch_module
from training.scope_round14.adapters.base import CapabilityAdapter

_RECOVER_OPS = frozenset({"RECOVER", "REPLAN", "ROLLBACK_TO", "ROLLBACK"})


class RollbackLiteAdapter(CapabilityAdapter):
  capability_id = "rollback_lite"

  def __init__(self) -> None:
    super().__init__("rollback_lite")

  @staticmethod
  def remap_operation(operation: str | None) -> str:
    if operation is None:
      return "CONTINUE"
    u = str(operation).upper()
    if u in _RECOVER_OPS:
      return "RECOVER"
    return "CONTINUE"

  def build_decision_state(self, raw: dict[str, Any]) -> dict[str, Any]:
    ds = dict(raw.get("decision_state") or {})
    # Strip checkpoint pointer fields from Stage1 decision state.
    for drop in (
      "checkpoint_registry",
      "candidate_checkpoint_ids",
      "gold_checkpoint_id",
      "gold_checkpoint_global_id",
      "checkpoint_metadata",
    ):
      ds.pop(drop, None)
    for key in self.schema.decision_state_fields:
      ds.setdefault(key, raw.get(key))
    return ds

  def shadow_label(self, raw: dict[str, Any]) -> str:
    gold = (
      raw.get("gold_action")
      or raw.get("gold_operation")
      or raw.get("operation")
      or (raw.get("target_action") or {}).get("operation")
    )
    return self.remap_operation(gold)

  def capability_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "recover_recall": float(rollout_metrics.get("RecoverRecall") or rollout_metrics.get("recover_recall") or 0),
      "continue_recall": float(rollout_metrics.get("ContinueRecall") or rollout_metrics.get("continue_recall") or 0),
      "recover_trigger_rate": float(rollout_metrics.get("recover_trigger_rate") or 0),
    }

  def side_effect_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "mean_turns": float(rollout_metrics.get("mean_turns") or 0),
      "task_recall": float(rollout_metrics.get("recall") or 0),
      "balanced_accuracy": float(rollout_metrics.get("balanced_accuracy") or 0),
    }

  def module_enable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(harness_cfg, "recovery", enabled=True)

  def module_disable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(harness_cfg, "recovery", enabled=False)
