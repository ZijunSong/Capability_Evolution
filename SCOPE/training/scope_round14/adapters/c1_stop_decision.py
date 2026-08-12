"""C1 stop_decision adapter — STOP/CONTINUE via premature_stop."""

from __future__ import annotations

from typing import Any

from training.scope_round14.adapters._harness_patch import patch_module
from training.scope_round14.adapters.base import CapabilityAdapter

_STOP_LABELS = frozenset({"STOP", "PREMATURE_STOP", "SHOULD_STOP"})
_CONTINUE_LABELS = frozenset({"CONTINUE", "NO_STOP"})


class StopDecisionAdapter(CapabilityAdapter):
  capability_id = "stop_decision"

  def __init__(self) -> None:
    super().__init__("stop_decision")

  def build_decision_state(self, raw: dict[str, Any]) -> dict[str, Any]:
    ds = dict(raw.get("decision_state") or {})
    for key in self.schema.decision_state_fields:
      ds.setdefault(key, raw.get(key))
    return ds

  def shadow_label(self, raw: dict[str, Any]) -> str:
    gold = (
      raw.get("gold_action")
      or raw.get("gold_operation")
      or raw.get("shadow_reason_code")
      or raw.get("target_action", {}).get("operation")
    )
    if gold:
      u = str(gold).upper()
      if u in _STOP_LABELS or "STOP" in u:
        return "STOP"
      if u in _CONTINUE_LABELS:
        return "CONTINUE"
    route = str(raw.get("shadow_route") or raw.get("route") or "").upper()
    if route == "CORRECT" and "STOP" in str(raw.get("shadow_reason_code", "")).upper():
      return "STOP"
    return "CONTINUE"

  def capability_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "premature_stop_fp": float(rollout_metrics.get("premature_stop_fp") or 0),
      "should_stop_miss": float(rollout_metrics.get("should_stop_miss") or 0),
      "stop_trigger_rate": float(rollout_metrics.get("stop_trigger_rate") or 0),
    }

  def side_effect_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "mean_turns": float(rollout_metrics.get("mean_turns") or 0),
      "task_recall": float(rollout_metrics.get("recall") or 0),
      "answer_acc": float(rollout_metrics.get("answer_acc") or 0),
    }

  def module_enable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(
      harness_cfg,
      "context_budget",
      flags={"stop_budget_hint": True},
    )

  def module_disable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(
      harness_cfg,
      "context_budget",
      flags={"stop_budget_hint": False},
    )
