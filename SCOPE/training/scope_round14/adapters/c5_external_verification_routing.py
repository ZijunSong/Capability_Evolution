"""C5 external_verification_routing — VERIFY_EXTERNALLY / DO_NOT."""

from __future__ import annotations

from typing import Any

from training.scope_round14.adapters._harness_patch import patch_module
from training.scope_round14.adapters.base import CapabilityAdapter


class ExternalVerificationRoutingAdapter(CapabilityAdapter):
  capability_id = "external_verification_routing"

  def __init__(self) -> None:
    super().__init__("external_verification_routing")

  def build_decision_state(self, raw: dict[str, Any]) -> dict[str, Any]:
    ds = dict(raw.get("decision_state") or {})
    for key in self.schema.decision_state_fields:
      ds.setdefault(key, raw.get(key))
    ds.setdefault("tool_schema_visible", True)
    return ds

  def shadow_label(self, raw: dict[str, Any]) -> str:
    gold = raw.get("gold_action") or raw.get("gold_operation")
    if gold:
      u = str(gold).upper()
      if u in {"VERIFY_EXTERNALLY", "VERIFY", "YES", "EXTERNAL_VERIFY"}:
        return "VERIFY_EXTERNALLY"
      if u in {"DO_NOT", "NO_VERIFY", "SKIP", "NO"}:
        return "DO_NOT"
    if raw.get("used_external_verify") or raw.get("verify_tool_called"):
      return "VERIFY_EXTERNALLY"
    return "DO_NOT"

  def capability_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "external_verify_route_rate": float(
        rollout_metrics.get("external_verify_route_rate") or 0
      ),
      "missed_external_verify": float(rollout_metrics.get("missed_external_verify") or 0),
      "unnecessary_external_verify": float(
        rollout_metrics.get("unnecessary_external_verify") or 0
      ),
    }

  def side_effect_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "mean_tool_calls": float(rollout_metrics.get("mean_tool_calls") or 0),
      "task_recall": float(rollout_metrics.get("recall") or 0),
      "answer_acc": float(rollout_metrics.get("answer_acc") or 0),
    }

  def module_enable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(
      harness_cfg,
      "verification",
      enabled=True,
      flags={"expose_verify_tool": True, "harness_auto_verify_decision": True},
    )

  def module_disable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    # Tool execution stays; only routing policy module OFF.
    return patch_module(
      harness_cfg,
      "verification",
      enabled=True,
      flags={"expose_verify_tool": True, "harness_auto_verify_decision": False},
    )
