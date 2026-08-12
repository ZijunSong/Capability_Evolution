"""C2 verification_routing — VERIFY / NO_VERIFY."""

from __future__ import annotations

from typing import Any

from training.scope_round14.adapters._harness_patch import patch_module
from training.scope_round14.adapters.base import CapabilityAdapter


class VerificationRoutingAdapter(CapabilityAdapter):
  capability_id = "verification_routing"

  def __init__(self) -> None:
    super().__init__("verification_routing")

  def build_decision_state(self, raw: dict[str, Any]) -> dict[str, Any]:
    ds = dict(raw.get("decision_state") or {})
    for key in self.schema.decision_state_fields:
      ds.setdefault(key, raw.get(key))
    return ds

  def shadow_label(self, raw: dict[str, Any]) -> str:
    gold = raw.get("gold_action") or raw.get("gold_operation")
    if gold:
      u = str(gold).upper()
      if u in {"VERIFY", "SHOULD_VERIFY", "YES"}:
        return "VERIFY"
      if u in {"NO_VERIFY", "SKIP_VERIFY", "NO"}:
        return "NO_VERIFY"
    reason = str(raw.get("shadow_reason_code") or "").upper()
    if "VERIFY" in reason and "NO_" not in reason:
      return "VERIFY"
    return "NO_VERIFY"

  def capability_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "missed_needed_verification": float(
        rollout_metrics.get("missed_needed_verification") or 0
      ),
      "unnecessary_verification": float(
        rollout_metrics.get("unnecessary_verification") or 0
      ),
      "verify_trigger_rate": float(rollout_metrics.get("verify_trigger_rate") or 0),
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
      flags={"verification_aware_curation": True, "harness_auto_verify_decision": True},
    )

  def module_disable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(
      harness_cfg,
      "verification",
      flags={"verification_aware_curation": False, "harness_auto_verify_decision": False},
    )
