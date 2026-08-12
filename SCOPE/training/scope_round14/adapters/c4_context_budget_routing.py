"""C4 context_budget_routing — KEEP_CONTEXT / COMPRESS_OR_DROP."""

from __future__ import annotations

from typing import Any

from training.scope_round14.adapters._harness_patch import patch_module
from training.scope_round14.adapters.base import CapabilityAdapter


class ContextBudgetRoutingAdapter(CapabilityAdapter):
  capability_id = "context_budget_routing"

  def __init__(self) -> None:
    super().__init__("context_budget_routing")

  def build_decision_state(self, raw: dict[str, Any]) -> dict[str, Any]:
    ds = dict(raw.get("decision_state") or {})
    for key in self.schema.decision_state_fields:
      ds.setdefault(key, raw.get(key))
    return ds

  def shadow_label(self, raw: dict[str, Any]) -> str:
    gold = raw.get("gold_action") or raw.get("gold_operation")
    if gold:
      u = str(gold).upper()
      if u in {"KEEP_CONTEXT", "KEEP"}:
        return "KEEP_CONTEXT"
      if u in {"COMPRESS_OR_DROP", "COMPRESS", "DROP", "TRUNCATE"}:
        return "COMPRESS_OR_DROP"
    near = raw.get("near_budget_boundary") or raw.get("token_pressure")
    if near:
      return "COMPRESS_OR_DROP"
    return "KEEP_CONTEXT"

  def capability_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "hard_truncation_rate": float(rollout_metrics.get("hard_truncation_rate") or 0),
      "budget_violation_rate": float(rollout_metrics.get("budget_violation_rate") or 0),
      "positive_evidence_lost_rate": float(
        rollout_metrics.get("positive_evidence_lost_rate") or 0
      ),
    }

  def side_effect_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "mean_tokens": float(rollout_metrics.get("mean_tokens") or 0),
      "task_recall": float(rollout_metrics.get("recall") or 0),
      "answer_acc": float(rollout_metrics.get("answer_acc") or 0),
    }

  def module_enable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(
      harness_cfg,
      "context_budget",
      enabled=True,
      flags={
        "sentence_compression": True,
        "structured_context_rendering": True,
        "token_budget_marker": True,
        "deterministic_truncation": True,
      },
    )

  def module_disable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    # Routing OFF but keep hard safety truncation in runtime.
    return patch_module(
      harness_cfg,
      "context_budget",
      enabled=True,
      flags={
        "sentence_compression": False,
        "structured_context_rendering": False,
        "token_budget_marker": False,
        "deterministic_truncation": True,
      },
    )
