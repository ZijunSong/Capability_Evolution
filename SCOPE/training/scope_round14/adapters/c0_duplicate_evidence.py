"""C0 duplicate_evidence adapter — DupDecisionPoint / DupBilateralShadow."""

from __future__ import annotations

from typing import Any

from harness.capability.dup_operation import DupOperation
from harness.shadow.dup_bilateral_shadow import DupBilateralShadow
from training.scope_round14.adapters._harness_patch import patch_module
from training.scope_round14.adapters.base import CapabilityAdapter


class DuplicateEvidenceAdapter(CapabilityAdapter):
  capability_id = "duplicate_evidence"

  def __init__(self) -> None:
    super().__init__("duplicate_evidence")
    self._shadow = DupBilateralShadow()

  def build_decision_state(self, raw: dict[str, Any]) -> dict[str, Any]:
    ds = dict(raw.get("decision_state") or {})
    target = raw.get("target_action") or {}
    cid = target.get("candidate_id") or raw.get("candidate_evidence_id")
    if cid:
      ds.setdefault("candidate_evidence_id", cid)
    return ds

  def shadow_label(self, raw: dict[str, Any]) -> str:
    gold = raw.get("gold_operation") or raw.get("gold_action")
    if gold:
      return str(gold)
    op, _ = DupBilateralShadow.from_serialized_student_state(raw)
    return op.value

  def capability_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    tel = rollout_metrics.get("dup_telemetry") or rollout_metrics
    return {
      "balanced_accuracy": float(
        rollout_metrics.get("balanced_accuracy") or tel.get("balanced_accuracy") or 0
      ),
      "duplicate_reject_rate": float(
        rollout_metrics.get("duplicate_reject_rate") or tel.get("duplicate_reject_rate") or 0
      ),
      "false_skip_rate": float(
        rollout_metrics.get("false_skip_rate") or tel.get("false_skip_rate") or 0
      ),
      "keep_recall": float(
        rollout_metrics.get("keep_recall")
        or (tel.get("KEEP_EVIDENCE") or {}).get("recall")
        or 0
      ),
      "skip_recall": float(
        rollout_metrics.get("skip_recall")
        or (tel.get("SKIP_DUPLICATE") or {}).get("recall")
        or 0
      ),
      "dup_skip_rate": float(rollout_metrics.get("dup_skip_rate") or tel.get("duplicate_reject_rate") or 0),
      "dup_false_admit_rate": float(rollout_metrics.get("dup_false_admit_rate") or tel.get("false_skip_rate") or 0),
      "dup_module_trigger_rate": float(rollout_metrics.get("dup_module_trigger_rate") or 0),
    }

  def side_effect_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "task_recall": float(rollout_metrics.get("recall") or rollout_metrics.get("task_recall") or 0),
      "answer_acc": float(rollout_metrics.get("answer_acc") or 0),
      "mean_turns": float(rollout_metrics.get("mean_turns") or 0),
    }

  def module_enable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(
      harness_cfg,
      "evidence_state",
      enabled=True,
      flags={"content_dedup": True, "subtractive_curation": True},
    )

  def module_disable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(
      harness_cfg,
      "evidence_state",
      enabled=False,
      flags={"content_dedup": False, "subtractive_curation": False},
    )

  def map_training_label(self, label: str) -> str:
    """Map shadow KEEP/SKIP to DupOperation enum values."""
    u = label.upper()
    if u in {DupOperation.KEEP_EVIDENCE.value, "ADMIT", "KEEP"}:
      return DupOperation.KEEP_EVIDENCE.value
    if u in {DupOperation.SKIP_DUPLICATE.value, "DROP", "SKIP"}:
      return DupOperation.SKIP_DUPLICATE.value
    return label
