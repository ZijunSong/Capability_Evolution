"""C3 evidence_admission — ADMIT / DROP (local curation decision)."""

from __future__ import annotations

from typing import Any

from harness.shadow.dup_bilateral_shadow import DupBilateralShadow
from training.scope_round14.adapters._harness_patch import patch_module
from training.scope_round14.adapters.base import CapabilityAdapter


class EvidenceAdmissionAdapter(CapabilityAdapter):
  capability_id = "evidence_admission"

  def __init__(self) -> None:
    super().__init__("evidence_admission")

  def build_decision_state(self, raw: dict[str, Any]) -> dict[str, Any]:
    ds = dict(raw.get("decision_state") or {})
    target = raw.get("target_action") or {}
    cid = target.get("candidate_id") or raw.get("candidate_evidence_id")
    if cid:
      ds.setdefault("candidate_evidence_id", cid)
    return ds

  def shadow_label(self, raw: dict[str, Any]) -> str:
    gold = raw.get("gold_action") or raw.get("gold_operation")
    if gold:
      u = str(gold).upper()
      if u in {"ADMIT", "KEEP_EVIDENCE", "KEEP"}:
        return "ADMIT"
      if u in {"DROP", "SKIP_DUPLICATE", "SKIP", "REJECT"}:
        return "DROP"
    op, _ = DupBilateralShadow.from_serialized_student_state(raw)
    return "DROP" if op.value == "SKIP_DUPLICATE" else "ADMIT"

  def capability_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "irrelevant_admission_rate": float(
        rollout_metrics.get("irrelevant_admission_rate") or 0
      ),
      "duplicate_admission_rate": float(
        rollout_metrics.get("duplicate_admission_rate") or 0
      ),
      "positive_evidence_retention": float(
        rollout_metrics.get("positive_evidence_retention") or 0
      ),
    }

  def side_effect_metric(self, rollout_metrics: dict[str, Any]) -> dict[str, float]:
    return {
      "n_curated_mean": float(rollout_metrics.get("n_curated_mean") or 0),
      "task_recall": float(rollout_metrics.get("recall") or 0),
      "mean_tokens": float(rollout_metrics.get("mean_tokens") or 0),
    }

  def module_enable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(
      harness_cfg,
      "evidence_state",
      enabled=True,
      flags={"subtractive_curation": True, "candidate_pool": True},
    )

  def module_disable(self, harness_cfg: dict[str, Any]) -> dict[str, Any]:
    return patch_module(
      harness_cfg,
      "evidence_state",
      enabled=False,
      flags={"subtractive_curation": False, "candidate_pool": False},
    )
