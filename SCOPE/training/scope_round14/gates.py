"""CapabilityInternalizationEvidence + ModuleRetirementGate (Round14)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class InternalizationStatus(str, Enum):
  PROVEN_INTERNALIZED = "PROVEN_INTERNALIZED"
  HYBRID_INTERNALIZED = "HYBRID_INTERNALIZED"
  UNRESOLVED = "UNRESOLVED"
  CURRENTLY_NOT_INTERNALIZED = "CURRENTLY_NOT_INTERNALIZED"
  RUNTIME_EXECUTION_REQUIRED = "RUNTIME_EXECUTION_REQUIRED"
  ALREADY_INTRINSIC = "ALREADY_INTRINSIC"


@dataclass
class CapabilityInternalizationEvidence:
  capability_id: str
  event_support: dict[str, Any] = field(default_factory=dict)
  bilateral_support: dict[str, Any] = field(default_factory=dict)
  information_safe: dict[str, Any] = field(default_factory=dict)
  state_observability: dict[str, Any] = field(default_factory=dict)
  base_gap: dict[str, Any] = field(default_factory=dict)
  local_learnability: dict[str, Any] = field(default_factory=dict)
  fresh_live_transfer: dict[str, Any] = field(default_factory=dict)
  retirement_behavior_gain: dict[str, Any] = field(default_factory=dict)
  task_retention: dict[str, Any] = field(default_factory=dict)
  seed_stability: dict[str, Any] = field(default_factory=dict)
  status: str = InternalizationStatus.UNRESOLVED.value
  gate_a_pass: bool = False
  gate_b_pass: bool = False
  gate_c_pass: bool = False
  notes: list[str] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


class ModuleRetirementGate:
  """Gate A/B/C thresholds from 0810-todo2 §1.1–1.3."""

  GATE_A_MIN_TRAIN = 1000
  GATE_A_MIN_VALID = 300
  GATE_A_MIN_CLASS_VALID = 100
  GATE_A_MIN_QUERIES = 50
  GATE_A_MAX_CONFLICT_RATE = 0.01
  GATE_A_BASE_INTRINSIC_BAL = 0.90

  GATE_B_MIN_BAL = 0.75
  GATE_B_MIN_CLASS_RECALL = 0.70
  GATE_B_MAX_SEED_SPAN = 0.05
  GATE_B_REQUIRED_SEEDS = 3

  def evaluate_gate_a(self, dataset_stats: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    n_train = int(dataset_stats.get("n_train") or 0)
    n_valid = int(dataset_stats.get("n_valid") or 0)
    n_queries = int(dataset_stats.get("n_unique_queries") or dataset_stats.get("train_queries") or 0)
    conflict = float(dataset_stats.get("label_conflict_rate") or 0)
    info_violations = int(dataset_stats.get("info_safe_violations") or 0)
    base_bal = dataset_stats.get("base_balanced_accuracy")

    if n_train < self.GATE_A_MIN_TRAIN:
      reasons.append(f"n_train={n_train}<{self.GATE_A_MIN_TRAIN}")
    if n_valid < self.GATE_A_MIN_VALID:
      reasons.append(f"n_valid={n_valid}<{self.GATE_A_MIN_VALID}")
    if n_queries < self.GATE_A_MIN_QUERIES:
      reasons.append(f"n_unique_queries={n_queries}<{self.GATE_A_MIN_QUERIES}")
    if conflict > self.GATE_A_MAX_CONFLICT_RATE:
      reasons.append(f"label_conflict_rate={conflict}>{self.GATE_A_MAX_CONFLICT_RATE}")
    if info_violations > 0:
      reasons.append(f"info_safe_violations={info_violations}>0")

    per_class = dataset_stats.get("valid_class_counts") or {}
    if len(per_class) < 2:
      reasons.append(f"bilateral_fail n_classes={len(per_class)}<{2}")
    for action, count in per_class.items():
      if int(count) < self.GATE_A_MIN_CLASS_VALID:
        reasons.append(f"valid.{action}={count}<{self.GATE_A_MIN_CLASS_VALID}")
    train_class = dataset_stats.get("train_class_counts") or {}
    if train_class and len(train_class) < 2:
      reasons.append(f"train_bilateral_fail n_classes={len(train_class)}<{2}")

    if base_bal is not None and float(base_bal) >= self.GATE_A_BASE_INTRINSIC_BAL:
      reasons.append(f"ALREADY_INTRINSIC base_bal={base_bal}")

    return (len(reasons) == 0), reasons

  def evaluate_gate_b(self, seed_metrics: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if len(seed_metrics) < self.GATE_B_REQUIRED_SEEDS:
      reasons.append(f"n_seeds={len(seed_metrics)}<{self.GATE_B_REQUIRED_SEEDS}")
      return False, reasons

    bals: list[float] = []
    for i, m in enumerate(seed_metrics):
      bal = float(m.get("balanced_accuracy") or 0)
      bals.append(bal)
      if bal < self.GATE_B_MIN_BAL:
        reasons.append(f"seed{i} bal={bal:.3f}<{self.GATE_B_MIN_BAL}")
      for key, val in (m.get("per_class_recall") or m.get("class_recall") or {}).items():
        if float(val) < self.GATE_B_MIN_CLASS_RECALL:
          reasons.append(f"seed{i} {key}_recall={val}<{self.GATE_B_MIN_CLASS_RECALL}")
      parser = float(m.get("parser_success") or 1.0)
      if parser < 1.0 - 1e-9:
        reasons.append(f"seed{i} parser_success={parser}")
      parity = float((m.get("canonical_parity") or {}).get("operation_agreement") or 1.0)
      if parity < 1.0 - 1e-9:
        reasons.append(f"seed{i} parity={parity}")

    if bals:
      span = max(bals) - min(bals)
      if span > self.GATE_B_MAX_SEED_SPAN:
        reasons.append(f"seed_span={span:.3f}>{self.GATE_B_MAX_SEED_SPAN}")

    return (len(reasons) == 0), reasons

  def evaluate_gate_c(
    self,
    comparisons: dict[str, dict[str, Any]],
    *,
    hybrid: bool = False,
  ) -> tuple[bool, list[str]]:
    """comparisons keys: B_OFF, B_ON, T_OFF, optional T_ON."""
    reasons: list[str] = []
    b_off = comparisons.get("B_OFF") or {}
    t_off = comparisons.get("T_OFF") or {}
    cap_delta = float(t_off.get("capability_delta_vs_b_off") or 0)
    if cap_delta <= 0:
      reasons.append(f"T_OFF capability_delta={cap_delta}<=0")
    task_delta = float(t_off.get("task_delta_vs_b_off") or 0)
    if task_delta < -0.02:
      reasons.append(f"task_regression task_delta={task_delta}")
    side = float(t_off.get("side_effect_delta") or 0)
    if side > 0.05:
      reasons.append(f"side_effect_delta={side}>0.05")
    seeds_ok = comparisons.get("seed_consistency", True)
    if not seeds_ok:
      reasons.append("seed direction inconsistent")
    if hybrid and cap_delta > 0:
      return True, ["HYBRID: routing internalized, execution remains runtime"]
    return (len(reasons) == 0), reasons

  def classify_status(
    self,
    *,
    gate_a: bool,
    gate_b: bool,
    gate_c: bool,
    hybrid: bool = False,
    runtime_execution: bool = False,
    unresolved: bool = False,
    already_intrinsic: bool = False,
    objective_repair_attempted: int = 0,
  ) -> InternalizationStatus:
    if already_intrinsic:
      return InternalizationStatus.ALREADY_INTRINSIC
    if runtime_execution:
      return InternalizationStatus.RUNTIME_EXECUTION_REQUIRED
    if unresolved or not gate_a:
      return InternalizationStatus.UNRESOLVED
    if gate_a and gate_b and gate_c:
      return (
        InternalizationStatus.HYBRID_INTERNALIZED
        if hybrid
        else InternalizationStatus.PROVEN_INTERNALIZED
      )
    if gate_a and gate_b and not gate_c and objective_repair_attempted >= 2:
      return InternalizationStatus.CURRENTLY_NOT_INTERNALIZED
    if gate_a and not gate_b:
      return InternalizationStatus.UNRESOLVED
    return InternalizationStatus.UNRESOLVED

  def build_evidence(
    self,
    capability_id: str,
    *,
    dataset_stats: dict[str, Any] | None = None,
    local_metrics: list[dict[str, Any]] | None = None,
    retirement: dict[str, Any] | None = None,
    hybrid: bool = False,
  ) -> CapabilityInternalizationEvidence:
    ds = dataset_stats or {}
    gate_a, a_reasons = self.evaluate_gate_a(ds) if ds else (False, ["no dataset stats"])
    gate_b, b_reasons = (
      self.evaluate_gate_b(local_metrics or []) if local_metrics else (False, ["no local metrics"])
    )
    gate_c, c_reasons = (
      self.evaluate_gate_c(retirement or {}) if retirement else (False, ["no retirement eval"])
    )

    already_intrinsic = any("ALREADY_INTRINSIC" in r for r in a_reasons)
    status = self.classify_status(
      gate_a=gate_a,
      gate_b=gate_b,
      gate_c=gate_c,
      hybrid=hybrid,
      already_intrinsic=already_intrinsic,
      unresolved=not gate_a,
    )

    ev = CapabilityInternalizationEvidence(
      capability_id=capability_id,
      event_support={
        "n_train": ds.get("n_train"),
        "n_valid": ds.get("n_valid"),
        "n_unique_queries": ds.get("n_unique_queries"),
        "collection_modes": ds.get("collection_modes"),
      },
      bilateral_support=ds.get("bilateral_support") or {},
      information_safe={
        "violations": ds.get("info_safe_violations", 0),
        "label_conflict_rate": ds.get("label_conflict_rate"),
      },
      state_observability=ds.get("state_observability") or {},
      base_gap={"base_balanced_accuracy": ds.get("base_balanced_accuracy")},
      local_learnability={
        "per_seed": local_metrics or [],
        "gate_b_pass": gate_b,
        "fail_reasons": b_reasons,
      },
      fresh_live_transfer=retirement.get("fresh_transfer") if retirement else {},
      retirement_behavior_gain=retirement.get("comparisons") if retirement else {},
      task_retention=retirement.get("task_retention") if retirement else {},
      seed_stability=retirement.get("seed_stability") if retirement else {},
      status=status.value,
      gate_a_pass=gate_a,
      gate_b_pass=gate_b,
      gate_c_pass=gate_c,
      notes=a_reasons + b_reasons + c_reasons,
    )
    return ev

  def write_gate_json(self, evidence: CapabilityInternalizationEvidence, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "DATASET_GATE.json"
    payload = {
      "schema_version": "scope.round14.dataset_gate.v1",
      "capability_id": evidence.capability_id,
      "gate_a_pass": evidence.gate_a_pass,
      "gate_b_pass": evidence.gate_b_pass,
      "gate_c_pass": evidence.gate_c_pass,
      "status": evidence.status,
      "evidence": evidence.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
