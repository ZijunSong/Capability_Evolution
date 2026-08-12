"""Gate A/B/C logic with synthetic metrics."""

from __future__ import annotations

from training.scope_round14.gates import InternalizationStatus, ModuleRetirementGate


def test_gate_a_pass_synthetic():
  gate = ModuleRetirementGate()
  stats = {
    "n_train": 1200,
    "n_valid": 350,
    "n_unique_queries": 60,
    "label_conflict_rate": 0.0,
    "info_safe_violations": 0,
    "valid_class_counts": {"KEEP": 120, "SKIP": 110},
  }
  ok, reasons = gate.evaluate_gate_a(stats)
  assert ok is True
  assert reasons == []


def test_gate_a_fail_low_train():
  gate = ModuleRetirementGate()
  stats = {"n_train": 50, "n_valid": 350, "n_unique_queries": 60, "valid_class_counts": {}}
  ok, reasons = gate.evaluate_gate_a(stats)
  assert ok is False
  assert any("n_train" in r for r in reasons)


def test_gate_b_three_seed_pass():
  gate = ModuleRetirementGate()
  seeds = [
    {
      "balanced_accuracy": 0.80,
      "per_class_recall": {"A": 0.75, "B": 0.72},
      "parser_success": 1.0,
      "canonical_parity": {"operation_agreement": 1.0},
    }
  ] * 3
  ok, reasons = gate.evaluate_gate_b(seeds)
  assert ok is True


def test_gate_b_fail_span():
  gate = ModuleRetirementGate()
  seeds = [
    {"balanced_accuracy": 0.80, "per_class_recall": {"A": 0.75, "B": 0.72}},
    {"balanced_accuracy": 0.70, "per_class_recall": {"A": 0.75, "B": 0.72}},
    {"balanced_accuracy": 0.78, "per_class_recall": {"A": 0.75, "B": 0.72}},
  ]
  ok, reasons = gate.evaluate_gate_b(seeds)
  assert ok is False
  assert any("bal=" in r or "seed_span" in r for r in reasons)


def test_gate_c_positive_delta():
  gate = ModuleRetirementGate()
  ok, _ = gate.evaluate_gate_c(
    {
      "B_OFF": {},
      "T_OFF": {"capability_delta_vs_b_off": 0.05, "task_delta_vs_b_off": 0.0, "side_effect_delta": 0.0},
      "seed_consistency": True,
    }
  )
  assert ok is True


def test_classify_proven():
  gate = ModuleRetirementGate()
  status = gate.classify_status(gate_a=True, gate_b=True, gate_c=True)
  assert status == InternalizationStatus.PROVEN_INTERNALIZED


def test_classify_hybrid():
  gate = ModuleRetirementGate()
  status = gate.classify_status(gate_a=True, gate_b=True, gate_c=True, hybrid=True)
  assert status == InternalizationStatus.HYBRID_INTERNALIZED
