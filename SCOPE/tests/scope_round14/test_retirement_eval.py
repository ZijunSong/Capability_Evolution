"""Round14 retirement eval + aggregate tests."""

from __future__ import annotations

import json
from pathlib import Path

from training.scope_round14.aggregate_dup_anchor import build_comparison, load_condition
from training.scope_round14.gates import ModuleRetirementGate
from training.scope_round14.run_module_retirement_eval import aggregate_rollout_dir


def test_aggregate_rollout_dir_from_summary(tmp_path: Path):
  out = tmp_path / "rollout"
  out.mkdir()
  summary = {
    "n_queries": 10,
    "dup_telemetry": {
      "balanced_accuracy": 0.82,
      "duplicate_reject_rate": 0.75,
      "false_skip_rate": 0.05,
      "KEEP_EVIDENCE": {"recall": 0.80},
      "SKIP_DUPLICATE": {"recall": 0.84},
    },
  }
  (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
  (out / "episodes.jsonl").write_text(
    json.dumps({"recall": 0.5, "turns": 12}) + "\n"
    + json.dumps({"recall": 0.7, "turns": 8}) + "\n",
    encoding="utf-8",
  )
  agg = aggregate_rollout_dir(out)
  assert agg["balanced_accuracy"] == 0.82
  assert agg["duplicate_reject_rate"] == 0.75
  assert agg["recall"] == 0.6
  assert agg["mean_turns"] == 10.0
  assert agg["keep_recall"] == 0.80


def test_build_comparison_positive_delta():
  b_off = {
    "dup_telemetry": {"balanced_accuracy": 0.70, "duplicate_reject_rate": 0.5},
    "recall": 0.4,
    "mean_turns": 15,
  }
  b_on = {"dup_telemetry": {"balanced_accuracy": 0.72}}
  t42 = {
    "dup_telemetry": {"balanced_accuracy": 0.80, "duplicate_reject_rate": 0.7},
    "recall": 0.42,
    "mean_turns": 14,
  }
  t43 = {
    "dup_telemetry": {"balanced_accuracy": 0.78, "duplicate_reject_rate": 0.68},
    "recall": 0.41,
    "mean_turns": 14,
  }
  cmp = build_comparison(b_off, b_on, {42: t42, 43: t43})
  gate = ModuleRetirementGate()
  ok, _ = gate.evaluate_gate_c(
    {
      "B_OFF": cmp["comparisons"]["B_OFF"],
      "T_OFF": cmp["comparisons"]["T_OFF"],
      "seed_consistency": cmp["seed_consistency"],
    }
  )
  assert cmp["comparisons"]["T_OFF"]["capability_delta_vs_b_off"] > 0
  assert ok is True


def test_load_condition_reads_done_dir(tmp_path: Path):
  cond = tmp_path / "B_OFF"
  cond.mkdir()
  (cond / "summary.json").write_text(
    json.dumps({"dup_telemetry": {"balanced_accuracy": 0.5}}), encoding="utf-8"
  )
  (cond / "DONE").write_text("ok\n", encoding="utf-8")
  loaded = load_condition(tmp_path, "B_OFF")
  assert loaded is not None
  assert loaded["balanced_accuracy"] == 0.5
