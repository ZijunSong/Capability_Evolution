"""rollback_lite remapping integration test."""

from __future__ import annotations

import json
from pathlib import Path

from training.scope_round14.adapters.c6_rollback_lite import RollbackLiteAdapter


def test_remap_strips_checkpoint_fields():
  adapter = RollbackLiteAdapter()
  row = {
    "event_id": "e1",
    "query_id": "q1",
    "decision_state": {
      "turn_index": 2,
      "checkpoint_registry": ["a", "b"],
      "candidate_checkpoint_ids": ["a"],
    },
    "gold_operation": "REPLAN",
    "gold_checkpoint_id": "a",
  }
  norm = adapter.normalize_row(row)
  assert norm["gold_action"] == "RECOVER"
  ds = norm["decision_state"]
  assert "checkpoint_registry" not in ds
  assert "candidate_checkpoint_ids" not in ds


def test_remap_file_if_exists():
  path = Path("/data/ppnm/Capability_Evolution/SCOPE/artifacts/datasets/scope_round14/rollback_lite/train.jsonl")
  if not path.exists():
    return
  line = path.read_text(encoding="utf-8").splitlines()[0]
  row = json.loads(line)
  assert row["gold_action"] in {"RECOVER", "CONTINUE"}
  assert row["capability_id"] == "rollback_lite"
