"""Adapter candidate_actions and label remap tests."""

from __future__ import annotations

import pytest

from training.scope_round14.adapters.c0_duplicate_evidence import DuplicateEvidenceAdapter
from training.scope_round14.adapters.c6_rollback_lite import RollbackLiteAdapter
from training.scope_round14.adapters.registry import get_adapter, list_adapters


def test_list_adapters_has_seven():
  caps = list_adapters()
  assert len(caps) == 7
  assert "duplicate_evidence" in caps
  assert "rollback_lite" in caps


@pytest.mark.parametrize(
  "capability,expected_actions",
  [
    ("duplicate_evidence", ["KEEP_EVIDENCE", "SKIP_DUPLICATE"]),
    ("stop_decision", ["STOP", "CONTINUE"]),
    ("verification_routing", ["VERIFY", "NO_VERIFY"]),
    ("evidence_admission", ["ADMIT", "DROP"]),
    ("context_budget_routing", ["KEEP_CONTEXT", "COMPRESS_OR_DROP"]),
    ("external_verification_routing", ["VERIFY_EXTERNALLY", "DO_NOT"]),
    ("rollback_lite", ["RECOVER", "CONTINUE"]),
  ],
)
def test_candidate_actions(capability, expected_actions):
  adapter = get_adapter(capability)
  assert adapter.candidate_actions() == expected_actions


def test_rollback_lite_remap():
  adapter = RollbackLiteAdapter()
  assert adapter.remap_operation("REPLAN") == "RECOVER"
  assert adapter.remap_operation("ROLLBACK_TO") == "RECOVER"
  assert adapter.remap_operation("CONTINUE") == "CONTINUE"
  row = {
    "decision_state": {"checkpoint_registry": [1, 2], "turn_index": 3},
    "gold_operation": "ROLLBACK_TO",
    "gold_checkpoint_id": "ckpt_5",
  }
  out = adapter.normalize_row(row)
  assert out["gold_action"] == "RECOVER"
  assert "checkpoint_registry" not in out["decision_state"]


def test_dup_label_map():
  adapter = DuplicateEvidenceAdapter()
  assert adapter.map_training_label("ADMIT") == "KEEP_EVIDENCE"
  assert adapter.map_training_label("SKIP") == "SKIP_DUPLICATE"
