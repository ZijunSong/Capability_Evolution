"""Tests for compact Dup target and ActionRealizer operation path."""

from __future__ import annotations

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.dup_operation import DupOperation
from harness.shadow.action_realizer import ActionRealizer
from training.scope.compact_target import (
    apply_compact_target_to_sample,
    compact_target_from_sample,
    infer_operation_from_action,
)
from tests.scope.conftest import make_state


def test_skip_curate_maps_to_skip_duplicate():
    action = {
        "action_type": "curate_document",
        "arguments": {"add_ids": [], "remove_ids": []},
    }
    assert infer_operation_from_action(action) == DupOperation.SKIP_DUPLICATE


def test_curate_add_maps_to_keep():
    action = {
        "action_type": "curate_document",
        "arguments": {"add_ids": ["d1"], "remove_ids": []},
    }
    assert infer_operation_from_action(action) == DupOperation.KEEP_EVIDENCE


def test_action_realizer_skip_duplicate_no_hidden_teacher():
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    realizer = ActionRealizer()
    cand = realizer.realize_operation(
        state, DupOperation.SKIP_DUPLICATE, candidate_id="d1", student_action=student
    )
    assert cand.source == "operation_realized"
    assert cand.action.arguments.get("add_ids") == []


def test_compact_target_roundtrip():
    sample = {
        "target_action": {
            "action_type": "curate_document",
            "arguments": {"add_ids": [], "remove_ids": []},
        },
        "artifact": {"recommended_operation": "skip_curate"},
    }
    compact = compact_target_from_sample(sample)
    assert compact is not None
    assert compact.operation == DupOperation.SKIP_DUPLICATE
    out = apply_compact_target_to_sample(sample)
    assert out["metadata"]["target_format"] == "compact_operation"
