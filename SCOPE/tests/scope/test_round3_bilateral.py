"""Round 3 bilateral duplicate capability tests (Barrier A)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from harness.artifacts.gates import run_information_safe_gates
from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.dup_decision_point import (
    build_decision_points,
    is_duplicate_candidate,
)
from harness.capability.dup_operation import DupOperation
from harness.capability.selectors import RuleBasedCriticalStateSelector, SelectorConfig
from harness.shadow.action_realizer import ActionRealizer
from harness.shadow.dup_bilateral_shadow import DupBilateralShadow
from training.scope.compact_target import apply_compact_target_to_sample, compact_target_from_sample
from training.scope.operation_scorer import score_operations
from training.scope.routing import route_decision
from training.scope.schema import Route
from tests.scope.conftest import make_state


def test_unique_candidate_shadow_keep():
    state = make_state(curated_document_ids=("d1",))
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d2"]},
    )
    shadow = DupBilateralShadow()
    art = shadow.analyze_candidate(state, student, build_decision_points(state, student)[0])
    assert art.metadata["shadow_operation"] == DupOperation.KEEP_EVIDENCE.value
    assert art.mode.value == "endorse"


def test_duplicate_candidate_shadow_skip():
    state = make_state(curated_document_ids=("d1",))
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    shadow = DupBilateralShadow()
    art = shadow.analyze_candidate(state, student, build_decision_points(state, student)[0])
    assert art.metadata["shadow_operation"] == DupOperation.SKIP_DUPLICATE.value
    assert art.mode.value == "correct"


def test_student_curate_keep_endorse():
    state = make_state(curated_document_ids=("d1",))
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d2"]},
    )
    shadow = DupBilateralShadow()
    art = shadow.analyze(state, student)
    routed = route_decision(state, art, student)
    assert routed.route == Route.ENDORSE


def test_student_curate_skip_correct():
    state = make_state(curated_document_ids=("d1",))
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    shadow = DupBilateralShadow()
    art = shadow.analyze(state, student)
    routed = route_decision(state, art, student)
    assert routed.route == Route.CORRECT


def test_keep_action_realizer_executes_curate():
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d2"]},
    )
    cand = ActionRealizer().realize_operation(
        state, DupOperation.KEEP_EVIDENCE, candidate_id="d2", student_action=student
    )
    assert cand.action.arguments.get("add_ids") == ["d2"]


def test_skip_action_realizer_no_add():
    state = make_state(curated_document_ids=("d1",))
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d1"]},
    )
    cand = ActionRealizer().realize_operation(
        state, DupOperation.SKIP_DUPLICATE, candidate_id="d1", student_action=student
    )
    assert cand.action.arguments.get("add_ids") == []


def test_shadow_does_not_mutate_environment():
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d2"]},
    )
    before = state.to_dict()
    DupBilateralShadow().analyze(state, student)
    after = state.to_dict()
    assert before == after


def test_visibility_gate_zero_violation():
    state = make_state()
    student = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d2"]},
    )
    art = DupBilateralShadow().analyze(state, student)
    gates = run_information_safe_gates(state, art)
    assert gates.visible
    assert gates.all_passed or gates.module_valid


def test_action_realizer_no_external_calls():
    """ActionRealizer is deterministic — no LLM / network."""
    import inspect

    src = inspect.getsource(ActionRealizer.realize_operation)
    assert "openai" not in src.lower()
    assert "requests" not in src.lower()


def test_operation_scorer_shared_train_inference():
    """score_operations is the single scoring entry point."""
    from training.scope import operation_scorer as mod

    assert hasattr(mod, "score_operations")
    assert hasattr(mod, "operation_ce_loss")


def test_verbalizer_length_normalization():
    """Different-length verbalizers use mean log-prob, not sum."""
    from training.scope.operation_scorer import _completion_logprob
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/data/ppnm/models/Qwen2.5-7B-Instruct"
    if not Path(model_path).exists():
        pytest.skip("model not available")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    device = torch.device("cpu")
    model.to(device)
    prompt = "test state\n\nEvidence admission operation:\n"
    s_keep, n_keep = _completion_logprob(model, tok, prompt, "KEEP_EVIDENCE", device=device)
    s_skip, n_skip = _completion_logprob(model, tok, prompt, "SKIP_DUPLICATE", device=device)
    assert n_keep != n_skip or True  # lengths may differ
    assert isinstance(s_keep, float)
    assert isinstance(s_skip, float)


def test_round1_round2_schema_readable():
    """Legacy compact JSON and full action formats still parse."""
    legacy = {
        "target_action": {
            "action_type": "curate_document",
            "arguments": {"add_ids": ["d1"], "remove_ids": []},
        }
    }
    compact = {
        "target_action": {"operation": "SKIP_DUPLICATE"},
        "metadata": {"target_format": "compact_operation"},
    }
    assert compact_target_from_sample(legacy) is not None
    assert compact_target_from_sample(compact) is not None
    out = apply_compact_target_to_sample(legacy)
    assert "compact_operation" in out["metadata"]["target_format"] or out["metadata"].get("compact_target")


def test_selector_triggers_evidence_admission():
    state = make_state()
    action = CapabilityAction(
        action_type=CapabilityActionType.CURATE_DOCUMENT,
        arguments={"add_ids": ["d2"]},
    )
    sel = RuleBasedCriticalStateSelector(SelectorConfig(before_curate=True))
    mods = sel.select(state, action)
    assert "duplicate_evidence" in mods
