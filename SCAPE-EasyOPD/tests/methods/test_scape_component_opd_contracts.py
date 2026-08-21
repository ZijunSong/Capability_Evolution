from __future__ import annotations

import pytest

from easyopd import EasyOPD
from easyopd.methods.scape_component_opd.action_projection import project_curated_delta
from easyopd.methods.scape_component_opd.component_registry import audit_component
from easyopd.methods.scape_component_opd.controls import assert_query_disjoint, query_disjoint_splits, shuffled_targets_preserve_marginal
from easyopd.methods.scape_component_opd.scape_agent_loop import SCAPEAgentLoop
from easyopd.methods.scape_component_opd.state_snapshot import SCAPEStateSnapshot, assert_same_state_before_component_fork
from easyopd.methods.scape_component_opd.tool_span import audit_tool_call_span, require_parsable_tool_calls


def test_easyopd_registry_discovers_scape_component_opd():
    assert "scape_component_opd" in EasyOPD.list_methods()
    inst = EasyOPD.from_hparams("scape_component_opd", auto_resolve_data=False)
    assert inst.method_name == "scape_component_opd"


def test_verify_tool_non_realizable_refuses_by_default():
    audit = audit_component("verify_tool")
    assert audit["can_train"] is False
    assert audit["decision_code"] == "NON_REALIZABLE_ACTION_SPACE_MISMATCH"


def test_content_dedup_zero_event_support_blocks_training():
    audit = audit_component("content_dedup", event_support=0)
    assert audit["can_train"] is False
    assert audit["decision_code"] == "STOP_NO_ACTIVE_EVENT_SUPPORT"


def test_auto_projection_uses_visible_curated_delta():
    action, audit = project_curated_delta(curated_ids_pre=["d0"], curated_ids_post=["d0", "d1"], visible_doc_ids=["d0", "d1"])
    assert action is not None
    assert action.name == "curate"
    assert action.arguments == {"add_ids": ["d1"], "remove_ids": []}
    assert audit["projection_valid"] is True
    bad_action, bad_audit = project_curated_delta(curated_ids_pre=["d0"], curated_ids_post=["d0", "hidden"], visible_doc_ids=["d0"])
    assert bad_action is None
    assert bad_audit["projection_valid"] is False


def test_tool_span_parser_requires_legal_tool_and_json_args():
    audit = audit_tool_call_span('to=curate\n{"add_ids":["d1"],"remove_ids":[]}\n</tool_call>')
    assert audit.parsable is True
    assert audit.tool_name == "curate"
    with pytest.raises(AssertionError):
        require_parsable_tool_calls(["to=not_a_tool\n{}"])


def test_state_fork_same_hash_before_component_effect():
    snap = SCAPEStateSnapshot(query_id="q", turn_id=1, curated_ids=["d0"], documents=[{"id": "d0"}], component_masks={"evidence_graph": False})
    hashes = assert_same_state_before_component_fork(snap)
    assert hashes["state_hash_student"] == hashes["state_hash_teacher"]


def test_scape_agent_loop_no_privilege_and_verify_gating():
    loop = SCAPEAgentLoop("evidence_graph", student_inference_privilege=False)
    assert "verify" not in loop.available_tools()
    assert "verify" in loop.available_tools(include_verify=True)
    view = loop.build_student_view({"evidence_graph": {"secret": 1}, "curated_importance": {"d": "high"}})
    assert "evidence_graph" not in view
    assert "curated_importance" not in view
    assert view["student_inference_privilege"] is False
    with pytest.raises(ValueError):
        SCAPEAgentLoop("evidence_graph", student_inference_privilege=True)


def test_query_disjoint_split_and_shuffle_marginal():
    splits = query_disjoint_splits([f"q{i}" for i in range(50)], seed=7)
    assert_query_disjoint(splits)
    rows = [{"query_id": f"q{i}", "projected_action": {"name": "curate", "arguments": {"add_ids": [str(i)], "remove_ids": []}}} for i in range(5)]
    shuffled = shuffled_targets_preserve_marginal(rows)
    assert [r["projected_action"] for r in shuffled] != [r["projected_action"] for r in rows]
    assert sorted(str(r["projected_action"]) for r in shuffled) == sorted(str(r["projected_action"]) for r in rows)
