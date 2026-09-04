from __future__ import annotations

from trim.adapters.components import minus_mask
from trim.state.snapshot import capture_snapshot
from trim.training.action_codec import canonicalize_action, render_action, validate_roundtrip
from trim.training.opd_realizability import (
    check_action_realizability,
    decision_state_signature,
)
from trim.training.tool_mask import legal_tool_names, validate_action_arguments


def _snap(component_id: str, **wm):
    return capture_snapshot(
        query_id="q",
        step=0,
        harness_mask=minus_mask(component_id),
        working_memory={
            "documents": [{"id": "d1", "text": "full"}, {"id": "d2", "text": "full"}],
            "curated_ids": ["d1"],
            "accessible_doc_ids": ["d1", "d2"],
            **wm,
        },
    )


def test_verify_absent_when_verify_tool_false():
    mask = minus_mask("verify_tool")
    assert mask["verify_tool"] is False
    tools = legal_tool_names(harness_mask=mask)
    assert "verify" not in tools
    assert "importance_tagging" not in tools


def test_curate_cannot_carry_importance_when_mask_off():
    ok, reason = validate_action_arguments(
        "curate",
        {"add_ids": ["d1"], "remove_ids": [], "importance": {"d1": "high"}},
        harness_mask=minus_mask("importance_tagging"),
    )
    assert ok is False
    assert reason == "INVALID_ARGUMENT_SCHEMA"


def test_decision_state_ignores_raw_pool_supersets():
    snap = _snap(
        "content_dedup",
        documents=[
            {"id": "d1"},
            {"id": "d2"},
            {"id": "d3_dup"},
            {"id": "d4"},
            {"id": "d5_dup"},
        ],
        accessible_doc_ids=["d1", "d2", "d3_dup", "d4", "d5_dup"],
        curated_ids=["d1", "d2", "d4"],
    )
    sig = decision_state_signature(snap)
    assert "d1" in sig.accessible_doc_ids
    assert "d4" in sig.accessible_doc_ids
    report = check_action_realizability(
        action={"name": "curate", "arguments": {"add_ids": ["d1"], "remove_ids": []}},
        student_snapshot=snap,
        student_mask=snap.harness_mask,
        component_id="content_dedup",
    )
    assert report.passed is True


def test_inaccessible_curate_fails_r3():
    snap = _snap("auto_populate_first_search", accessible_doc_ids=["d1"], curated_ids=["d1"])
    report = check_action_realizability(
        action={"name": "curate", "arguments": {"add_ids": ["d17"], "remove_ids": []}},
        student_snapshot=snap,
        student_mask=snap.harness_mask,
        component_id="auto_populate_first_search",
    )
    assert report.passed is False
    assert "DOC_NOT_ACCESSIBLE" in report.reason_codes


def test_action_codec_roundtrip():
    action = canonicalize_action({"name": "curate", "arguments": {"add_ids": ["d2", "d1"], "remove_ids": []}})
    assert action["arguments"]["add_ids"] == ["d1", "d2"]
    assert validate_roundtrip(action)
    assert validate_roundtrip({"name": "search_corpus", "arguments": {"query": "q"}})
    text = render_action(action)
    assert text.startswith("to=curate")
