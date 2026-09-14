"""Regression tests for Harness-G auditfix2 review (2026-09-13)."""

from __future__ import annotations

from trim.adapters.harness_profiles import full_mask_for, zero_mask_for
from trim.eval.harness_g_env import (
    _lookup_sids,
    _text_to_sentence_parts,
    allowed_menu_targets,
    execute_tool,
    new_state,
)
from trim.eval.harness_g_runtime import is_protocol_feedback, make_protocol_feedback
from trim.eval.model_tokenizer import parse_qwen_tool_call
from trim.training.parse_rollout_action import parse_generated_action


def test_length_rejects_menu_fallback_before_a0():
    mask = zero_mask_for("Harness-G")
    action, ok = parse_generated_action(
        "A0",
        None,
        None,
        harness_mask=mask,
        action_map={"A0": {"name": "answer"}},
        finish_reason="length",
    )
    assert ok is False
    assert action["name"] == "truncated"


def test_cross_message_harmony_lookup_not_spliced_with_analysis_answer():
    mask = zero_mask_for("Harness-G")
    raw = (
        '<|channel|>analysis<|message|>Do not use to=answer.'
        '<|end|><|start|>assistant to=functions.lookup'
        '<|channel|>commentary<|message|>{"eid":"e:alice"}<|call|>'
    )
    action, ok = parse_generated_action(raw, None, None, harness_mask=mask, finish_reason="stop")
    assert ok is True
    assert action["name"] == "lookup"
    assert action["arguments"]["eid"] == "e:alice"


def test_lookup_shows_fresh_doc_when_old_doc_fills_window():
    mask = full_mask_for("Harness-G")
    st = new_state(
        "Alice Smith evidence",
        {
            "old": {"text": " ".join(f"Old irrelevant passage number {i}." for i in range(8))},
            "fresh": {"text": "Alice Smith found the decisive evidence."},
        },
        harness_mask=mask,
    )
    visible = _lookup_sids(st, "e:alice_smith", new_doc_ids=["old", "fresh"])
    assert any(sid.startswith("fresh:") for sid in visible)


def test_lookup_rejects_eid_not_in_menu():
    mask = full_mask_for("Harness-G")
    st = new_state(
        "Who?",
        {"d": {"text": "Alice visited Paris in May. Bob stayed in Madrid."}},
        harness_mask=mask,
    )
    st, _, _ = execute_tool(st, "init", {})
    st, _, _ = execute_tool(st, "select", {"sid": "d:s0"})
    may_in_menu = any(x.get("eid") == "e:may" for x in st["action_map"].values())
    assert may_in_menu is False
    _, obs, ok = execute_tool(st, "lookup", {"eid": "e:may"})
    assert ok is False
    assert "target_not_in_menu" in obs


def test_lookup_without_init_does_not_mutate_state():
    mask = zero_mask_for("Harness-G")
    st = new_state("Who?", {"d": {"text": "Alice visited Paris."}}, harness_mask=mask)
    before = dict(st)
    _, obs, ok = execute_tool(st, "lookup", {"eid": "e:alice"})
    assert ok is False
    assert "not_initialized" in obs
    assert st.get("initialized") == before.get("initialized")
    assert st.get("visible_sids") == before.get("visible_sids")


def test_paris_sentence_not_dropped_by_metadata_filter():
    parts = _text_to_sentence_parts(
        "Author: Ada Lovelace\nThe answer is Paris. Additional neutral material follows."
    )
    joined = " ".join(parts)
    assert "Paris" in joined


def test_qwen_rejects_multiple_tool_calls():
    text = (
        '<tool_call>{"name":"init","arguments":{}}</tool_call>'
        '<tool_call>{"name":"select","arguments":{"sid":"d:s0"}}</tool_call>'
    )
    parsed = parse_qwen_tool_call(text)
    assert parsed.parsed is False
    assert parsed.error == "multiple_qwen_tool_calls"


def test_qwen_rejects_bad_json_arguments():
    text = '<tool_call>{"name":"init","arguments":"not-json"}</tool_call>'
    parsed = parse_qwen_tool_call(text)
    assert parsed.parsed is False
    assert parsed.error == "json_missing_or_invalid"


def test_protocol_feedback_is_not_a_tool_call():
    fb = make_protocol_feedback("ERROR [parse_failed]: bad output")
    assert is_protocol_feedback(fb) is True
    assert "name" not in fb


def test_allowed_menu_targets_match_action_map():
    mask = zero_mask_for("Harness-G")
    st = new_state("query", {"d": {"text": "Alice Smith wrote the report."}}, harness_mask=mask)
    st, _, _ = execute_tool(st, "init", {})
    menu_sids, menu_eids = allowed_menu_targets(st)
    for action in st["action_map"].values():
        if action.get("sid"):
            assert str(action["sid"]) in menu_sids
        if action.get("eid"):
            assert str(action["eid"]) in menu_eids
