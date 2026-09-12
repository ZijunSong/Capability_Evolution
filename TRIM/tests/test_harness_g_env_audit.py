"""Regression tests from Harness-G zero/all audit (2026-09-12)."""

from __future__ import annotations

from trim.adapters.harness_profiles import zero_mask_for
from trim.eval.harness_g_env import (
    _hybrid_rank_sids,
    build_action_map,
    execute_tool,
    new_state,
    wm_text,
)
from trim.eval.harness_g_runtime import parse_harness_g_action
from trim.training.parse_rollout_action import parse_generated_action


def _state_with_entity():
    store = {
        "d1": {"id": "d1", "text": "Alice Smith visited Paris in 2019."},
        "d2": {"id": "d2", "text": "Bob Jones wrote about London."},
    }
    st = new_state("Who is Alice Smith?", store, harness_mask=zero_mask_for("Harness-G"))
    return st


def test_wm_shows_copyable_eid_not_surface_only():
    st = _state_with_entity()
    st["initialized"] = True
    st["visible_sids"] = list(st["sentences"])[:1]
    st["frontier_eids"] = ["e:alice_smith"]
    st["action_map"] = build_action_map(st, include_answer=True)
    wm = wm_text(st)
    assert 'eid="e:alice_smith"' in wm
    assert 'surface="Alice Smith"' in wm


def test_lookup_surface_name_rejected():
    st = _state_with_entity()
    st2, obs, ok = execute_tool(st, "lookup", {"eid": "Alice Smith"})
    assert ok is False
    assert "eid_not_found" in obs


def test_lookup_valid_eid_works():
    st = _state_with_entity()
    st2, obs, ok = execute_tool(st, "lookup", {"eid": "e:alice_smith"}, searcher=None)
    assert ok is True
    assert "e:alice_smith" in obs


def test_lookup_empty_args_rejected():
    st = _state_with_entity()
    st2, obs, ok = execute_tool(st, "lookup", {})
    assert ok is False
    assert "missing_eid" in obs


def test_lookup_dedup_blocks_repeat():
    mask = zero_mask_for("Harness-G")
    mask["lookup_dedup"] = True
    st = _state_with_entity()
    st["harness_mask"] = mask
    st, _, ok1 = execute_tool(st, "lookup", {"eid": "e:alice_smith"}, searcher=None)
    assert ok1 is True
    st, obs, ok2 = execute_tool(st, "lookup", {"eid": "e:alice_smith"}, searcher=None)
    assert ok2 is False
    assert "eid_already_visited" in obs


def test_answer_with_invalid_sid_fails():
    mask = zero_mask_for("Harness-G")
    mask["answer_with"] = True
    st = _state_with_entity()
    st["harness_mask"] = mask
    st["initialized"] = True
    st2, obs, ok = execute_tool(st, "answer_with", {"sid": "missing:s0"})
    assert ok is False
    assert not st2.get("ended")
    assert "sid_not_found" in obs


def test_hybrid_differs_from_lexical_only():
    sentences = {
        "d1:s0": {"sid": "d1:s0", "doc_id": "d1", "text": "common words here"},
        "d2:s0": {"sid": "d2:s0", "doc_id": "d2", "text": "Alice Smith entity hit"},
    }
    entities = {
        "e:alice_smith": {"eid": "e:alice_smith", "surface": "Alice Smith", "sids": ["d2:s0"], "synonyms": []},
    }
    query = "Alice Smith biography"
    lexical = _hybrid_rank_sids(query, sentences, {}, 2)
    hybrid = _hybrid_rank_sids(query, sentences, entities, 2, doc_order=["d1", "d2"])
    assert hybrid != lexical or hybrid[0] == "d2:s0"


def test_parse_rejects_invalid_json_select():
    action, ok = parse_harness_g_action('to=select {"sid":')
    assert ok is False


def test_parse_rejects_analysis_only_tool_mention():
    mask = zero_mask_for("Harness-G")
    text = "I should to=answer {\"reason\": \"done\"} in analysis only."
    action, ok = parse_generated_action(text, None, enc=None, harness_mask=mask)
    assert ok is False


def test_parse_a0_menu_id_with_action_map():
    st = _state_with_entity()
    st["initialized"] = True
    st["visible_sids"] = ["d1:s0"]
    st["action_map"] = build_action_map(st, include_answer=True)
    action, ok = parse_harness_g_action("A0", action_map=st["action_map"])
    assert ok is True
    assert action["name"] == "select"
    assert action["arguments"]["sid"] == "d1:s0"
