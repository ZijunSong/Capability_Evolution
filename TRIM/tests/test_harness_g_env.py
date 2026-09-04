"""TRIM-native Harness-G env: thin runtime vs mask-gated advanced menu."""

from __future__ import annotations

from trim.adapters.components import full_mask, zero_mask
from trim.eval.harness_g_env import build_action_map, execute_tool, new_state
from trim.eval.harness_g_runtime import parse_harness_g_action

STORE = {
    "d1": {
        "id": "d1",
        "text": "Alice Smith visited Paris. Bob Jones lived nearby. The treaty was signed later.",
    },
    "d2": {
        "id": "d2",
        "text": "Carol Adams founded the company. Alice Smith later joined.",
    },
}


def test_zero_mask_hides_answer_with():
    st = new_state("Who is Alice Smith?", STORE, harness_mask=zero_mask("Harness-G"))
    st, obs, ok = execute_tool(st, "init", {})
    assert ok is True
    assert st["initialized"] is True
    assert st["visible_sids"]
    types = {a.get("type") for a in (st.get("action_map") or {}).values()}
    assert "SELECT" in types
    assert "ANSWER_WITH" not in types
    st2, _obs, ok2 = execute_tool(st, "answer_with", {"sid": st["visible_sids"][0]})
    assert ok2 is False
    assert st2["invalid_tools"] >= 1


def test_full_mask_exposes_answer_with():
    st = new_state("Who is Alice Smith?", STORE, harness_mask=full_mask("Harness-G"))
    st, _obs, ok = execute_tool(st, "init", {})
    assert ok is True
    types = {a.get("type") for a in (st.get("action_map") or {}).values()}
    assert "ANSWER_WITH" in types
    assert "SELECT" in types


def test_select_lookup_answer_transitions():
    st = new_state("Alice Smith Paris", STORE, harness_mask=zero_mask("Harness-G"))
    st, _obs, ok = execute_tool(st, "init", {})
    assert ok and st["visible_sids"]
    sid = st["visible_sids"][0]
    st, obs, ok = execute_tool(st, "select", {"sid": sid})
    assert ok is True
    assert sid in st["selected_sids"]
    assert "SELECT" in obs
    if st.get("frontier_eids"):
        eid = st["frontier_eids"][0]
        st, _obs, ok = execute_tool(st, "lookup", {"eid": eid})
        assert ok is True
        assert eid in st["visited_eids"]
        assert st["n_search_calls"] >= 2
    st, _obs, ok = execute_tool(st, "answer", {"reason": "enough evidence"})
    assert ok is True
    assert st["ended"] is True
    assert st["end_reason"]


def test_a0_menu_maps_onto_named_select():
    st = new_state("Alice Smith", STORE, harness_mask=zero_mask("Harness-G"))
    st, _obs, _ok = execute_tool(st, "init", {})
    aid = next(k for k, v in st["action_map"].items() if v.get("type") == "SELECT")
    mapped = st["action_map"][aid]
    st, _obs, ok = execute_tool(st, aid, {})
    assert ok is True
    assert mapped["sid"] in st["selected_sids"]


def test_parse_named_tools_and_a0():
    parsed, ok = parse_harness_g_action('to=select {"sid": "d1:s0"}')
    assert ok is True
    assert parsed["name"] == "select"
    assert parsed["arguments"]["sid"] == "d1:s0"
    menu = {"A0": {"type": "LOOKUP", "eid": "e:alice_smith", "name": "lookup"}}
    parsed, ok = parse_harness_g_action("A0", action_map=menu)
    assert ok is True
    assert parsed["name"] == "lookup"
    assert parsed["arguments"]["eid"] == "e:alice_smith"


def test_snc_preview_only_when_mask_on():
    off = new_state("Alice", STORE, harness_mask=zero_mask("Harness-G"))
    off["visible_sids"] = list(_sids(off)[:2])
    menu_off = build_action_map(off, include_answer=True)
    assert all("snc_preview" not in a for a in menu_off.values())
    on = new_state("Alice", STORE, harness_mask=full_mask("Harness-G"))
    on["visible_sids"] = list(_sids(on)[:2])
    menu_on = build_action_map(on, include_answer=True)
    selects = [a for a in menu_on.values() if a.get("type") == "SELECT"]
    assert selects
    assert all("snc_preview" in a for a in selects)


def _sids(state):
    return list((state.get("sentences") or {}).keys())
