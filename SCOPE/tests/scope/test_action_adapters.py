"""CapabilityAction adapter round-trip tests."""

from __future__ import annotations

from harness.capability.action_space import CapabilityAction, CapabilityActionType
from harness.capability.adapters import (
    parse_action_from_tools,
    parse_policy_action,
    render_capability_action,
)


def test_parse_search():
    a = parse_policy_action('{"tool":"search_corpus","query":"acme"}')
    assert a is not None
    assert a.action_type == CapabilityActionType.SEARCH
    assert a.arguments.get("query") == "acme"


def test_parse_fan_out():
    a = parse_action_from_tools(
        ["fan_out_search"],
        [{"queries": ["q1", "q2"]}],
    )
    assert a is not None
    assert a.arguments.get("fan_out") is True


def test_parse_verify_curate_end():
    v = parse_action_from_tools(["verify"], [{"doc_ids": ["d1"], "claim": "c"}])
    assert v.action_type == CapabilityActionType.VERIFY_CLAIM
    c = parse_action_from_tools(["curate"], [{"add_ids": ["d1"], "remove_ids": []}])
    assert c.action_type == CapabilityActionType.CURATE_DOCUMENT
    e = parse_action_from_tools(["end_search"], [{"reasoning": "done"}])
    assert e.action_type == CapabilityActionType.STOP_AND_ANSWER
    a = parse_action_from_tools(["user_text"], [{"text": "42"}])
    assert a.action_type == CapabilityActionType.ANSWER
    assert a.arguments.get("text") == "42"


def test_round_trip_render_parse():
    action = CapabilityAction(
        action_type=CapabilityActionType.VERIFY_CLAIM,
        arguments={"doc_ids": ["d1"], "claim": "founded by X"},
        target_claim_id="claim1",
    )
    text = render_capability_action(action)
    parsed = parse_policy_action(text)
    assert parsed is not None
    assert parsed.action_type == CapabilityActionType.VERIFY_CLAIM


def test_unparseable_returns_none():
    assert parse_policy_action("not a tool call at all ???") is None
