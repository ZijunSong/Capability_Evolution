from __future__ import annotations

from trim.training.opd_events import EventKind
from trim.training.sft_ablation.teacher_archive import events_from_native_turn


def _events(**kwargs):
    base = dict(
        trajectory_id="q:r0",
        turn_id=1,
        observation={"text": "OBS_SECRET"},
        pre_curated=[],
        post_curated=[],
        pre_state_hash="pre",
        post_state_hash="post",
        prompt_ids=[11, 12],
        executed_ok=True,
        start_index=0,
    )
    base.update(kwargs)
    return events_from_native_turn(**base)


def test_search_does_not_carry_auto_curate_and_ids_are_stable():
    first = _events(
        action={"name": "search_corpus", "arguments": {"query": "q"}},
        post_curated=["d1"],
    )
    second = _events(
        action={"name": "search_corpus", "arguments": {"query": "q"}},
        post_curated=["d1"],
    )
    assert [item["event_id"] for item in first] == [item["event_id"] for item in second]
    action = first[0]
    assert action["kind"] == EventKind.MODEL_ACTION.value
    assert "curated_delta" not in action["state_delta"]
    assert action["prompt_ids"] == [11, 12]
    assert first[1]["kind"] == EventKind.TOOL_OBSERVATION.value
    mutation = first[2]
    assert mutation["kind"] == EventKind.HARNESS_MUTATION.value
    assert mutation["event_index"] > action["event_index"]
    assert mutation["state_delta"]["before_curated"] == []
    assert mutation["state_delta"]["after_curated"] == ["d1"]


def test_native_curate_delta_is_not_emitted_twice():
    events = _events(
        action={"name": "curate", "arguments": {"add_ids": ["d1"], "remove_ids": [], "importance": {"d1": 3}}},
        pre_curated=[],
        post_curated=["d1"],
    )
    assert [item["kind"] for item in events] == [EventKind.MODEL_ACTION.value, EventKind.TOOL_OBSERVATION.value]
    assert events[0]["arguments"]["importance"] == {"d1": 3}
    assert events[0]["state_delta"]["curated_delta"]["add_ids"] == ["d1"]
    assert all(item["kind"] != EventKind.HARNESS_MUTATION.value for item in events)
