"""H0 profile + context window tests (refactor guide §5, §14)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsi.contracts.action import ToolCall
from rsi.contracts.events import CostRecord, Event, EventKind, Visibility
from rsi.contracts.observation import Observation
from rsi.contracts.task import PublicTask
from rsi.runtime.context import ContextPolicy, prompt_hash, render_view, truncate_head_tail, view_digest
from rsi.runtime.profiles import H0Profile, load_profile, profile_from_runtime_config


def _task() -> PublicTask:
    return PublicTask(
        task_id="toy:t1", domain="toy", instruction="do it", group_id="toy:g", metadata={}
    )


def _event(order: int, obs: str) -> Event:
    return Event(
        event_id=f"e{order}",
        order=order,
        component_id="h0",
        event_kind=EventKind.TOOL_CALL,
        pre_state_digest="a",
        post_state_digest="b",
        action=ToolCall(name="python_exec", arguments={"code": "print(1)"}),
        observation=Observation(content=obs),
        visibility=Visibility.STUDENT,
        cost=CostRecord(tool_calls=1),
    )


def test_oversized_newest_observation_still_visible():
    """Guide §14: the newest observation exceeding the window must still be
    shown truncated with an elision marker — never an empty history."""
    view = _view_with_events([_event(0, "old"), _event(1, "X" * 20000)])
    prompt = render_view(view, max_chars=2000)
    assert "History:" in prompt
    hist = prompt.split("History:\n", 1)[1]
    assert hist.strip(), "empty history is the bug this test pins"
    assert "TRUNCATED" in hist and "chars of" in hist


def _view_with_events(events):
    from rsi.harness.component import PublicView

    return PublicView(
        task=_task(),
        tool_catalog=(),
        history=tuple(events),
        remaining_action_budget=10,
    )


def test_truncate_head_tail_marks_elision():
    out = truncate_head_tail("A" * 500 + "B" * 500, max_chars=200)
    assert out.startswith("A") and out.endswith("B")
    assert "TRUNCATED" in out


def test_profile_changes_prompt_and_hash():
    """Guide §14: changing the profile text or window changes the protocol
    identity — cached baselines never cross versions."""
    react = load_profile("react_zero_shot_v1")
    act = load_profile("act_only_v1")
    view = _view_with_events([_event(0, "obs")])

    p_react = ContextPolicy(max_chars=8000, profile=react)
    p_act = ContextPolicy(max_chars=8000, profile=act)
    assert p_react.render(view) != p_act.render(view)
    assert prompt_hash(p_react) != prompt_hash(p_act)

    p_small = ContextPolicy(max_chars=1000, profile=react)
    assert prompt_hash(p_react) != prompt_hash(p_small)

    # view_digest uses the policy's actual window, not a default re-render:
    # with a large observation the 1000-char window truncates where the
    # 8000-char one does not, so digests differ.
    big = _view_with_events([_event(0, "Y" * 3000)])
    assert view_digest(big, policy=p_small) != view_digest(big, policy=p_react)


def test_profile_text_appears_in_render_header_order():
    """Fixed ordering: profile → env doc → task → tools → protocol → bank."""
    react = load_profile("react_zero_shot_v1", environment_doc_text="ENVDOC-XYZ")
    view = _view_with_events([_event(0, "obs")])
    prompt = render_view(view, max_chars=8000, profile=react)
    i_react = prompt.index("You interact with an environment")
    i_env = prompt.index("ENVDOC-XYZ")
    i_task = prompt.index("Task (toy): do it")
    assert i_react < i_env < i_task


def test_missing_profile_is_hard_error():
    """Guide §5.2: no silent legacy fallback — a config without
    runtime.profile must fail to load."""
    with pytest.raises(Exception, match="runtime.profile is REQUIRED"):
        profile_from_runtime_config({"max_chars": 8000})
    with pytest.raises(Exception, match="unknown H0 profile"):
        profile_from_runtime_config({"profile": "typo_id"})


def test_profile_digest_changes_with_text():
    a = H0Profile(profile_id="x", instruction_text="do A")
    b = H0Profile(profile_id="x", instruction_text="do B")
    assert a.digest() != b.digest()
