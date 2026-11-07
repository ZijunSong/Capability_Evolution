"""PR-06 regression tests (guide §12): snapshot restores the OLD decision
state and the public prefix excludes the future.

``test_snapshot_at_old_decision``: from a later state, forking the t=2
snapshot must reproduce the t=2 state — NOT the current state (the pre-fix
real adapters replayed the FULL history unconditionally, so every teacher
branch silently started from the episode's future).

``test_public_prefix_causality``: a decision recorded at public_event_end=k
must never see events >= k in its prefix; collect_opd_rows must pass exactly
``trace.events[:k]`` to the teacher and skip legacy records without the
boundary.
"""

from __future__ import annotations

import pytest

from rsi.contracts.action import ToolCall
from rsi.contracts.observation import SnapshotRef
from rsi.environments.toy import ToyFileWorld


def _advance(env: ToyFileWorld, path: str, content: str) -> None:
    env.step(ToolCall(name="write_file", arguments={"path": path, "content": content}))


def test_snapshot_at_old_decision():
    env = ToyFileWorld()
    from rsi.contracts.task import PublicTask

    task = PublicTask(
        task_id="toy-snap-1", domain="toy_file", instruction="test", group_id="g",
        metadata={"files": {"a.txt": "seed", "b.txt": "seed"}},
    )
    env.reset(task, seed=17)
    # t=0 state
    snap0 = env.snapshot()
    digest0 = env.state_digest()
    # t=1: mutate
    _advance(env, "a.txt", "one")
    snap1 = env.snapshot()
    digest1 = env.state_digest()
    assert digest1 != digest0
    # t=2: mutate again
    _advance(env, "b.txt", "two")
    digest2 = env.state_digest()

    # fork the OLD snapshot (t=0): child must reproduce the t=0 state, not t=2
    child0 = env.fork(snap0)
    assert child0.state_digest() == digest0, "fork(old snapshot) must restore the old state"
    # base must still be live and at the CURRENT state
    assert env.state_digest() == digest2
    # fork the middle snapshot too
    child1 = env.fork(snap1)
    assert child1.state_digest() == digest1
    # a full-history fork equals the current state
    child_now = env.fork(env.snapshot())
    assert child_now.state_digest() == digest2


def test_public_prefix_causality_in_decision_record():
    """DecisionRecord.public_event_end marks the sampling-time boundary; the
    prefix slice must contain no later events."""
    from rsi.contracts.trace import DecisionRecord

    dr = DecisionRecord(
        campaign_id="c", round_index=0, batch_index=0,
        episode_id="e", decision_index=1,
        environment_snapshot_ref="replay:2",
        public_view_digest="d", runtime_contract_digest="r",
        policy_digest="p", tokenizer_digest="t", chat_template_digest="ct",
        bank_digest="b",
        student_prompt_token_ids=(1, 2, 3),
        sampled_action_token_ids=(4,),
        old_logprobs=(-0.5,),
        action_mask=(1,),
        public_event_end=3,
    )
    assert dr.public_event_end == 3
    assert dr.to_json()["public_event_end"] == 3
    # legacy record: boundary -1 → consumers must skip
    legacy = DecisionRecord(
        campaign_id="c", round_index=0, batch_index=0,
        episode_id="e", decision_index=0,
        environment_snapshot_ref="replay:0",
        public_view_digest="d", runtime_contract_digest="r",
        policy_digest="p", tokenizer_digest="t", chat_template_digest="ct",
        bank_digest="b",
        student_prompt_token_ids=(), sampled_action_token_ids=(),
        old_logprobs=(), action_mask=(),
    )
    assert legacy.public_event_end == -1


def test_collect_opd_rows_skips_legacy_and_uses_prefix():
    """collect_opd_rows: (a) skips records without a public_event_end,
    (b) passes the EXACT prefix (not the full history) to the teacher."""
    from rsi.contracts.task import PublicTask
    from rsi.contracts.trace import DecisionRecord, EpisodeTrace, Event
    from rsi.contracts.events import EventKind
    from rsi.opd.pipeline import collect_opd_rows
    from rsi.harness.loader import load_empty_bank

    seen_prefixes: list[int] = []

    class RecordingRunner:
        def run_branch(self, *, base_env, snapshot_ref, task, student_events, decision_index):
            seen_prefixes.append(len(student_events))
            from rsi.opd.teacher import TeacherBranch
            from rsi.contracts.events import CostRecord

            return TeacherBranch(
                events=(), context_ref="x", snapshot_ref=str(snapshot_ref),
                triggered_components={}, cost=CostRecord(),
                terminated_reason="finish", pre_target_event_count=0,
            )

    class NullEnv:
        def tool_catalog(self):
            return ()

    task = PublicTask(task_id="t", domain="toy_file", instruction="i", group_id="g", metadata={})

    def mk(i, end):
        return DecisionRecord(
            campaign_id="c", round_index=0, batch_index=0,
            episode_id="e", decision_index=i,
            environment_snapshot_ref="replay:0",
            public_view_digest="d", runtime_contract_digest="r",
            policy_digest="p", tokenizer_digest="t", chat_template_digest="ct",
            bank_digest="b",
            student_prompt_token_ids=(), sampled_action_token_ids=(),
            old_logprobs=(), action_mask=(),
            public_event_end=end,
        )

    legacy = DecisionRecord(
        campaign_id="c", round_index=0, batch_index=0,
        episode_id="e", decision_index=0,
        environment_snapshot_ref="replay:0",
        public_view_digest="d", runtime_contract_digest="r",
        policy_digest="p", tokenizer_digest="t", chat_template_digest="ct",
        bank_digest="b",
        student_prompt_token_ids=(), sampled_action_token_ids=(),
        old_logprobs=(), action_mask=(),
    )
    events = tuple(
        Event(
            event_id=f"ev{k}", order=k, component_id="h0",
            event_kind=EventKind.TOOL_CALL,
            pre_state_digest="d", post_state_digest="d",
        )
        for k in range(5)
    )
    trace = EpisodeTrace(
        episode_id="e", task_id="t", domain="toy_file",
        policy_digest="p", bank_digest="b",
        events=events,
        decisions=(legacy, mk(1, 2), mk(2, 5)),
        final_answer=None, outcome_reward=None, terminated_reason="finish",
    )
    out, stats = collect_opd_rows(
        policy=None, teacher_policy=None, env=NullEnv(), bank=load_empty_bank(),
        task=task, trace=trace, teacher_runner=RecordingRunner(),
    )
    # legacy record skipped loudly, two valid records got exact prefixes
    assert [r.reject_reason for r in out[:1]] == ["no_public_event_end_recorded"]
    assert seen_prefixes == [2, 5]
    # the legacy skip + the two empty-branch alignment skips (no target in an
    # empty TeacherBranch) — the boundary skip is what matters here
    assert stats.teacher_fork_success == 2
    assert out[0].teacher_branch_ok is False
    assert out[1].teacher_branch_ok is True and out[2].teacher_branch_ok is True
