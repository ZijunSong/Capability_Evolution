from trim.adapters.components import zero_mask
from trim.state.snapshot import EnvironmentSnapshot
from trim.training.opd_events import EventKind, HarnessEvent
from trim.training.rl_opd_types import StudentDecisionPoint
from trim.training.teacher_branch import SOURCE_CAPABILITY, TeacherBranchResult
from trim.training.tinker_rl_opd_trainer import project_on_policy_decisions


def _point(prompt: list[int]) -> StudentDecisionPoint:
    snap = EnvironmentSnapshot(
        query_id="q",
        step=0,
        harness_mask=zero_mask("Harness-1"),
        working_memory={"query": "q", "accessible_doc_ids": ["d1"]},
        query_text="q",
    )
    return StudentDecisionPoint(
        episode_id="ep",
        query_id="q",
        rollout_idx=0,
        turn_id=0,
        policy_version="v1",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input=None,
        student_action_tokens=[1],
        student_action_text="",
        action_tool_names=["search_corpus"],
        student_prompt_token_ids=[9],
        teacher_prompt_token_ids=prompt,
    )


def _event(cid: str, eid: str) -> HarnessEvent:
    return HarnessEvent(
        event_id=eid,
        turn_id=0,
        kind=EventKind.MODEL_ACTION,
        component_id=cid,
        action_name="search_corpus",
        arguments={"query": "visible"},
    )


def test_joint_trace_stays_in_one_group_and_keeps_branch_prompt():
    point = _point([1, 2, 3])

    def teacher(_point):
        return TeacherBranchResult(
            decision_state_id="ep:0",
            component_id="bank",
            triggered=True,
            source_type=SOURCE_CAPABILITY,
            events=[_event("evo.a", "a"), _event("evo.b", "b")],
            teacher_condition={
                "continuous": True,
                "teacher_prompt_token_ids": [7, 7],
                "bank_sha256": "abc",
            },
        )

    steps, _audit, extras = project_on_policy_decisions(
        [point], teacher_event_fn=teacher, component_id="all"
    )
    assert extras["n_attempt"] >= 0
    assert steps
    assert all(step.metadata["teacher_prompt_token_ids"] == [7, 7] for step in steps)
    assert all(step.metadata["teacher_prefix_source"] == "branch" for step in steps)
    assert all(step.metadata["selection_rule"] == "continuous_bank_joint_trace" for step in steps)
    assert all(step.metadata["bank_sha256"] == "abc" for step in steps)


def test_independent_traces_keep_their_own_prompts():
    first = _point([1])
    second = _point([2])

    def teacher(point):
        prompt = [11] if point.teacher_prompt_token_ids == [1] else [22]
        cid = "sentence_compress" if prompt == [11] else "verify_tool"
        return TeacherBranchResult(
            decision_state_id=point.decision_point_id,
            component_id=cid,
            triggered=True,
            source_type=SOURCE_CAPABILITY,
            events=[_event(cid, cid)],
            teacher_condition={"continuous": False, "teacher_prompt_token_ids": prompt},
        )

    steps, _audit, _extras = project_on_policy_decisions(
        [first, second], teacher_event_fn=teacher, component_id="all"
    )
    prompts = {tuple(step.metadata["teacher_prompt_token_ids"]) for step in steps}
    assert (11,) in prompts
    assert (22,) in prompts
