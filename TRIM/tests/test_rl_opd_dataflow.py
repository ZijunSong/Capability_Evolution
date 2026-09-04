from __future__ import annotations

import asyncio

from trim.adapters.components import minus_mask
from trim.state.snapshot import capture_snapshot
from trim.training.opd_events import harness_mutation, model_action
from trim.training.rl_opd_observer import DecisionObserver
from trim.training.rl_opd_types import (
    UPDATE_OPD_ONLY_ZERO_RL,
    HybridRolloutGroup,
    StudentDecisionPoint,
)
from trim.training.tinker_rl_opd_trainer import (
    HybridLoopState,
    prepare_hybrid_batch,
    run_hybrid_training_step,
    sample_decision_points,
    split_hybrid_substeps,
)


def _snap(qid: str = "q0"):
    return capture_snapshot(
        query_id=qid,
        step=0,
        harness_mask=minus_mask("auto_populate_first_search"),
        working_memory={
            "curated_ids": ["d1"],
            "accessible_doc_ids": ["d1", "d2"],
            "pool": ["d1", "d2"],
        },
        metadata={"component_id": "auto_populate_first_search"},
    )


def _point(
    *,
    episode: str,
    qid: str,
    turn: int,
    reward: float,
    valid: bool = True,
    tools: list[str] | None = None,
    policy: str = "v0",
) -> StudentDecisionPoint:
    snap = _snap(qid)
    return StudentDecisionPoint(
        episode_id=episode,
        query_id=qid,
        rollout_idx=0,
        turn_id=turn,
        policy_version=policy,
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input=None,
        student_action_tokens=[],
        student_action_text="search_corpus",
        action_tool_names=tools or ["search_corpus"],
        reward=reward,
        structurally_valid=valid,
    )


def _teacher_auto(_point: StudentDecisionPoint):
    return [
        harness_mutation(
            "auto_populate_first_search",
            {"before_curated": ["d1"], "after_curated": ["d1", "d2"]},
        )
    ]


class _SpyClient:
    def __init__(self) -> None:
        self.calls: list = []

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None):
        self.calls.append(("fb", loss_fn, len(list(data))))
        return {"loss": 0.0}

    async def optim_step_async(self, adam):
        del adam
        self.calls.append(("opt",))


def test_lambda_zero_skips_teacher_and_projector():
    called = []

    def teacher(p):
        called.append(p)
        return _teacher_auto(p)

    group = HybridRolloutGroup(
        query_id="q0",
        policy_version="v0",
        trajectory_group=None,
        decision_points=[_point(episode="e0", qid="q0", turn=0, reward=1.0)],
        terminal_rewards=[1.0, 0.0],
    )
    batch = prepare_hybrid_batch(
        groups=[group],
        rl_datums_by_query={"q0": [{"n_tokens": 4}, {"n_tokens": 4}]},
        policy_version="v0",
        lambda_opd=0.0,
        component_id="auto_populate_first_search",
        teacher_event_fn=teacher,
    )
    assert called == []
    assert batch.opd_datums == []
    assert batch.skipped_teacher is True
    assert batch.rl_datums


def test_constant_reward_keeps_opd_states():
    group = HybridRolloutGroup(
        query_id="q0",
        policy_version="v0",
        trajectory_group=None,
        decision_points=[_point(episode="e0", qid="q0", turn=0, reward=0.2)],
        terminal_rewards=[0.2, 0.2],
    )
    batch = prepare_hybrid_batch(
        groups=[group],
        rl_datums_by_query={"q0": [{"n_tokens": 5}, {"n_tokens": 5}]},
        policy_version="v0",
        lambda_opd=0.1,
        component_id="auto_populate_first_search",
        teacher_event_fn=_teacher_auto,
        remove_constant_reward_groups=True,
    )
    assert batch.rl_datums == []
    assert batch.opd_datums
    assert batch.update_type == UPDATE_OPD_ONLY_ZERO_RL


def test_valid_failures_allowed_format_errors_dropped():
    fail = _point(episode="e0", qid="q0", turn=0, reward=0.0, valid=True)
    bad = _point(episode="e0", qid="q0", turn=1, reward=-1.0, valid=False)
    picked = sample_decision_points(
        [fail, bad],
        per_trajectory=4,
        seed=0,
        include_valid_failures=True,
        include_format_errors=False,
    )
    assert [p.turn_id for p in picked] == [0]


def test_negative_k_keeps_every_action_on_the_trajectory():
    points = [
        _point(episode="e0", qid="q0", turn=t, reward=0.1, tools=["search_corpus"])
        for t in range(6)
    ]
    picked = sample_decision_points(points, per_trajectory=-1, seed=0)
    assert [p.turn_id for p in picked] == list(range(6))
    subset = sample_decision_points(points, per_trajectory=3, seed=0)
    assert len(subset) == 3


def test_zero_k_samples_nothing():
    points = [_point(episode="e0", qid="q0", turn=0, reward=0.1)]
    assert sample_decision_points(points, per_trajectory=0, seed=0) == []


def test_num_substeps_not_doubled():
    pairs = split_hybrid_substeps([1, 2, 3, 4], ["a", "b"], num_substeps=4)
    assert len(pairs) == 4
    assert sum(len(r) for r, _ in pairs) == 4
    assert sum(len(o) for _, o in pairs) == 2


def test_teacher_shadow_does_not_change_rl_reward():
    rewards = [0.4, 0.1]
    group = HybridRolloutGroup(
        query_id="q0",
        policy_version="v0",
        trajectory_group=None,
        decision_points=[_point(episode="e0", qid="q0", turn=0, reward=0.4)],
        terminal_rewards=list(rewards),
    )
    before = list(group.terminal_rewards)
    prepare_hybrid_batch(
        groups=[group],
        rl_datums_by_query={"q0": [{"n_tokens": 2}]},
        policy_version="v0",
        lambda_opd=0.1,
        component_id="auto_populate_first_search",
        teacher_event_fn=_teacher_auto,
        remove_constant_reward_groups=False,
    )
    assert group.terminal_rewards == before == rewards


def test_closed_loop_bumps_policy_version():
    class Client(_SpyClient):
        pass

    group = HybridRolloutGroup(
        query_id="q0",
        policy_version="v0",
        trajectory_group=None,
        decision_points=[_point(episode="e0", qid="q0", turn=0, reward=1.0)],
        terminal_rewards=[1.0, 0.0],
    )
    state = HybridLoopState(policy_version="v0")
    asyncio.run(
        run_hybrid_training_step(
            training_client=Client(),
            groups=[group],
            rl_datums_by_query={"q0": [{"n_tokens": 2}]},
            policy_version="v0",
            lambda_opd=0.1,
            component_id="auto_populate_first_search",
            teacher_event_fn=_teacher_auto,
            num_substeps=1,
            loop_state=state,
        )
    )
    assert state.policy_version == "v1"
    assert state.global_optimizer_step == 1


def test_observer_does_not_need_live_teacher():
    class _Env:
        query_id = "q0"
        query_text = "q"
        rollout_idx = 0
        _current_turn = 0
        _episode_id = "q0_r0"
        _action_records = []
        scape_decision_points = []
        wm = type("W", (), {"curated_ids": ["d1"], "pool_ids": ["d1", "d2"]})()

        def export_visible_state(self):
            return {
                "episode_id": self._episode_id,
                "task_id": self.query_id,
                "turn_id": self._current_turn,
                "query": self.query_text,
                "pool_document_ids": ["d1", "d2"],
                "curated_document_ids": ["d1"],
                "visible_document_ids": ["d1", "d2"],
            }

    class _Action:
        tools = [type("T", (), {"tool_schema": type("S", (), {"name": "search_corpus"})()})()]

    env = _Env()
    obs = DecisionObserver(policy_version="v0", component_id="auto_populate_first_search")
    curated_before = list(env.wm.curated_ids)
    obs.on_pre_action(env, _Action())
    obs.on_post_action(env, _Action(), reward=0.3)
    assert env.wm.curated_ids == curated_before
    assert obs.points[0].structurally_valid
    assert obs.points[0].pre_action_snapshot.working_memory["curated_ids"] == ["d1"]


def test_reject_teacher_still_keeps_rl_datums():
    def reject_teacher(_p):
        return [model_action("verify", {"doc_id": "ghost"}, component_id="verify_tool")]

    snap = capture_snapshot(
        query_id="q0",
        step=0,
        harness_mask=minus_mask("verify_tool"),
        working_memory={"curated_ids": [], "accessible_doc_ids": ["d2"], "documents": [{"id": "d2"}]},
        metadata={"component_id": "verify_tool"},
    )
    point = StudentDecisionPoint(
        episode_id="e0",
        query_id="q0",
        rollout_idx=0,
        turn_id=0,
        policy_version="v0",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input=None,
        student_action_tokens=[],
        student_action_text="search_corpus",
        action_tool_names=["search_corpus"],
        reward=1.0,
        structurally_valid=True,
    )
    group = HybridRolloutGroup(
        query_id="q0",
        policy_version="v0",
        trajectory_group=None,
        decision_points=[point],
        terminal_rewards=[1.0, 0.0],
    )
    batch = prepare_hybrid_batch(
        groups=[group],
        rl_datums_by_query={"q0": [{"n_tokens": 8}]},
        policy_version="v0",
        lambda_opd=0.1,
        component_id="verify_tool",
        teacher_event_fn=reject_teacher,
        remove_constant_reward_groups=False,
    )
    assert batch.rl_datums
    assert batch.opd_datums == []


def test_scape_rl_sampled_gap_skips_projector_and_keeps_sampled_tokens():
    called = []

    def teacher(p):
        called.append(p)
        return _teacher_auto(p)

    point = _point(episode="e0", qid="q0", turn=0, reward=1.0)
    point.student_action_tokens = [11, 12, 13]
    point.student_prompt_token_ids = [1, 2, 3]
    group = HybridRolloutGroup(
        query_id="q0",
        policy_version="v0",
        trajectory_group=None,
        decision_points=[point],
        terminal_rewards=[1.0, 0.0],
    )
    batch = prepare_hybrid_batch(
        groups=[group],
        rl_datums_by_query={"q0": [{"n_tokens": 8}]},
        policy_version="v0",
        lambda_opd=0.01,
        component_id="auto_populate_first_search",
        teacher_event_fn=teacher,
        remove_constant_reward_groups=False,
        opd_loss="sr_opd_sampled_gap",
    )
    assert called == []
    assert batch.opd_datums
    datum = batch.opd_datums[0]
    assert datum.target_tokens[-3:] == [11, 12, 13]
    assert datum.prompt_token_ids == [1, 2, 3]
    assert datum.teacher_prompt_token_ids
    assert datum.metadata["sampled_action"] is True
    assert datum.metadata["lambda_opd"] == 0.01
    assert datum.metadata["gate_beta"] == 5.0
    assert batch.projection_stats.get("projector_used") is False
    assert batch.n_opd_tokens == 3


def test_scape_rl_sampled_gap_does_not_require_teacher_fn():
    point = _point(episode="e0", qid="q0", turn=0, reward=0.4)
    point.student_action_tokens = [7, 8]
    group = HybridRolloutGroup(
        query_id="q0",
        policy_version="v0",
        trajectory_group=None,
        decision_points=[point],
        terminal_rewards=[0.4, 0.1],
    )
    batch = prepare_hybrid_batch(
        groups=[group],
        rl_datums_by_query={"q0": [{"n_tokens": 2}]},
        policy_version="v0",
        lambda_opd=0.01,
        component_id="auto_populate_first_search",
        teacher_event_fn=None,
        remove_constant_reward_groups=False,
        opd_loss="sr_opd_sampled_gap",
    )
    assert batch.opd_datums
    assert batch.rl_datums


def test_scape_seed_projects_then_uses_seed_gap_weights():
    called = []

    def teacher(p):
        called.append(p)
        return _teacher_auto(p)

    point = _point(episode="e0", qid="q0", turn=0, reward=1.0)
    point.student_action_tokens = [11, 12, 13]
    group = HybridRolloutGroup(
        query_id="q0",
        policy_version="v0",
        trajectory_group=None,
        decision_points=[point],
        terminal_rewards=[1.0, 0.0],
    )
    batch = prepare_hybrid_batch(
        groups=[group],
        rl_datums_by_query={"q0": [{"n_tokens": 8}]},
        policy_version="v0",
        lambda_opd=0.01,
        component_id="auto_populate_first_search",
        teacher_event_fn=teacher,
        remove_constant_reward_groups=False,
        opd_loss="sr_opd_projected_gap",
        opd_states_per_trajectory=-1,
    )
    assert called
    assert batch.opd_datums
    datum = batch.opd_datums[0]
    assert datum.metadata["sampled_action"] is False
    assert datum.metadata["projector_used"] is True
    assert datum.metadata["lambda_opd"] == 0.01
    assert datum.teacher_prompt_token_ids
    assert all(w in {0.0, 1.0} for w in datum.weights)
    assert batch.projection_stats.get("projector_used") is True
    assert batch.projection_stats.get("sampled_action_opd") is False
