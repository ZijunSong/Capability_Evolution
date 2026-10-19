from __future__ import annotations

import pytest

from trim.training.sft_ablation.contracts import PROFILES, SFTBatchRow
from trim.training.student_onpolicy_sft.runtime import LoopState, Policy, run_online_ce
from trim.training.tinker_rl_opd_trainer import sample_decision_points


class Ports:
    def __init__(self, *, empty: bool = False, stale: bool = False) -> None:
        self.empty = empty
        self.stale = stale
        self.calls: list[tuple] = []
        self.digest = "adapter-v0"

    def restore_or_initialize(self) -> LoopState:
        return LoopState()

    def publish_and_verify_policy(self, state: LoopState) -> Policy:
        self.calls.append(("publish", state.optimizer_step, self.digest))
        return Policy(version=f"student:step{state.optimizer_step}:{self.digest}", digest=self.digest)

    def collect_fresh_student_points(self, policy: Policy, round_id: int):
        self.calls.append(("collect", policy.version, policy.digest))
        version = "stale" if self.stale else policy.version
        return [type("Point", (), {"policy_version": version, "reward": 0.0})()]

    def build_legal_ce_targets(self, points, policy: Policy):
        self.calls.append(("label", points[0].reward if points else None))
        if self.empty:
            return [], {"reason": "no_legal_branch"}
        profile = PROFILES["student_onpolicy_sft"]
        row = SFTBatchRow(
            row_id=f"row-{policy.digest}",
            experiment="student_onpolicy_sft",
            query_id="q",
            episode_id="ep",
            decision_id="d",
            prompt_ids=[1],
            response_ids=[2],
            response_mask=[True],
            weight=1.0,
            policy_version=policy.version,
            prompt_owner=profile.prompt_owner,
            target_source=profile.supervision_provider,
        )
        return [row], {"collection_semantics": "student_states_for_ce"}

    def persist_round_data(self, state, policy, rows, audit) -> None:
        self.calls.append(("persist", policy.digest, len(rows)))

    def record_empty_round(self, state, audit) -> None:
        self.calls.append(("empty", state.empty_rounds))

    def ce_optimizer_update(self, rows):
        self.calls.append(("update", self.digest))
        self.digest = "adapter-v1"
        return type("Result", (), {"updated": True, "n_rows": len(rows)})()

    def save_and_publish_checkpoint(self, state, result) -> None:
        self.calls.append(("save", state.optimizer_step, self.digest))

    def persist_loop_state(self, state) -> None:
        self.calls.append(("state", state.optimizer_step, state.round_id))

    def summary(self, state) -> dict:
        return {"optimizer_step": state.optimizer_step, "digest": self.digest}


def test_second_round_uses_the_updated_adapter_and_reward_zero_still_labels():
    ports = Ports()
    summary = run_online_ce(ports, max_updates=2, max_empty_rounds=3)
    assert summary["optimizer_step"] == 2
    assert ("collect", "student:step0:adapter-v0", "adapter-v0") in ports.calls
    assert ("collect", "student:step1:adapter-v1", "adapter-v1") in ports.calls
    assert ("label", 0.0) in ports.calls
    assert [name for name, *_rest in ports.calls].count("update") == 2
    assert "execute" not in {name for name, *_rest in ports.calls}


def test_stale_policy_is_refused():
    with pytest.raises(ValueError, match="mixed rollout"):
        run_online_ce(Ports(stale=True), max_updates=1)


def test_empty_rounds_do_not_update_and_then_fail():
    ports = Ports(empty=True)
    with pytest.raises(RuntimeError, match="consecutive rounds"):
        run_online_ce(ports, max_updates=1, max_empty_rounds=2)
    assert "update" not in {name for name, *_rest in ports.calls}
    assert ("empty", 1) in ports.calls
    assert ("empty", 2) in ports.calls


def test_zero_reward_points_stay_eligible():
    from trim.state.snapshot import EnvironmentSnapshot

    snap = EnvironmentSnapshot(query_id="q", step=0, harness_mask={}, working_memory={})
    point = type(
        "P",
        (),
        {
            "episode_id": "ep",
            "turn_id": 0,
            "reward": 0.0,
            "structurally_valid": True,
            "action_tool_names": ["search_corpus"],
        },
    )()
    del snap
    chosen = sample_decision_points([point], per_trajectory=-1, seed=1, include_valid_failures=True)
    assert chosen == [point]
