"""Rollback hard-control contract tests (Round 8)."""

from __future__ import annotations

import copy

import pytest

from harness.capability.rollback_operation import RollbackOperation
from harness.recovery.checkpoint_store import CheckpointStore
from harness.recovery.recovery_budget import RecoveryBudget
from harness.recovery.rollback_runtime import RollbackRuntime
from training.scope.decide_rollback_operation import decide_rollback_operation


class _FakeWM:
    def __init__(self) -> None:
        self.turn_number = 0
        self.curated_ids = ["d1"]
        self.curated_notes = {}
        self.pool_ids = ["d1", "d2"]
        self.pool_id_set = set(self.pool_ids)
        self.search_history = ["q1"]
        self.observation_lineage = [{"observation_id": "obs1"}]
        self.claim_states = {}
        self.curated_observation_ids = {}
        self.verification_records = []
        self.doc_store = {"d1": {"full_text": "x"}, "d2": {"full_text": "y"}}

    def snapshot_hash(self) -> str:
        from harness.telemetry.state_hash import hash_working_memory_fields

        return hash_working_memory_fields(
            curated_ids=self.curated_ids,
            pool_ids=self.pool_ids,
            search_history=self.search_history,
            observation_ids=["obs1"],
            turn_number=self.turn_number,
        )


class _FakeEnv:
    def __init__(self) -> None:
        self.wm = _FakeWM()
        self._current_turn = 0
        self.max_turns = 35


def test_rollback_runtime_restores_state_hash():
    env = _FakeEnv()
    store = CheckpointStore()
    cp = store.save_from_env(env, turn_id=0)
    env.wm.curated_ids.append("bad")
    env.wm.pool_ids.append("bad_doc")
    runtime = RollbackRuntime(store, RecoveryBudget(max_rollbacks=2))
    runtime.execute(env, RollbackOperation.ROLLBACK_TO, checkpoint_id=cp.checkpoint_id)
    assert env.wm.snapshot_hash() == cp.state_hash


def test_rollback_missing_checkpoint_fail_closed():
    env = _FakeEnv()
    runtime = RollbackRuntime(CheckpointStore(), RecoveryBudget(max_rollbacks=1))
    with pytest.raises(ValueError):
        runtime.execute(env, RollbackOperation.ROLLBACK_TO, checkpoint_id="missing")


def test_rollback_budget_enforcement():
    env = _FakeEnv()
    store = CheckpointStore()
    cp = store.save_from_env(env, turn_id=0)
    runtime = RollbackRuntime(store, RecoveryBudget(max_rollbacks=0))
    with pytest.raises(RuntimeError):
        runtime.execute(env, RollbackOperation.ROLLBACK_TO, checkpoint_id=cp.checkpoint_id)


def test_decide_rollback_operation_argmax():
    d = decide_rollback_operation(
        score_continue=0.1,
        score_replan=0.2,
        score_rollback=0.9,
        candidate_checkpoint_id="ckpt_0",
    )
    assert d.predicted_operation == RollbackOperation.ROLLBACK_TO
    assert d.checkpoint_id == "ckpt_0"
