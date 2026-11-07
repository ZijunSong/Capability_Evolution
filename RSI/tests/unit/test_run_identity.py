"""PR-02 tests (guide §12): run artifact identity + actor RNG streams."""

from __future__ import annotations

import json

import pytest

from rsi.runtime.run_identity import RunContext, atomic_write_json, stable_seed


def _ctx(**over) -> RunContext:
    base = dict(
        run_id="run1", campaign_id="c", phase="eval", arm_id="A",
        bank_digest="b" * 16, policy_digest="p" * 16, replicate_id=0,
        episode_nonce="n0",
    )
    base.update(over)
    return RunContext(**base)


def test_ab_arms_get_different_identities():
    a = _ctx(arm_id="A")
    b = _ctx(arm_id="B")
    assert a.experiment_hash() != b.experiment_hash()
    # and both are colon-free (AppWorld checkpoint constraint)
    assert ":" not in a.experiment_hash()
    da = a.episode_dir("/runs", "task:1")
    db = b.episode_dir("/runs", "task:1")
    assert da != db
    assert "task_1" in str(da)  # colon sanitized
    assert "/A/" in str(da) and "/B/" in str(db)


def test_replicate_and_nonce_change_paths_not_seeds():
    """Output identity varies per replicate/nonce; the TASK seed derivation
    (environment stream) is identity-free."""
    a0 = _ctx(replicate_id=0, episode_nonce="n0")
    a1 = _ctx(replicate_id=1, episode_nonce="n1")
    assert a0.experiment_hash() != a1.experiment_hash()
    env_seed_a = stable_seed(17, task_id="t1", stream="environment", decision_index=0)
    env_seed_b = stable_seed(17, task_id="t1", stream="environment", decision_index=0)
    assert env_seed_a == env_seed_b  # environment seed independent of identity


def test_stable_seed_deterministic_and_process_independent():
    s1 = stable_seed(17, task_id="t", stream="actor", decision_index=3)
    s2 = stable_seed(17, task_id="t", stream="actor", decision_index=3)
    assert s1 == s2
    # never the salted builtin hash (different processes would disagree)
    assert s1 != hash((17, "t", 0, "actor", 3, 0))


def test_streams_are_independent():
    """teacher/aux requests never consume the student's stream: same decision
    under different streams yields different seeds; a teacher call at
    decision i does not change the actor seed at i+1."""
    actor_i = stable_seed(17, task_id="t", stream="actor", decision_index=0)
    teacher_i = stable_seed(17, task_id="t", stream="teacher", decision_index=0)
    aux_i = stable_seed(17, task_id="t", stream="auxiliary", decision_index=0)
    assert len({actor_i, teacher_i, aux_i}) == 3
    actor_next = stable_seed(17, task_id="t", stream="actor", decision_index=1)
    assert actor_next != actor_i
    # and the actor seed is unaffected by "having run" the teacher call
    actor_i_again = stable_seed(17, task_id="t", stream="actor", decision_index=0)
    assert actor_i_again == actor_i


def test_unknown_stream_rejected():
    with pytest.raises(ValueError):
        stable_seed(1, task_id="t", stream="bogus")


def test_atomic_write_collisions(tmp_path):
    p = tmp_path / "artifacts" / "final_artifact.json"
    atomic_write_json(p, {"a": 1})
    # same content: no-op
    atomic_write_json(p, {"a": 1})
    # different content on the same path: refuses (the A/B overwrite bug)
    with pytest.raises(FileExistsError):
        atomic_write_json(p, {"a": 2})
    # explicit owner may overwrite
    atomic_write_json(p, {"a": 2}, overwrite_ok=True)
    assert json.loads(p.read_text()) == {"a": 2}
    # no tmp litter left behind
    assert not list(p.parent.glob("*.tmp"))
