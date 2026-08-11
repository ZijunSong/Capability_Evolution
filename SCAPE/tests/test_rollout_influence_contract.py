from __future__ import annotations

from scape.probes.influence import assert_reduced_rollout_owns_state, score_influence_on_snapshot
from scape.probes.rollout import (
    FakeSearchEnv,
    full_teacher_score_only,
    replay_parity,
    student_rollout_collect,
)


def _student_policy(_view, snap):
    return {"name": "search", "arguments": {"query": snap.query_id}}


def test_reduced_rollout_owns_state_distribution():
    env = FakeSearchEnv(query_id="q-own", component_id="evidence_graph")
    snaps = student_rollout_collect(env, _student_policy, n_steps=2)
    assert_reduced_rollout_owns_state(snaps, component_id="evidence_graph")
    assert all(s.harness_mask.get("evidence_graph") is False for s in snaps)
    # Teacher scoring must not require / create a full-harness rollout
    outs = full_teacher_score_only(snaps, component_id="evidence_graph")
    assert len(outs) == len(snaps)


def test_full_vs_minus_replay_parity():
    env = FakeSearchEnv(query_id="q-parity", component_id="sentence_compress")
    snaps = student_rollout_collect(env, _student_policy, n_steps=1)
    report = replay_parity(snaps[-1], component_id="sentence_compress")
    assert report["same_snapshot"] is True
    # Views should differ when the component changes rendering
    assert report["views_differ"] is True

    # Influence on student-owned state
    def student_pol(view):
        return {
            "tool_name_probs": {"search": 0.7, "curate": 0.3},
            "decoded": {"name": "search", "arguments": {"query": "x"}},
        }

    def teacher_pol(view):
        return {
            "tool_name_probs": {"search": 0.2, "curate": 0.8},
            "decoded": {"name": "curate", "arguments": {"add_ids": ["d1"]}},
        }

    sample = score_influence_on_snapshot(
        snaps[-1],
        component_id="sentence_compress",
        student_policy=student_pol,
        teacher_policy=teacher_pol,
    )
    assert sample.I_name > 0
