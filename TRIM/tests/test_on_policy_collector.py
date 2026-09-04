from trim.adapters.components import minus_mask
from trim.state.snapshot import capture_snapshot
from trim.training.on_policy_collector import filter_component_states
from trim.training.rl_opd_types import StudentDecisionPoint


def _point(qid: str, text: str) -> StudentDecisionPoint:
    snap = capture_snapshot(
        query_id=qid,
        step=1,
        harness_mask=minus_mask("sentence_compress"),
        working_memory={"query": "q", "documents": [{"id": "d1", "text": text}]},
    )
    return StudentDecisionPoint(
        episode_id=qid,
        query_id=qid,
        rollout_idx=0,
        turn_id=1,
        policy_version="v0",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="",
        student_action_tokens=[],
        student_action_text="to=search_corpus",
        action_tool_names=["search_corpus"],
    )


def test_collector_keeps_long_observations_only():
    long = _point("a", "Long noisy retrieved passage. " * 20)
    short = _point("b", "short")
    kept = filter_component_states([long, short], component_id="sentence_compress")
    assert [p.query_id for p in kept] == ["a"]
