"""Regression tests for the 2026-09-14 TRIM training audit P0/P1 fixes."""

from __future__ import annotations

from pathlib import Path

from trim.adapters.components import minus_mask
from trim.eval.h1_component_runtime import MAX_CURATED_DOCS
from trim.eval.harmony_runtime import fit_actions_obs_to_budget, fit_prompt_ids_to_context
from trim.state.snapshot import capture_snapshot
from trim.training.auto_populate_teacher import (
    is_first_search_trigger,
    teacher_events_from_wm,
)
from trim.training.four_cell_runtime import (
    encode_aligned_teacher_prompt,
    freeze_train_state,
    terminal_reward,
    terminal_reward_breakdown,
)
from trim.training.opd_events import harness_mutation, model_action
from trim.training.opd_projection import StudentActionSpaceProjector
from trim.training.opd_realizability import (
    REJECT_CURATED_CAPACITY,
    apply_student_action,
    check_action_realizability,
)
from trim.training.rl_opd_types import StudentDecisionPoint
from trim.training.tinker_rl_opd_trainer import project_on_policy_decisions
from trim.training.train_checkpoint import training_output_occupied
from trim.training.train_query_sampler import QuerySampler


def _snap(**wm):
    return capture_snapshot(
        query_id="q",
        step=int(wm.get("step") or 0),
        harness_mask=minus_mask("auto_populate_first_search"),
        working_memory={
            "query": "Apple FY2023 filing date",
            "curated_ids": [],
            "accessible_doc_ids": [],
            "pool": {},
            **wm,
        },
    )


def _point(snap, **kwargs):
    return StudentDecisionPoint(
        episode_id="e0",
        query_id="q",
        rollout_idx=0,
        turn_id=int(kwargs.pop("turn_id", 0)),
        policy_version="v0",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="student",
        student_action_tokens=kwargs.pop("tokens", [1, 2]),
        student_action_text=kwargs.pop("text", "to=search_corpus\n{}\n"),
        action_tool_names=kwargs.pop("tools", ["search_corpus"]),
        structurally_valid=True,
        student_prompt_token_ids=[9, 9],
        teacher_prompt_token_ids=[8, 8],
        **kwargs,
    )


def test_auto_populate_empty_pool_emits_first_search():
    events = teacher_events_from_wm(
        {"query": "q", "pool": {}, "curated_ids": [], "first_search_pending": True, "search_count": 0}
    )
    assert events[0].action_name == "search_corpus"
    assert events[0].arguments["query"] == "q"


def test_auto_populate_just_after_first_search_emits_delta():
    wm = {
        "query": "q",
        "first_search_pending": False,
        "search_count": 1,
        "last_tool_name": "search_corpus",
        "curated_ids": [],
        "pool": {
            "d1": {"id": "d1", "score": 2.0, "text": "a"},
            "d2": {"id": "d2", "score": 1.0, "text": "b"},
        },
    }
    assert is_first_search_trigger(wm) == "apply_auto_populate"
    events = teacher_events_from_wm(wm)
    assert events[0].state_delta["after_curated"] == ["d1", "d2"]


def test_auto_populate_skips_late_leftover_and_repeat_search():
    leftover = {
        "query": "q",
        "first_search_pending": False,
        "search_count": 1,
        "last_tool_name": "curate",
        "step": 30,
        "curated_ids": ["d1"],
        "pool": {"d1": {"score": 1.0}, "d2": {"score": 0.9}},
    }
    assert is_first_search_trigger(leftover) is None
    assert teacher_events_from_wm(leftover)[0].metadata["skip_reason"] == "not_first_search_trigger"

    done = {
        "query": "q",
        "first_search_pending": False,
        "search_count": 2,
        "last_tool_name": "search_corpus",
        "curated_ids": ["d1", "d2"],
        "pool": {"d1": {"score": 1.0}, "d2": {"score": 0.9}},
    }
    assert is_first_search_trigger(done) is None


def test_capacity_add_only_when_full_is_not_realizable():
    snap = _snap(
        curated_ids=[f"c{i}" for i in range(MAX_CURATED_DOCS)],
        accessible_doc_ids=[f"c{i}" for i in range(MAX_CURATED_DOCS)] + ["new"],
        pool={"new": {"id": "new", "text": "x"}},
    )
    action = {"name": "curate", "arguments": {"add_ids": ["new"], "remove_ids": []}}
    report = check_action_realizability(
        action=action,
        student_snapshot=snap,
        student_mask=snap.harness_mask,
        component_id="auto_populate_first_search",
    )
    assert report.passed is False
    assert REJECT_CURATED_CAPACITY in report.reason_codes
    after = apply_student_action(snap, action)
    assert len(after.working_memory["curated_ids"]) == MAX_CURATED_DOCS
    assert "new" not in after.working_memory["curated_ids"]


def test_capacity_legal_replace_passes():
    ids = [f"c{i}" for i in range(MAX_CURATED_DOCS)]
    snap = _snap(
        curated_ids=ids,
        accessible_doc_ids=ids + ["new"],
        pool={"new": {"id": "new", "text": "x"}},
    )
    action = {"name": "curate", "arguments": {"add_ids": ["new"], "remove_ids": [ids[-1]]}}
    report = check_action_realizability(
        action=action,
        student_snapshot=snap,
        student_mask=snap.harness_mask,
        component_id="auto_populate_first_search",
    )
    assert report.passed is True
    after = apply_student_action(snap, action)
    assert "new" in after.working_memory["curated_ids"]
    assert ids[-1] not in after.working_memory["curated_ids"]
    assert len(after.working_memory["curated_ids"]) == MAX_CURATED_DOCS


def test_independent_component_projection_does_not_let_auto_steal():
    snap = _snap(
        first_search_pending=False,
        search_count=3,
        last_tool_name="read_document",
        curated_ids=["d1"],
        accessible_doc_ids=["d1", "d2"],
        pool={"d1": {"id": "d1", "text": "Alice lectured at a university. " * 20, "score": 1.0}},
        documents=[
            {"id": "d1", "text": "Alice lectured at a university. " * 20},
            {"id": "d2", "text": "The author lectured from 2018 until his death. " * 16},
        ],
    )
    point = _point(snap, turn_id=5)

    def teacher(p):
        from trim.training.auto_populate_teacher import teacher_events_from_point as auto_fn
        from trim.training.sentence_compress_teacher import teacher_events_from_point as sc_fn

        return list(auto_fn(p)) + list(sc_fn(p))

    steps, audit, extras = project_on_policy_decisions(
        [point],
        teacher_event_fn=teacher,
        component_id="all",
        projector=StudentActionSpaceProjector(),
    )
    assert extras["component_stats"]["auto_populate_first_search"]["skip_count"] >= 1
    assert extras["component_stats"]["sentence_compress"]["align_count"] >= 1
    assert any(s.target_action.get("name") == "curate" for s in steps)
    assert extras["rl_opd_same_tool_rate"] != extras.get("rl_opd_exact_target_overlap_rate") or extras[
        "rl_opd_same_supervised_token_rate"
    ] == extras["rl_opd_exact_target_overlap_rate"]


def test_zero_recall_shaping_is_capped():
    st = {
        "curated": {"d1": {"text": "unrelated"}, "d2": {"text": "also unrelated"}},
        "pool": {"d1": {}, "d2": {}, "d3": {}},
        "ended": True,
    }
    actions = [
        {"name": "search_corpus", "arguments": {"query": "when was the apple fy2023 10-k filed"}},
        {"name": "curate", "arguments": {"add_ids": ["d1", "d2"]}},
        {"name": "end_search", "arguments": {}},
    ]
    parts = terminal_reward_breakdown(
        st,
        query="when was the apple fy2023 10-k filed",
        gold_ids=["gold"],
        valids=[True, True, True],
        actions=actions,
    )
    assert parts["task_recall"] == 0.0
    assert parts["shaping"] <= 0.10
    assert parts["total"] <= 0.20
    assert terminal_reward(st, query="q", gold_ids=["gold"], valids=[True, True, True], actions=actions) == parts["total"]


def test_teacher_prompt_uses_frozen_pre_state(tmp_path):
    frozen = {
        "query": "q",
        "pool": {},
        "curated": {},
        "first_search_pending": True,
        "search_count": 0,
        "harness_mask": minus_mask("sentence_compress"),
    }
    copied = freeze_train_state(frozen)
    copied["pool"]["new"] = {"id": "new", "text": "appeared after action"}
    assert "new" not in frozen["pool"]

    class Enc:
        def build_first_turn_prompt_ids(self, query):
            return [1, 2, 3]

        def build_continuation_prompt_ids(self, query, actions_obs=None, wm_text=""):
            blob = str(wm_text or "")
            assert "appeared after action" not in blob
            return [7, 8, 9] + ([10] if actions_obs else [])

    ids, wm = encode_aligned_teacher_prompt(
        Enc(),
        "q",
        frozen_st=frozen,
        frozen_acts=[],
        component_id="sentence_compress",
    )
    assert ids
    assert "appeared after action" not in wm


def test_fit_actions_obs_drops_oldest_complete_pairs():
    pairs = [(i, i) for i in range(8)]

    def encode(kept):
        return list(range(1000 * len(kept)))

    fitted = fit_actions_obs_to_budget(
        pairs, encode_fn=encode, max_model_len=2500, max_new_tokens=100
    )
    assert fitted[0][0] > 0
    assert fitted[-1] == pairs[-1]


def test_fit_prompt_ids_still_fits_budget():
    ids = list(range(40000))
    fitted = fit_prompt_ids_to_context(ids, max_model_len=32768, max_new_tokens=2048, keep_prefix=4096)
    assert len(fitted) + 2048 <= 32768


def test_sampler_checkpoint_step_matches_after_bump():
    pool = [{"query_id": f"q{i}"} for i in range(64)]
    sampler = QuerySampler(pool, base_seed=42, groups_per_step=32)
    sampler.sample_for_rollout()
    sampler.note_rollout_start()
    sampler.note_update_complete()
    assert sampler.state.global_optimizer_step == 1
    payload = sampler.state.to_dict()
    assert payload["global_optimizer_step"] == 1


def test_occupied_output_dir_detects_checkpoints(tmp_path: Path):
    assert training_output_occupied(tmp_path) is False
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "rl").mkdir()
    assert training_output_occupied(tmp_path) is True


class _EmptyLiveSearcher:
    name = "pyserini_lucene"

    def search(self, query: str, k: int = 5):
        del query, k
        return []


def test_live_empty_search_does_not_prefetch_seeded_store():
    from trim.eval.local_search_env import execute_tool, new_state

    store = {"d1": {"id": "d1", "text": "Paris is the capital of France."}}
    st, _obs, ok = execute_tool(
        new_state("capital of France", store),
        "search_corpus",
        {"query": "capital of France"},
        searcher=_EmptyLiveSearcher(),
    )
    assert ok is True
    assert not st.get("pool")


def test_local_legacy_without_searcher_ranks_episode_store():
    from trim.eval.local_search_env import execute_tool, new_state

    store = {"d1": {"id": "d1", "text": "Paris is the capital of France."}}
    st, obs, ok = execute_tool(
        new_state("capital of France", store),
        "search_corpus",
        {"query": "capital of France"},
        searcher=None,
    )
    assert ok is True
    assert "d1" in st["pool"]
    assert "Paris" in obs
