"""T01 DualView teacher mask, T03 query text, T04 skip heuristic, T05 isolation."""

from __future__ import annotations

import pytest

from trim.adapters.role_masks import student_mask_for_ids, teacher_mask_for_ids
from trim.rendering.dual_view import DualViewRenderer
from trim.state.snapshot import capture_snapshot
from trim.training.opd_dataset import render_student_prompt, render_teacher_prompt
from trim.training.parse_rollout_action import parse_generated_action
from trim.training.teacher_isolation import COMPONENT_KIND, run_teacher_branch_isolated
from trim.upstream_harness1.v8d_flags import all_enabled_mask


def test_dual_view_all_uses_ten_teacher_flags():
    student = student_mask_for_ids(list(all_enabled_mask("Harness-1")), harness="Harness-1")
    teacher = teacher_mask_for_ids(list(all_enabled_mask("Harness-1")), harness="Harness-1", preset="all")
    snap = capture_snapshot(
        query_id="q1",
        query_text="What year did X happen?",
        step=1,
        harness_mask=student,
        working_memory={"query": "What year did X happen?", "documents": [{"id": "d1", "text": "hello" * 500}]},
        metadata={"teacher_mask": teacher},
    )
    dual = DualViewRenderer().render_pair(snap, component_id="all", student_mask=student, teacher_mask=teacher)
    assert sum(dual.full_mask.values()) == 10
    assert dual.full_mask["chunk_neighbors"] is True
    assert dual.full_mask["adaptive_rerank_instruction"] is True
    assert sum(dual.student_mask.values()) == 0


def test_dual_view_zero_does_not_restore_full_view():
    zero = teacher_mask_for_ids([], harness="Harness-1", preset="zero")
    snap = capture_snapshot(
        query_id="q0",
        query_text="plain question",
        step=0,
        harness_mask=zero,
        working_memory={"query": "plain question", "documents": []},
        metadata={"teacher_mask": zero},
    )
    dual = DualViewRenderer().render_pair(snap, component_id="zero", student_mask=zero, teacher_mask=zero)
    assert sum(dual.full_mask.values()) == 0
    assert dual.full_mask == dual.student_mask


def test_opd_prompts_use_query_text_not_id():
    snap = capture_snapshot(
        query_id="bcplus_007",
        query_text="Who wrote the 1998 paper on Y?",
        step=2,
        harness_mask=teacher_mask_for_ids([], preset="zero"),
        working_memory={"query": "Who wrote the 1998 paper on Y?", "documents": []},
        metadata={"teacher_mask": teacher_mask_for_ids([], preset="zero")},
    )
    student = render_student_prompt(snap, component_id="zero")
    teacher = render_teacher_prompt(snap, component_id="zero")
    assert "Who wrote the 1998 paper on Y?" in student
    assert "Who wrote the 1998 paper on Y?" in teacher
    assert "Query-id: bcplus_007" in student


def test_teacher_isolation_does_not_mutate_student():
    snap = capture_snapshot(
        query_id="q",
        query_text="qtext",
        step=1,
        harness_mask=teacher_mask_for_ids([], preset="zero"),
        working_memory={"query": "qtext", "documents": [{"id": "d1", "text": "abc"}]},
    )
    before = snap.content_hash()

    def _touch(forked):
        forked.working_memory["documents"][0]["text"] = "MUTATED"
        return "ok"

    result, original = run_teacher_branch_isolated(snap, _touch)
    assert result == "ok"
    assert original.content_hash() == before
    assert original.working_memory["documents"][0]["text"] == "abc"


def test_component_kinds_are_classified():
    assert COMPONENT_KIND["sentence_compress"] == "view"
    assert COMPONENT_KIND["auto_populate_first_search"] == "auto_state"
    assert COMPONENT_KIND["verify_tool"] == "callable_tool"
    assert COMPONENT_KIND["content_dedup"] == "auto_state"


def test_teacher_mode_allows_verify_student_mask_does_not():
    text = (
        "<|start|>assistant to=functions.verify<|channel|>commentary "
        '<|constrain|>json<|message|>{"claim": "X happened in 1999"}<|call|>'
    )
    zero = {cid: False for cid in all_enabled_mask("Harness-1")}
    student, student_ok = parse_generated_action(text, None, enc=None, harness_mask=zero, teacher_mode=False)
    teacher, teacher_ok = parse_generated_action(text, None, enc=None, harness_mask=zero, teacher_mode=True)
    assert student["name"] == "verify"
    assert student_ok is False
    assert teacher_ok is True
    assert teacher["arguments"]["claim"] == "X happened in 1999"


def test_stable_seed_does_not_depend_on_process_hash():
    pytest.importorskip("torch")
    from trim.training.four_cell_runtime import stable_rollout_seed

    a = stable_rollout_seed(42, "scape_seed_rollout0")
    b = stable_rollout_seed(42, "scape_seed_rollout0")
    c = stable_rollout_seed(42, "scape_seed_rollout1")
    assert a == b
    assert a != c


def test_snapshot_keeps_full_document_text():
    pytest.importorskip("torch")
    from trim.training.four_cell_runtime import snap_from_state

    long_text = "Z" * 5000
    st = {
        "query": "long doc question",
        "curated": {},
        "pool": {"d1": True},
        "doc_store": {"d1": {"id": "d1", "text": long_text}},
        "step": 1,
    }
    snap = snap_from_state("qid", st, "all")
    stored = (snap.working_memory.get("doc_store") or {})["d1"]["text"]
    assert stored == long_text
    assert len(snap.working_memory["documents"][0]["text"]) == 5000
    assert snap.query_text == "long doc question"


def test_on_policy_refresh_includes_trim_cells():
    pytest.importorskip("torch")
    from trim.training.vllm_hybrid import plan_cell_phases

    phases = plan_cell_phases("scape_seed", train_steps=2, on_policy_refresh=True, use_frozen_states=False)
    assert phases.count("vllm_rollout") == 2
    assert phases.count("hf_train") == 2
    frozen = plan_cell_phases("pure_opd", train_steps=2, on_policy_refresh=True, use_frozen_states=True)
    assert "vllm_rollout" not in frozen


def test_unregistered_teacher_does_not_use_heuristic_by_default():
    pytest.importorskip("torch")
    from trim.training.four_cell_runtime import teacher_for
    from trim.training.rl_opd_types import StudentDecisionPoint

    snap = capture_snapshot(
        query_id="q",
        query_text="qtext",
        step=0,
        harness_mask=teacher_mask_for_ids([], preset="zero"),
        working_memory={"query": "qtext", "documents": [{"id": "d1", "text": "abc"}]},
    )
    point = StudentDecisionPoint(
        episode_id="e",
        query_id="q",
        rollout_idx=0,
        turn_id=0,
        policy_version="v0",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="",
        student_action_tokens=[],
        student_action_text="",
        action_tool_names=[],
        post_action_snapshot=snap,
    )
    events = teacher_for("content_dedup", teacher_kind="upstream")(point)
    assert events
    assert events[0].metadata.get("teacher_kind") == "skip_unregistered"
    heuristic = teacher_for("content_dedup", teacher_kind="heuristic_generic")(point)
    assert any(getattr(e, "metadata", {}).get("teacher_kind") == "heuristic_generic" for e in heuristic)
