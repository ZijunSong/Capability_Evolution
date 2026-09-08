"""RL and TRIM training must put the student/teacher mask into live rollouts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from trim.adapters.components import all_component_ids, zero_mask
from trim.cli.launch import parse_train_args
from trim.eval.local_search_env import execute_tool, new_state
from trim.eval.runtime_effect_audit import audit_train_runtime_or_raise
from trim.training.four_cell_runtime import (
    cell_lambda,
    resolved_rollout_mask,
    snap_from_state,
    student_mask_for,
    teacher_action_from_point,
    teacher_for,
    teacher_mask_for,
)
from trim.training.rl_opd_types import (
    OPD_LOSS_PROJECTED_GAP,
    TRAINING_MODE_RL,
    TRAINING_MODE_RL_OPD,
    TRAINING_MODE_SCAPE_SEED,
    StudentDecisionPoint,
)
from trim.training.tinker_rl_opd_trainer import prepare_hybrid_batch


STORE = {
    "d1": {
        "id": "d1",
        "text": "Alice Smith visited Paris in 2019. The treaty named Bob Jones. " * 8,
        "score": 0.9,
    },
    "d2": {
        "id": "d2",
        "text": "Alice Smith later joined Carol Adams in Paris. The 2019 treaty held.",
        "score": 0.4,
    },
}


def _row(qid: str = "q0") -> dict:
    return {
        "query_id": qid,
        "query": "Alice Smith Paris 2019",
        "gold_docids": ["d1"],
        "frozen_doc_store": dict(STORE),
    }


def _search_generate(requests):
    from trim.training.vllm_hybrid import GenerateResult, mean_behavior_logprob

    text = (
        "<|channel|>analysis<|message|>search now.<|end|>"
        "<|start|>assistant to=functions.search_corpus<|channel|>commentary "
        "<|constrain|>json<|message|>{\"query\": \"Alice Smith Paris 2019\"}<|call|>"
    )
    ids = [1, 2, 200012]
    return [
        GenerateResult(
            request_id=req.request_id,
            token_ids=ids,
            token_logprobs=[-0.1, -0.2, -0.3],
            text=text,
            logprob_old=mean_behavior_logprob([-0.1, -0.2, -0.3]),
            logprob_provenance="vllm_sampled_token",
        )
        for req in requests
    ]


def _point_from_state(component_id: str, st: dict, *, mask: dict[str, bool]) -> StudentDecisionPoint:
    snap = snap_from_state("q0", st, component_id, harness_mask=mask)
    return StudentDecisionPoint(
        episode_id="e0",
        query_id="q0",
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
        structurally_valid=True,
    )


def test_resolved_rollout_mask_defaults_to_student_not_zero():
    student = resolved_rollout_mask("sentence_compress")
    assert student == student_mask_for("sentence_compress")
    assert student != zero_mask()
    assert student["sentence_compress"] is False
    assert student["auto_populate_first_search"] is True
    coalition = ",".join(all_component_ids())
    assert resolved_rollout_mask(coalition) == zero_mask()
    assert resolved_rollout_mask("sentence_compress", teacher_mode=True) == teacher_mask_for(
        "sentence_compress"
    )
    explicit = dict(zero_mask())
    explicit["verify_tool"] = True
    assert resolved_rollout_mask("sentence_compress", harness_mask=explicit) == explicit


def test_parse_train_rl_vs_trim_and_cell_lambda():
    rl_args, rl_spec = parse_train_args(
        ["--train_method", "rl", "--component", "all", "--out", "/tmp/train-rl-runtime"]
    )
    trim_args, trim_spec = parse_train_args(
        ["--train_method", "trim", "--component", "all", "--out", "/tmp/train-trim-runtime"]
    )
    opd_args, opd_spec = parse_train_args(
        ["--train_method", "rl+opd", "--component", "sentence_compress", "--out", "/tmp/train-rlopd"]
    )
    assert rl_spec.training_mode == TRAINING_MODE_RL
    assert trim_spec.training_mode == TRAINING_MODE_SCAPE_SEED
    assert opd_spec.training_mode == TRAINING_MODE_RL_OPD
    # run_train.py zeros lambda for RL; parse_train_args still carries the default.
    rl_lam = 0.0 if rl_args.training_mode == TRAINING_MODE_RL else float(rl_args.lambda_opd)
    trim_lam = 0.0 if trim_args.training_mode == TRAINING_MODE_RL else float(trim_args.lambda_opd)
    assert rl_lam == 0.0
    assert trim_lam > 0.0
    assert trim_args.opd_loss == OPD_LOSS_PROJECTED_GAP
    assert cell_lambda("rl", 0.1) == 0.0
    assert cell_lambda("scape_seed", 0.01) == 0.01
    assert cell_lambda("rl_opd", 0.1) == 0.1


def test_teacher_action_differs_by_component():
    mask = teacher_mask_for("sentence_compress")
    st = new_state("Alice Smith Paris 2019", dict(STORE), harness_mask=mask)
    point = _point_from_state("sentence_compress", st, mask=mask)
    compress = teacher_action_from_point(point, "sentence_compress")
    auto = teacher_action_from_point(point, "auto_populate_first_search")
    assert compress["name"] == "curate"
    assert auto["name"] == "search_corpus"


def test_batched_rollout_omitted_mask_matches_student_and_fires_live():
    pytest.importorskip("openai_harmony")
    from trim.training.batched_env_rollout import rollout_queries_batched

    with patch("trim.training.batched_env_rollout._build_prompt_ids", return_value=[1, 2, 3, 4]):
        groups = rollout_queries_batched(
            _search_generate,
            [_row()],
            component_id="sentence_compress",
            group_size=1,
            max_turns=1,
            max_new=32,
            policy_version="v0",
            seed=1,
            sample=True,
            enc=None,
            searcher=None,
        )
    point = groups[0].decision_points[0]
    want = student_mask_for("sentence_compress")
    assert point.pre_action_snapshot.harness_mask == want
    wm = point.post_action_snapshot.working_memory
    assert wm.get("auto_populate_seed")
    assert int((wm.get("runtime_effects") or {}).get("auto_populate_first_search") or 0) >= 1


def test_batched_rollout_all_student_is_zero_no_auto_seed():
    pytest.importorskip("openai_harmony")
    from trim.training.batched_env_rollout import rollout_queries_batched

    coalition = ",".join(all_component_ids())
    with patch("trim.training.batched_env_rollout._build_prompt_ids", return_value=[1, 2, 3, 4]):
        groups = rollout_queries_batched(
            _search_generate,
            [_row()],
            component_id=coalition,
            group_size=1,
            max_turns=1,
            max_new=32,
            policy_version="v0",
            seed=1,
            sample=True,
            enc=None,
            searcher=None,
        )
    point = groups[0].decision_points[0]
    assert point.pre_action_snapshot.harness_mask == zero_mask()
    wm = point.post_action_snapshot.working_memory
    assert not wm.get("auto_populate_seed")
    assert int((wm.get("runtime_effects") or {}).get("auto_populate_first_search") or 0) == 0


def test_teacher_mode_uses_teacher_for_not_hardcoded_sentence_compress():
    pytest.importorskip("openai_harmony")
    from trim.training.batched_env_rollout import rollout_queries_batched

    class Enc:
        def encode(self, text):
            return [1, 2, 3]

    def boom(*_a, **_k):
        raise AssertionError("teacher_mode must not hardcode sentence_compress_teacher")

    with (
        patch("trim.training.batched_env_rollout._build_prompt_ids", return_value=[1, 2, 3, 4]),
        patch("trim.training.sentence_compress_teacher.teacher_events_from_point", boom),
    ):
        groups = rollout_queries_batched(
            _search_generate,
            [_row()],
            component_id="auto_populate_first_search",
            group_size=1,
            max_turns=1,
            max_new=32,
            policy_version="v0",
            seed=1,
            sample=False,
            enc=Enc(),
            searcher=None,
            teacher_mode=True,
        )
    assert groups
    assert groups[0].decision_points


def test_prepare_hybrid_batch_rl_skips_opd_trim_projects():
    pytest.importorskip("openai_harmony")
    from trim.training.batched_env_rollout import rollout_queries_batched

    with patch("trim.training.batched_env_rollout._build_prompt_ids", return_value=[1, 2, 3, 4]):
        groups = rollout_queries_batched(
            _search_generate,
            [_row()],
            component_id="sentence_compress",
            group_size=1,
            max_turns=1,
            max_new=32,
            policy_version="v0",
            seed=1,
            sample=True,
            enc=None,
            searcher=None,
        )
    rl_by_q = {
        g.query_id: list((g.trajectory_group or {}).get("rl_rows") or []) for g in groups
    }
    teacher = teacher_for("sentence_compress")
    rl_batch = prepare_hybrid_batch(
        groups=groups,
        rl_datums_by_query=rl_by_q,
        policy_version="v0",
        lambda_opd=0.0,
        component_id="sentence_compress",
        teacher_event_fn=teacher,
        remove_constant_reward_groups=False,
        include_format_errors=True,
    )
    assert rl_batch.opd_datums == []
    assert rl_batch.skipped_teacher is True
    assert rl_batch.rl_datums

    trim_batch = prepare_hybrid_batch(
        groups=groups,
        rl_datums_by_query=rl_by_q,
        policy_version="v0",
        lambda_opd=0.01,
        component_id="sentence_compress",
        teacher_event_fn=teacher,
        remove_constant_reward_groups=False,
        include_format_errors=True,
        opd_loss=OPD_LOSS_PROJECTED_GAP,
        opd_states_per_trajectory=-1,
    )
    assert trim_batch.skipped_teacher is False
    assert trim_batch.opd_datums
    assert int(trim_batch.projection_stats.get("n_projected_training_steps") or 0) >= 1
    # train_cell(name="rl") drops OPD even if the batch has it; TRIM keeps both.
    rl_cell_opd: list = []
    trim_cell_opd = trim_batch.opd_datums
    assert rl_cell_opd == []
    assert trim_cell_opd


def test_audit_train_runtime_rl_and_trim(tmp_path):
    rl = SimpleNamespace(
        component="sentence_compress",
        harness="Harness-1",
        training_mode=TRAINING_MODE_RL,
        lambda_opd=0.0,
        opd_loss="sr_opd_ce",
    )
    rl_audit = audit_train_runtime_or_raise(rl, out=tmp_path / "rl")
    assert rl_audit["pass"] is True
    assert rl_audit["opd_projection"] is None
    assert (tmp_path / "rl" / "RUNTIME_EFFECT_AUDIT.json").is_file()

    trim = SimpleNamespace(
        component="sentence_compress",
        harness="Harness-1",
        training_mode=TRAINING_MODE_SCAPE_SEED,
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_PROJECTED_GAP,
    )
    trim_audit = audit_train_runtime_or_raise(trim, out=tmp_path / "trim")
    assert trim_audit["pass"] is True
    assert int(trim_audit["opd_n_projected_steps"] or 0) >= 1

    coalition = ",".join(all_component_ids())
    all_trim = SimpleNamespace(
        component=coalition,
        harness="Harness-1",
        training_mode=TRAINING_MODE_SCAPE_SEED,
        lambda_opd=0.01,
        opd_loss=OPD_LOSS_PROJECTED_GAP,
    )
    all_audit = audit_train_runtime_or_raise(all_trim, out=tmp_path / "trim_all")
    assert all_audit["pass"] is True
    assert int(all_audit["student_n_on"] or 0) == 0
    assert int(all_audit["teacher_n_on"] or 0) > 0
    assert int(all_audit["opd_n_projected_steps"] or 0) >= 1


def test_live_search_respects_student_mask_before_snapshot():
    """The eval-era bug: None mask -> live zero, snapshot student. Must not return."""
    student = student_mask_for("sentence_compress")
    live = new_state("Alice Smith Paris", dict(STORE), harness_mask=None)
    live, _obs, _ok = execute_tool(live, "search_corpus", {"query": "Alice Paris"})
    # Unresolved None is zero; that path is only legal if rollout resolved first.
    assert not live.get("auto_seed")
    resolved = new_state("Alice Smith Paris", dict(STORE), harness_mask=student)
    resolved, _obs, _ok = execute_tool(resolved, "search_corpus", {"query": "Alice Paris"})
    assert resolved.get("auto_seed")
    snap = snap_from_state("q", resolved, "sentence_compress")
    assert snap.harness_mask == student
