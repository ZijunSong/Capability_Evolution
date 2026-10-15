from pathlib import Path

import pytest

from trim.harness_bank.loader import load_bank
from trim.harness_bank.shadow import StudentEnv, fork_teacher, restore_private_state
from trim.harness_bank.runtime import EpisodeBankRuntime
from tests.rsi_support import write_bank


def test_teacher_search_and_curate_do_not_touch_student(tmp_path: Path):
    bank = load_bank(write_bank(tmp_path / "bank", component_ids=["evo.query_repair"]))
    student = StudentEnv(
        {
            "query": "revenue",
            "pool": {"d1": {"id": "d1"}},
            "curated": {},
            "files": {"notes.txt": "student"},
            "sampled_action": {"name": "search_corpus", "arguments": {"query": "revenue"}},
            "step": 1,
        },
        reward=0.25,
    )
    before = student.export()
    fork = fork_teacher(student, bank)
    fork.search("revenue filing", ["d9"])
    fork.curate(["d9"])
    fork.files["notes.txt"] = "teacher"
    assert student.export() == before
    assert fork.student_unchanged(student)
    assert "d9" in fork.visible["pool"]
    assert "d9" not in student.visible["pool"]


def test_bank_teacher_runs_hooks_without_mutating_the_snapshot(tmp_path: Path):
    from trim.adapters.components import zero_mask
    from trim.state.snapshot import EnvironmentSnapshot
    from trim.training.four_cell_runtime import teacher_events_from_wm_for, teacher_for
    from trim.training.rl_opd_types import StudentDecisionPoint

    bank_dir = write_bank(tmp_path / "bank", component_ids=["evo.query_repair"])
    snap = EnvironmentSnapshot(
        query_id="q",
        step=1,
        harness_mask=zero_mask("Harness-1"),
        working_memory={"query": "revenue", "accessible_doc_ids": ["d1"], "curated_ids": []},
        query_text="revenue",
    )
    before = snap.content_hash()
    point = StudentDecisionPoint(
        episode_id="ep",
        query_id="q",
        rollout_idx=0,
        turn_id=1,
        policy_version="v1",
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=before,
        student_model_input=None,
        student_action_tokens=[1],
        student_action_text="",
        action_tool_names=["search_corpus"],
    )
    branch = teacher_for("all", bank=str(bank_dir))(point)
    assert branch.teacher_condition["continuous"] is True
    assert branch.teacher_condition["teacher_guidance"]
    assert branch.triggered is True
    assert snap.content_hash() == before
    with pytest.raises(Exception, match="UNREGISTERED_BANK_COMPONENT"):
        teacher_events_from_wm_for("all", {}, bank=str(bank_dir))


def test_future_observation_is_rejected(tmp_path: Path):
    bank = load_bank(write_bank(tmp_path / "bank", component_ids=["evo.query_repair"]))
    runtime = EpisodeBankRuntime(bank, role="teacher")
    with pytest.raises(RuntimeError, match="after decision"):
        restore_private_state(runtime, [{"step": 4, "action": {"name": "search_corpus"}}], step=2)
