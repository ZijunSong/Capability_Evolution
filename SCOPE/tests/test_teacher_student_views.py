"""Tests for teacher/student views."""

from __future__ import annotations

import pytest

from harness.modules.verification import VerificationRecord
from harness.views.privileged_artifacts import PrivilegedArtifacts
from harness.views.student_view import StudentView
from harness.views.teacher_view import TeacherView
from training.opd.shadow_harness import ShadowHarness
from harness.harness_config import config_path, load_harness_config


class _StubWorkingMemory:
    def __init__(self, query: str):
        self.query = query
        self.verification_records: list = []
        self.pool_ids: list = []

    def get_structured_state(self, **kwargs) -> str:
        text = f"Query: {self.query}"
        if kwargs.get("include_verification") and self.verification_records:
            text += "\n" + "\n".join(
                r.compact_text() for r in self.verification_records
            )
        return text


def test_student_hides_verification_rationale():
    wm = _StubWorkingMemory("query")
    wm.verification_records.append(
        VerificationRecord(
            turn_id=1,
            claim="test claim",
            doc_ids=["d1"],
            judgments={"d1": True},
            rationales={"d1": "secret rationale"},
        )
    )
    view = StudentView.from_episode_state("query", wm, "recent", include_verification=False)
    assert "secret rationale" not in view.render()


def test_teacher_includes_privileged_artifact():
    student = StudentView(query="q", recent_trajectory="action")
    artifact = PrivilegedArtifacts(
        module_id="verification",
        turn_id=1,
        compact_text="claim supported",
        future_leakage=False,
    )
    teacher = TeacherView(student_view=student, privileged_artifacts=[artifact])
    assert "claim supported" in teacher.render()


def test_shadow_does_not_mutate_student_wm():
    wm = _StubWorkingMemory("query")
    original_pool = list(wm.pool_ids)
    shadow = ShadowHarness(load_harness_config(config_path("modules_full.yaml")), offline=True)
    shadow.run_verification_shadow(
        turn_id=0,
        claim="claim",
        doc_ids=["d1"],
        doc_texts={"d1": "text"},
        student_wm=wm,
    )
    assert wm.pool_ids == original_pool
    assert wm.verification_records == []


def test_reject_future_leakage():
    artifact = PrivilegedArtifacts(
        module_id="verification",
        turn_id=1,
        compact_text="leaked",
        future_leakage=True,
    )
    teacher = TeacherView(
        student_view=StudentView("q", ""),
        privileged_artifacts=[artifact],
    )
    with pytest.raises(ValueError):
        teacher.render()
