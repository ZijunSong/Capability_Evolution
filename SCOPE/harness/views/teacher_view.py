"""Teacher view: student prefix plus privileged module artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.views.privileged_artifacts import PrivilegedArtifacts
from harness.views.student_view import StudentView


@dataclass
class TeacherView:
    student_view: StudentView
    privileged_artifacts: list[PrivilegedArtifacts] = field(default_factory=list)
    remaining_budget: int | None = None

    def render(self) -> str:
        parts = [self.student_view.render()]
        for artifact in self.privileged_artifacts:
            artifact.validate()
            if artifact.compact_text:
                parts.append(
                    f"[{artifact.module_id}@{artifact.turn_id}]\n{artifact.compact_text}"
                )
        if self.remaining_budget is not None:
            parts.append(f"Remaining budget: {self.remaining_budget} turns")
        return "\n\n".join(parts)

    def privileged_token_estimate(self) -> int:
        return sum(len(a.compact_text) for a in self.privileged_artifacts) // 4
