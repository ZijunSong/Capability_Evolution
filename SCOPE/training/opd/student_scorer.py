"""Student scoring utilities."""

from __future__ import annotations

from training.opd._policy_backend import OPDTransition, TrainBackend


class StudentScorer:
    def __init__(self, student: TrainBackend) -> None:
        self.student = student

    def score_transition(self, transition: OPDTransition) -> list[float]:
        return self.student.score_tokens(
            transition.student_input_ids, transition.action_ids
        )
