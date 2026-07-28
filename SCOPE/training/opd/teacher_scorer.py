"""Teacher scoring for OPD (frozen ref TrainBackend)."""

from __future__ import annotations

from typing import Any

from training.opd._policy_backend import OPDTransition, TrainBackend


def teacher_advantage_weights(
    student_logps: list[float],
    teacher_logps: list[float],
    *,
    max_weight: float = 5.0,
) -> list[float]:
    weights = []
    for s_lp, t_lp in zip(student_logps, teacher_logps):
        adv = t_lp - s_lp
        weights.append(max(0.0, min(max_weight, adv)))
    return weights


class TeacherScorer:
    def __init__(self, teacher: TrainBackend) -> None:
        self.teacher = teacher

    def score_transition(
        self,
        transition: OPDTransition,
        *,
        weighting: str = "teacher-advantage",
    ) -> dict[str, Any]:
        student_logps = self.teacher.score_tokens(
            transition.student_input_ids, transition.action_ids
        )
        teacher_logps = self.teacher.score_tokens(
            transition.teacher_input_ids, transition.action_ids
        )
        if weighting == "uniform":
            weights = [1.0 if m else 0.0 for m in transition.action_mask]
        elif weighting == "teacher-advantage":
            raw = teacher_advantage_weights(student_logps, teacher_logps)
            weights = [w if m else 0.0 for w, m in zip(raw, transition.action_mask)]
        else:
            weights = [
                1.0 if m and transition.success else 0.0
                for m in transition.action_mask
            ]
        return {
            "student_logps": student_logps,
            "teacher_logps": teacher_logps,
            "weights": weights,
        }
