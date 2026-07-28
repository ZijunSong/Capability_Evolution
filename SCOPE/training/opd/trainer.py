"""OPD trainer: TrainBackend for student + frozen ref teacher scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.opd._policy_backend import MockTrainBackend, OPDTransition, TrainBackend
from training.opd.loss import compute_opd_loss, compute_sampled_nll_loss
from training.opd.replay_buffer import OPDReplayBuffer
from training.opd.teacher_scorer import TeacherScorer


class OPDTrainer:
    def __init__(
        self,
        student: TrainBackend | None = None,
        teacher: TrainBackend | None = None,
        *,
        output_dir: Path | None = None,
    ) -> None:
        self.student = student or MockTrainBackend()
        self.teacher = teacher or self.student
        self.teacher_scorer = TeacherScorer(self.teacher)
        self.buffer = OPDReplayBuffer()
        self.output_dir = output_dir or Path("outputs/opd")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_transitions(self, transitions: list[OPDTransition]) -> None:
        self.buffer.extend(transitions)

    def train_epoch(
        self,
        *,
        batch_size: int = 8,
        weighting: str = "teacher-advantage",
        use_kl: bool = False,
    ) -> dict[str, float]:
        batch = self.buffer.sample(batch_size)
        if not batch:
            return {"loss": 0.0, "batch_size": 0.0}

        losses = []
        for transition in batch:
            scored = self.teacher_scorer.score_transition(
                transition, weighting=weighting
            )
            student_logps = self.student.score_tokens(
                transition.student_input_ids, transition.action_ids
            )
            if use_kl:
                loss = compute_opd_loss(
                    student_logps,
                    scored["teacher_logps"],
                    scored["weights"],
                )
            else:
                loss = compute_sampled_nll_loss(student_logps, scored["weights"])
            losses.append(loss)

        mean_loss = sum(losses) / len(losses)
        metrics = self.student.train_step(batch, {"loss": mean_loss})
        metrics["opd_loss"] = mean_loss
        return metrics

    def save_checkpoint(self, name: str = "checkpoint.json") -> Path:
        path = self.output_dir / name
        path.write_text(
            json.dumps({"transitions": len(self.buffer), "status": "saved"}),
            encoding="utf-8",
        )
        return path
