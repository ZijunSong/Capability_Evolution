from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SCAPEComponentOPDConfig:
    component_name: str = "evidence_graph"
    loss: str = "projected_action_ce"
    event_only: bool = True
    student_inference_privilege: bool = False
    teacher_mode: str = "same_weights_privileged_view"
    reference_mode: str = "frozen_init"
    reference_coef: float = 0.05
