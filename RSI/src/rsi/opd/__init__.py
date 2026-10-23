"""HarnessOPD: teacher forks, alignment, realizability, scoring, losses."""

from __future__ import annotations

from rsi.opd.alignment import AlignmentContext, align_decision
from rsi.opd.losses import (
    cispo_token_loss,
    gated_projected_gap_loss,
    joint_loss,
)
from rsi.opd.records import OPDRow, RLRow, build_opd_row, build_rl_row
from rsi.opd.teacher import TeacherBranch, TeacherForkRunner

__all__ = [
    "AlignmentContext",
    "align_decision",
    "cispo_token_loss",
    "gated_projected_gap_loss",
    "joint_loss",
    "OPDRow",
    "RLRow",
    "build_opd_row",
    "build_rl_row",
    "TeacherBranch",
    "TeacherForkRunner",
]
