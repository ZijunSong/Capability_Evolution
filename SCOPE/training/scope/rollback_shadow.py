"""Rollback bilateral shadow labels for typed hard-control supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.capability.rollback_operation import RollbackOperation, RollbackReasonCode
from harness.recovery.stagnation_detector import FailureEvent


@dataclass
class RollbackShadowLabel:
    operation: RollbackOperation
    checkpoint_id: str | None
    reason_code: RollbackReasonCode
    route: str  # ENDORSE | CORRECT


class RollbackBilateralShadow:
    module_id = "rollback_decision"

    def label_failure_event(
        self,
        event: FailureEvent,
        *,
        healthy_continue: bool = False,
    ) -> RollbackShadowLabel:
        if healthy_continue or event.reason_code == RollbackReasonCode.NONE:
            return RollbackShadowLabel(
                operation=RollbackOperation.CONTINUE,
                checkpoint_id=None,
                reason_code=RollbackReasonCode.NONE,
                route="ENDORSE",
            )
        if event.suggested_checkpoint_id:
            return RollbackShadowLabel(
                operation=RollbackOperation.ROLLBACK_TO,
                checkpoint_id=event.suggested_checkpoint_id,
                reason_code=event.reason_code,
                route="CORRECT",
            )
        return RollbackShadowLabel(
            operation=RollbackOperation.REPLAN,
            checkpoint_id=None,
            reason_code=event.reason_code,
            route="CORRECT",
        )

    def label_from_dict(self, row: dict[str, Any]) -> RollbackShadowLabel:
        op = RollbackOperation(str(row.get("operation", "CONTINUE")))
        reason = RollbackReasonCode(str(row.get("reason_code", "NONE")))
        ck = row.get("checkpoint_id")
        route = str(row.get("route", "ENDORSE"))
        return RollbackShadowLabel(
            operation=op,
            checkpoint_id=str(ck) if ck else None,
            reason_code=reason,
            route=route,
        )
