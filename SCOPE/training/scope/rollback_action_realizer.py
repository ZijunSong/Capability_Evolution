"""Bridge rollback decisions to RollbackRuntime hard executor."""

from __future__ import annotations

from typing import Any

from harness.capability.rollback_operation import RollbackOperation
from harness.recovery.rollback_runtime import RollbackRuntime
from training.scope.decide_rollback_operation import RollbackDecision


class RollbackActionRealizer:
    def realize(
        self,
        env: Any,
        decision: RollbackDecision,
        runtime: RollbackRuntime,
    ) -> bool:
        try:
            return runtime.execute(
                env,
                decision.predicted_operation,
                checkpoint_id=decision.checkpoint_id,
            )
        except (RuntimeError, ValueError):
            return False

    def supports_operation(self, operation: RollbackOperation) -> bool:
        return operation in (
            RollbackOperation.CONTINUE,
            RollbackOperation.REPLAN,
            RollbackOperation.ROLLBACK_TO,
        )
