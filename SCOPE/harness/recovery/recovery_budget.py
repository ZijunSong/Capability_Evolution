"""Hard rollback budget enforcement (I2, I8)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RecoveryBudget:
    max_rollbacks: int = 3
    used: int = 0
    violations: list[str] = field(default_factory=list)

    def can_rollback(self) -> bool:
        return self.used < self.max_rollbacks

    def consume(self, event_id: str) -> None:
        if not self.can_rollback():
            self.violations.append(f"budget_exceeded:{event_id}")
            raise RuntimeError(f"rollback budget exhausted before {event_id}")
        self.used += 1

    def remaining(self) -> int:
        return max(0, self.max_rollbacks - self.used)
