"""Rollback hard-control telemetry aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RollbackTelemetryAggregator:
    events: list[dict[str, Any]] = field(default_factory=list)
    n_continue: int = 0
    n_replan: int = 0
    n_rollback: int = 0
    n_invalid_checkpoint: int = 0

    def record(self, row: dict[str, Any]) -> None:
        self.events.append(row)
        op = str(row.get("operation", ""))
        if op == "CONTINUE":
            self.n_continue += 1
        elif op == "REPLAN":
            self.n_replan += 1
        elif op == "ROLLBACK_TO":
            self.n_rollback += 1
        if row.get("invalid_checkpoint"):
            self.n_invalid_checkpoint += 1

    def summary(self) -> dict[str, Any]:
        return {
            "n_events": len(self.events),
            "n_continue": self.n_continue,
            "n_replan": self.n_replan,
            "n_rollback": self.n_rollback,
            "n_invalid_checkpoint": self.n_invalid_checkpoint,
        }

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self.events:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
