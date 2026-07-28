"""Telemetry schema for modular Harness episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TurnTelemetry:
    turn_id: int
    student_observation: str = ""
    action_tokens: list[int] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    node_events: list[dict[str, Any]] = field(default_factory=list)
    module_artifacts: dict[str, Any] = field(default_factory=dict)
    working_memory_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeTelemetry:
    episode_id: str
    query_id: str
    model_id: str
    module_config_hash: str
    task_metrics: dict[str, Any] = field(default_factory=dict)
    cost_metrics: dict[str, Any] = field(default_factory=dict)
    behavior_metrics: dict[str, Any] = field(default_factory=dict)
    turns: list[TurnTelemetry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["turns"] = [asdict(t) for t in self.turns]
        return data
