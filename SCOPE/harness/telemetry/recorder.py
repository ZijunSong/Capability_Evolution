"""Episode telemetry recorder."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from harness.graph.execution_context import ExecutionContext
from harness.harness_config import HarnessConfig
from harness.telemetry.jsonl_writer import JsonlWriter
from harness.telemetry.schema import EpisodeTelemetry, TurnTelemetry


class TelemetryRecorder:
    def __init__(
        self,
        *,
        query_id: str,
        model_id: str,
        harness_config: HarnessConfig,
        output_dir: Path | None = None,
    ) -> None:
        self.episode_id = str(uuid.uuid4())
        self.query_id = query_id
        self.model_id = model_id
        self.harness_config = harness_config
        self.output_dir = output_dir
        self.turns: list[TurnTelemetry] = []
        self.task_metrics: dict[str, Any] = {}
        self.cost_metrics: dict[str, Any] = {}
        self.behavior_metrics: dict[str, Any] = {}

    def start_turn(self, turn_id: int, student_observation: str = "") -> TurnTelemetry:
        turn = TurnTelemetry(turn_id=turn_id, student_observation=student_observation)
        self.turns.append(turn)
        return turn

    def record_context(self, context: ExecutionContext, turn: TurnTelemetry) -> None:
        turn.node_events = context.turn_node_events()
        if context.working_memory is not None and hasattr(
            context.working_memory, "snapshot"
        ):
            turn.working_memory_snapshot = context.working_memory.snapshot().__dict__
        context.reset_turn_events()

    def set_task_metrics(self, metrics: dict[str, Any]) -> None:
        self.task_metrics = metrics

    def compute_behavior_metrics(self) -> dict[str, Any]:
        queries: list[str] = []
        verify_calls = 0
        for turn in self.turns:
            for tc in turn.tool_calls:
                name = tc.get("name", "")
                if name == "search_corpus":
                    args = tc.get("arguments", {})
                    q = args.get("query") if isinstance(args, dict) else None
                    if q:
                        queries.append(str(q))
                if name == "verify":
                    verify_calls += 1
        repeated = 0
        if queries:
            repeated = len(queries) - len(set(queries))
        self.behavior_metrics = {
            "query_repeat_count": repeated,
            "verify_call_count": verify_calls,
            "turn_count": len(self.turns),
            "forced_max_turn_end": self.task_metrics.get("forced_max_turn_end", False),
        }
        return self.behavior_metrics

    def finalize(self) -> EpisodeTelemetry:
        self.compute_behavior_metrics()
        return EpisodeTelemetry(
            episode_id=self.episode_id,
            query_id=self.query_id,
            model_id=self.model_id,
            module_config_hash=self.harness_config.config_hash(),
            task_metrics=self.task_metrics,
            cost_metrics=self.cost_metrics,
            behavior_metrics=self.behavior_metrics,
            turns=self.turns,
        )

    def save(self, path: Path | None = None) -> Path:
        episode = self.finalize()
        out = path or (self.output_dir / f"{self.query_id}.jsonl")
        assert out is not None
        writer = JsonlWriter(out)
        writer.write(episode.to_dict())
        return out
