"""Per-episode execution context for modular Harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from harness.graph.events import ModuleMetrics, NodeEvent

if TYPE_CHECKING:
    from harness.harness_config import HarnessConfig


@dataclass
class ExecutionContext:
    episode_id: str
    query_id: str
    turn_id: int = 0
    harness_config: HarnessConfig | None = None
    working_memory: Any = None
    node_events: list[NodeEvent] = field(default_factory=list)
    module_metrics: dict[str, ModuleMetrics] = field(default_factory=dict)
    unavailable_tool_calls: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)

    def record_node_event(self, **kwargs: Any) -> None:
        event = NodeEvent(**kwargs)
        self.node_events.append(event)
        metrics = self.module_metrics.setdefault(
            event.module_id,
            ModuleMetrics(module_id=event.module_id),
        )
        metrics.module_invocation_count += 1
        metrics.module_latency_ms += event.latency_ms
        metrics.module_output_tokens += event.token_delta
        if event.changed_state:
            metrics.module_state_changes += 1
        if event.fallback_used:
            metrics.module_fallback_count += 1

    def record_unavailable_tool(self, tool_name: str) -> None:
        self.unavailable_tool_calls += 1
        for module_id in ("verification", "evidence_state"):
            metrics = self.module_metrics.setdefault(
                module_id,
                ModuleMetrics(module_id=module_id),
            )
            metrics.unavailable_tool_calls += 1
        _ = tool_name

    def turn_node_events(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.node_events]

    def reset_turn_events(self) -> None:
        self.node_events.clear()
