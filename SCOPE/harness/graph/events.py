"""Node and module telemetry events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NodeStatus(str, Enum):
    ENABLED = "enabled"
    FALLBACK = "fallback"
    DISABLED = "disabled"


@dataclass
class NodeEvent:
    node_id: str
    module_id: str
    status: str
    invoked: bool = True
    fallback_used: bool = False
    changed_state: bool = False
    latency_ms: float = 0.0
    input_digest: str = ""
    output_digest: str = ""
    input_size: int = 0
    output_size: int = 0
    token_delta: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModuleMetrics:
    module_id: str
    module_invocation_count: int = 0
    module_latency_ms: float = 0.0
    module_output_tokens: float = 0.0
    module_state_changes: int = 0
    module_fallback_count: int = 0
    unavailable_tool_calls: int = 0
    module_artifact_usage: int = 0
    module_cost_per_success: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
