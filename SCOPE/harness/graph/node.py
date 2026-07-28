"""Unified Harness node interface."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from harness.graph.events import NodeStatus

if TYPE_CHECKING:
    from harness.graph.execution_context import ExecutionContext


@dataclass
class NodeResult:
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    changed_state: bool = False
    cost: dict[str, float] = field(default_factory=dict)


def _digest(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        payload = repr(value)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class HarnessNode(ABC):
    """A single Harness operation unit with enabled and fallback paths."""

    node_id: str
    module_id: str

    def __init__(self, *, enabled: bool = True, on_error: str = "fallback") -> None:
        self._enabled = enabled
        self._on_error = on_error  # "fallback" | "raise"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def execute(self, payload: Any, context: "ExecutionContext") -> NodeResult:
        """Run node with telemetry; route to fallback on disable or error."""
        start = time.perf_counter()
        status = NodeStatus.ENABLED if self._enabled else NodeStatus.FALLBACK
        fallback_used = not self._enabled
        error: str | None = None
        try:
            if self._enabled:
                result = self.run(payload, context)
            else:
                result = self.fallback(payload, context)
        except Exception as exc:
            error = traceback.format_exc()
            if self._on_error == "raise":
                raise
            status = NodeStatus.FALLBACK
            fallback_used = True
            result = self.fallback(payload, context)
            result.metadata["error"] = str(exc)

        latency_ms = (time.perf_counter() - start) * 1000.0
        context.record_node_event(
            node_id=self.node_id,
            module_id=self.module_id,
            status=status.value,
            invoked=True,
            fallback_used=fallback_used,
            changed_state=result.changed_state,
            latency_ms=latency_ms,
            input_digest=_digest(payload),
            output_digest=_digest(result.output),
            input_size=_size_hint(payload),
            output_size=_size_hint(result.output),
            token_delta=float(result.metadata.get("token_delta", 0.0)),
            error=error,
        )
        return result

    @abstractmethod
    def run(self, payload: Any, context: "ExecutionContext") -> NodeResult:
        ...

    @abstractmethod
    def fallback(self, payload: Any, context: "ExecutionContext") -> NodeResult:
        ...


def _size_hint(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (list, tuple, dict, set)):
        return len(value)
    return len(str(value))
