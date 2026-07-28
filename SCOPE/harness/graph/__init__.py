"""Harness graph: nodes, modules, registry, and execution context."""

from harness.graph.events import NodeEvent, NodeStatus
from harness.graph.execution_context import ExecutionContext
from harness.graph.module import HarnessModule, ModuleConfig
from harness.graph.node import HarnessNode, NodeResult

__all__ = [
    "ExecutionContext",
    "HarnessModule",
    "HarnessNode",
    "ModuleConfig",
    "NodeEvent",
    "NodeResult",
    "NodeStatus",
]


def __getattr__(name: str):
    if name == "ModuleRegistry":
        from harness.graph.registry import ModuleRegistry
        return ModuleRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
