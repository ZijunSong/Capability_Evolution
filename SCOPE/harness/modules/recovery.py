"""M4: Recovery control module (Phase 2 stubs)."""

from __future__ import annotations

from typing import Any

from harness.graph.execution_context import ExecutionContext
from harness.graph.module import HarnessModule, ModuleConfig
from harness.graph.node import HarnessNode, NodeResult

RECOVERY_NODE_IDS = ("X1", "X2", "X3", "X4", "X5", "X6")


class _RecoveryStubNode(HarnessNode):
    def __init__(self, node_id: str) -> None:
        super().__init__(enabled=False)
        self.node_id = node_id
        self.module_id = "recovery"

    def run(self, payload: Any, context: ExecutionContext) -> NodeResult:
        return NodeResult(output=None, changed_state=False)

    def fallback(self, payload: Any, context: ExecutionContext) -> NodeResult:
        return NodeResult(output=None, metadata={"stub": True})


def build_recovery_module(config: ModuleConfig) -> HarnessModule:
    nodes = [_RecoveryStubNode(node_id) for node_id in RECOVERY_NODE_IDS]
    return HarnessModule(module_id="recovery", nodes=nodes, config=config)
