"""M0: Retrieval interface module nodes."""

from __future__ import annotations

from typing import Any

from harness.graph.execution_context import ExecutionContext
from harness.graph.module import HarnessModule, ModuleConfig
from harness.graph.node import HarnessNode, NodeResult

RETRIEVAL_NODE_IDS = ("R1", "R2", "R3", "R4", "R5", "R6")


class _RetrievalPassthroughNode(HarnessNode):
    def __init__(
        self,
        node_id: str,
        *,
        enabled: bool = True,
        required: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.node_id = node_id
        self.module_id = "retrieval"
        self.required = required

    def run(self, payload: Any, context: ExecutionContext) -> NodeResult:
        return NodeResult(
            output=payload,
            metadata={"node_id": self.node_id, "enabled": True},
            changed_state=False,
        )

    def fallback(self, payload: Any, context: ExecutionContext) -> NodeResult:
        if self.required:
            return self.run(payload, context)
        return NodeResult(
            output=payload,
            metadata={"node_id": self.node_id, "fallback_used": True},
            changed_state=False,
        )


def build_retrieval_module(config: ModuleConfig) -> HarnessModule:
    nodes: list[HarnessNode] = []
    optional = {"R5", "R6"}
    for node_id in RETRIEVAL_NODE_IDS:
        enabled = config.is_node_enabled(node_id, default=True)
        if node_id in optional and not config.options.get(
            "rerank" if node_id == "R5" else "chunk_neighbors", node_id != "R6"
        ):
            enabled = False
        nodes.append(
            _RetrievalPassthroughNode(
                node_id,
                enabled=enabled,
                required=node_id not in optional,
            )
        )
    return HarnessModule(module_id="retrieval", nodes=nodes, config=config)
