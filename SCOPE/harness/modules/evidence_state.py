"""M1: Evidence state module."""

from __future__ import annotations

from typing import Any

from harness.graph.execution_context import ExecutionContext
from harness.graph.module import HarnessModule, ModuleConfig
from harness.graph.node import HarnessNode, NodeResult

EVIDENCE_NODE_IDS = (
    "E1",
    "E2",
    "E3",
    "E4",
    "E5",
    "E6",
    "E7",
    "E8",
    "E9",
)

_NODE_OPTION_MAP = {
    "E1": "preserve_minimal_selection",
    "E2": "candidate_pool",
    "E3": "content_dedup",
    "E4": "evidence_graph",
    "E5": "importance_tagging",
    "E6": "subtractive_curation",
    "E7": "auto_seed",
    "E8": "review_memory",
    "E9": "render_structured_state",
}


class _EvidenceStateNode(HarnessNode):
    def __init__(self, node_id: str, *, enabled: bool) -> None:
        super().__init__(enabled=enabled)
        self.node_id = node_id
        self.module_id = "evidence_state"

    def run(self, payload: Any, context: ExecutionContext) -> NodeResult:
        wm = context.working_memory
        if wm is None:
            return NodeResult(output=payload, changed_state=False)
        if self.node_id == "E9" and hasattr(wm, "get_structured_state"):
            output = wm.get_structured_state()
        elif self.node_id == "E1" and hasattr(wm, "get_minimal_state"):
            output = wm.get_minimal_state()
        else:
            output = payload
        return NodeResult(output=output, changed_state=self.node_id != "E9")

    def fallback(self, payload: Any, context: ExecutionContext) -> NodeResult:
        wm = context.working_memory
        if self.node_id == "E1" and wm is not None and hasattr(wm, "get_minimal_state"):
            return NodeResult(output=wm.get_minimal_state(), changed_state=False)
        if self.node_id == "E9":
            # Minimal: only curated selection in context
            if wm is not None and hasattr(wm, "get_minimal_state"):
                return NodeResult(output=wm.get_minimal_state(), changed_state=False)
        return NodeResult(output=payload, metadata={"fallback_used": True})


def build_evidence_state_module(config: ModuleConfig) -> HarnessModule:
    nodes: list[HarnessNode] = []
    for node_id in EVIDENCE_NODE_IDS:
        opt_key = _NODE_OPTION_MAP[node_id]
        default = config.options.get(opt_key, True)
        if node_id == "E1":
            default = True  # always preserve minimal selection
        enabled = config.enabled and config.options.get(opt_key, default)
        nodes.append(_EvidenceStateNode(node_id, enabled=bool(enabled)))
    return HarnessModule(module_id="evidence_state", nodes=nodes, config=config)
