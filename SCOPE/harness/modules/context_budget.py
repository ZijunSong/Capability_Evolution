"""M3: Context and budget module."""

from __future__ import annotations

from typing import Any

from harness.graph.execution_context import ExecutionContext
from harness.graph.module import HarnessModule, ModuleConfig
from harness.graph.node import HarnessNode, NodeResult

_DEFAULT_PROMPT_TOKEN_BUDGET = 30720

CONTEXT_NODE_IDS = ("C1", "C2", "C3", "C4", "C5", "C6")

_NODE_OPTION_MAP = {
    "C1": "sentence_compression",
    "C2": "structured_context_rendering",
    "C3": "recent_window",
    "C4": "token_budget_marker",
    "C5": "deterministic_truncation",
    "C6": "stop_budget_hint",
}


class _ContextBudgetNode(HarnessNode):
    def __init__(self, node_id: str, *, enabled: bool) -> None:
        super().__init__(enabled=enabled)
        self.node_id = node_id
        self.module_id = "context_budget"

    def run(self, payload: Any, context: ExecutionContext) -> NodeResult:
        if self.node_id == "C5":
            text = payload if isinstance(payload, str) else str(payload)
            limit = int(context.artifacts.get("context_char_limit", _DEFAULT_PROMPT_TOKEN_BUDGET * 3))
            if len(text) > limit:
                text = text[:limit] + "\n...(truncated)"
            return NodeResult(output=text, changed_state=True)
        return NodeResult(output=payload, changed_state=False)

    def fallback(self, payload: Any, context: ExecutionContext) -> NodeResult:
        if self.node_id == "C5" or not self.enabled:
            text = payload if isinstance(payload, str) else str(payload)
            limit = int(context.artifacts.get("context_char_limit", _DEFAULT_PROMPT_TOKEN_BUDGET * 3))
            if len(text) > limit:
                text = text[:limit] + "\n...(truncated)"
            return NodeResult(
                output=text,
                metadata={"fallback_used": True, "deterministic_truncation": True},
            )
        return NodeResult(output=payload, metadata={"fallback_used": True})


def build_context_budget_module(config: ModuleConfig) -> HarnessModule:
    nodes: list[HarnessNode] = []
    for node_id in CONTEXT_NODE_IDS:
        opt_key = _NODE_OPTION_MAP[node_id]
        if node_id == "C5":
            enabled = (not config.enabled) or config.options.get("deterministic_truncation", False)
        else:
            enabled = config.enabled and config.options.get(opt_key, True)
        nodes.append(_ContextBudgetNode(node_id, enabled=bool(enabled)))
    return HarnessModule(module_id="context_budget", nodes=nodes, config=config)
