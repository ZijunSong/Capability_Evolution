"""Harness module definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.graph.node import HarnessNode


@dataclass
class ModuleConfig:
    module_id: str
    enabled: bool = True
    lifecycle_managed: bool = False
    required: bool = False
    node_overrides: dict[str, bool] = field(default_factory=dict)
    fallback_mode: str = "minimal"
    options: dict[str, Any] = field(default_factory=dict)

    def is_node_enabled(self, node_id: str, default: bool = True) -> bool:
        if not self.enabled:
            return False
        if node_id in self.node_overrides:
            return self.node_overrides[node_id]
        return default


@dataclass
class HarnessModule:
    module_id: str
    nodes: list[HarnessNode]
    config: ModuleConfig

    def node(self, node_id: str) -> HarnessNode | None:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
