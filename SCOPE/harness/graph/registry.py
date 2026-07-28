"""Module registry and default module graph."""

from __future__ import annotations

from harness.graph.module import HarnessModule, ModuleConfig
from harness.harness_config import HarnessConfig
from harness.modules.context_budget import build_context_budget_module
from harness.modules.evidence_state import build_evidence_state_module
from harness.modules.recovery import build_recovery_module
from harness.modules.retrieval import build_retrieval_module
from harness.modules.verification import build_verification_module


class ModuleRegistry:
    """Registry of all Harness modules for a resolved configuration."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.modules: dict[str, HarnessModule] = {
            "retrieval": build_retrieval_module(config.retrieval),
            "evidence_state": build_evidence_state_module(config.evidence_state),
            "verification": build_verification_module(config.verification),
            "context_budget": build_context_budget_module(config.context_budget),
            "recovery": build_recovery_module(config.recovery),
        }

    def get(self, module_id: str) -> HarnessModule:
        if module_id not in self.modules:
            raise KeyError(f"Unknown module: {module_id}")
        return self.modules[module_id]

    def all_modules(self) -> list[HarnessModule]:
        return list(self.modules.values())

    def lifecycle_managed_modules(self) -> list[HarnessModule]:
        return [m for m in self.modules.values() if m.config.lifecycle_managed]

    @classmethod
    def from_config(cls, config: HarnessConfig) -> ModuleRegistry:
        return cls(config)
