"""Shadow module registry."""

from __future__ import annotations

from harness.shadow.base import ShadowModule
from harness.shadow.budget_shadow import BudgetShadow
from harness.shadow.evidence_shadow import EvidenceShadow
from harness.shadow.verification_shadow import VerificationShadow


class ShadowRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, ShadowModule] = {}

    def register(self, module: ShadowModule) -> None:
        self._modules[module.module_id] = module

    def get(self, module_id: str) -> ShadowModule:
        if module_id not in self._modules:
            raise KeyError(f"Unknown shadow module: {module_id}")
        return self._modules[module_id]

    def has(self, module_id: str) -> bool:
        return module_id in self._modules

    def ids(self) -> list[str]:
        return list(self._modules.keys())


def build_default_registry(
    *,
    evidence_state: bool = True,
    verification: bool = True,
    budget_control: bool = False,
) -> ShadowRegistry:
    reg = ShadowRegistry()
    if verification:
        reg.register(VerificationShadow())
    if evidence_state:
        reg.register(EvidenceShadow())
    if budget_control:
        reg.register(BudgetShadow())
    return reg
