from __future__ import annotations

from typing import Any

from .component_registry import audit_component, get_component_spec, list_component_specs
from .types import ComponentTransitionRecord


class SCAPEComponentOPD:
    method_name = "scape_component_opd"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        component = self.config.get("component") or {}
        self.component_name = component.get("name") or (component.get("names") or ["evidence_graph"])[0]
        self.spec = get_component_spec(self.component_name)

    def audit(self, *, event_support: int | None = None, student_has_tool: bool = False) -> dict[str, Any]:
        return audit_component(self.component_name, event_support=event_support, student_has_tool=student_has_tool)

    def build_transition_record(self, **kwargs: Any) -> ComponentTransitionRecord:
        kwargs.setdefault("component_name", self.spec.name)
        kwargs.setdefault("component_effect_type", self.spec.effect_type)
        kwargs.setdefault("realizability", self.spec.realizability)
        return ComponentTransitionRecord(**kwargs)

    @staticmethod
    def list_components() -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "effect_type": spec.effect_type,
                "realizability": spec.realizability,
                "default_loss_mode": spec.default_loss_mode,
                "mechanism_metrics": list(spec.mechanism_metrics),
            }
            for spec in list_component_specs()
        ]
