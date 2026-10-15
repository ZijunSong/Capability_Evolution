"""Ordered hook dispatch. Zero and student roles never enter component code."""

from __future__ import annotations

from typing import Any

from trim.harness_bank.types import ComponentSpec

_SILENT_ROLES = {"zero", "student"}


class HookBudget:
    def __init__(self, spec: ComponentSpec) -> None:
        self.llm_limit = int(spec.max_extra_llm_calls_per_episode)
        self.tool_limit = int(spec.max_extra_tool_calls_per_episode)
        self.llm_used = 0
        self.tool_used = 0
        self.exceeded = False

    def charge(self, result: dict[str, Any]) -> None:
        self.llm_used += int(result.get("extra_llm_calls") or 0)
        self.tool_used += int(result.get("extra_tool_calls") or 0)
        if self.llm_used > self.llm_limit or self.tool_used > self.tool_limit:
            self.exceeded = True


def hooks_enabled(role: str) -> bool:
    return str(role or "zero") not in _SILENT_ROLES


def call_hook(
    component: Any,
    spec: ComponentSpec,
    phase: str,
    view: Any,
    private_state: dict[str, Any],
    *,
    tool_result: dict[str, Any] | None = None,
    proposed_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase == "on_reset":
        raw = component.on_reset(view, private_state)
    elif phase == "before_model":
        raw = component.before_model(view, private_state)
    elif phase == "after_tool":
        raw = component.after_tool(view, tool_result or {}, private_state)
    elif phase == "before_finish":
        raw = component.before_finish(view, proposed_action or {}, private_state)
    else:
        raise ValueError(f"unknown hook phase {phase}")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"{spec.id}.{phase} must return a dict")
    allowed = set(spec.writes) | {"extra_llm_calls", "extra_tool_calls", "events"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{spec.id} wrote undeclared fields {unknown}")
    return raw
