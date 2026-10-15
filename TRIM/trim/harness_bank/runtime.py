"""Per-episode bank runtime. Budgets and private state live on the instance."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from trim.harness_bank.hooks import HookBudget, call_hook, hooks_enabled
from trim.harness_bank.loader import instantiate_components
from trim.harness_bank.types import BankSpec, ComponentSpec
from trim.rsi.canonical import sha256_json


class EpisodeBankRuntime:
    def __init__(self, bank: BankSpec, *, role: str) -> None:
        self.bank = bank
        self.role = str(role)
        self.components = instantiate_components(bank) if hooks_enabled(self.role) else []
        self.specs: dict[str, ComponentSpec] = {spec.id: spec for spec in bank.components}
        self.private: dict[str, dict[str, Any]] = {spec.id: {} for spec in bank.components}
        self.budgets = {spec.id: HookBudget(spec) for spec in bank.components}
        self.events: list[dict[str, Any]] = []
        self.visible: dict[str, Any] = {}
        self.cost = {
            "extra_llm_calls": 0,
            "extra_tool_calls": 0,
            "component_wall_s": 0.0,
            "hidden_tool_calls": 0,
        }

    def bind_visible(self, visible_state: dict[str, Any]) -> None:
        self.visible = visible_state

    def on_reset(self, visible_state: dict[str, Any]) -> dict[str, Any]:
        self.visible = visible_state
        return self._dispatch("on_reset", visible_state)

    def render(self, visible_state: dict[str, Any], *, role: str | None = None) -> dict[str, Any]:
        active_role = self.role if role is None else str(role)
        rendered = deepcopy(visible_state)
        rendered.pop("teacher_guidance", None)
        if not hooks_enabled(active_role):
            return rendered
        effect = self._dispatch("before_model", visible_state)
        guidance = effect.get("teacher_guidance")
        if guidance:
            rendered["teacher_guidance"] = guidance
        return rendered

    def step(self, action: dict[str, Any], *, role: str | None = None) -> dict[str, Any]:
        active_role = self.role if role is None else str(role)
        if not hooks_enabled(active_role):
            return {"events": [], "cost": dict(self.cost)}
        phase = "before_finish" if str(action.get("name") or "") == "end_search" else "after_tool"
        effect = self._dispatch(phase, self.visible, tool_result=action, proposed_action=action)
        return {"events": list(self.events), "effect": effect, "cost": dict(self.cost)}

    def export_state(self) -> dict[str, Any]:
        return {
            "bank_sha256": self.bank.bank_sha256,
            "role": self.role,
            "private": deepcopy(self.private),
            "events": deepcopy(self.events),
            "cost": dict(self.cost),
            "budgets": {
                cid: {"llm_used": budget.llm_used, "tool_used": budget.tool_used, "exceeded": budget.exceeded}
                for cid, budget in self.budgets.items()
            },
        }

    def fork_for_teacher(self, student_snapshot: dict[str, Any]) -> "EpisodeBankRuntime":
        """New runtime. Private state is replayed from student-visible events only."""
        forked = EpisodeBankRuntime(self.bank, role="teacher")
        visible = deepcopy(student_snapshot.get("visible") or student_snapshot)
        history = list(student_snapshot.get("visible_events") or [])
        forked.on_reset(visible)
        for event in history:
            if int(event.get("step") or 0) > int(visible.get("step") or 0):
                raise RuntimeError("teacher fork was given an observation from after the decision")
            forked.visible = deepcopy(event.get("visible") or visible)
            forked.step(dict(event.get("action") or {}), role="teacher")
        forked.visible = visible
        return forked

    def private_hash(self) -> str:
        return sha256_json(self.private)

    def _dispatch(
        self,
        phase: str,
        visible_state: dict[str, Any],
        *,
        tool_result: dict[str, Any] | None = None,
        proposed_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not hooks_enabled(self.role):
            return {}
        merged: dict[str, Any] = {}
        for component in self.components:
            spec = self.specs[component.component_id]
            if phase != "on_reset" and phase != spec.phase:
                continue
            budget = self.budgets[spec.id]
            if budget.exceeded:
                continue
            private = self.private.setdefault(spec.id, {})
            result = call_hook(
                component,
                spec,
                phase,
                visible_state,
                private,
                tool_result=tool_result,
                proposed_action=proposed_action,
            )
            budget.charge(result)
            self.cost["extra_llm_calls"] += int(result.get("extra_llm_calls") or 0)
            self.cost["extra_tool_calls"] += int(result.get("extra_tool_calls") or 0)
            self.cost["hidden_tool_calls"] += int(result.get("extra_tool_calls") or 0)
            for event in result.get("events") or []:
                self.events.append(dict(event))
            for key, value in result.items():
                if key in {"extra_llm_calls", "extra_tool_calls", "events"}:
                    continue
                merged[key] = value
                if key == "teacher_guidance":
                    visible_state[key] = value
        return merged


def attach_bank(state: dict[str, Any], bank: BankSpec, *, role: str) -> EpisodeBankRuntime:
    runtime = EpisodeBankRuntime(bank, role=role)
    visible = {
        "query": state.get("query") or "",
        "visible_search_history": list(state.get("tool_history") or []),
        "visible_doc_ids": list((state.get("pool") or {}).keys()) if isinstance(state.get("pool"), dict) else [],
        "curated_ids": list((state.get("curated") or {}).keys()) if isinstance(state.get("curated"), dict) else [],
        "step": int(state.get("step") or 0),
    }
    runtime.on_reset(visible)
    state["_rsi_bank"] = {
        "path": bank.root,
        "sha256": bank.bank_sha256,
        "role": role,
        "bank_id": bank.bank_id,
    }
    state["_bank_runtime"] = runtime
    return runtime


def render_guidance(state: dict[str, Any]) -> str:
    runtime = state.get("_bank_runtime")
    if runtime is None or not hooks_enabled(runtime.role):
        return ""
    rendered = runtime.render(dict(runtime.visible), role=runtime.role)
    return str(rendered.get("teacher_guidance") or "")
