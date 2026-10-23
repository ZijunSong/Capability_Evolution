"""Deterministic composition of bank components over the H0 loop hooks.

Hook order is ``(hook_order, component_id)``.  A component that raises is
recorded as a ``component_runtime_error`` event and disabled for the rest of
the episode — it is never silently ignored while still counted as active.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from rsi.contracts.action import Action, Finish
from rsi.contracts.errors import ComponentRuntimeError
from rsi.contracts.events import Event
from rsi.contracts.common import JSONValue
from rsi.harness.component import (
    EpisodeContext,
    FinishEffect,
    ProposalEffect,
    PublicView,
    RuntimeAPI,
    StatePatch,
    ViewPatch,
)
from rsi.harness.loader import LoadedBank, LoadedComponent


@dataclass
class CompositionState:
    active: dict[str, bool] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    trigger_counts: dict[str, int] = field(default_factory=dict)

    def record_trigger(self, component_id: str) -> None:
        self.trigger_counts[component_id] = self.trigger_counts.get(component_id, 0) + 1


class HarnessComposition:
    """Runs the ordered component pipeline for one episode."""

    def __init__(self, bank: LoadedBank, api: RuntimeAPI):
        self.bank = bank
        self.api = api
        self.state = CompositionState(
            active={c.manifest.id: True for c in bank.components},
            errors={},
            trigger_counts={},
        )
        self._components: tuple[LoadedComponent, ...] = bank.components

    def initialize_all(self, ctx: EpisodeContext) -> None:
        for comp in self._components:
            self._guard(comp, "initialize", lambda c=comp: c.instance.initialize(ctx))

    def _guard(self, comp: LoadedComponent, hook: str, fn) -> Any:
        if not self.state.active.get(comp.manifest.id, False):
            return None
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001 - component code
            self.state.active[comp.manifest.id] = False
            self.state.errors[comp.manifest.id] = f"{hook}: {e!r}"
            self.api.log(
                "component_runtime_error",
                component_id=comp.manifest.id,
                hook=hook,
                error=repr(e),
            )
            raise ComponentRuntimeError(
                f"component {comp.manifest.id} failed in {hook}: {e!r}"
            ) from e
        self.state.record_trigger(comp.manifest.id)
        return result

    def apply_before_model(self, view: PublicView) -> PublicView:
        instructions: list[str] = []
        keep: set[int] | None = None
        extra: dict[str, JSONValue] = {}
        for comp in self._components:
            patch = self._guard(
                comp,
                "before_model",
                lambda c=comp: c.instance.before_model(view, self.api),
            )
            if patch is None:
                continue
            if patch.append_instructions:
                instructions.append(patch.append_instructions)
            if patch.history_filter is not None:
                wanted = set(patch.history_filter)
                keep = wanted if keep is None else (keep & wanted)
            extra.update(dict(patch.extra))
        history = view.history
        if keep is not None:
            history = tuple(e for e in view.history if e.order in keep)
        if not instructions and history is view.history and not extra:
            return view
        merged_extra = dict(view.extra)
        merged_extra.update(extra)
        if instructions:
            merged_extra["appended_instructions"] = "\n".join(instructions)
        return PublicView(
            task=view.task,
            tool_catalog=view.tool_catalog,
            history=history,
            remaining_action_budget=view.remaining_action_budget,
            extra=merged_extra,
        )

    def apply_after_proposal(self, action: Action, view: PublicView) -> ProposalEffect:
        current = action
        veto = False
        veto_reason: str | None = None
        notes: dict[str, JSONValue] = {}
        for comp in self._components:
            eff = self._guard(
                comp,
                "after_proposal",
                lambda c=comp, a=current: c.instance.after_proposal(a, view, self.api),
            )
            if eff is None:
                continue
            if eff.action is not None:
                current = eff.action
            if eff.veto:
                veto = True
                veto_reason = eff.veto_reason or "vetoed"
            notes.update(dict(eff.notes))
        return ProposalEffect(action=current if current is not action else None, veto=veto, veto_reason=veto_reason, notes=notes)

    def apply_after_observation(self, event: Event) -> None:
        for comp in self._components:
            patch = self._guard(
                comp,
                "after_observation",
                lambda c=comp: c.instance.after_observation(event, self.api),
            )
            if patch is None:
                continue
            if patch.set:
                self.api.set_state({comp.manifest.id: dict(patch.set)})
            if patch.notes:
                self.api.log("component_note", component_id=comp.manifest.id, **dict(patch.notes))

    def apply_before_finish(self, action: Finish, view: PublicView) -> FinishEffect:
        block = False
        reason: str | None = None
        new_action: Finish | None = None
        for comp in self._components:
            eff = self._guard(
                comp,
                "before_finish",
                lambda c=comp: c.instance.before_finish(action, view, self.api),
            )
            if eff is None:
                continue
            if eff.block:
                block = True
                reason = eff.block_reason or "blocked"
            if eff.action is not None:
                new_action = eff.action
        return FinishEffect(block=block, block_reason=reason, action=new_action)

    def active_component_ids(self) -> tuple[str, ...]:
        return tuple(sorted(cid for cid, on in self.state.active.items() if on))

    def trigger_report(self) -> Mapping[str, int]:
        return dict(self.state.trigger_counts)
