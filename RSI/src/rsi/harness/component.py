"""Harness component protocol.

Components are *evolved external policy code*: they may wrap/compose tools,
patch views before a model call, veto/modify proposals, absorb observations
into private state and gate finishing.  They run against a controlled
:class:`RuntimeAPI` that auto-accounts every extra call.  A component may not
silently fail and still count as enabled.

The initial bank is empty.  Nothing in this module is a pre-installed solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from rsi.contracts.action import Action, Finish, ToolCall
from rsi.contracts.common import JSONValue
from rsi.contracts.events import Event
from rsi.contracts.observation import Observation, ToolSpec
from rsi.contracts.task import PublicTask


@dataclass
class EpisodeContext:
    """Per-episode state handed to ``initialize``."""

    episode_id: str
    task: PublicTask
    tool_catalog: tuple[ToolSpec, ...]
    # Component-private state store managed by the runtime (per component id).
    private_state: dict[str, Any] = field(default_factory=dict)
    rng_seed: int = 0
    extra_llm_call_budget: int = 0


@dataclass(frozen=True)
class PublicView:
    """What the model is about to see: task + tool schemas + legal history."""

    task: PublicTask
    tool_catalog: tuple[ToolSpec, ...]
    history: tuple[Event, ...]
    remaining_action_budget: int
    extra: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewPatch:
    """Delta a component applies to the view before the model call."""

    append_instructions: str | None = None
    history_filter: tuple[int, ...] | None = None  # event orders to keep
    extra: Mapping[str, JSONValue] = field(default_factory=dict)
    changed: bool = False


@dataclass(frozen=True)
class ProposalEffect:
    """What ``after_proposal`` may do: replace, veto, or annotate an action."""

    action: Action | None = None  # None = keep original
    veto: bool = False
    veto_reason: str | None = None
    notes: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class StatePatch:
    """Private-state update after a tool observation."""

    set: Mapping[str, JSONValue] = field(default_factory=dict)
    notes: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class FinishEffect:
    """Gate/adjust the finish action.  ``block`` forces the loop to continue."""

    block: bool = False
    block_reason: str | None = None
    action: Finish | None = None


class RuntimeAPI(Protocol):
    """Controlled services for components.

    Every call through this API is metered.  Components cannot forge logs or
    costs.  ``call_model`` is restricted to the current Actor model in the main
    configuration (auxiliary external models are a separate experiment config).
    """

    def call_tool(self, call: ToolCall) -> Observation: ...
    def call_model(self, prompt: str, *, max_new_tokens: int = 256) -> str: ...
    def get_state(self) -> Mapping[str, Any]: ...
    def set_state(self, patch: Mapping[str, JSONValue]) -> None: ...
    def budget(self) -> Mapping[str, int]: ...
    def log(self, message: str, **fields: JSONValue) -> None: ...


@runtime_checkable
class HarnessComponent(Protocol):
    """The evolvable component interface.

    Implementations may be async; the composition layer normalizes sync/async
    to one convention (a sync facade that must not block on its own loop).
    """

    def initialize(self, ctx: EpisodeContext) -> None: ...

    def before_model(self, view: PublicView, api: RuntimeAPI) -> ViewPatch: ...

    def after_proposal(self, action: Action, view: PublicView, api: RuntimeAPI) -> ProposalEffect: ...

    def after_observation(self, event: Event, api: RuntimeAPI) -> StatePatch: ...

    def before_finish(self, action: Finish, view: PublicView, api: RuntimeAPI) -> FinishEffect: ...


class NullComponent:
    """No-op reference implementation (not part of any bank; tests only)."""

    def initialize(self, ctx: EpisodeContext) -> None:
        return None

    def before_model(self, view: PublicView, api: RuntimeAPI) -> ViewPatch:
        return ViewPatch(changed=False)

    def after_proposal(self, action: Action, view: PublicView, api: RuntimeAPI) -> ProposalEffect:
        return ProposalEffect(action=None)

    def after_observation(self, event: Event, api: RuntimeAPI) -> StatePatch:
        return StatePatch()

    def before_finish(self, action: Finish, view: PublicView, api: RuntimeAPI) -> FinishEffect:
        return FinishEffect()
