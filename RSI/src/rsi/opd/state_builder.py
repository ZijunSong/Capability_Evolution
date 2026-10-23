"""TeacherStateBuilder (task book §8.7).

The Student never ran the advanced bank, so Teacher private state cannot be
assumed to exist on the Student's trajectory.  The builder takes the legal
event prefix up to the current decision and replays the components' pure state
updates in an isolated namespace (or calls an explicit
``rebuild_from_public_history``).  The runtime keeps immutable raw public
events; rebuild never reads later events, future rewards or final answers.

If rebuilding needs extra environment tool execution, that execution must
happen inside the Teacher fork, and the resulting extra observations are
recorded — subsequent anchors still pass the §8.3 information checks.  Never
replay side-effecting history tools in the real Student environment, and never
paste a full-rollout Teacher's internal state onto a different Student
trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rsi.contracts.common import JSONValue
from rsi.contracts.events import Event
from rsi.contracts.errors import ProtocolViolation
from rsi.harness.component import EpisodeContext, PublicView
from rsi.harness.loader import LoadedBank, LoadedComponent


@dataclass
class TeacherStateBundle:
    """Isolated per-component state reconstructed from the public prefix."""

    component_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    rebuilt_from_events: int = 0
    used_explicit_rebuild: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "component_states": {k: dict(v) for k, v in self.component_states.items()},
            "rebuilt_from_events": self.rebuilt_from_events,
            "used_explicit_rebuild": list(self.used_explicit_rebuild),
            "notes": list(self.notes),
        }


class TeacherStateBuilder:
    def __init__(self, bank: LoadedBank):
        self.bank = bank

    def rebuild_from_public_history(
        self,
        *,
        public_events: Sequence[Event],
        task,
        catalog,
        forbidden_suffix_events: Sequence[Event] = (),
    ) -> TeacherStateBundle:
        if forbidden_suffix_events:
            # Defensive: callers must not hand future events in.
            raise ProtocolViolation("TeacherStateBuilder cannot read events after the decision point")
        bundle = TeacherStateBundle()
        # Fresh isolated component instances for the replay namespace.
        ctx = EpisodeContext(episode_id="teacher-state-rebuild", task=task, tool_catalog=catalog)
        for comp in self.bank.components:
            instance = comp.instance
            explicit = getattr(instance, "rebuild_from_public_history", None)
            if callable(explicit):
                state = explicit(list(public_events))
                bundle.component_states[comp.manifest.id] = dict(state or {})
                bundle.used_explicit_rebuild += (comp.manifest.id,)
                continue
            # Default: replay pure after_observation state updates in isolation.
            state: dict[str, Any] = {}
            fake_api = _ReplayAPI(state)
            instance.initialize(ctx)
            for ev in public_events:
                try:
                    patch = instance.after_observation(ev, fake_api)
                except Exception as e:  # noqa: BLE001 - component replay must not kill training
                    bundle.notes += (f"{comp.manifest.id}: replay error {e!r}",)
                    continue
                if patch is not None and patch.set:
                    state.setdefault(comp.manifest.id, {}).update(dict(patch.set))
            bundle.component_states[comp.manifest.id] = state
            bundle.rebuilt_from_events += 1
        return bundle


class _ReplayAPI:
    """Minimal RuntimeAPI stand-in for pure state replay: no tool/model calls,
    no budget spend.  Components that try side-effecting calls during rebuild
    are refused (state replay is pure by contract)."""

    def __init__(self, state: dict[str, Any]):
        self._state = state

    def call_tool(self, call):  # noqa: ANN001
        raise ProtocolViolation("state rebuild may not execute tools; use the Teacher fork")

    def call_model(self, prompt: str, *, max_new_tokens: int = 256) -> str:
        raise ProtocolViolation("state rebuild may not call models")

    def get_state(self) -> Mapping[str, Any]:
        return dict(self._state)

    def set_state(self, patch: Mapping[str, JSONValue]) -> None:
        for k, v in dict(patch).items():
            if isinstance(v, dict):
                self._state.setdefault(k, {}).update(v) if isinstance(self._state.get(k), dict) else self._state.update({k: v})
            else:
                self._state[k] = v

    def budget(self) -> Mapping[str, int]:
        return {"actions_used": 0}

    def log(self, message: str, **fields: JSONValue) -> None:
        return None
