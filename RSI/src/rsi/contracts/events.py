"""Generic episode events.

Every runtime occurrence is an :class:`Event` with a stable id, ordering,
owning component, pre/post state digests, information provenance and cost.
Events are the substrate for OPD alignment (explicit map / native anchor /
skip / runtime_only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from rsi.contracts.common import JSONValue, digest_of
from rsi.contracts.action import Action
from rsi.contracts.observation import Observation


class EventKind(str, Enum):
    MODEL_PROPOSAL = "model_proposal"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMPONENT_HOOK = "component_hook"
    VIEW_PATCH = "view_patch"
    FINISH = "finish"
    ERROR = "error"
    EPISODE_BOUNDARY = "episode_boundary"


class Visibility(str, Enum):
    STUDENT = "student"  # in the Student's H0-visible history
    TEACHER_PRIVATE = "teacher_private"  # only inside a Teacher fork
    RUNTIME = "runtime"  # bookkeeping only


@dataclass(frozen=True)
class CostRecord:
    """All extra spend is auto-accounted by the runtime; components cannot forge it."""

    actor_input_tokens: int = 0
    actor_output_tokens: int = 0
    aux_input_tokens: int = 0
    aux_output_tokens: int = 0
    tool_calls: int = 0
    extra_llm_calls: int = 0
    wall_ms: float = 0.0

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "actor_input_tokens": self.actor_input_tokens,
            "actor_output_tokens": self.actor_output_tokens,
            "aux_input_tokens": self.aux_input_tokens,
            "aux_output_tokens": self.aux_output_tokens,
            "tool_calls": self.tool_calls,
            "extra_llm_calls": self.extra_llm_calls,
            "wall_ms": self.wall_ms,
        }


@dataclass(frozen=True)
class Event:
    event_id: str
    order: int
    component_id: str  # "h0" for base-loop events, else the component id
    event_kind: EventKind
    pre_state_digest: str
    post_state_digest: str
    action: Action | None = None
    observation: Observation | None = None
    visibility: Visibility = Visibility.STUDENT
    effect_type: str = "none"  # none | state_change | context_change | decision_change
    # Where the information in this event came from (runtime-produced).
    information_provenance: tuple[str, ...] = ()
    cost: CostRecord = field(default_factory=CostRecord)
    data: Mapping[str, JSONValue] = field(default_factory=dict)

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "event_id": self.event_id,
            "order": self.order,
            "component_id": self.component_id,
            "event_kind": self.event_kind.value,
            "pre_state_digest": self.pre_state_digest,
            "post_state_digest": self.post_state_digest,
            "action": self.action.to_json() if self.action is not None else None,
            "observation": self.observation.to_public_dict() if self.observation is not None else None,
            "visibility": self.visibility.value,
            "effect_type": self.effect_type,
            "information_provenance": list(self.information_provenance),
            "cost": self.cost.to_json(),
            "data": dict(self.data),
        }

    @staticmethod
    def make_id(episode_id: str, order: int) -> str:
        return digest_of({"episode_id": episode_id, "order": order})[:16]


def total_cost(events: Sequence[Event]) -> CostRecord:
    return CostRecord(
        actor_input_tokens=sum(e.cost.actor_input_tokens for e in events),
        actor_output_tokens=sum(e.cost.actor_output_tokens for e in events),
        aux_input_tokens=sum(e.cost.aux_input_tokens for e in events),
        aux_output_tokens=sum(e.cost.aux_output_tokens for e in events),
        tool_calls=sum(e.cost.tool_calls for e in events),
        extra_llm_calls=sum(e.cost.extra_llm_calls for e in events),
        wall_ms=sum(e.cost.wall_ms for e in events),
    )
