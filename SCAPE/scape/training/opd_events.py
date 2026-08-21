"""Normalized Teacher / Harness events for SR-OPD projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4


class EventKind(str, Enum):
    MODEL_ACTION = "model_action"
    TOOL_OBSERVATION = "tool_observation"
    HARNESS_MUTATION = "harness_mutation"
    OBS_TRANSFORM = "obs_transform"


@dataclass
class HarnessEvent:
    event_id: str
    turn_id: int
    kind: EventKind
    component_id: str | None = None

    action_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)

    state_delta: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] | None = None

    visible_to_student: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HarnessEvent":
        payload = dict(data)
        return cls(
            event_id=str(payload.get("event_id") or uuid4().hex),
            turn_id=int(payload.get("turn_id") or 0),
            kind=EventKind(payload["kind"]),
            component_id=payload.get("component_id"),
            action_name=payload.get("action_name") or payload.get("action"),
            arguments=dict(payload.get("arguments") or {}),
            state_delta=dict(payload.get("state_delta") or {}),
            observation=payload.get("observation"),
            visible_to_student=bool(payload.get("visible_to_student", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


def _eid(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def model_action(
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    turn_id: int = 0,
    component_id: str | None = None,
    visible_to_student: bool = True,
    state_delta: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    event_id: str | None = None,
) -> HarnessEvent:
    return HarnessEvent(
        event_id=event_id or _eid("act"),
        turn_id=turn_id,
        kind=EventKind.MODEL_ACTION,
        component_id=component_id,
        action_name=name,
        arguments=dict(arguments or {}),
        state_delta=dict(state_delta or {}),
        visible_to_student=visible_to_student,
        metadata=dict(metadata or {}),
    )


def tool_observation(
    *,
    turn_id: int = 0,
    component_id: str | None = None,
    observation: Mapping[str, Any] | None = None,
    visible_to_student: bool = False,
    metadata: Mapping[str, Any] | None = None,
    event_id: str | None = None,
) -> HarnessEvent:
    return HarnessEvent(
        event_id=event_id or _eid("obs"),
        turn_id=turn_id,
        kind=EventKind.TOOL_OBSERVATION,
        component_id=component_id,
        observation=dict(observation or {}),
        visible_to_student=visible_to_student,
        metadata=dict(metadata or {}),
    )


def harness_mutation(
    component_id: str,
    state_delta: Mapping[str, Any],
    *,
    turn_id: int = 0,
    visible_to_student: bool = False,
    metadata: Mapping[str, Any] | None = None,
    event_id: str | None = None,
) -> HarnessEvent:
    return HarnessEvent(
        event_id=event_id or _eid("mut"),
        turn_id=turn_id,
        kind=EventKind.HARNESS_MUTATION,
        component_id=component_id,
        state_delta=dict(state_delta),
        visible_to_student=visible_to_student,
        metadata=dict(metadata or {}),
    )


def obs_transform(
    component_id: str,
    *,
    turn_id: int = 0,
    state_delta: Mapping[str, Any] | None = None,
    observation: Mapping[str, Any] | None = None,
    visible_to_student: bool = False,
    metadata: Mapping[str, Any] | None = None,
    event_id: str | None = None,
) -> HarnessEvent:
    return HarnessEvent(
        event_id=event_id or _eid("xfm"),
        turn_id=turn_id,
        kind=EventKind.OBS_TRANSFORM,
        component_id=component_id,
        state_delta=dict(state_delta or {}),
        observation=dict(observation or {}),
        visible_to_student=visible_to_student,
        metadata=dict(metadata or {}),
    )
