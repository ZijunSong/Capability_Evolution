"""Observation / tool protocol shared by environments and the H0 loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from rsi.contracts.common import JSONValue, digest_of
from rsi.contracts.action import Action


@dataclass(frozen=True)
class ToolSpec:
    """A basic tool exposed by an environment.

    ``capabilities`` are coarse tags (e.g. ``"file_write"``, ``"python_exec"``)
    used by components for conditional triggering.  They describe what the
    tool *is*, never benchmark identity.
    """

    name: str
    description: str
    parameters_schema: Mapping[str, JSONValue]
    capabilities: tuple[str, ...] = ()

    def digest(self) -> str:
        return digest_of(
            {
                "name": self.name,
                "description": self.description,
                "parameters_schema": dict(self.parameters_schema),
                "capabilities": list(self.capabilities),
            }
        )


@dataclass(frozen=True)
class ObservationProvenance:
    """Where an observation's information came from.

    Produced by the runtime/adapter, never trusted from candidate declarations.
    ``source_ref`` identifies the environment subsystem that produced the value
    (tool name + opaque resource id).  ``visible_to_student`` is set by the
    runtime according to the visibility protocol.
    """

    source_tool: str
    source_ref: str
    produced_by: str = "environment"  # environment | model | component
    visible_to_student: bool = True


@dataclass(frozen=True)
class Observation:
    content: str
    ok: bool = True
    error_kind: str | None = None
    provenance: ObservationProvenance | None = None
    structured: Mapping[str, JSONValue] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, JSONValue]:
        return {
            "content": self.content,
            "ok": self.ok,
            "error_kind": self.error_kind,
            "structured": dict(self.structured),
        }


@dataclass(frozen=True)
class SnapshotRef:
    """Reference to a complete environment snapshot.

    A snapshot must cover the mutable filesystem layer, application DB,
    interpreter/session state, tool caches, RNG and clock state.  Adapters that
    cannot do this must set ``supports_shadow_fork=False`` on the adapter (see
    EnvironmentAdapter) and must not pretend to support isolated Teacher forks.
    """

    ref_id: str
    digest: str
    kind: str = "full"  # full | replay_prefix


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    done: bool = False
    snapshot: SnapshotRef | None = None
    # Opaque, adapter-defined step cost record (tool calls, wall time, tokens).
    cost: Mapping[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreRecord:
    """Official task metric on [0, 1].  Scorer-side; never a runtime obs."""

    task_id: str
    domain: str
    metric_name: str
    value: float
    success: bool
    details: Mapping[str, JSONValue] = field(default_factory=dict)
    scorer_version: str = "v1"
    synthetic: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.value) <= 1.0):
            raise ValueError(f"score must be in [0,1], got {self.value}")


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a final artifact produced by an episode."""

    ref_id: str
    path: str | None = None
    digest: str | None = None
    kind: str = "answer"


def tool_catalog_digest(catalog: tuple[ToolSpec, ...]) -> str:
    return digest_of([{"name": t.name, "digest": t.digest()} for t in sorted(catalog, key=lambda t: t.name)])


__all__ = [
    "ToolSpec",
    "ObservationProvenance",
    "Observation",
    "SnapshotRef",
    "StepResult",
    "ScoreRecord",
    "ArtifactRef",
    "tool_catalog_digest",
    "Action",
]
