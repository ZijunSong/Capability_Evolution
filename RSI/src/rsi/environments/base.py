"""Environment adapter & scorer protocols.

The adapter is the task world (basic tools, file/API state, legal observations,
real execution).  The scorer is a separate evaluator service that keeps
``PrivateTaskRef`` out of Actor/Teacher/Proposer workers.

Snapshotting must cover the mutable filesystem layer, application DB,
interpreter/session state, tool caches, RNG and clock simulation state.  A
control-object ``deepcopy`` sharing an external DB is not acceptable.  Adapters
that cannot snapshot/replay reliably must set ``supports_shadow_fork = False``
and must not claim isolated Teacher rollouts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rsi.contracts.action import Action
from rsi.contracts.common import JSONValue
from rsi.contracts.observation import (
    ArtifactRef,
    Observation,
    ScoreRecord,
    SnapshotRef,
    StepResult,
    ToolSpec,
)
from rsi.contracts.task import PrivateTaskRef, PublicTask


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """One task world.  Construction args carry paths/version identity only —
    never benchmark-name routing inside core code."""

    name: str
    version: str
    supports_shadow_fork: bool

    def reset(self, public_task: PublicTask, seed: int) -> Observation: ...

    def tool_catalog(self) -> tuple[ToolSpec, ...]: ...

    def step(self, action: Action) -> StepResult: ...

    def snapshot(self) -> SnapshotRef: ...

    def fork(self, snapshot: SnapshotRef) -> "EnvironmentAdapter": ...

    def state_digest(self) -> str: ...

    def close(self) -> None: ...

    def environment_identity(self) -> dict[str, JSONValue]: ...


@runtime_checkable
class Scorer(Protocol):
    """Official-metric evaluator.  Holds private task refs."""

    scorer_version: str

    def evaluate(self, final_artifact: ArtifactRef, private_task_ref: PrivateTaskRef) -> ScoreRecord: ...


@runtime_checkable
class TaskProvider(Protocol):
    """Supplies public tasks and private refs for one domain/dataset."""

    domain: str

    def public_tasks(self, role: str, round_index: int | None = None) -> tuple[PublicTask, ...]: ...

    def private_ref(self, task_id: str) -> PrivateTaskRef: ...

    def groups(self) -> dict[str, tuple[str, ...]]: ...
