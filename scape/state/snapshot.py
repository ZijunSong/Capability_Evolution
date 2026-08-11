"""Same-environment-state snapshots.

A snapshot `xi_t` captures the environment state produced by a *student*
rollout under a reduced harness. Full-view teachers must render from the same
snapshot without stepping the environment forward.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(_canonical_bytes(obj)).hexdigest()


@dataclass
class EnvironmentSnapshot:
    """Frozen environment state at step t.

    Fields are intentionally generic so SCAPE can wrap Harness-1 WorkingMemory
    (or a test double) without forking upstream.
    """

    query_id: str
    step: int
    harness_mask: dict[str, bool]
    working_memory: dict[str, Any]
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Monotonic creation counter used to prove "no future information"
    created_at_step: int | None = None

    def __post_init__(self) -> None:
        if self.created_at_step is None:
            self.created_at_step = int(self.step)
        if self.created_at_step != self.step:
            raise ValueError(
                f"snapshot created_at_step={self.created_at_step} != step={self.step}; "
                "snapshots must not contain future information"
            )
        # Deep-freeze mutable payloads
        self.working_memory = deepcopy(self.working_memory)
        self.tool_history = deepcopy(self.tool_history)
        self.observations = deepcopy(self.observations)
        self.harness_mask = dict(self.harness_mask)
        self.metadata = dict(self.metadata)

    def content_dict(self) -> dict[str, Any]:
        """Serializable content used for hashing (excludes ephemeral metadata keys)."""
        meta = {k: v for k, v in self.metadata.items() if not str(k).startswith("_")}
        return {
            "query_id": self.query_id,
            "step": self.step,
            "created_at_step": self.created_at_step,
            "harness_mask": self.harness_mask,
            "working_memory": self.working_memory,
            "tool_history": self.tool_history,
            "observations": self.observations,
            "metadata": meta,
        }

    def content_hash(self) -> str:
        return stable_hash(self.content_dict())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["content_hash"] = self.content_hash()
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvironmentSnapshot":
        payload = dict(data)
        payload.pop("content_hash", None)
        return cls(
            query_id=str(payload["query_id"]),
            step=int(payload["step"]),
            harness_mask=dict(payload["harness_mask"]),
            working_memory=dict(payload.get("working_memory") or {}),
            tool_history=list(payload.get("tool_history") or []),
            observations=list(payload.get("observations") or []),
            metadata=dict(payload.get("metadata") or {}),
            created_at_step=payload.get("created_at_step"),
        )

    def assert_no_future(self, *, max_known_step: int) -> None:
        if self.step > max_known_step or (self.created_at_step or 0) > max_known_step:
            raise AssertionError(
                f"snapshot leaks future info: step={self.step} "
                f"created_at_step={self.created_at_step} max_known_step={max_known_step}"
            )
        for obs in self.observations:
            obs_step = obs.get("step")
            if obs_step is not None and int(obs_step) > max_known_step:
                raise AssertionError(f"observation from future step {obs_step}")
        for act in self.tool_history:
            act_step = act.get("step")
            if act_step is not None and int(act_step) > max_known_step:
                raise AssertionError(f"tool history from future step {act_step}")


def snapshot_roundtrip_ok(snap: EnvironmentSnapshot) -> bool:
    restored = EnvironmentSnapshot.from_dict(snap.to_dict())
    return restored.content_hash() == snap.content_hash()


def capture_snapshot(
    *,
    query_id: str,
    step: int,
    harness_mask: Mapping[str, bool],
    working_memory: Mapping[str, Any],
    tool_history: list[dict[str, Any]] | None = None,
    observations: list[dict[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        query_id=query_id,
        step=step,
        harness_mask=dict(harness_mask),
        working_memory=dict(working_memory),
        tool_history=list(tool_history or []),
        observations=list(observations or []),
        metadata=dict(metadata or {}),
        created_at_step=step,
    )
