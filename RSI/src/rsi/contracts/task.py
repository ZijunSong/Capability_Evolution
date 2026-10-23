"""Task contracts.

``PublicTask`` is everything the Actor / Teacher / Proposer may see.
``PrivateTaskRef`` is an opaque handle held by the evaluator service and never
enters those workers.  Training reward is an allowed supervision channel;
hidden answer *content* is not a runtime observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from rsi.contracts.common import JSONValue, digest_of


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    domain: str
    instruction: str
    # Cluster/group id: all tasks sharing a scenario, workbook problem, repo
    # issue family... must carry the same group_id and land in one split role.
    group_id: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def digest(self) -> str:
        return digest_of(
            {
                "task_id": self.task_id,
                "domain": self.domain,
                "instruction": self.instruction,
                "group_id": self.group_id,
                "metadata": dict(self.metadata),
            }
        )


@dataclass(frozen=True)
class PrivateTaskRef:
    """Opaque evaluator-side handle.  Contains no answer text."""

    ref_id: str
    domain: str


# Data roles from the task book §11.2.  ``pilot_reused_dev`` is a statistics
# mode of the whole suite, not a per-task role.
DATA_ROLES = (
    "work_train",
    "harness_select",
    "weight_dev",
    "confirm_h",
    "confirm_w",
    "final_test",
    "transfer_test",
)


@dataclass(frozen=True)
class SplitAssignment:
    role: str
    task_ids: tuple[str, ...]
    round_index: int | None = None  # for confirm_h[k] / confirm_w[k]

    def __post_init__(self) -> None:
        if self.role not in DATA_ROLES:
            raise ValueError(f"unknown data role {self.role!r}")


@dataclass(frozen=True)
class SplitManifest:
    """Frozen split identity: role -> task ids, plus group-disjointness proof."""

    manifest_id: str
    domain: str
    assignments: tuple[SplitAssignment, ...]
    groups: Mapping[str, tuple[str, ...]]  # group_id -> task_ids

    def digest(self) -> str:
        return digest_of(
            {
                "manifest_id": self.manifest_id,
                "domain": self.domain,
                "assignments": [
                    {"role": a.role, "task_ids": list(a.task_ids), "round_index": a.round_index}
                    for a in self.assignments
                ],
                "groups": {k: list(v) for k, v in sorted(self.groups.items())},
            }
        )

    def role_tasks(self, role: str, round_index: int | None = None) -> tuple[str, ...]:
        out: list[str] = []
        for a in self.assignments:
            if a.role != role:
                continue
            if role in ("confirm_h", "confirm_w") and round_index is not None:
                if a.round_index != round_index:
                    continue
            out.extend(a.task_ids)
        return tuple(out)

    def check_group_disjoint(self) -> None:
        """Refuse a split where one group lands in two different roles.

        The same role across rounds (confirm_h[k] reusing a development reserve
        in ``pilot_reused_dev`` mode) is allowed; two *different* roles never
        share a group.
        """
        seen: dict[str, str] = {}
        for a in self.assignments:
            for tid in a.task_ids:
                gid = None
                for g, members in self.groups.items():
                    if tid in members:
                        gid = g
                        break
                if gid is None:
                    raise ValueError(f"task {tid} not covered by any group")
                if gid in seen and seen[gid] != a.role:
                    raise ValueError(
                        f"group {gid} appears in both {seen[gid]} and {a.role}; "
                        "group-disjointness violated"
                    )
                seen[gid] = a.role
