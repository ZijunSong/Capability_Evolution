"""Sync-mode policy version contract for RL+SR-OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class PolicyVersionMismatch(ValueError):
    """Raised when rollout / train / teacher versions disagree in sync mode."""


@dataclass(frozen=True)
class PolicyVersion:
    rollout_policy: str
    train_policy: str
    harness_teacher_policy: str
    kl_reference_policy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_policy": self.rollout_policy,
            "train_policy": self.train_policy,
            "harness_teacher_policy": self.harness_teacher_policy,
            "kl_reference_policy": self.kl_reference_policy,
        }


def assert_policy_versions_match(
    *,
    rollout_policy: str,
    train_policy: str,
    harness_teacher_policy: str,
    sync: bool = True,
) -> None:
    """Canonical MVP is sync on-policy only. Mismatch is a hard fail."""
    if not sync:
        return
    if not rollout_policy or not train_policy or not harness_teacher_policy:
        raise PolicyVersionMismatch(
            f"missing policy version: rollout={rollout_policy!r} "
            f"train={train_policy!r} teacher={harness_teacher_policy!r}"
        )
    if not (rollout_policy == train_policy == harness_teacher_policy):
        raise PolicyVersionMismatch(
            f"sync mode requires rollout==train==teacher; "
            f"got rollout={rollout_policy} train={train_policy} "
            f"teacher={harness_teacher_policy}"
        )
