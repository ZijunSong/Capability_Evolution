"""Registration of toy domains (no optional dependencies)."""

from __future__ import annotations

from typing import Mapping

from rsi.contracts.task import PrivateTaskRef, PublicTask
from rsi.environments.base import TaskProvider
from rsi.environments.registry import EnvironmentBundle, register_bundle
from rsi.environments.toy import (
    ToyFileWorld,
    ToyKVWorld,
    ToyScorer,
    build_toy_file_tasks,
    build_toy_kv_tasks,
)


def _role_index(role: str) -> int:
    return {
        "work_train": 0,
        "harness_select": 1,
        "weight_dev": 2,
        "confirm_h": 3,
        "confirm_w": 4,
        "final_test": 5,
        "transfer_test": 6,
    }[role]


def _group_index(group_id: str) -> int:
    return int(group_id.rsplit("-g", 1)[-1])


class _ToyFileTaskProvider:
    domain = "toy_file"

    def __init__(self, n_tasks: int = 48, seed: int = 17):
        pairs = build_toy_file_tasks(n_tasks, seed)
        self._tasks = tuple(p for p, _ in pairs)
        self._expected = {p.task_id: e for p, e in pairs}

    def public_tasks(self, role: str, round_index: int | None = None) -> tuple[PublicTask, ...]:
        # Role slicing is by *group* so clusters never straddle roles
        # (test_split_group_disjointness).  confirm_h/confirm_w reuse the same
        # development reserve in pilot mode (statistics_mode=pilot_reused_dev).
        idx = _role_index(role)
        return tuple(t for t in self._tasks if (_group_index(t.group_id) % 7) == idx)

    def private_ref(self, task_id: str) -> PrivateTaskRef:
        return PrivateTaskRef(ref_id=task_id, domain=self.domain)

    def groups(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for t in self._tasks:
            out.setdefault(t.group_id, []).append(t.task_id)
        return {k: tuple(v) for k, v in out.items()}

    def expected_map(self) -> Mapping[str, str]:
        return dict(self._expected)

    def all_tasks(self) -> tuple[PublicTask, ...]:
        return self._tasks


class _ToyKVTaskProvider:
    domain = "toy_kv"

    def __init__(self, n_tasks: int = 30, seed: int = 17):
        pairs = build_toy_kv_tasks(n_tasks, seed)
        self._tasks = tuple(p for p, _ in pairs)
        self._expected = {p.task_id: e for p, e in pairs}

    def public_tasks(self, role: str, round_index: int | None = None) -> tuple[PublicTask, ...]:
        idx = _role_index(role)
        return tuple(t for t in self._tasks if (_group_index(t.group_id) % 7) == idx)

    def private_ref(self, task_id: str) -> PrivateTaskRef:
        return PrivateTaskRef(ref_id=task_id, domain=self.domain)

    def groups(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for t in self._tasks:
            out.setdefault(t.group_id, []).append(t.task_id)
        return {k: tuple(v) for k, v in out.items()}

    def expected_map(self) -> Mapping[str, str]:
        return dict(self._expected)

    def all_tasks(self) -> tuple[PublicTask, ...]:
        return self._tasks


class _ToyKVScorer(ToyScorer):
    scorer_version = "toy_kv_scorer_v1"


register_bundle(
    EnvironmentBundle(
        domain="toy_file",
        adapter_factory=lambda: ToyFileWorld(),
        task_provider_factory=lambda: _ToyFileTaskProvider(),
        scorer_factory=lambda: ToyScorer(_ToyFileTaskProvider().expected_map()),
        synthetic=True,
    )
)

register_bundle(
    EnvironmentBundle(
        domain="toy_kv",
        adapter_factory=lambda: ToyKVWorld(),
        task_provider_factory=lambda: _ToyKVTaskProvider(),
        scorer_factory=lambda: _ToyKVScorer(_ToyKVTaskProvider().expected_map()),
        synthetic=True,
    )
)
