"""Manifest-authoritative task resolution (audit doc §10.5).

The eval/evolve main paths previously called ``provider.public_tasks(role)``,
which re-enumerated the dataset directory on every invocation — the frozen
manifest existed but nothing read it, so new files appearing under
``data/tasks/`` silently changed the evaluation task set and ROLE_DATASET
constants never took effect.

``TaskResolver`` inverts that: when the config declares
``suite.split_manifest`` (plus ``manifest_is_authoritative: true``), every task
request goes through the frozen manifest; the provider is used only to
materialize the exact task ids the manifest names.  A manifest without an
assignment for the requested (role, round) is a hard ProtocolViolation — the
split cannot silently grow.

``Task.load`` failures are surfaced by the provider wrapper: a placeholder
instruction ("(appworld task 123)") must never enter a formal panel, so
materialization errors raise instead of being swallowed (audit doc §10.5).
"""

from __future__ import annotations

from typing import Any

from rsi.contracts.errors import ProtocolViolation
from rsi.contracts.task import PublicTask
from rsi.evaluation.data_prep import load_split_manifests


class TaskResolver:
    """Resolve (domain, role, round_index) -> tuple[PublicTask] from the frozen manifest."""

    def __init__(self, cfg: Any):
        self._cfg = cfg
        suite = cfg.section("suite")
        self._authoritative = bool(suite.get("manifest_is_authoritative", False))
        self._manifests = load_split_manifests(cfg) if self._authoritative else {}
        self._manifest_path = cfg.resolve_path(str(suite.get("split_manifest", "")))

    @property
    def authoritative(self) -> bool:
        return self._authoritative

    def manifest_digest(self, domain: str) -> str:
        manifest = self._manifests.get(domain)
        if manifest is None:
            raise ProtocolViolation(f"no frozen split manifest for domain {domain!r}")
        return manifest.digest()

    def resolve(
        self,
        provider: Any,
        *,
        domain: str,
        role: str,
        round_index: int | None = None,
    ) -> tuple[PublicTask, ...]:
        """Return the manifest's task ids for (domain, role, round), materialized.

        ``provider`` is only the materializer: its ``materialize_task_id`` is
        called for each frozen id; its enumeration is never consulted.
        """
        manifest = self._manifests.get(domain)
        if manifest is None:
            raise ProtocolViolation(
                f"no frozen split manifest for domain {domain!r} "
                f"(manifest: {self._manifest_path})"
            )
        matches = [
            a
            for a in manifest.assignments
            if a.role == role and (round_index is None or a.round_index == round_index)
        ]
        if not matches:
            raise ProtocolViolation(
                f"frozen manifest {manifest.manifest_id} has no assignment for "
                f"role={role!r} round={round_index!r}; roles present: "
                f"{sorted({a.role for a in manifest.assignments})}"
            )
        materialize = getattr(provider, "materialize_task_id", None)
        if materialize is None:
            raise ProtocolViolation(
                f"provider for {domain!r} does not implement materialize_task_id; "
                "manifest-authoritative mode requires exact-id materialization"
            )
        tasks: list[PublicTask] = []
        for assignment in matches:
            for tid in assignment.task_ids:
                tasks.append(materialize(tid))
        return tuple(tasks)


def install_materialize_task_id(provider: Any) -> Any:
    """Expose ``materialize_task_id(tid)`` on a provider whose only public
    constructor path is public_tasks(); a strict wrapper around its internal
    materializer that FAILS LOUDLY instead of returning placeholder tasks."""
    if hasattr(provider, "materialize_task_id"):
        return provider

    inner_names = ("_materialize", "_materialize_task")
    inner = None
    for name in inner_names:
        inner = getattr(provider, name, None)
        if callable(inner):
            break
    if inner is None:
        raise ProtocolViolation(
            f"provider {type(provider).__name__} has no materializer to expose"
        )

    def materialize_task_id(tid: str) -> PublicTask:
        task = inner(tid)
        instruction = str(getattr(task, "instruction", "") or "")
        if "(appworld task" in instruction or not instruction.strip():
            raise ProtocolViolation(
                f"task {tid!r} materialized with a placeholder instruction — "
                "Task.load failed upstream; data preflight must block, not substitute"
            )
        return task

    provider.materialize_task_id = materialize_task_id  # type: ignore[attr-defined]
    return provider
