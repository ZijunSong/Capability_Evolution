"""EvidenceStore: indexed, role-gated access to real execution evidence.

PR-4 (audit doc §5.2): the evolution packet previously carried only failure
LABELS plus a canned hint, so the proposer generated families of same-shaped
cosmetic components.  This store gives the proposer (analyst role) on-demand
access to concrete evidence: per-task telemetry, trace fragments, component
effects, and cross-arm comparisons.

Access control (audit doc §5.2): only evolution_train / harness_select roles
are readable.  confirm_*/final/transfer content is never mounted into the
proposer workspace.  Fragments are capped so a proposal stays inside its
output budget — index first, read slices on demand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from rsi.contracts.errors import ProtocolViolation

# Roles whose per-task evidence the evolver may read (audit §5.2: default
# open = evolution_train + harness_select only).
EVOLUTION_READABLE_ROLES = frozenset({"evolution_train", "work_train", "harness_select"})


class EvidenceStore:
    """Read-side index over panel results (per-task telemetry + traces)."""

    def __init__(self, *, allowed_roles: Iterable[str] = EVOLUTION_READABLE_ROLES):
        self._allowed_roles = frozenset(allowed_roles)
        # (role, task_id) -> per-task telemetry dict
        self._cases: dict[tuple[str, str], dict[str, Any]] = {}
        # trace_ref -> {"events": [...], "component_effects": {...}}
        self._traces: dict[str, dict[str, Any]] = {}

    # -- ingestion ---------------------------------------------------------
    def add_panel_result(
        self,
        *,
        role: str,
        per_task: Mapping[str, Mapping[str, Any]],
        max_tasks: int = 200,
    ) -> int:
        """Ingest a panel's per-task telemetry.  Returns tasks added."""
        if role not in self._allowed_roles:
            raise ProtocolViolation(
                f"EvidenceStore: role {role!r} is not evolver-readable "
                f"(allowed: {sorted(self._allowed_roles)})"
            )
        added = 0
        for tid, meta in dict(per_task).items():
            if added >= max_tasks:
                break
            self._cases[(role, tid)] = dict(meta)
            added += 1
        return added

    def add_trace(self, trace_ref: str, payload: Mapping[str, Any]) -> None:
        self._traces[str(trace_ref)] = dict(payload)

    # -- read side ---------------------------------------------------------
    def list_cases(
        self,
        *,
        role: str,
        failure_kind: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Summarized cases (telemetry only, no trace bodies)."""
        if role not in self._allowed_roles:
            raise ProtocolViolation(f"EvidenceStore: role {role!r} is not evolver-readable")
        out: list[dict[str, Any]] = []
        for (r, tid), meta in sorted(self._cases.items()):
            if r != role:
                continue
            if failure_kind and meta.get("terminated_reason") != failure_kind:
                continue
            out.append({"task_id": tid, **{k: meta.get(k) for k in (
                "value", "terminated_reason", "n_decisions", "wall_ms",
                "component_errors", "partial_test_pass_rate",
            )}})
            if len(out) >= limit:
                break
        return out

    def read_trace(
        self,
        trace_ref: str,
        *,
        start_decision: int = 0,
        end_decision: int | None = None,
    ) -> dict[str, Any]:
        trace = self._traces.get(trace_ref)
        if trace is None:
            raise ProtocolViolation(f"EvidenceStore: unknown trace_ref {trace_ref!r}")
        events = list(trace.get("events") or [])
        end = end_decision if end_decision is not None else len(events)
        return {
            "trace_ref": trace_ref,
            "events": events[start_decision:end],
            "component_effects": dict(trace.get("component_effects") or {}),
        }

    def read_component_effects(self, trace_ref: str, component_id: str) -> dict[str, Any]:
        trace = self._traces.get(trace_ref)
        if trace is None:
            raise ProtocolViolation(f"EvidenceStore: unknown trace_ref {trace_ref!r}")
        effects = dict(trace.get("component_effects") or {})
        return effects.get(component_id, {})

    def compare_arms(self, task_id: str, arm_a: dict, arm_b: dict) -> dict[str, Any]:
        """Paired delta on one task across two arm result dicts
        ({role -> per_task}); returns the raw numbers, interpretation is the
        analyst's job, not the store's."""
        a = (arm_a.get("per_task") or {}).get(task_id) or {}
        b = (arm_b.get("per_task") or {}).get(task_id) or {}
        return {
            "task_id": task_id,
            "arm_a": {k: a.get(k) for k in ("value", "terminated_reason", "n_decisions", "wall_ms")},
            "arm_b": {k: b.get(k) for k in ("value", "terminated_reason", "n_decisions", "wall_ms")},
        }

    def failure_fragments(
        self,
        *,
        role: str,
        limit: int = 5,
        max_chars_per_task: int = 400,
    ) -> list[str]:
        """Bounded text fragments for the packet builder: one line per failed
        task with the concrete telemetry, so the proposer sees EVIDENCE."""
        cases = self.list_cases(role=role, limit=limit)
        fragments: list[str] = []
        for c in cases:
            comp = c.get("component_errors") or {}
            comp_s = f" component_errors={comp}" if comp else ""
            fragments.append(
                f"{c['task_id']}: value={c.get('value')} "
                f"terminated={c.get('terminated_reason')} "
                f"n_decisions={c.get('n_decisions')} "
                f"wall_ms={c.get('wall_ms')}{comp_s}"
            )
        return [f[:max_chars_per_task] for f in fragments]
