"""Minimal student rollout utilities for same-state sample collection.

Production runs should wrap Harness-1 agent loops. This module provides the
SCAPE contract: reduced harness owns state occupancy; full teacher only renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from scape.adapters.components import coalition_minus_mask, minus_mask
from scape.adapters.harness_mask import apply_component_mask
from scape.rendering.dual_view import DualViewRenderer
from scape.state.snapshot import EnvironmentSnapshot, capture_snapshot
from scape.training.teacher import FullViewTeacher


PolicyFn = Callable[[Mapping[str, Any], EnvironmentSnapshot], dict[str, Any]]
EnvStepFn = Callable[[EnvironmentSnapshot, Mapping[str, Any]], EnvironmentSnapshot]


@dataclass
class FakeSearchEnv:
    """Deterministic env for unit tests / dry-runs."""

    query_id: str
    component_id: str
    component_ids: list[str] | None = None
    max_steps: int = 3
    docs: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0

    def __post_init__(self) -> None:
        if not self.docs:
            self.docs = [
                {"id": "d1", "text": "alpha evidence about topic"},
                {"id": "d2", "text": "beta evidence about topic"},
            ]

    def _student_mask(self) -> dict[str, bool]:
        if self.component_ids:
            return coalition_minus_mask(self.component_ids)
        return minus_mask(self.component_id)

    def initial_snapshot(self) -> EnvironmentSnapshot:
        mask = self._student_mask()
        return capture_snapshot(
            query_id=self.query_id,
            step=0,
            harness_mask=mask,
            working_memory={
                "documents": self.docs,
                "curated_docs": self.docs[:1],
                "curated_ids": [self.docs[0]["id"]],
                "curated_importance": {self.docs[0]["id"]: "high"},
                "evidence_graph": {"nodes": [self.docs[0]["id"]], "edges": []},
                "token_budget_marker": "budget=1024",
                "auto_populate_seed": ["seed_q"],
            },
            tool_history=[],
            observations=[],
            metadata={"owner": "student_reduced"},
        )

    def step(self, snap: EnvironmentSnapshot, action: Mapping[str, Any]) -> EnvironmentSnapshot:
        self.step_count += 1
        hist = list(snap.tool_history)
        hist.append({"step": snap.step, "action": dict(action)})
        obs = list(snap.observations)
        obs.append({"step": self.step_count, "ok": True})
        wm = dict(snap.working_memory)
        return capture_snapshot(
            query_id=snap.query_id,
            step=self.step_count,
            harness_mask=snap.harness_mask,
            working_memory=wm,
            tool_history=hist,
            observations=obs,
            metadata=snap.metadata,
        )


def student_rollout_collect(
    env: FakeSearchEnv,
    student_policy: PolicyFn,
    *,
    n_steps: int = 2,
) -> list[EnvironmentSnapshot]:
    """Roll out under H_-m; return snapshots that own the state distribution."""
    snaps: list[EnvironmentSnapshot] = []
    with apply_component_mask(minus_mask(env.component_id)):
        snap = env.initial_snapshot()
        snaps.append(snap)
        for _ in range(n_steps):
            action = student_policy({}, snap)
            snap = env.step(snap, action)
            snaps.append(snap)
    return snaps


def full_teacher_score_only(
    snapshots: list[EnvironmentSnapshot],
    *,
    component_id: str,
    teacher: FullViewTeacher | None = None,
) -> list[dict[str, Any]]:
    """Score snapshots with full-view teacher without env stepping."""
    t = teacher or FullViewTeacher(renderer=DualViewRenderer())
    env_before = t.renderer.environment_steps
    outs = []
    for snap in snapshots:
        dual = t.dual_view(snap, component_id=component_id)
        outs.append(t.score(dual.full_view))
    t.assert_no_env_step()
    if t.renderer.environment_steps != env_before:
        raise RuntimeError("full teacher stepped environment")
    return outs


def replay_parity(
    snap: EnvironmentSnapshot,
    *,
    component_id: str,
    renderer: DualViewRenderer | None = None,
) -> dict[str, Any]:
    """Full vs minus render from same snapshot must share snapshot hash."""
    rend = renderer or DualViewRenderer()
    dual = rend.render_pair(snap, component_id=component_id)
    return {
        "same_snapshot": dual.snapshot_hash == snap.content_hash(),
        "student_hash": dual.student_view.get("render_hash"),
        "full_hash": dual.full_view.get("render_hash"),
        "views_differ": dual.student_view.get("render_hash") != dual.full_view.get("render_hash"),
    }
