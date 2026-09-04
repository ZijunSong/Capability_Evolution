"""Teacher policies for SCAPE OPD.

Canonical strategy (fixed in manifests; do not swap across seeds):
  teacher = EMA / lagged copy of student parameters under FULL harness view.

The teacher is never allowed to step the environment; it only scores / decodes
actions on dual-view renders of student-owned snapshots.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trim.rendering.dual_view import DualView, DualViewRenderer
from trim.state.snapshot import EnvironmentSnapshot


DecodeFn = Callable[[Mapping[str, Any]], dict[str, Any]]
ScoreFn = Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass
class TeacherConfig:
    strategy: str = "ema"  # ema | lagged_copy
    ema_decay: float = 0.99
    lag_steps: int = 100
    # Locked for a run — write into RUN_MANIFEST and never change per seed silently
    strategy_lock_id: str = "scape_teacher_v0_ema"


@dataclass
class FullViewTeacher:
    """Teacher that reads full-view renders without env stepping."""

    config: TeacherConfig = field(default_factory=TeacherConfig)
    renderer: DualViewRenderer = field(default_factory=DualViewRenderer)
    _shadow_params: dict[str, Any] = field(default_factory=dict)
    _updates: int = 0
    _env_steps_at_init: int = 0
    decode_fn: DecodeFn | None = None
    score_fn: ScoreFn | None = None

    def __post_init__(self) -> None:
        self._env_steps_at_init = self.renderer.environment_steps

    def update_from_student(self, student_params: Mapping[str, Any]) -> None:
        self._updates += 1
        if self.config.strategy == "lagged_copy":
            if self._updates % max(1, self.config.lag_steps) == 0 or not self._shadow_params:
                self._shadow_params = deepcopy(dict(student_params))
            return
        if self.config.strategy != "ema":
            raise ValueError(f"unknown teacher strategy: {self.config.strategy}")
        decay = float(self.config.ema_decay)
        if not self._shadow_params:
            self._shadow_params = deepcopy(dict(student_params))
            return
        # Numeric EMA for float tensors-as-lists; identity copy otherwise
        out: dict[str, Any] = {}
        for k, v in student_params.items():
            prev = self._shadow_params.get(k)
            if isinstance(v, (int, float)) and isinstance(prev, (int, float)):
                out[k] = decay * float(prev) + (1.0 - decay) * float(v)
            else:
                out[k] = deepcopy(v)
        self._shadow_params = out

    def dual_view(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        component_id: str,
    ) -> DualView:
        before = self.renderer.environment_steps
        view = self.renderer.render_pair(snapshot, component_id=component_id)
        after = self.renderer.environment_steps
        if after != before:
            raise RuntimeError("full teacher must not step the environment")
        return view

    def decode_tool_call(self, full_view: Mapping[str, Any]) -> dict[str, Any]:
        if self.decode_fn is not None:
            return self.decode_fn(full_view)
        # Deterministic stub decoder for unit tests / dry-runs
        docs = full_view.get("documents") or []
        if docs:
            return {
                "name": "curate",
                "arguments": {"add_ids": [docs[0].get("id")], "remove_ids": []},
            }
        return {"name": "search", "arguments": {"query": full_view.get("query_id", "")}}

    def score(self, full_view: Mapping[str, Any]) -> dict[str, Any]:
        if self.score_fn is not None:
            return self.score_fn(full_view)
        call = self.decode_tool_call(full_view)
        # Uniform-ish stub distribution over legal names with peak on decoded name
        names = ["search", "grep", "read_document", "curate", "verify", "end_search"]
        probs = {n: 0.05 for n in names}
        probs[str(call["name"])] = 0.75
        z = sum(probs.values())
        probs = {k: v / z for k, v in probs.items()}
        return {"tool_name_probs": probs, "decoded": call, "confidence": max(probs.values())}

    def assert_no_env_step(self) -> None:
        if self.renderer.environment_steps != self._env_steps_at_init:
            raise AssertionError("teacher stepped the environment")

    def manifest_fields(self) -> dict[str, Any]:
        return {
            "teacher_strategy": self.config.strategy,
            "teacher_ema_decay": self.config.ema_decay,
            "teacher_lag_steps": self.config.lag_steps,
            "teacher_strategy_lock_id": self.config.strategy_lock_id,
            "teacher_updates": self._updates,
        }
