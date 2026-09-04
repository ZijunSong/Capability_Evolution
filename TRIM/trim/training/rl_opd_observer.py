"""Read-only Student decision observer for SlidingWindowSearchEnv.

Must never mutate working memory, toolset, reward, or observations.
Teacher inference is forbidden here; it runs later on frozen snapshots.
"""

from __future__ import annotations

from typing import Any, Callable

from trim.adapters.components import minus_mask
from trim.state.snapshot import EnvironmentSnapshot, capture_snapshot
from trim.training.rl_opd_types import StudentDecisionPoint


def snapshot_from_student_env(
    env: Any,
    *,
    component_id: str,
) -> EnvironmentSnapshot:
    """Build a SCAPE snapshot from Student-visible env fields only."""
    vis = {}
    if hasattr(env, "export_visible_state"):
        vis = dict(env.export_visible_state() or {})
    wm = getattr(env, "wm", None)
    curated = list(vis.get("curated_document_ids") or getattr(wm, "curated_ids", []) or [])
    pool = list(vis.get("pool_document_ids") or getattr(wm, "pool_ids", []) or [])
    harness_mask = dict(getattr(env, "scape_harness_mask", None) or minus_mask(component_id))
    history: list[dict[str, Any]] = []
    for rec in list(getattr(env, "_action_records", []) or []):
        if hasattr(rec, "to_dict"):
            history.append(dict(rec.to_dict()))
        elif isinstance(rec, dict):
            history.append(dict(rec))
        else:
            history.append(
                {
                    "turn_id": getattr(rec, "turn_id", 0),
                    "action_type": getattr(rec, "action_type", ""),
                }
            )
    return capture_snapshot(
        query_id=str(getattr(env, "query_id", vis.get("task_id") or "q")),
        step=int(getattr(env, "_current_turn", vis.get("turn_id") or 0)),
        harness_mask=harness_mask,
        working_memory={
            "curated_ids": [str(x) for x in curated],
            "accessible_doc_ids": [str(x) for x in (vis.get("visible_document_ids") or pool)],
            "pool": [str(x) for x in pool],
            "query": getattr(env, "query_text", "") or vis.get("query"),
        },
        tool_history=history,
        observations=[],
        metadata={
            "component_id": component_id,
            "owner": "student_reduced",
            "episode_id": vis.get("episode_id") or getattr(env, "_episode_id", ""),
        },
    )


def _tool_names_from_action(action: Any) -> list[str]:
    names: list[str] = []
    tools = getattr(action, "tools", None) or []
    for tool in tools:
        schema = getattr(tool, "tool_schema", None)
        name = getattr(schema, "name", None) or getattr(tool, "name", None)
        if name:
            names.append(str(name))
    if not names and isinstance(action, dict):
        names = [str(action.get("name") or "")]
    return [n for n in names if n]


class DecisionObserver:
    """Collects StudentDecisionPoint rows without touching the live env."""

    def __init__(
        self,
        *,
        policy_version: str,
        component_id: str,
        on_capture: Callable[[StudentDecisionPoint], None] | None = None,
    ) -> None:
        self.policy_version = policy_version
        self.component_id = component_id
        self.on_capture = on_capture
        self.points: list[StudentDecisionPoint] = []
        self._pending: dict[int, StudentDecisionPoint] = {}

    def on_pre_action(self, env: Any, action: Any) -> None:
        snap = snapshot_from_student_env(env, component_id=self.component_id)
        env_id = id(env)
        names = _tool_names_from_action(action)
        point = StudentDecisionPoint(
            episode_id=str(getattr(env, "_episode_id", f"{snap.query_id}_r0")),
            query_id=snap.query_id,
            rollout_idx=int(getattr(env, "rollout_idx", 0)),
            turn_id=int(getattr(env, "_current_turn", 0)),
            policy_version=self.policy_version,
            pre_action_snapshot=snap,
            pre_action_snapshot_hash=snap.content_hash(),
            student_model_input=None,
            student_action_tokens=[],
            student_action_text=" ".join(names),
            action_tool_names=names,
            structurally_valid=True,
        )
        self._pending[env_id] = point

    def on_post_action(
        self,
        env: Any,
        action: Any,
        *,
        reward: float | None,
        structurally_valid: bool = True,
    ) -> None:
        del action
        env_id = id(env)
        point = self._pending.pop(env_id, None)
        if point is None:
            return
        try:
            post = snapshot_from_student_env(env, component_id=self.component_id)
        except Exception:
            post = None
        point.post_action_snapshot = post
        point.reward = None if reward is None else float(reward)
        point.structurally_valid = bool(structurally_valid)
        self.points.append(point)
        if hasattr(env, "scape_decision_points"):
            env.scape_decision_points.append(point)
        if self.on_capture is not None:
            self.on_capture(point)

    def on_format_error(self, env: Any, action: Any) -> None:
        self.on_pre_action(env, action)
        self.on_post_action(env, action, reward=None, structurally_valid=False)
