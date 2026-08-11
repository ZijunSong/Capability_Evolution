"""Dual-view rendering from a shared environment snapshot.

student view = r_-m(xi_t)
full view    = r_F(xi_t)

Both views MUST be produced from the same snapshot. The full-view teacher
must not step the environment.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from scape.adapters.components import full_mask, minus_mask
from scape.state.snapshot import EnvironmentSnapshot, stable_hash


RenderFn = Callable[[EnvironmentSnapshot, Mapping[str, bool]], dict[str, Any]]


@dataclass
class DualView:
    snapshot_hash: str
    query_id: str
    step: int
    student_mask: dict[str, bool]
    full_mask: dict[str, bool]
    student_view: dict[str, Any]
    full_view: dict[str, Any]
    null_controls: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_hash": self.snapshot_hash,
            "query_id": self.query_id,
            "step": self.step,
            "student_mask": self.student_mask,
            "full_mask": self.full_mask,
            "student_view": self.student_view,
            "full_view": self.full_view,
            "null_controls": self.null_controls,
        }


def default_render(snapshot: EnvironmentSnapshot, mask: Mapping[str, bool]) -> dict[str, Any]:
    """Deterministic text-ish render of snapshot under a component mask.

    This is a SCAPE-side renderer used for tests and offline probes. Production
    runs can inject a Harness-1-backed RenderFn that reads WorkingMemory fields
    conditionally from the mask without stepping the env.
    """
    wm = snapshot.working_memory
    docs = list(wm.get("curated_docs") or wm.get("documents") or [])
    graph = wm.get("evidence_graph")
    importance = wm.get("curated_importance") or {}
    budget = wm.get("token_budget_marker")
    rerank = wm.get("rerank_instruction")

    rendered_docs: list[dict[str, Any]] = []
    for doc in docs:
        item = {"id": doc.get("id"), "text": doc.get("text") or doc.get("content") or ""}
        if mask.get("importance_tagging") and importance:
            item["importance"] = importance.get(str(item["id"]))
        if mask.get("sentence_compress") and isinstance(item["text"], str):
            # Cheap deterministic stand-in for sentence compress
            item["text"] = item["text"][: max(32, len(item["text"]) // 2)]
        if mask.get("content_dedup"):
            item["dedup_key"] = stable_hash(item["text"])[:12]
        rendered_docs.append(item)

    if mask.get("subtractive_curation"):
        # Keep only explicitly curated ids when present
        curated_ids = set(wm.get("curated_ids") or [])
        if curated_ids:
            rendered_docs = [d for d in rendered_docs if d.get("id") in curated_ids]

    payload: dict[str, Any] = {
        "query_id": snapshot.query_id,
        "step": snapshot.step,
        "mask": dict(mask),
        "documents": rendered_docs,
        "tool_history": deepcopy(snapshot.tool_history),
    }
    if mask.get("evidence_graph") and graph is not None:
        payload["evidence_graph"] = deepcopy(graph)
    if mask.get("token_budget_marker") and budget is not None:
        payload["token_budget_marker"] = budget
    if mask.get("adaptive_rerank_instruction") and rerank is not None:
        payload["rerank_instruction"] = rerank
    if mask.get("auto_populate_first_search"):
        payload["auto_seed"] = wm.get("auto_populate_seed")
    if mask.get("verify_tool"):
        payload["verify_available"] = True
    else:
        payload["verify_available"] = False
    if mask.get("chunk_neighbors"):
        payload["chunk_neighbors"] = wm.get("chunk_neighbors") or []

    payload["render_hash"] = stable_hash(payload)
    return payload


def field_order_perturb(view: Mapping[str, Any]) -> dict[str, Any]:
    """Null control: same content, different key insertion order."""
    items = list(view.items())
    items_rev = list(reversed(items))
    # Rebuild via json roundtrip with sorted keys to keep semantic equality,
    # then re-insert in reverse order for a distinct object identity/order.
    base = json.loads(json.dumps(view, sort_keys=True))
    out: dict[str, Any] = {}
    for k, _ in items_rev:
        if k in base:
            out[k] = base[k]
    for k, v in base.items():
        out.setdefault(k, v)
    return out


class DualViewRenderer:
    def __init__(self, render_fn: RenderFn | None = None):
        self.render_fn = render_fn or default_render
        self._env_steps: int = 0  # teacher must not increment this

    @property
    def environment_steps(self) -> int:
        return self._env_steps

    def render_pair(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        component_id: str | None = None,
        student_mask: Mapping[str, bool] | None = None,
        include_null_controls: bool = True,
    ) -> DualView:
        if student_mask is None:
            if component_id is None:
                raise ValueError("component_id or student_mask required")
            student_mask = minus_mask(component_id)
        fmask = full_mask()
        # Render without stepping environment
        before = self._env_steps
        student_view = self.render_fn(snapshot, student_mask)
        full_view = self.render_fn(snapshot, fmask)
        if self._env_steps != before:
            raise RuntimeError("renderer stepped the environment; forbidden for teacher view")

        nulls: dict[str, dict[str, Any]] = {}
        if include_null_controls:
            nulls["same_render"] = deepcopy(student_view)
            nulls["field_order_only"] = field_order_perturb(student_view)

        return DualView(
            snapshot_hash=snapshot.content_hash(),
            query_id=snapshot.query_id,
            step=snapshot.step,
            student_mask=dict(student_mask),
            full_mask=dict(fmask),
            student_view=student_view,
            full_view=full_view,
            null_controls=nulls,
        )

    def assert_same_snapshot(self, view: DualView, snapshot: EnvironmentSnapshot) -> None:
        if view.snapshot_hash != snapshot.content_hash():
            raise AssertionError("dual view not bound to provided snapshot")
        if view.query_id != snapshot.query_id or view.step != snapshot.step:
            raise AssertionError("dual view query/step mismatch vs snapshot")
