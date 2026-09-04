"""Load real sentence_compress TRAIN_STATES / event rows into Student snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from trim.adapters.components import minus_mask
from trim.eval.official_query_pool import default_sentence_train_states
from trim.state.snapshot import capture_snapshot
from trim.training.rl_opd_types import HybridRolloutGroup, StudentDecisionPoint


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def _as_docs(raw: Any) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if isinstance(raw, dict):
        for did, rec in raw.items():
            if isinstance(rec, dict):
                text = str(rec.get("text") or rec.get("content") or rec.get("snippet") or "")
            else:
                text = str(rec or "")
            if text:
                docs.append({"id": str(did), "text": text})
        return docs
    if isinstance(raw, list):
        for rec in raw:
            if isinstance(rec, dict):
                did = str(rec.get("id") or rec.get("doc_id") or rec.get("docid") or "")
                text = str(rec.get("text") or rec.get("content") or rec.get("snippet") or "")
                if did and text:
                    docs.append({"id": did, "text": text})
    return docs


def working_memory_from_state(row: dict[str, Any]) -> dict[str, Any]:
    wm = dict(row.get("working_memory") or row.get("student_observable_env_state") or row.get("observable_env_state") or {})
    payload = row.get("payload") or row.get("event_payload") or {}
    docs = _as_docs(wm.get("documents") or row.get("documents") or row.get("doc_texts") or payload.get("doc_texts"))
    if not docs:
        ids = list(row.get("visible_doc_ids") or payload.get("search_result_doc_ids") or payload.get("visible_doc_ids") or [])
        obs = str(row.get("observation") or payload.get("observation") or payload.get("compressed_teacher_view") or "")
        if ids and obs:
            docs = [{"id": str(did), "text": obs} for did in ids[:12]]
        elif obs:
            docs = [{"id": "obs0", "text": obs}]
    curated = [str(x) for x in (wm.get("curated_ids") or row.get("curated_ids") or [])]
    query = str(row.get("query") or row.get("question") or wm.get("query") or row.get("query_id") or "")
    return {
        "query": query,
        "documents": docs,
        "curated_ids": curated,
        "pool": {d["id"]: d for d in docs},
        "accessible_doc_ids": [d["id"] for d in docs],
        "doc_store": {d["id"]: d for d in docs},
    }


def snapshot_from_state_row(row: dict[str, Any], *, component_id: str = "sentence_compress"):
    wm = working_memory_from_state(row)
    return capture_snapshot(
        query_id=str(row.get("query_id") or row.get("id") or "q"),
        step=int(row.get("turn_id") or row.get("step_id") or row.get("step") or 0),
        harness_mask=minus_mask(component_id),
        working_memory=wm,
        tool_history=list(row.get("tool_history") or wm.get("tool_history") or []),
        metadata={"component_id": component_id, "source": "train_states_5k", "event_active": bool(row.get("event_active", True))},
    )


def decision_point_from_state_row(
    row: dict[str, Any],
    *,
    component_id: str = "sentence_compress",
    policy_version: str = "v0",
) -> StudentDecisionPoint:
    snap = snapshot_from_state_row(row, component_id=component_id)
    qid = snap.query_id
    return StudentDecisionPoint(
        episode_id=str(row.get("rollout_id") or f"{qid}_frozen"),
        query_id=qid,
        rollout_idx=int(row.get("rollout_seed") or 0),
        turn_id=int(row.get("turn_id") or row.get("step_id") or 0),
        policy_version=policy_version,
        pre_action_snapshot=snap,
        pre_action_snapshot_hash=snap.content_hash(),
        student_model_input="",
        student_action_tokens=[],
        student_action_text=str(row.get("student_action_text") or ""),
        action_tool_names=list(row.get("action_tool_names") or []),
        structurally_valid=True,
        reward=0.0,
    )


def load_train_states(
    path: Path | None = None,
    *,
    component_id: str = "sentence_compress",
    limit: int | None = None,
    policy_version: str = "v0",
) -> tuple[list[StudentDecisionPoint], dict[str, Any]]:
    src = path or default_sentence_train_states()
    if src is None or not Path(src).is_file():
        return [], {"found": False, "path": str(src) if src else None, "n_states": 0}
    points: list[StudentDecisionPoint] = []
    for i, row in enumerate(iter_jsonl(Path(src))):
        if limit is not None and i >= limit:
            break
        if row.get("event_active") is False and not row.get("working_memory") and not row.get("documents"):
            continue
        points.append(decision_point_from_state_row(row, component_id=component_id, policy_version=policy_version))
    return points, {
        "found": True,
        "path": str(src),
        "n_states": len(points),
        "n_queries": len({p.query_id for p in points}),
        "source": "train_states_5k",
    }


def groups_from_frozen_points(points: list[StudentDecisionPoint]) -> list[HybridRolloutGroup]:
    by_q: dict[str, list[StudentDecisionPoint]] = {}
    for point in points:
        by_q.setdefault(point.query_id, []).append(point)
    groups: list[HybridRolloutGroup] = []
    for qid, rows in by_q.items():
        groups.append(
            HybridRolloutGroup(
                query_id=qid,
                policy_version=rows[0].policy_version,
                trajectory_group={"rl_rows": [], "source": "train_states_5k", "n_states": len(rows)},
                decision_points=rows,
                terminal_rewards=[0.0],
                metadata={"frozen": True},
            )
        )
    return groups


def doc_store_from_points(points: list[StudentDecisionPoint], query_id: str) -> dict[str, Any]:
    store: dict[str, Any] = {}
    for point in points:
        if point.query_id != query_id:
            continue
        for rec in point.pre_action_snapshot.working_memory.get("documents") or []:
            if isinstance(rec, dict) and rec.get("id") and rec.get("text"):
                store[str(rec["id"])] = {"id": str(rec["id"]), "text": str(rec["text"])}
    return store
