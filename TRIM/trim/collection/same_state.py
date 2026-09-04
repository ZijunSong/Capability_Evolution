"""Same-environment-state collection for true SCAPE OPD.

Produces structured `EnvironmentSnapshot` records under H_-S (coalition mask).
For plumbing smoke we generate deterministic WM states with Harness-1-shaped
tool calls. Production can swap in a Harness-1 agent loop without changing the
schema.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from trim.adapters.components import coalition_minus_mask, full_mask
from trim.probes.rollout import FakeSearchEnv
from trim.rendering.dual_view import DualViewRenderer
from trim.state.snapshot import EnvironmentSnapshot, capture_snapshot, snapshot_roundtrip_ok
from trim.training.action_codec import format_tool_call_text
from trim.training.tool_mask import legal_tool_names

SNAPSHOT_SCHEMA_VERSION = "scape_snapshot_v1"
TOOL_MASK_VERSION = "scape_tool_mask_v1"


HARNESS1_TOOL_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "fan_out_search",
        "arguments": {"query": "evidence for topic {qid}", "n": 3},
    },
    {
        "name": "search_corpus",
        "arguments": {"query": "corpus lookup {qid}"},
    },
    {
        "name": "grep_corpus",
        "arguments": {"pattern": "key.?fact"},
    },
    {
        "name": "read_document",
        "arguments": {"doc_id": "d1"},
    },
    {
        "name": "review_docs",
        "arguments": {"doc_ids": ["d1", "d2"]},
    },
    {
        "name": "curate",
        "arguments": {"add_ids": ["d1"], "remove_ids": []},
    },
    {
        "name": "verify",
        "arguments": {"claim": "fact about {qid}", "evidence_ids": ["d1"]},
    },
    {
        "name": "end_search",
        "arguments": {"reason": "sufficient evidence"},
    },
]


def resolve_component_ids(
    *,
    component_id: str | None = None,
    component_ids: Sequence[str] | None = None,
) -> list[str]:
    """Normalize single- or multi-component coalition targets."""
    if component_ids is not None:
        ids = [str(cid) for cid in component_ids]
        if not ids:
            raise ValueError("component_ids must be non-empty")
        return ids
    if component_id is not None:
        return [str(component_id)]
    raise ValueError("component_id or component_ids required")


def coalition_label(component_ids: Sequence[str]) -> str:
    return ",".join(component_ids)


def _snapshot_with_mask(
    snap: EnvironmentSnapshot,
    student_mask: Mapping[str, bool],
    *,
    component_ids: Sequence[str],
) -> EnvironmentSnapshot:
    return capture_snapshot(
        query_id=snap.query_id,
        step=snap.step,
        harness_mask=dict(student_mask),
        working_memory=snap.working_memory,
        tool_history=snap.tool_history,
        observations=snap.observations,
        metadata={
            **dict(snap.metadata),
            "component_ids": list(component_ids),
            "owner": "student_reduced",
        },
    )


def build_paired_state(
    *,
    query_id: str,
    step: int,
    component_id: str | None = None,
    component_ids: Sequence[str] | None = None,
    rng: random.Random,
) -> dict[str, Any]:
    """One student-owned same-state record with full/reduced dual views + action text."""
    resolved_ids = resolve_component_ids(component_id=component_id, component_ids=component_ids)
    student_mask = coalition_minus_mask(resolved_ids)
    env = FakeSearchEnv(
        query_id=query_id,
        component_id=resolved_ids[0],
        component_ids=resolved_ids,
        max_steps=max(1, step + 1),
    )
    snap0 = env.initial_snapshot()
    # Advance a few student steps so tool_history is non-empty
    snap = snap0
    for s in range(max(1, step)):
        tmpl = HARNESS1_TOOL_TEMPLATES[rng.randrange(len(HARNESS1_TOOL_TEMPLATES) - 1)]
        args = {
            k: (v.format(qid=query_id) if isinstance(v, str) else v)
            for k, v in tmpl["arguments"].items()
        }
        action = {"name": tmpl["name"], "arguments": args}
        snap = env.step(snap, action)

    tmpl = HARNESS1_TOOL_TEMPLATES[rng.randrange(len(HARNESS1_TOOL_TEMPLATES))]
    args = {
        k: (v.format(qid=query_id) if isinstance(v, str) else v)
        for k, v in tmpl["arguments"].items()
    }
    student_action = {"name": tmpl["name"], "arguments": args}
    response_text = format_tool_call_text(student_action["name"], student_action["arguments"])
    if student_action["name"] != "end_search" and rng.random() < 0.25:
        response_text += format_tool_call_text(
            "end_search", {"reason": "sufficient evidence"}
        )

    snap = _snapshot_with_mask(snap, student_mask, component_ids=resolved_ids)

    renderer = DualViewRenderer()
    dual = renderer.render_pair(snap, student_mask=student_mask)
    assert dual.snapshot_hash == snap.content_hash()
    assert renderer.environment_steps == 0

    minus_text = coalition_label(resolved_ids)
    prompt_reduced = (
        f"System: Harness reduced view (minus {{{minus_text}}}).\n"
        f"Query: {query_id}\n"
        f"State:\n{json.dumps(dual.student_view, ensure_ascii=False)}\n"
        f"Assistant:"
    )
    prompt_full = (
        f"System: Harness full view.\n"
        f"Query: {query_id}\n"
        f"State:\n{json.dumps(dual.full_view, ensure_ascii=False)}\n"
        f"Assistant:"
    )

    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "tool_mask_version": TOOL_MASK_VERSION,
        "component_id": resolved_ids[0],
        "component_ids": list(resolved_ids),
        "student_mask": dict(student_mask),
        "full_mask": dict(full_mask()),
        "query_id": query_id,
        "step": snap.step,
        "snapshot": snap.to_dict(),
        "snapshot_hash": snap.content_hash(),
        "student_action": student_action,
        "response_text": response_text,
        "prompt_reduced": prompt_reduced,
        "prompt_full": prompt_full,
        "student_view_hash": dual.student_view.get("render_hash"),
        "full_view_hash": dual.full_view.get("render_hash"),
        "views_differ_by_harness_only": dual.student_view.get("render_hash")
        != dual.full_view.get("render_hash"),
        "teacher_does_not_step_environment": True,
        "no_future_observation": True,
        "same_snapshot_hash": True,
        "legacy_scope_path_used": False,
        "legal_tools": legal_tool_names(),
    }


def collect_same_state_dataset(
    *,
    n_states: int,
    component_id: str | None = "evidence_graph",
    component_ids: Sequence[str] | None = None,
    seed: int = 42,
    out_path: Path | None = None,
    query_prefix: str = "smoke_q",
) -> list[dict[str, Any]]:
    resolved_ids = resolve_component_ids(component_id=component_id, component_ids=component_ids)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n_states):
        qid = f"{query_prefix}{i:04d}"
        step = 1 + (i % 3)
        row = build_paired_state(
            query_id=qid,
            step=step,
            component_ids=resolved_ids,
            rng=rng,
        )
        assert snapshot_roundtrip_ok(EnvironmentSnapshot.from_dict(row["snapshot"]))
        rows.append(row)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def build_query_disjoint_splits(
    *,
    component_id: str | None = "evidence_graph",
    component_ids: Sequence[str] | None = None,
    out_dir: Path,
    train_n: int = 8000,
    valid_n: int = 1000,
    test_n: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Build EG_TRAIN / EG_VALID / EG_TEST with query-disjoint pools."""
    resolved_ids = resolve_component_ids(component_id=component_id, component_ids=component_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "EG_TRAIN_8K": ("train_q", train_n, seed),
        "EG_VALID_1K": ("valid_q", valid_n, seed + 1),
        "EG_TEST_1K": ("test_q", test_n, seed + 2),
    }
    meta: dict[str, Any] = {
        "component_id": resolved_ids[0],
        "component_ids": list(resolved_ids),
        "query_disjoint": True,
        "splits": {},
    }
    for name, (prefix, n, split_seed) in splits.items():
        path = out_dir / f"{name}.jsonl"
        rows = collect_same_state_dataset(
            n_states=n,
            component_ids=resolved_ids,
            seed=split_seed,
            out_path=path,
            query_prefix=prefix,
        )
        audit = audit_same_state(rows)
        meta["splits"][name] = {
            "path": str(path),
            "n_states": len(rows),
            "query_prefix": prefix,
            "seed": split_seed,
            "audit": audit,
        }
    (out_dir / "DATA_AUDIT.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def load_same_state_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def audit_same_state(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    same = sum(1 for r in rows if r.get("same_snapshot_hash"))
    no_step = sum(1 for r in rows if r.get("teacher_does_not_step_environment"))
    no_future = sum(1 for r in rows if r.get("no_future_observation"))
    views_differ = sum(1 for r in rows if r.get("views_differ_by_harness_only"))
    legacy = sum(1 for r in rows if r.get("legacy_scope_path_used"))
    coalition_rows = sum(1 for r in rows if len(r.get("component_ids") or []) > 1)
    has_student_mask = sum(1 for r in rows if isinstance(r.get("student_mask"), Mapping))
    return {
        "n_states": n,
        "same_snapshot_hash_rate": same / max(1, n),
        "teacher_does_not_step_rate": no_step / max(1, n),
        "no_future_observation_rate": no_future / max(1, n),
        "full_reduced_differ_rate": views_differ / max(1, n),
        "legacy_scope_path_used": legacy > 0,
        "coalition_rows": coalition_rows,
        "student_mask_recorded_rate": has_student_mask / max(1, n),
        "pass": same == n and no_step == n and no_future == n and legacy == 0,
    }
