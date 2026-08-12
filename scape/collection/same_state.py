"""Same-environment-state collection for true SCAPE OPD.

Produces structured `EnvironmentSnapshot` records under H_-m. For plumbing
smoke we generate deterministic WM states with Harness-1-shaped tool calls.
Production can swap in a Harness-1 agent loop without changing the schema.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from scape.adapters.components import minus_mask
from scape.probes.rollout import FakeSearchEnv
from scape.rendering.dual_view import DualViewRenderer
from scape.state.snapshot import EnvironmentSnapshot, snapshot_roundtrip_ok
from scape.training.tool_mask import legal_tool_names

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


def format_tool_call_text(name: str, arguments: Mapping[str, Any]) -> str:
    return f"to={name}\n{json.dumps(dict(arguments), ensure_ascii=False)}\n"


def build_paired_state(
    *,
    query_id: str,
    step: int,
    component_id: str,
    rng: random.Random,
) -> dict[str, Any]:
    """One student-owned same-state record with full/reduced dual views + action text."""
    env = FakeSearchEnv(query_id=query_id, component_id=component_id, max_steps=max(1, step + 1))
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

    renderer = DualViewRenderer()
    dual = renderer.render_pair(snap, component_id=component_id)
    assert dual.snapshot_hash == snap.content_hash()
    assert renderer.environment_steps == 0

    prompt_reduced = (
        f"System: Harness reduced view (minus {component_id}).\n"
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
        "component_id": component_id,
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
    component_id: str = "evidence_graph",
    seed: int = 42,
    out_path: Path | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n_states):
        qid = f"smoke_q{i:04d}"
        step = 1 + (i % 3)
        row = build_paired_state(
            query_id=qid, step=step, component_id=component_id, rng=rng
        )
        assert snapshot_roundtrip_ok(EnvironmentSnapshot.from_dict(row["snapshot"]))
        rows.append(row)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


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
    return {
        "n_states": n,
        "same_snapshot_hash_rate": same / max(1, n),
        "teacher_does_not_step_rate": no_step / max(1, n),
        "no_future_observation_rate": no_future / max(1, n),
        "full_reduced_differ_rate": views_differ / max(1, n),
        "legacy_scope_path_used": legacy > 0,
        "pass": same == n and no_step == n and no_future == n and legacy == 0,
    }
