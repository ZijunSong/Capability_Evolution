"""Graph-hybrid same-state collection: student V2, teacher V3.

V2 = GRAPH_STATE_ONLY  — graph state retained externally; no full renderer in prompt
V3 = GRAPH_STATE_PLUS_MINIMAL_RENDER — minimal graph-aware render for teacher
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from scape.adapters.components import full_mask, minus_mask
from scape.collection.same_state import (
    audit_same_state,
    build_paired_state,
    collect_same_state_dataset,
    format_tool_call_text,
)
from scape.rendering.dual_view import DualViewRenderer, default_render
from scape.state.snapshot import EnvironmentSnapshot, stable_hash


def minimal_graph_render(snapshot: Any, mask: Any) -> dict[str, Any]:
  """V3 teacher: graph state + compact render summary without full renderer."""
  base = default_render(snapshot, mask)
  graph = base.get("evidence_graph")
  out: dict[str, Any] = {
    "query_id": base.get("query_id"),
    "step": base.get("step"),
    "mask": dict(mask),
    "documents": base.get("documents") or [],
    "tool_history": base.get("tool_history") or [],
    "graph_state_external": True,
  }
  if graph is not None:
    nodes = graph.get("nodes") or graph.get("edges") or graph
    if isinstance(nodes, list):
      out["graph_node_count"] = len(nodes)
    elif isinstance(nodes, dict):
      out["graph_node_count"] = len(nodes.get("nodes", []))
    else:
      out["graph_node_count"] = 1
    out["minimal_graph_render"] = {
      "summary": f"graph_nodes={out['graph_node_count']}",
      "edge_hint": bool(graph.get("edges")),
    }
  else:
    out["graph_node_count"] = 0
    out["minimal_graph_render"] = {"summary": "empty_graph"}
  out["render_hash"] = stable_hash(out)
  return out


def graph_state_only_render(snapshot: Any, mask: Any) -> dict[str, Any]:
  """V2 student: external graph store; prompt carries state pointer only."""
  base = default_render(snapshot, mask)
  graph = base.get("evidence_graph")
  out: dict[str, Any] = {
    "query_id": base.get("query_id"),
    "step": base.get("step"),
    "mask": dict(mask),
    "documents": base.get("documents") or [],
    "tool_history": base.get("tool_history") or [],
    "graph_state_external": True,
    "graph_state_ref": stable_hash(graph)[:16] if graph else "none",
  }
  out["render_hash"] = stable_hash(out)
  return out


def build_graph_hybrid_row(
  *,
  query_id: str,
  step: int,
  rng: random.Random,
) -> dict[str, Any]:
  """One graph-hybrid row with V2 student / V3 teacher prompts."""
  row = build_paired_state(
    query_id=query_id,
    step=step,
    component_id="evidence_graph",
    rng=rng,
  )
  snap = EnvironmentSnapshot.from_dict(row["snapshot"])
  renderer = DualViewRenderer(render_fn=default_render)
  student_mask = minus_mask("evidence_graph")
  full_mask_dict = full_mask()

  v2_renderer = DualViewRenderer(render_fn=graph_state_only_render)
  v3_renderer = DualViewRenderer(render_fn=minimal_graph_render)

  v2_view = v2_renderer.render_fn(snap, student_mask)
  v3_view = v3_renderer.render_fn(snap, full_mask_dict)

  prompt_v2 = (
    f"System: Harness graph-hybrid student V2 (GRAPH_STATE_ONLY).\n"
    f"Query: {query_id}\n"
    f"State:\n{json.dumps(v2_view, ensure_ascii=False)}\n"
    f"Assistant:"
  )
  prompt_v3 = (
    f"System: Harness graph-hybrid teacher V3 (GRAPH_STATE_PLUS_MINIMAL_RENDER).\n"
    f"Query: {query_id}\n"
    f"State:\n{json.dumps(v3_view, ensure_ascii=False)}\n"
    f"Assistant:"
  )

  teacher_action = _parse_teacher_from_response(row["response_text"])
  student_tool = row.get("student_action", {}).get("name")
  teacher_tool = teacher_action.get("name")

  from scape.eval.learnability_metrics_v2 import tool_name_distribution_js
  from scape.training.tool_opd import normalize_probs

  legal = [
    "fan_out_search", "search_corpus", "grep_corpus", "read_document",
    "review_docs", "curate", "verify", "end_search",
  ]
  t_logits = {n: (2.0 if n == teacher_tool else 0.0) for n in legal}
  s_logits = {n: (1.5 if n == student_tool else 0.0) for n in legal}
  js_name = tool_name_distribution_js(t_logits, s_logits)

  minimal_tokens = len(json.dumps(v3_view.get("minimal_graph_render", {})))
  full_tokens = len(json.dumps(v3_view))

  gh = {
    **row,
    "prompt_reduced": prompt_v2,
    "prompt_full": prompt_v3,
    "student_view": "V2_GRAPH_STATE_ONLY",
    "teacher_view": "V3_GRAPH_STATE_PLUS_MINIMAL_RENDER",
    "graph_state_size": v2_view.get("graph_node_count", 0),
    "minimal_render_tokens": minimal_tokens,
    "full_render_tokens": full_tokens,
    "turn": snap.step,
    "student_tool": student_tool,
    "teacher_tool": teacher_tool,
    "teacher_entropy": 0.0,
    "JS_name": js_name,
    "component_id": "evidence_graph_hybrid",
    "legacy_scope_path_used": False,
  }
  return gh


def _parse_teacher_from_response(response_text: str) -> dict[str, Any]:
  lines = response_text.strip().splitlines()
  if not lines:
    return {"name": None, "arguments": {}}
  name = lines[0].replace("to=", "").strip()
  args: dict[str, Any] = {}
  for line in lines[1:]:
    if line.strip().startswith("{") and line.strip().endswith("}"):
      args = json.loads(line.strip())
      break
  return {"name": name, "arguments": args}


def collect_graph_hybrid_dataset(
  *,
  n_states: int,
  seed: int = 42,
  out_path: Path | None = None,
  query_prefix: str = "gh_q",
) -> list[dict[str, Any]]:
  rng = random.Random(seed)
  rows: list[dict[str, Any]] = []
  for i in range(n_states):
    qid = f"{query_prefix}{i:04d}"
    step = 1 + (i % 3)
    rows.append(build_graph_hybrid_row(query_id=qid, step=step, rng=rng))
  if out_path is not None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
      for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
  return rows


def build_graph_hybrid_splits(
  *,
  out_dir: Path,
  seed: int = 42,
) -> dict[str, Any]:
  out_dir.mkdir(parents=True, exist_ok=True)
  splits = {
    "GH_TRAIN_8K": ("gh_train", 8000, seed),
    "GH_VALID_1K": ("gh_valid", 1000, seed + 1),
    "GH_TEST_1K": ("gh_test", 1000, seed + 2),
  }
  meta: dict[str, Any] = {
    "student_view": "V2_GRAPH_STATE_ONLY",
    "teacher_view": "V3_GRAPH_STATE_PLUS_MINIMAL_RENDER",
    "query_disjoint": True,
    "splits": {},
  }
  for name, (prefix, n, split_seed) in splits.items():
    path = out_dir / f"{name}.jsonl"
    rows = collect_graph_hybrid_dataset(
      n_states=n,
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
