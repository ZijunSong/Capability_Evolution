"""Candidate selector -> CAPABILITY_PLACEMENT_MAP / CANDIDATE_SELECTION.

Heuristic scheduler score (NOT a paper-final formula):
  score ~ Contribution × Influence_above_null × semantic_fraction / runtime_cost
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scape.adapters.components import COMPONENT_TAXONOMY, RUNTIME_ANCHORS


def _semantic_fraction(component_id: str) -> float:
    meta = COMPONENT_TAXONOMY.get(component_id) or {}
    kind = str(meta.get("semantic_or_runtime", "runtime"))
    if kind == "semantic":
        return 1.0
    if kind == "hybrid":
        return 0.5
    return 0.0


def placement_score(row: Mapping[str, Any]) -> float:
    contrib = float(row.get("contribution", 0.0))
    influence = float(row.get("influence_above_null", 0.0))
    sem = float(row.get("semantic_fraction", _semantic_fraction(str(row["component_id"]))))
    raw_cost = float(row.get("runtime_cost", 1.0))
    # Non-positive cost means removing the component does not save runtime in the
    # current estimate; do not let that become an artificially huge priority.
    cost = raw_cost if raw_cost > 0 else float("inf")
    return (max(0.0, contrib) * max(0.0, influence) * sem) / cost


def is_forced_runtime_anchor(component_id: str) -> bool:
    if component_id in RUNTIME_ANCHORS:
        return True
    meta = COMPONENT_TAXONOMY.get(component_id) or {}
    return bool(meta.get("runtime_anchor"))


def select_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 2,
    exclude_content_dedup_as_a: bool = True,
) -> dict[str, Any]:
    enriched: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        cid = str(item["component_id"])
        item.setdefault("semantic_fraction", _semantic_fraction(cid))
        item["score"] = placement_score(item)
        item["runtime_anchor"] = is_forced_runtime_anchor(cid)
        # Priority buckets
        quality_ok = bool(item.get("quality_positive", float(item.get("contribution", 0.0)) > 0))
        influence_ok = float(item.get("influence_above_null", 0.0)) > 0
        semantic_ok = float(item["semantic_fraction"]) > 0
        if item["runtime_anchor"]:
            item["priority"] = "Runtime"
        elif quality_ok and influence_ok and semantic_ok:
            item["priority"] = "A"
        elif (not quality_ok) and float(item.get("runtime_cost", 0.0)) > 0:
            item["priority"] = "B"
        else:
            item["priority"] = "Hybrid"
        enriched.append(item)

    # Sort for map
    enriched.sort(key=lambda x: x["score"], reverse=True)

    # Select top-k from Priority A only; never fully-internalize runtime anchors
    pool = [
        x
        for x in enriched
        if x["priority"] == "A" and not x["runtime_anchor"]
    ]
    if exclude_content_dedup_as_a:
        pool = [x for x in pool if x["component_id"] != "content_dedup"]

    selected = pool[:top_k]
    labels = ["A", "B", "C", "D"]
    candidates = {}
    for i, row in enumerate(selected):
        candidates[labels[i]] = {
            "component_id": row["component_id"],
            "score": row["score"],
            "priority": row["priority"],
            "contribution": row.get("contribution"),
            "influence_above_null": row.get("influence_above_null"),
            "semantic_fraction": row.get("semantic_fraction"),
            "runtime_cost": row.get("runtime_cost"),
        }

    return {
        "rows": enriched,
        "candidates": candidates,
        "n_selected": len(candidates),
    }


def write_placement_map(result: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "CAPABILITY_PLACEMENT_MAP.csv"
    md_path = out_dir / "CAPABILITY_PLACEMENT_MAP.md"
    json_path = out_dir / "CAPABILITY_PLACEMENT_MAP.json"
    sel_path = out_dir / "CANDIDATE_SELECTION.json"

    rows = list(result["rows"])
    fieldnames = [
        "component_id",
        "priority",
        "score",
        "contribution",
        "influence_above_null",
        "semantic_fraction",
        "runtime_cost",
        "runtime_anchor",
        "quality_positive",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        "# Capability Placement Map",
        "",
        "| component | priority | score | contrib | influenceΔnull | semantic | cost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['component_id']} | {r['priority']} | {r['score']:.4f} | "
            f"{float(r.get('contribution') or 0):.4f} | "
            f"{float(r.get('influence_above_null') or 0):.4f} | "
            f"{float(r.get('semantic_fraction') or 0):.2f} | "
            f"{float(r.get('runtime_cost') or 0):.2f} |"
        )
    lines.append("")
    lines.append("## Selected candidates (max 2)")
    for label, c in (result.get("candidates") or {}).items():
        lines.append(f"- Candidate {label}: `{c['component_id']}` (score={c['score']:.4f})")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    sel_path.write_text(
        json.dumps(result.get("candidates") or {}, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "csv": csv_path,
        "md": md_path,
        "json": json_path,
        "selection": sel_path,
    }
