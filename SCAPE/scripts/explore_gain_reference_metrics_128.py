#!/usr/bin/env python3
"""Explore paired reference metrics from frozen 128-state artifacts.

This is a diagnostic scorer. It never replaces the formal qrel recall gates;
rows with complete endpoint provenance are preferred, while compact artifacts
are included only when the needed fields are present.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "outputs/0821_gain_reference_metrics_128"

FILES = {
    "importance_tagging": ROOT / "outputs/0820_importance_tagging_recall_128/scored/IMPORTANCE_RECALL_PER_STATE.jsonl",
    "adaptive_rerank_instruction": ROOT / "outputs/0820_adaptive_rerank_instruction_recall_128/ADAPTIVE_RERANK_RECALL_PER_STATE.jsonl",
    "content_dedup": ROOT / "outputs/0820_content_dedup_real_recall_128/CONTENT_DEDUP_RECALL_PER_STATE.jsonl",
    "evidence_graph": ROOT / "outputs/0820_evidence_graph_recall_formal_20260820/scored/EVIDENCE_GRAPH_EVIDENCE_RECALL_PER_STATE.jsonl",
    "subtractive_curation": ROOT / "outputs/0820_subtractive_curation_recall_128_final/SUBTRACTIVE_CANDIDATE_ACTIVATED_RECALL_PER_STATE.jsonl",
    "importance_tagging_plus_subtractive_curation": ROOT / "outputs/0820_joint_importance_subtractive_recall_128_final/JOINT_RECALL_PER_STATE.jsonl",
    "sentence_compress": ROOT / "outputs/0820_sentence_compress_formal_fork_k128_frozen_pool1024/SENTENCE_COMPRESS_USABLE_EVIDENCE_RECALL_PER_STATE.jsonl",
    "auto_populate_first_search": ROOT / "outputs/0820_auto_populate_first_search_recall_128_rerun/AUTO_VALUE_CONFIRM/AUTO_VALUE_PER_STATE.jsonl",
}

# Some legacy paired artifacts store a row-level ``tool_search_cost`` that is
# already T-S.  Keep the branch-level source explicit for the reference cost
# metric rather than allowing that delta to be mistaken for both branch values.
COST_SOURCES = {
    "subtractive_curation": FILES["subtractive_curation"],
    "importance_tagging_plus_subtractive_curation": ROOT / "outputs/0820_joint_importance_subtractive_preopd_fork_pilot128_retry/JOINT_PREOPD_VALUE_PER_STATE.jsonl",
}


def as_set(value):
    if not isinstance(value, list):
        return set()
    return {str(x) for x in value}


def action_name(action):
    if not isinstance(action, dict):
        return None
    return action.get("name")


def endpoint(row, branch):
    label = "student" if branch == "S" else "teacher"
    candidates = [
        f"branch_{branch}_endpoint",
        f"{branch}_endpoint",
        f"{label}_endpoint",
        branch,
    ]
    for key in candidates:
        if isinstance(row.get(key), dict):
            return row[key]
    return None


def branch_metrics(row, branch):
    for key in (f"branch_{branch}_metrics", f"{branch}_metrics"):
        if isinstance(row.get(key), dict):
            return row[key]
    return {}


def branch_trace(row, branch):
    value = row.get(f"branch_{branch}_trace")
    return value if isinstance(value, list) else []


def action(row, branch):
    for key in (f"a_{branch}", f"{branch}_action", f"action_{branch}"):
        if key in row:
            return row[key]
    return None


def normalize_endpoint(row, branch):
    ep = endpoint(row, branch)
    if ep is not None:
        gold = as_set(ep.get("gold_evidence_ids", row.get("gold_evidence_ids", [])))
        candidates = as_set(ep.get("final_candidate_evidence_ids", ep.get("candidate_ids", [])))
        activated = as_set(ep.get("final_activated_evidence_ids", ep.get("activated_ids", [])))
        if not activated:
            activated = as_set(ep.get("final_usable_evidence_ids", []))
        reads = as_set(ep.get("successful_read_ids_within_k", ep.get("read_evidence_ids_within_k", [])))
        attempts = ep.get("read_attempt_ids_within_k", [])
        retained = as_set(ep.get("read_ids_retained_at_endpoint", ep.get("read_ids_entered_context", [])))
        context = ep.get("context_evidence_ids_by_step", [])
        return {"gold": gold, "candidate": candidates, "activated": activated,
                "reads": reads, "attempts": len(attempts) if isinstance(attempts, list) else 0,
                "retained": retained, "context": context if isinstance(context, list) else []}

    # compact importance artifact uses teacher/student directly
    obj = row.get("teacher" if branch == "T" else "student")
    if isinstance(obj, dict):
        return {"gold": as_set(row.get("gold_evidence_ids", [])),
                "candidate": as_set(obj.get("final_candidate_evidence_ids", [])),
                "activated": as_set(obj.get("final_activated_evidence_ids", [])),
                "reads": as_set(obj.get("successful_read_ids_within_k", [])),
                "attempts": len(obj.get("read_attempt_ids_within_k", [])),
                "retained": as_set(obj.get("read_ids_retained_at_endpoint", [])),
                "context": obj.get("context_evidence_ids_by_step", []) or []}

    # evidence_graph compact scorer
    prefix = "teacher" if branch == "T" else "student"
    if f"{prefix}_candidate_ids" in row:
        return {"gold": as_set(row.get("gold_evidence_ids", [])),
                "candidate": as_set(row.get(f"{prefix}_candidate_ids", [])),
                "activated": as_set(row.get(f"{prefix}_activated_ids", [])),
                "reads": set(), "attempts": 0, "retained": set(), "context": []}
    return None


def value_metric(row, branch, name):
    metrics = branch_metrics(row, branch)
    if name == "cost" and isinstance(metrics.get("tool_search_cost"), (int, float)):
        return float(metrics["tool_search_cost"])
    if name in metrics and isinstance(metrics[name], (int, float)):
        return float(metrics[name])
    ep = endpoint(row, branch)
    if ep and isinstance(ep.get(name), (int, float)):
        return float(ep[name])
    aliases = {
        "cost": ["tool_cost", "tool_search_cost"],
        "utility": ["objective_utility", "utility"],
        "duplicate": ["duplicate_read_count"],
        "coverage": ["evidence_coverage"],
        "redundancy": ["redundancy"],
        "unsupported": ["unsupported_claims"],
        "successful_reads": ["successful_read_count"],
    }
    for alias in aliases.get(name, []):
        if alias in row and isinstance(row[alias], (int, float)):
            return float(row[alias])
        if ep and isinstance(ep.get(alias), (int, float)):
            return float(ep[alias])
        if alias in metrics and isinstance(metrics[alias], (int, float)):
            return float(metrics[alias])
    return None


def paired_row(row):
    s = normalize_endpoint(row, "S")
    t = normalize_endpoint(row, "T")
    if not s or not t:
        return None
    gold = t["gold"] or s["gold"] or as_set(row.get("gold_evidence_ids", []))
    actions_differ = action_name(action(row, "T")) != action_name(action(row, "S"))
    t_context = t["context"]
    s_context = s["context"]

    def first_gt(context):
        for i, docs in enumerate(context, 1):
            if as_set(docs) & gold:
                return i
        return None

    def context_auc(context):
        if not context:
            return None
        seen = set()
        vals = []
        for docs in context:
            seen |= as_set(docs)
            vals.append(len(seen & gold) / len(gold) if gold else 0.0)
        return sum(vals) / len(vals) if vals else None

    out = {
        "query_id": row.get("query_id"), "state_id": row.get("state_id"),
        "component": row.get("component"), "K": row.get("K"),
        "candidate_set_changed": int(t["candidate"] != s["candidate"]),
        "activated_set_changed": int(t["activated"] != s["activated"]),
        "candidate_jaccard": (len(t["candidate"] & s["candidate"]) / len(t["candidate"] | s["candidate"])) if (t["candidate"] | s["candidate"]) else 1.0,
        "activated_jaccard": (len(t["activated"] & s["activated"]) / len(t["activated"] | s["activated"])) if (t["activated"] | s["activated"]) else 1.0,
        "candidate_gt_added": len((t["candidate"] - s["candidate"]) & gold),
        "candidate_gt_lost": len((s["candidate"] - t["candidate"]) & gold),
        "activated_gt_added": len((t["activated"] - s["activated"]) & gold),
        "activated_gt_lost": len((s["activated"] - t["activated"]) & gold),
        "candidate_delta": (len(t["candidate"] & gold) - len(s["candidate"] & gold)) / len(gold) if gold else 0.0,
        "activated_delta": (len(t["activated"] & gold) - len(s["activated"] & gold)) / len(gold) if gold else 0.0,
        "activated_precision_delta": ((len(t["activated"] & gold) / len(t["activated"]) if t["activated"] else 0.0) - (len(s["activated"] & gold) / len(s["activated"]) if s["activated"] else 0.0)),
        "action_disagreement": int(actions_differ),
        "first_gt_context_step_T": first_gt(t_context), "first_gt_context_step_S": first_gt(s_context),
        "context_auc_T": context_auc(t_context), "context_auc_S": context_auc(s_context),
    }
    for name in ("cost", "utility", "duplicate", "coverage", "redundancy", "unsupported", "successful_reads"):
        tv, sv = value_metric(row, "T", name), value_metric(row, "S", name)
        if tv is not None and sv is not None:
            out[f"{name}_delta"] = tv - sv
    out["read_attempt_delta"] = t["attempts"] - s["attempts"]
    out["successful_read_set_delta"] = len(t["reads"]) - len(s["reads"])
    out["retained_read_set_delta"] = len(t["retained"]) - len(s["retained"])
    return out


def load_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = paired_row(row)
            if item is not None:
                rows.append(item)
    return rows


def cost_summary(path):
    """Aggregate branch-level costs, never the row-level T-S convenience field."""
    if not path.exists():
        return None
    groups = defaultdict(lambda: {"deltas": [], "teacher": [], "student": []})
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = branch_metrics(row, "T").get("tool_search_cost")
            s = branch_metrics(row, "S").get("tool_search_cost")
            if isinstance(t, (int, float)) and isinstance(s, (int, float)):
                key = str(row.get("K"))
                groups[key]["teacher"].append(float(t))
                groups[key]["student"].append(float(s))
                groups[key]["deltas"].append(float(t) - float(s))

    def one(values):
        deltas = values["deltas"]
        teacher = values["teacher"]
        student = values["student"]
        return {
            "mean": mean(deltas),
            "std": pstdev(deltas),
            "nonzero": sum(abs(v) > 1e-12 for v in deltas),
            "positive": sum(v > 1e-12 for v in deltas),
            "negative": sum(v < -1e-12 for v in deltas),
            "n": len(deltas),
            "teacher_mean": mean(teacher),
            "student_mean": mean(student),
        }

    if not groups:
        return None
    all_values = {"deltas": [], "teacher": [], "student": []}
    for values in groups.values():
        for key in all_values:
            all_values[key].extend(values[key])
    result = one(all_values)
    result["by_K"] = {key: one(groups[key]) for key in sorted(groups)}
    return result


def summarize(rows):
    numeric = [k for k in rows[0] if isinstance(rows[0].get(k), (int, float))] if rows else []
    out = {"n": len(rows)}
    for key in numeric:
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(float(r[key]))]
        if not vals:
            continue
        out[key] = {"mean": mean(vals), "std": pstdev(vals), "nonzero": sum(abs(v) > 1e-12 for v in vals), "positive": sum(v > 1e-12 for v in vals), "negative": sum(v < -1e-12 for v in vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], {}
    for component, path in FILES.items():
        rows = load_rows(path)
        for row in rows:
            row["source"] = str(path)
            row["component"] = row.get("component") or component
        all_rows.extend(rows)
        summary = summarize(rows)
        if component in COST_SOURCES:
            # The joint cost source is the complete 512-row pilot artifact;
            # the later recall replay is incomplete and is not used here.
            summary["cost_delta"] = cost_summary(COST_SOURCES[component])
            summary["cost_source"] = str(COST_SOURCES[component])
        summaries[component] = {"path": str(path), "summary": summary}
    with (args.output_dir / "GAIN_REFERENCE_METRICS_PER_STATE.jsonl").open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    result = {"definition": {
        "action_disagreement": "first-action tool name differs",
        "candidate_set_changed": "endpoint candidate ID set differs",
        "activated_set_changed": "endpoint activated ID set differs",
        "candidate_gt_added_lost": "qrel IDs in directional endpoint set difference",
        "context_auc": "mean cumulative qrel coverage over retained context steps; diagnostic, not endpoint recall",
        "cost_utility_components": "paired Teacher minus Student values when raw artifact provides both branches",
    }, "components": summaries, "total_rows": len(all_rows)}
    (args.output_dir / "GAIN_REFERENCE_METRICS_SUMMARY.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
