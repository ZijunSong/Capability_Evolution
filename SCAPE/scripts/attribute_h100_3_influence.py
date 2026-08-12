#!/usr/bin/env python3
"""Attribute H100-3 real influence by turn, tool, and argument class."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
TARGET_COMPONENTS = ("evidence_graph", "importance_tagging", "verify_tool")
TOOL_CANONICAL = {
    "fan_out_search": "search",
    "search_corpus": "search",
    "grep_corpus": "grep",
    "read_document": "read",
    "review_docs": "review",
    "curate": "curate",
    "verify": "verify",
    "end_search": "end",
    "end": "end",
}


def fnum(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def entropy(probs: Any) -> float:
    if not isinstance(probs, dict):
        return 0.0
    vals = [fnum(v) for v in probs.values() if fnum(v) > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return -sum((v / total) * math.log(v / total, 2) for v in vals)


def tool_name(action: Any) -> str:
    if isinstance(action, dict):
        name = str(action.get("name") or action.get("tool") or "unknown")
    else:
        name = str(action or "unknown")
    return TOOL_CANONICAL.get(name, name)


def args_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, dict) and isinstance(action.get("arguments"), dict):
        return action["arguments"]
    return {}


def turn_bucket(step: Any) -> str:
    s = int(fnum(step))
    if s <= 1:
        return "early"
    if s <= 4:
        return "middle"
    return "late"


def arg_class_from_keys(*arg_maps: dict[str, Any]) -> str:
    keys: set[str] = set()
    for arg_map in arg_maps:
        keys.update(str(k) for k in arg_map.keys())
    if not keys:
        return "termination reason"
    if keys & {"query", "queries", "pattern", "keywords"}:
        return "query text"
    if keys & {"doc_id", "doc_ids", "ids", "source_ids", "target_ids"}:
        return "doc ids"
    if keys & {"add_ids", "remove_ids", "keep_ids", "drop_ids", "curated_ids"}:
        return "add/remove ids"
    if keys & {"importance", "importance_value", "importance_values", "score", "scores", "priority"}:
        return "importance value"
    if keys & {"claim", "claims", "hypothesis", "answer", "evidence", "rationale"}:
        return "claim text"
    if keys & {"reason", "termination_reason", "done", "final"}:
        return "termination reason"
    return "other arguments"


def disagreement(student: dict[str, Any], teacher: dict[str, Any]) -> str:
    sn, tn = tool_name(student), tool_name(teacher)
    if sn != tn:
        return "name change"
    sa, ta = args_dict(student), args_dict(teacher)
    if json.dumps(sa, sort_keys=True, ensure_ascii=False) != json.dumps(ta, sort_keys=True, ensure_ascii=False):
        return "args-only"
    return "no meaningful change"


def add_metric(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["n"] += 1
    for k in ("I_name_normalized", "I_name_raw", "I_args_raw", "I_arg_key", "I_arg_value"):
        bucket[k].append(fnum(row.get(k)))
    bucket["teacher_entropy"].append(entropy(row.get("P_tool_name_full")))
    bucket["student_entropy"].append(entropy(row.get("P_tool_name_reduced")))
    d = row["_disagreement"]
    bucket["disagreements"][d] += 1


def summarize_bucket(key: tuple[str, ...], bucket: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    out = {field: val for field, val in zip(fields, key)}
    out.update(
        {
            "n_states": bucket["n"],
            "I_name_mean": mean(bucket["I_name_normalized"]),
            "I_name_raw_mean": mean(bucket["I_name_raw"]),
            "I_args_mean": mean(bucket["I_args_raw"]),
            "I_arg_key_mean": mean(bucket["I_arg_key"]),
            "I_arg_value_mean": mean(bucket["I_arg_value"]),
            "teacher_entropy_mean": mean(bucket["teacher_entropy"]),
            "student_entropy_mean": mean(bucket["student_entropy"]),
            "tool_name_disagreement_rate": bucket["disagreements"]["name change"] / bucket["n"] if bucket["n"] else 0.0,
            "args_only_disagreement_rate": bucket["disagreements"]["args-only"] / bucket["n"] if bucket["n"] else 0.0,
        }
    )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict[str, Any]], cols: list[str], limit: int | None = None) -> list[str]:
    use = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---" for _ in cols]) + "|"]
    for r in use:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def recommendation(component: str, tool_rows: list[dict[str, Any]], arg_rows: list[dict[str, Any]], totals: dict[str, Any]) -> list[str]:
    top_tools = sorted([r for r in tool_rows if r["component"] == component], key=lambda r: r["I_name_mean"] + r["I_args_mean"], reverse=True)[:4]
    top_args = sorted([r for r in arg_rows if r["component"] == component], key=lambda r: r["I_args_mean"], reverse=True)[:4]
    total = totals[component]
    i_name = mean(total["I_name_normalized"])
    i_args = mean(total["I_args_raw"])
    args_heavy = i_args > i_name * 1.5
    name_heavy = i_name > i_args * 1.5
    if args_heavy:
        weights = "name: medium, args: high"
    elif name_heavy:
        weights = "name: high, args: medium"
    else:
        weights = "name: high, args: high"
    return [
        f"### {component}",
        "",
        f"- mean I_name={i_name:.6f}; mean I_args={i_args:.6f}; recommended follow-up weights: **{weights}**.",
        f"- train event/tool focus: {', '.join(r['student_tool'] for r in top_tools) if top_tools else 'none'}.",
        f"- argument focus: {', '.join(r['argument_class'] for r in top_args) if top_args else 'none'}.",
        "- H20 V0 should still start with uniform name+args tool-token KL; this weighting is for later ablations.",
        "",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=REPO / "outputs" / "h100_3_real_influence" / "REAL_INFLUENCE_PER_STATE.jsonl")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "h100_3_influence_attribution")
    ap.add_argument("--components", nargs="*", default=list(TARGET_COMPONENTS))
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("component") not in set(args.components):
                continue
            student = row.get("student_executed_tool_action") or {}
            teacher = row.get("teacher_full_greedy_tool_call") or {}
            row["_student_tool"] = tool_name(student)
            row["_teacher_tool"] = tool_name(teacher)
            row["_disagreement"] = disagreement(student, teacher)
            row["_argument_class"] = arg_class_from_keys(args_dict(student), args_dict(teacher))
            row["_turn_bucket"] = turn_bucket(row.get("step"))
            rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(lambda: {"n": 0, "I_name_normalized": [], "I_name_raw": [], "I_args_raw": [], "I_arg_key": [], "I_arg_value": [], "teacher_entropy": [], "student_entropy": [], "disagreements": Counter()})
    by_tool = defaultdict(lambda: {"n": 0, "I_name_normalized": [], "I_name_raw": [], "I_args_raw": [], "I_arg_key": [], "I_arg_value": [], "teacher_entropy": [], "student_entropy": [], "disagreements": Counter()})
    by_arg = defaultdict(lambda: {"n": 0, "I_name_normalized": [], "I_name_raw": [], "I_args_raw": [], "I_arg_key": [], "I_arg_value": [], "teacher_entropy": [], "student_entropy": [], "disagreements": Counter()})
    by_turn = defaultdict(lambda: {"n": 0, "I_name_normalized": [], "I_name_raw": [], "I_args_raw": [], "I_arg_key": [], "I_arg_value": [], "teacher_entropy": [], "student_entropy": [], "disagreements": Counter()})

    for row in rows:
        c = row["component"]
        add_metric(totals[c], row)
        add_metric(by_tool[(c, row["_turn_bucket"], row["_student_tool"], row["_teacher_tool"])], row)
        add_metric(by_arg[(c, row["_argument_class"], row["_disagreement"])], row)
        add_metric(by_turn[(c, row["_turn_bucket"])], row)

    tool_rows = sorted([summarize_bucket(k, v, ["component", "turn_bucket", "student_tool", "teacher_tool"]) for k, v in by_tool.items()], key=lambda r: (r["component"], -r["I_name_mean"] - r["I_args_mean"]))
    arg_rows = sorted([summarize_bucket(k, v, ["component", "argument_class", "disagreement"]) for k, v in by_arg.items()], key=lambda r: (r["component"], -r["I_args_mean"]))
    turn_rows = sorted([summarize_bucket(k, v, ["component", "turn_bucket"]) for k, v in by_turn.items()], key=lambda r: (r["component"], r["turn_bucket"]))
    total_rows = sorted([summarize_bucket((k,), v, ["component"]) for k, v in totals.items()], key=lambda r: -r["I_name_mean"] - r["I_args_mean"])

    write_csv(args.out / "INFLUENCE_BY_TOOL.csv", tool_rows)
    write_csv(args.out / "INFLUENCE_BY_ARGUMENT_CLASS.csv", arg_rows)
    write_csv(args.out / "INFLUENCE_BY_TURN.csv", turn_rows)
    write_csv(args.out / "INFLUENCE_TOTALS.csv", total_rows)

    archetypes = sorted(rows, key=lambda r: fnum(r.get("I_name_normalized")) + fnum(r.get("I_args_raw")), reverse=True)[:10]
    with (args.out / "HIGH_INFLUENCE_ARCHETYPES.jsonl").open("w", encoding="utf-8") as f:
        for r in archetypes:
            f.write(json.dumps({
                "component": r.get("component"),
                "query_id": r.get("query_id"),
                "step": r.get("step"),
                "turn_bucket": r["_turn_bucket"],
                "student_tool": r["_student_tool"],
                "teacher_tool": r["_teacher_tool"],
                "disagreement": r["_disagreement"],
                "argument_class": r["_argument_class"],
                "I_name_normalized": fnum(r.get("I_name_normalized")),
                "I_args_raw": fnum(r.get("I_args_raw")),
                "student_executed_tool_action": r.get("student_executed_tool_action"),
                "teacher_full_greedy_tool_call": r.get("teacher_full_greedy_tool_call"),
            }, ensure_ascii=False) + "\n")

    for comp, filename in [
        ("evidence_graph", "EVIDENCE_GRAPH_ATTRIBUTION.md"),
        ("importance_tagging", "IMPORTANCE_TAGGING_ATTRIBUTION.md"),
        ("verify_tool", "VERIFY_TOOL_ATTRIBUTION.md"),
    ]:
        comp_tool = [r for r in tool_rows if r["component"] == comp]
        comp_arg = [r for r in arg_rows if r["component"] == comp]
        comp_turn = [r for r in turn_rows if r["component"] == comp]
        comp_arch = [json.loads(json.dumps({
            "query_id": r.get("query_id"), "step": r.get("step"), "turn_bucket": r["_turn_bucket"],
            "student_tool": r["_student_tool"], "teacher_tool": r["_teacher_tool"],
            "disagreement": r["_disagreement"], "argument_class": r["_argument_class"],
            "I_name_normalized": fnum(r.get("I_name_normalized")), "I_args_raw": fnum(r.get("I_args_raw")),
        }, ensure_ascii=False)) for r in sorted([x for x in rows if x["component"] == comp], key=lambda x: fnum(x.get("I_name_normalized")) + fnum(x.get("I_args_raw")), reverse=True)[:10]]
        lines = [f"# {comp} attribution", "", "## Totals", ""]
        lines += md_table([r for r in total_rows if r["component"] == comp], ["component", "n_states", "I_name_mean", "I_args_mean", "teacher_entropy_mean", "student_entropy_mean", "tool_name_disagreement_rate", "args_only_disagreement_rate"])
        lines += ["", "## By turn", ""] + md_table(comp_turn, ["component", "turn_bucket", "n_states", "I_name_mean", "I_args_mean", "tool_name_disagreement_rate", "args_only_disagreement_rate"])
        lines += ["", "## By tool", ""] + md_table(comp_tool, ["turn_bucket", "student_tool", "teacher_tool", "n_states", "I_name_mean", "I_args_mean", "teacher_entropy_mean", "student_entropy_mean", "tool_name_disagreement_rate", "args_only_disagreement_rate"], 20)
        lines += ["", "## By argument class", ""] + md_table(comp_arg, ["argument_class", "disagreement", "n_states", "I_args_mean", "I_arg_key_mean", "I_arg_value_mean", "args_only_disagreement_rate"], 20)
        lines += ["", "## Top-10 highest influence state archetypes", ""] + md_table(comp_arch, ["query_id", "step", "turn_bucket", "student_tool", "teacher_tool", "disagreement", "argument_class", "I_name_normalized", "I_args_raw"])
        (args.out / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

    rec = ["# H20 loss recommendation", "", "首轮 H20 V0 仍按调度要求使用 **uniform name+args tool-token KL**；以下只用于后续 ablation/stratification。", ""]
    for comp in args.components:
        rec += recommendation(comp, tool_rows, arg_rows, totals)
    (args.out / "H20_LOSS_RECOMMENDATION.md").write_text("\n".join(rec), encoding="utf-8")

    manifest = {
        "input": str(args.input),
        "n_rows": len(rows),
        "components": list(args.components),
        "gpu_rescore": "skipped: REAL_INFLUENCE_PER_STATE.jsonl already contains per-state P_tool_name_full/P_tool_name_reduced, I_name, I_args, and null statistics",
        "outputs": [
            "INFLUENCE_BY_TOOL.csv",
            "INFLUENCE_BY_ARGUMENT_CLASS.csv",
            "EVIDENCE_GRAPH_ATTRIBUTION.md",
            "IMPORTANCE_TAGGING_ATTRIBUTION.md",
            "VERIFY_TOOL_ATTRIBUTION.md",
            "HIGH_INFLUENCE_ARCHETYPES.jsonl",
            "H20_LOSS_RECOMMENDATION.md",
        ],
    }
    (args.out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
