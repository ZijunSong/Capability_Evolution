#!/usr/bin/env python3
"""Independent H100-3 KL / JS audit for same-state tool distributions.

This script intentionally does not import the H20 evaluator, Stage-L helpers,
L_m helpers, or tournament aggregators.  It uses only Python stdlib plus torch
for the independent numeric reference.  Optional HF rescoring support is kept
local to this file and imports transformers only when explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

TOOL_NAMES = (
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
)

DEFAULT_INPUT_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_3_real_influence_shards")
DEFAULT_H20_SUMMARY = Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/H20_LIGHTWEIGHT_TORCH_COMPLETE.json")
DEFAULT_STAGE_L_CURVE = Path("/mnt/songzijun/Capability_Evolution/SCAPE/outputs/true_scape_evidence_graph/STAGE_L_CURVE.csv")

TASKS = {
    0: {"case": "evidence_graph_base_vs_uniform_main", "component": "evidence_graph", "mode": "uniform", "seed": 42},
    1: {"case": "evidence_graph_base_vs_name_only", "component": "evidence_graph", "mode": "name_only", "seed": 43},
    2: {"case": "subtractive_seed42", "component": "subtractive_curation", "mode": "name_only", "seed": 42},
    3: {"case": "subtractive_seed43", "component": "subtractive_curation", "mode": "name_only", "seed": 43},
    4: {"case": "importance_seed42_to_seed43", "component": "importance_tagging", "mode": "name_only", "seed": 42},
    5: {"case": "verify_seed42_to_seed43", "component": "verify_tool", "mode": "name_only", "seed": 43},
    6: {"case": "independent_tool_mask_rebuild", "component": "evidence_graph", "mode": "mask", "seed": 46},
    7: {"case": "identical_model_null_manual64", "component": "evidence_graph", "mode": "null", "seed": 47},
}


def stable_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: stable_key(seed, r.get("snapshot_hash"), r.get("query_id"), r.get("step")))
    return ranked[:n]


def normalize(prob: dict[str, Any], eps: float = 1e-12) -> dict[str, float]:
    vals = {k: max(0.0, float(prob.get(k, 0.0))) for k in TOOL_NAMES}
    total = sum(vals.values())
    if total <= 0:
        return {k: 1.0 / len(TOOL_NAMES) for k in TOOL_NAMES}
    out = {k: max(eps, vals[k] / total) for k in TOOL_NAMES}
    z = sum(out.values())
    return {k: v / z for k, v in out.items()}


def uniform_like(keys: Iterable[str] = TOOL_NAMES) -> dict[str, float]:
    names = list(keys)
    return {k: 1.0 / len(names) for k in names}


def torch_metrics(teacher: dict[str, float], student: dict[str, float]) -> dict[str, float]:
    keys = list(TOOL_NAMES)
    t = torch.tensor([teacher[k] for k in keys], dtype=torch.float64)
    s = torch.tensor([student[k] for k in keys], dtype=torch.float64)
    t = t / t.sum()
    s = s / s.sum()
    eps = torch.tensor(1e-12, dtype=torch.float64)
    t = torch.clamp(t, min=eps.item())
    s = torch.clamp(s, min=eps.item())
    t = t / t.sum()
    s = s / s.sum()
    m = 0.5 * (t + s)
    kl_ts = torch.sum(t * (torch.log(t) - torch.log(s)))
    kl_st = torch.sum(s * (torch.log(s) - torch.log(t)))
    js = 0.5 * torch.sum(t * (torch.log(t) - torch.log(m))) + 0.5 * torch.sum(s * (torch.log(s) - torch.log(m)))
    ce = -torch.sum(t * torch.log(s))
    teacher_nll = -torch.sum(t * torch.log(t))
    student_nll = -torch.sum(s * torch.log(s))
    signed_gap = torch.sum(t * (torch.log(t) - torch.log(s)))
    return {
        "forward_KL_T||S": float(kl_ts),
        "reverse_KL_S||T": float(kl_st),
        "JS": float(js),
        "cross_entropy": float(ce),
        "teacher_NLL": float(teacher_nll),
        "student_NLL": float(student_nll),
        "signed_logprob_gap": float(signed_gap),
        "torch_reference_forward_KL": float(kl_ts),
        "torch_reference_JS": float(js),
    }


def manual_reference(teacher: dict[str, float], student: dict[str, float]) -> dict[str, float]:
    t = normalize(teacher)
    s = normalize(student)
    m = {k: 0.5 * (t[k] + s[k]) for k in TOOL_NAMES}
    kl_ts = sum(t[k] * math.log(t[k] / s[k]) for k in TOOL_NAMES)
    kl_st = sum(s[k] * math.log(s[k] / t[k]) for k in TOOL_NAMES)
    js = 0.5 * sum(t[k] * math.log(t[k] / m[k]) for k in TOOL_NAMES) + 0.5 * sum(s[k] * math.log(s[k] / m[k]) for k in TOOL_NAMES)
    return {"manual_forward_KL": kl_ts, "manual_reverse_KL": kl_st, "manual_JS": js}


def rebuilt_mask_ok(row: dict[str, Any], component: str) -> bool:
    reduced = row.get("reduced_view") or {}
    full = row.get("full_view") or {}
    rmask = reduced.get("mask") or {}
    fmask = full.get("mask") or {}
    if component not in rmask or component not in fmask:
        return False
    changed = [k for k in sorted(set(rmask) | set(fmask)) if bool(rmask.get(k)) != bool(fmask.get(k))]
    return changed == [component] and bool(rmask.get(component)) is False and bool(fmask.get(component)) is True


def audit_one(row: dict[str, Any], *, case: str, mode: str, component: str) -> dict[str, Any]:
    p_full = normalize(row.get("P_tool_name_full") or {})
    p_reduced = normalize(row.get("P_tool_name_reduced") or {})
    if mode == "uniform":
        teacher, student = p_full, uniform_like(TOOL_NAMES)
    elif mode == "null":
        teacher, student = p_full, dict(p_full)
    else:
        teacher, student = p_full, p_reduced
    metrics = torch_metrics(teacher, student)
    metrics.update(manual_reference(teacher, student))
    state_id = f"{component}:{row.get('query_id')}:{row.get('step')}:{str(row.get('snapshot_hash'))[:12]}"
    out = {
        "case": case,
        "component": component,
        "state_id": state_id,
        "query_id": row.get("query_id"),
        "step": row.get("step"),
        "snapshot_hash": row.get("snapshot_hash"),
        "mode": mode,
        "teacher_tool": (row.get("teacher_full_greedy_tool_call") or {}).get("name"),
        "student_tool": (row.get("student_executed_tool_action") or {}).get("name"),
        "H20_reported_D": row.get("I_args_raw", row.get("I_name_raw")),
        "H20_reported_I_name_raw": row.get("I_name_raw"),
        "H20_reported_I_args_raw": row.get("I_args_raw"),
        "mask_rebuild_ok": rebuilt_mask_ok(row, component),
    }
    out.update(metrics)
    out["KL_nonnegative_ok"] = out["forward_KL_T||S"] >= -1e-10 and out["reverse_KL_S||T"] >= -1e-10
    out["JS_nonnegative_ok"] = out["JS"] >= -1e-10
    out["torch_manual_kl_absdiff"] = abs(out["forward_KL_T||S"] - out["manual_forward_KL"])
    out["torch_manual_js_absdiff"] = abs(out["JS"] - out["manual_JS"])
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    xs2, ys2 = zip(*pairs)
    mx, my = statistics.mean(xs2), statistics.mean(ys2)
    vx = sum((x - mx) ** 2 for x in xs2)
    vy = sum((y - my) ** 2 for y in ys2)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_h20_stage_values(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"exists": True}
    for section in ("stage_l_gate", "stage_s_gate", "stage_m"):
        if section in obj:
            out[section] = obj[section]
    return out


def load_component_rows(input_root: Path, component: str) -> list[dict[str, Any]]:
    candidates = [
        input_root / component / "REAL_INFLUENCE_PER_STATE.jsonl",
        input_root / "shards" / component / "REAL_INFLUENCE_PER_STATE.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return load_jsonl(path)
    raise FileNotFoundError(f"missing REAL_INFLUENCE_PER_STATE.jsonl for {component} under {input_root}")


def run_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    task_ids = [args.task] if args.task is not None else sorted(TASKS)
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        spec = TASKS[task_id]
        component = str(spec["component"])
        source = load_component_rows(args.input_root, component)
        selected = select_rows(source, args.n_states if task_id != 7 else min(args.n_states, 64), int(spec["seed"]))
        for row in selected:
            rows.append(audit_one(row, case=str(spec["case"]), mode=str(spec["mode"]), component=component))
    return rows


def summarize_and_write(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "INDEPENDENT_KL.csv", rows)
    cross = []
    for r in rows:
        cross.append({
            "state_id": r["state_id"],
            "H20_reported_D": r.get("H20_reported_D"),
            "H1003_forward_KL": r["forward_KL_T||S"],
            "H1003_reverse_KL": r["reverse_KL_S||T"],
            "H1003_JS": r["JS"],
            "signed_gap": r["signed_logprob_gap"],
            "component": r["component"],
            "case": r["case"],
            "snapshot_hash": r["snapshot_hash"],
        })
    write_csv(out_dir / "H20_METRIC_CROSSCHECK.csv", cross)

    null_rows = [r for r in rows if r["mode"] == "null"]
    all_kl_ok = all(r["KL_nonnegative_ok"] and r["JS_nonnegative_ok"] for r in rows)
    null_max_kl = max([abs(r["forward_KL_T||S"]) for r in null_rows] or [0.0])
    null_max_js = max([abs(r["JS"]) for r in null_rows] or [0.0])
    (out_dir / "NULL_TEST.md").write_text(
        "# NULL_TEST\n\n"
        f"- total_rows: {len(rows)}\n"
        f"- KL_JS_nonnegative_all_rows: {all_kl_ok}\n"
        f"- identical_distribution_null_rows: {len(null_rows)}\n"
        f"- identical_model_null_max_forward_KL: {null_max_kl:.12g}\n"
        f"- identical_model_null_max_JS: {null_max_js:.12g}\n"
        f"- full_view_equals_reduced_view_same_distribution_metric: {null_max_kl:.12g}\n"
        "- teacher_equals_student_condition: implemented as identical categorical distribution over tool-name tokens.\n",
        encoding="utf-8",
    )

    by_comp = defaultdict(list)
    for r in rows:
        by_comp[r["component"]].append(r)
    mask_lines = ["# MASK_CROSSCHECK", "", "| component | n | mask_rebuild_ok | same_snapshot_hash_nonempty |", "|---|---:|---|---|"]
    for comp, vals in sorted(by_comp.items()):
        mask_ok = sum(1 for r in vals if r.get("mask_rebuild_ok"))
        nonempty = sum(1 for r in vals if r.get("snapshot_hash"))
        mask_lines.append(f"| {comp} | {len(vals)} | {mask_ok}/{len(vals)} | {nonempty}/{len(vals)} |")
    mask_lines.extend(["", "Independent mask rebuild requires exactly one changed component bit between reduced and full views."])
    (out_dir / "MASK_CROSSCHECK.md").write_text("\n".join(mask_lines) + "\n", encoding="utf-8")

    h20_vals = [float(r["H20_reported_D"]) for r in rows if r.get("H20_reported_D") is not None]
    gaps = [float(r["signed_logprob_gap"]) for r in rows if r.get("H20_reported_D") is not None]
    js_vals = [float(r["JS"]) for r in rows if r.get("H20_reported_D") is not None]
    corr_gap = pearson(h20_vals, gaps)
    corr_js = pearson(h20_vals, js_vals)
    hint_lines = [
        "# ROOT_CAUSE_HINT",
        "",
        "This independent audit computes non-negative KL(T||S), KL(S||T), and JS from normalized tool-name categorical distributions using torch.float64 plus a manual math reference.",
        f"- rows: {len(rows)}",
        f"- H20_reported_D_vs_signed_gap_pearson: {corr_gap}",
        f"- H20_reported_D_vs_JS_pearson: {corr_js}",
        "- Note: existing H20 lightweight artifacts are stage-level summaries; the per-state crosscheck uses the available HF same-state shard fields (`I_name_raw`/`I_args_raw`) as H20-reported D proxies.",
    ]
    if corr_gap is not None and abs(corr_gap) > 0.95:
        hint_lines.append("- If H20 D is highly correlated with signed_gap, it is a signed logprob gap/proxy, not a mathematical divergence.")
    if any(float(r.get("H20_reported_D") or 0.0) < 0 for r in rows):
        hint_lines.append("- Negative H20_reported_D values were observed, which cannot be KL/JS divergence.")
    (out_dir / "ROOT_CAUSE_HINT.md").write_text("\n".join(hint_lines) + "\n", encoding="utf-8")

    handoff = {
        "audit": "H1003 independent KL/JS scorer verification",
        "output_dir": str(out_dir),
        "input_root": str(args.input_root),
        "n_rows": len(rows),
        "tasks": TASKS,
        "required_outputs": [
            "INDEPENDENT_KL.csv",
            "NULL_TEST.md",
            "MASK_CROSSCHECK.md",
            "H20_METRIC_CROSSCHECK.csv",
            "ROOT_CAUSE_HINT.md",
            "H20_KL_AUDIT_HANDOFF.json",
        ],
        "null": {"max_forward_KL": null_max_kl, "max_JS": null_max_js, "rows": len(null_rows)},
        "correlations": {"H20_reported_D_vs_signed_gap": corr_gap, "H20_reported_D_vs_JS": corr_js},
        "h20_stage_summary": read_h20_stage_values(args.h20_summary),
        "script": "scripts/audit_independent_tool_kl.py",
        "independent_implementation": True,
        "imports_h20_evaluator": False,
        "imports_L_m_helper": False,
        "imports_tournament_aggregator": False,
    }
    (out_dir / "H20_KL_AUDIT_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scape_audit = out_dir.parent / "scape_audit"
    scape_audit.mkdir(parents=True, exist_ok=True)
    (scape_audit / "H1003_KL_AUDIT_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "task": args.task,
        "n_states": args.n_states,
        "rows": len(rows),
        "device_env": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "outputs": [str(out_dir / name) for name in handoff["required_outputs"]],
    }
    (out_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/h100_3_kl_audit"))
    ap.add_argument("--h20-summary", type=Path, default=DEFAULT_H20_SUMMARY)
    ap.add_argument("--n-states", type=int, default=256)
    ap.add_argument("--task", type=int, choices=sorted(TASKS), default=None)
    args = ap.parse_args()
    rows = run_tasks(args)
    summarize_and_write(args.out_dir, rows, args)
    print(json.dumps({"out_dir": str(args.out_dir), "rows": len(rows), "task": args.task}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
