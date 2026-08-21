#!/usr/bin/env python3
"""Finalize the 0816-2 importance_tagging proper K4/K8 fork gate."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "outputs" / "0816_2_importance_proper_fork_formal_recovered"
OUT = REPO / "outputs" / "0816_2_importance_proper_formal_0817"
SEEDS = (8423, 8424)
HORIZONS = (4, 8)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return float(ys[idx])


def ci95_mean(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    m = mean(xs)
    if len(xs) < 2:
        return (m, m)
    se = statistics.stdev(xs) / (len(xs) ** 0.5)
    return (m - 1.96 * se, m + 1.96 * se)


def action_name(row: dict, key: str) -> str:
    return str((row.get(key) or {}).get("name") or "")


def trace_names(row: dict, key: str) -> list[str]:
    out = []
    for item in row.get(key) or []:
        out.append(str(((item.get("action") or {}).get("name")) or ""))
    return out


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    shard_summaries = []
    for seed in SEEDS:
        for k in HORIZONS:
            path = SOURCE / f"K{k}_seed{seed}" / "shards" / f"importance_tagging_K{k}.jsonl"
            rows = read_rows(path)
            vals = [float(r["branch_T_minus_S"]) for r in rows]
            lo, hi = ci95_mean(vals)
            shard_summaries.append({
                "seed": seed,
                "K": k,
                "n_states": len(rows),
                "mean_branch_T_minus_S": mean(vals),
                "median_branch_T_minus_S": statistics.median(vals) if vals else 0.0,
                "ci95_low_normal_approx": lo,
                "ci95_high_normal_approx": hi,
                "positive_count": sum(v > 0 for v in vals),
                "negative_count": sum(v < 0 for v in vals),
                "zero_count": sum(v == 0 for v in vals),
                "p05": pct(vals, 0.05),
                "p95": pct(vals, 0.95),
                "mean_curated_evidence_gain": mean([float(r.get("curated_evidence_gain", 0.0)) for r in rows]),
                "mean_useful_unique_docs": mean([float(r.get("useful_unique_docs", 0.0)) for r in rows]),
                "mean_evidence_coverage": mean([float(r.get("evidence_coverage", 0.0)) for r in rows]),
                "mean_tool_search_cost_delta": mean([float(r.get("tool_search_cost", 0.0)) for r in rows]),
                "source_file": str(path),
            })
            all_rows.extend(rows)

    value_path = OUT / "IMPORTANCE_PROPER_VALUE_PER_STATE.jsonl"
    with value_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gate_passed = all(
        s["mean_branch_T_minus_S"] > 0 and s["ci95_low_normal_approx"] > 0
        for s in shard_summaries
    )
    k4_positive = all(s["mean_branch_T_minus_S"] > 0 for s in shard_summaries if s["K"] == 4)
    k8_consistent = all(s["mean_branch_T_minus_S"] > 0 for s in shard_summaries if s["K"] == 8)
    decision = "proper_k4_k8_gate_passed" if gate_passed else "proper_k4_k8_gate_failed"
    gate = {
        "status": decision,
        "component": "importance_tagging",
        "contract": "same xi_t; importance ON first branch vs OFF reduced branch; both continuations reduced policy; no full-harness takeover",
        "source_dir": str(SOURCE),
        "output_dir": str(OUT),
        "seeds": list(SEEDS),
        "K": list(HORIZONS),
        "n_rows": len(all_rows),
        "n_states_per_seed_k": 512,
        "k4_positive": k4_positive,
        "k8_direction_consistent_positive": k8_consistent,
        "gate_passed": gate_passed,
        "decision": "do_not_start_importance_lora_opd" if not gate_passed else "importance_lora_opd_allowed",
        "rows": shard_summaries,
    }
    write_json(OUT / "IMPORTANCE_K4_K8_GATE.json", gate)

    grouped = defaultdict(list)
    for r in all_rows:
        grouped[(int(r["K"]), int(r["seed"]))].append(r)
    case_rows = []
    for rows in grouped.values():
        ranked = sorted(rows, key=lambda r: float(r["branch_T_minus_S"]), reverse=True)
        selected = ranked[:25] + ranked[-25:]
        for r in selected:
            case_rows.append({
                "seed": r["seed"],
                "K": r["K"],
                "state_id": r["state_id"],
                "query_id": r["query_id"],
                "turn_id": r["turn_id"],
                "snapshot_hash": r["snapshot_hash"],
                "branch_T_minus_S": r["branch_T_minus_S"],
                "teacher_action": r.get("a_T"),
                "student_action": r.get("a_S"),
                "teacher_trace_actions": trace_names(r, "branch_T_trace"),
                "student_trace_actions": trace_names(r, "branch_S_trace"),
                "branch_T_metrics": r.get("branch_T_metrics"),
                "branch_S_metrics": r.get("branch_S_metrics"),
            })
    with (OUT / "IMPORTANCE_MECHANISM_CASES.jsonl").open("w", encoding="utf-8") as f:
        for r in case_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    first_pairs = Counter((action_name(r, "a_S"), action_name(r, "a_T")) for r in all_rows)
    t_names = Counter(action_name(r, "a_T") for r in all_rows)
    s_names = Counter(action_name(r, "a_S") for r in all_rows)
    positive = [r for r in all_rows if float(r["branch_T_minus_S"]) > 0]
    negative = [r for r in all_rows if float(r["branch_T_minus_S"]) < 0]
    tool_cost_delta = mean([float(r.get("tool_search_cost", 0.0)) for r in all_rows])
    analysis = [
        "# IMPORTANCE_MECHANISM_ANALYSIS",
        "",
        f"- status: `{decision}`",
        f"- formal rows: `{len(all_rows)}` = 2 seeds x K4/K8 x 512 states",
        "- contract: same xi_t, importance_tagging ON vs OFF first branch, reduced continuation after the first fork action, no full-harness takeover",
        f"- action: `{'do not start importance LoRA OPD from this gate result' if not gate_passed else 'importance LoRA OPD allowed by gate'}`",
        "",
        "## Formal K4/K8 summary",
        "",
        "| seed | K | n | mean T-S | median T-S | ci95 low | ci95 high | pos | neg | zero | mean tool-cost delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in shard_summaries:
        analysis.append(
            f"| {s['seed']} | {s['K']} | {s['n_states']} | {s['mean_branch_T_minus_S']:.6f} | {s['median_branch_T_minus_S']:.6f} | {s['ci95_low_normal_approx']:.6f} | {s['ci95_high_normal_approx']:.6f} | {s['positive_count']} | {s['negative_count']} | {s['zero_count']} | {s['mean_tool_search_cost_delta']:.6f} |"
        )
    analysis.extend([
        "",
        "## Mechanism counts",
        "",
        f"- first-action student marginal: `{dict(s_names)}`",
        f"- first-action teacher marginal: `{dict(t_names)}`",
        f"- positive states: `{len(positive)}`; negative states: `{len(negative)}`; zero states: `{len(all_rows) - len(positive) - len(negative)}`",
        f"- mean teacher-minus-student tool cost delta: `{tool_cost_delta:.6f}`",
        "- most common first-action pairs:",
    ])
    for (s_name, t_name), n in first_pairs.most_common(12):
        analysis.append(f"  - `{s_name} -> {t_name}`: {n}")
    analysis.extend([
        "",
        "## Interpretation",
        "",
        "The formal proper fork does not support the earlier approximate importance_tagging positive signal as a causal component. The full-view branch often changes the immediate route/action, but the branch value is not consistently positive under K4 and K8 reduced-policy continuation. Under the 0816-2 decision rule, importance_tagging LoRA OPD and real closed-loop training are blocked unless a separate contract audit finds a concrete bug or a new component-aligned target is justified from fresh evidence.",
    ])
    (OUT / "IMPORTANCE_MECHANISM_ANALYSIS.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")

    data_audit = {
        "status": "formal_recovered_complete",
        "source_dir": str(SOURCE),
        "n_rows": len(all_rows),
        "expected_rows": 2048,
        "query_ids": len({r["query_id"] for r in all_rows}),
        "snapshot_hashes": len({r["snapshot_hash"] for r in all_rows}),
        "full_harness_takeover_any": any(bool(r.get("full_harness_takeover")) for r in all_rows),
        "runner_values": sorted({str(r.get("runner")) for r in all_rows}),
    }
    write_json(OUT / "IMPORTANCE_DATA_AUDIT.json", data_audit)
    (OUT / "IMPORTANCE_DATA_AUDIT.md").write_text(
        "# IMPORTANCE_DATA_AUDIT\n\n"
        f"- status: `{data_audit['status']}`\n"
        f"- rows: `{data_audit['n_rows']}` / `{data_audit['expected_rows']}`\n"
        f"- unique query ids: `{data_audit['query_ids']}`\n"
        f"- unique snapshot hashes: `{data_audit['snapshot_hashes']}`\n"
        f"- full harness takeover any: `{data_audit['full_harness_takeover_any']}`\n"
        f"- runners: `{data_audit['runner_values']}`\n",
        encoding="utf-8",
    )
    target_contract = "# IMPORTANCE_TARGET_CONTRACT\n\nStatus: `not_started_gate_blocked`.\n\nThe 0816-2 gate requires proper K4 positive and K8 consistent before launching component-aligned LoRA OPD. The formal recovered 512-state x 2-seed x K4/K8 fork gate failed, so no route-KL, ranking, pointer, or evidence-mask target is authorized from this run.\n"
    (OUT / "IMPORTANCE_TARGET_CONTRACT.md").write_text(target_contract, encoding="utf-8")

    blocked_csvs = {
        "IMPORTANCE_LORA_TRAINING_CELLS.csv": ["status,reason\n", "not_started,proper_k4_k8_gate_failed\n"],
        "IMPORTANCE_REAL_CLOSED_LOOP.csv": ["status,reason\n", "not_started,importance_lora_not_authorized\n"],
        "IMPORTANCE_CAUSAL_CONTROL.csv": ["status,reason\n", "not_started,no_positive_importance_lora_to_control\n"],
        "IMPORTANCE_PAIRED_BOOTSTRAP.csv": ["status,reason\n", "not_started,no_real_closed_loop_result\n"],
    }
    for name, lines in blocked_csvs.items():
        (OUT / name).write_text("".join(lines), encoding="utf-8")
    (OUT / "IMPORTANCE_REAL_CLOSED_LOOP.md").write_text(
        "# IMPORTANCE_REAL_CLOSED_LOOP\n\nStatus: `not_started_gate_blocked`. The proper K4/K8 fork gate failed, so actual Student LoRA and no-privilege real closed-loop evaluation are not authorized for importance_tagging under the 0816-2 rules.\n",
        encoding="utf-8",
    )
    write_json(OUT / "BEST_IMPORTANCE_STUDENT.json", {"status": "none", "reason": "proper_k4_k8_gate_failed", "checkpoint": None})
    write_json(OUT / "H1002_IMPORTANCE_HANDOFF.json", {
        "status": decision,
        "component": "importance_tagging",
        "proper_k4_positive": k4_positive,
        "k8_direction_consistent": k8_consistent,
        "actual_lora_started": False,
        "real_closed_loop_started": False,
        "recommended_for_main_table": False,
        "source_dir": str(SOURCE),
        "output_dir": str(OUT),
    })

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    with (OUT / "SHA256SUMS").open("w", encoding="utf-8") as f:
        for p in files:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            f.write(f"{h}  {p.relative_to(OUT)}\n")
    print(json.dumps({"status": decision, "output_dir": str(OUT), "gate_passed": gate_passed}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
