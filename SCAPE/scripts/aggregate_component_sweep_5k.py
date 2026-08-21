#!/usr/bin/env python3
"""Aggregate the four-server 5K component sweep with fail-closed audits."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["verify_tool", "importance_tagging", "subtractive_curation", "auto_populate_first_search", "content_dedup", "chunk_neighbors", "evidence_graph", "sentence_compress", "token_budget_marker", "adaptive_rerank_instruction"]
EASYOPD = ROOT.parent / "SCAPE-EasyOPD"
HANDOFF_CANDIDATES = {
    1: [
        EASYOPD / "outputs" / "component_sweep_0818" / "h100_1_qwen3" / "H1001_COMPONENT_HANDOFF.json",
        EASYOPD / "outputs" / "component_sweep_0818" / "h100_1_qwen3" / "H1001_COMPONENT_HANDOFF_QWEN3.json",
        EASYOPD / "outputs" / "component_sweep_0818" / "h100_1" / "H1001_COMPONENT_HANDOFF.json",
        ROOT / "outputs" / "component_sweep_0818" / "h100_1" / "H1001_COMPONENT_HANDOFF.json",
    ],
    2: [
        EASYOPD / "outputs" / "component_sweep_0818" / "h100_2" / "H1002_COMPONENT_HANDOFF.json",
        ROOT / "outputs" / "component_sweep_0818" / "h100_2" / "H1002_COMPONENT_HANDOFF.json",
    ],
    3: [
        EASYOPD / "outputs" / "component_sweep_0818" / "h100_3_qwen3_faststart" / "H1003_COMPONENT_HANDOFF.json",
        EASYOPD / "outputs" / "component_sweep_0818" / "h100_3_rerun_realhook" / "H1003_COMPONENT_HANDOFF.json",
        EASYOPD / "outputs" / "component_sweep_0818" / "h100_3" / "H1003_COMPONENT_HANDOFF.json",
        ROOT / "outputs" / "component_sweep_0818" / "h100_3" / "H1003_COMPONENT_HANDOFF.json",
    ],
    4: [
        EASYOPD / "outputs" / "component_sweep_0818" / "h100_4" / "H1004_COMPONENT_HANDOFF.json",
        ROOT / "outputs" / "component_sweep_0818" / "h100_4" / "H1004_COMPONENT_HANDOFF.json",
    ],
}
FIELDS = ["Component", "Type", "Event Support (unique states)", "Train Queries", "Rollouts", "Teacher Reward", "Student Before Reward", "Student After PURE_OPD Reward", "Delta PURE vs Before", "Student After RL+OPD Reward", "Delta RL+OPD vs Before", "Best After", "Best Delta", "Decision"]
FULL_FIELDS = ["component", "effect_type", "realizability", "event_support", "n_train_unique_queries", "n_rollouts_total", "n_event_active_raw", "n_unique_event_active", "n_train_unique_states", "collection_status", "teacher_overall_reward", "before_overall_reward", "pure_seed42_overall_reward", "pure_seed43_overall_reward", "pure_mean_overall_reward", "pure_delta", "pure_ci_low", "pure_ci_high", "hybrid_seed42_overall_reward", "hybrid_seed43_overall_reward", "hybrid_mean_overall_reward", "hybrid_delta", "hybrid_ci_low", "hybrid_ci_high", "decision", "reason"]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def num(value: Any) -> Any:
    return value if isinstance(value, (int, float)) else "N/A"


def flatten(payload: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in payload.get("components", []) if isinstance(payload, dict) else []:
        component = row.get("component")
        if component:
            result[str(component)] = row
    return result


def main() -> int:
    out = EASYOPD / "outputs" / "component_sweep_0818" / "master"
    out.mkdir(parents=True, exist_ok=True)
    resolved_paths = {idx: next((path for path in paths if path.exists()), None) for idx, paths in HANDOFF_CANDIDATES.items()}
    available = {idx: read(path) for idx, path in resolved_paths.items() if path is not None}
    component_rows: dict[str, dict[str, Any]] = {}
    for payload in available.values():
        component_rows.update(flatten(payload))
    full_rows = []
    main_rows = []
    for component in ORDER:
        row = component_rows.get(component, {})
        data = row.get("data", {}) if isinstance(row.get("data"), dict) else {}
        teacher = row.get("teacher", {}) if isinstance(row.get("teacher"), dict) else {}
        before = row.get("student_before", {}) if isinstance(row.get("student_before"), dict) else {}
        pure = row.get("pure_opd", {}) if isinstance(row.get("pure_opd"), dict) else {}
        hybrid = row.get("rl_plus_opd", {}) if isinstance(row.get("rl_plus_opd"), dict) else {}
        p42 = pure.get("seed42", {}) if isinstance(pure.get("seed42"), dict) else {}
        p43 = pure.get("seed43", {}) if isinstance(pure.get("seed43"), dict) else {}
        h42 = hybrid.get("seed42", {}) if isinstance(hybrid.get("seed42"), dict) else {}
        h43 = hybrid.get("seed43", {}) if isinstance(hybrid.get("seed43"), dict) else {}
        pm = pure.get("mean", {}).get("overall_reward", "N/A") if isinstance(pure.get("mean"), dict) else "N/A"
        hm = hybrid.get("mean", {}).get("overall_reward", "N/A") if isinstance(hybrid.get("mean"), dict) else "N/A"
        b = before.get("overall_reward", "N/A")
        reason = row.get("reason", "missing H100 handoff")
        decision = row.get("decision", "MASTER_TABLE_INCOMPLETE" if not row else "INCONCLUSIVE")
        def delta(value: Any) -> Any:
            return value - b if isinstance(value, (int, float)) and isinstance(b, (int, float)) else "N/A"
        full_rows.append({"component": component, "effect_type": row.get("effect_type", "N/A"), "realizability": row.get("realizability", "N/A"), "event_support": data.get("n_unique_event_active", "N/A"), "n_train_unique_queries": data.get("n_train_unique_queries", data.get("n_queries_selected", "N/A")), "n_rollouts_total": data.get("n_rollouts_total", "N/A"), "n_event_active_raw": data.get("n_event_active_raw", "N/A"), "n_unique_event_active": data.get("n_unique_event_active", "N/A"), "n_train_unique_states": data.get("n_train_unique_states", data.get("train_states", "N/A")), "collection_status": data.get("collection_status", "N/A"), "teacher_overall_reward": teacher.get("overall_reward", "N/A"), "before_overall_reward": b, "pure_seed42_overall_reward": p42.get("overall_reward", "N/A"), "pure_seed43_overall_reward": p43.get("overall_reward", "N/A"), "pure_mean_overall_reward": pm, "pure_delta": delta(pm), "pure_ci_low": pure.get("paired_bootstrap_vs_before", {}).get("ci_low", "N/A") if isinstance(pure.get("paired_bootstrap_vs_before"), dict) else "N/A", "pure_ci_high": pure.get("paired_bootstrap_vs_before", {}).get("ci_high", "N/A") if isinstance(pure.get("paired_bootstrap_vs_before"), dict) else "N/A", "hybrid_seed42_overall_reward": h42.get("overall_reward", "N/A"), "hybrid_seed43_overall_reward": h43.get("overall_reward", "N/A"), "hybrid_mean_overall_reward": hm, "hybrid_delta": delta(hm), "hybrid_ci_low": hybrid.get("paired_bootstrap_vs_before", {}).get("ci_low", "N/A") if isinstance(hybrid.get("paired_bootstrap_vs_before"), dict) else "N/A", "hybrid_ci_high": hybrid.get("paired_bootstrap_vs_before", {}).get("ci_high", "N/A") if isinstance(hybrid.get("paired_bootstrap_vs_before"), dict) else "N/A", "decision": decision, "reason": reason})
        best = [v for v in (pm, hm) if isinstance(v, (int, float))]
        best_after = max(best) if best else "N/A"
        main_rows.append({"Component": component, "Type": row.get("effect_type", "N/A"), "Event Support (unique states)": data.get("n_unique_event_active", "N/A"), "Train Queries": data.get("n_train_unique_queries", data.get("n_queries_selected", "N/A")), "Rollouts": data.get("n_rollouts_total", "N/A"), "Teacher Reward": teacher.get("overall_reward", "N/A"), "Student Before Reward": b, "Student After PURE_OPD Reward": pm, "Delta PURE vs Before": delta(pm), "Student After RL+OPD Reward": hm, "Delta RL+OPD vs Before": delta(hm), "Best After": best_after, "Best Delta": delta(best_after), "Decision": decision})
    with (out / "COMPONENT_10_MAIN_TABLE.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader(); csv.DictWriter(f, fieldnames=FIELDS).writerows(main_rows)
    with (out / "COMPONENT_10_FULL_METRICS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FULL_FIELDS); writer.writeheader(); writer.writerows(full_rows)
    lines = ["| Component | Type | Event Support | Train Queries | Rollouts | Teacher | Before | After OPD | Delta OPD | After RL+OPD | Delta Hybrid | Best After | Best Delta | Decision |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in main_rows:
        lines.append("| " + " | ".join(str(r[k]) for k in FIELDS) + " |")
    (out / "COMPONENT_10_MAIN_TABLE.md").write_text("# COMPONENT_10_MAIN_TABLE\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    missing = [str(idx) for idx in range(1, 5) if idx not in available]
    resolved_manifest = {str(idx): str(path) for idx, path in resolved_paths.items() if path is not None}
    (out / "BASE_CONSISTENCY_AUDIT.md").write_text("# BASE_CONSISTENCY_AUDIT\n\n- status: `" + ("MASTER_TABLE_INCOMPLETE" if missing else "AUDIT_PENDING") + "`\n- missing handoffs: " + (", ".join(missing) if missing else "none") + "\n", encoding="utf-8")
    (out / "TEACHER_ISOLATION_AUDIT.md").write_text("# TEACHER_ISOLATION_AUDIT\n\n- status: `AUDIT_PENDING_UNTIL_ALL_HANDOFFS`\n", encoding="utf-8")
    (out / "LOSS_CONSISTENCY_AUDIT.md").write_text("# LOSS_CONSISTENCY_AUDIT\n\n- status: `AUDIT_PENDING_UNTIL_ALL_HANDOFFS`\n", encoding="utf-8")
    status = "MASTER_COMPONENT_SWEEP_READY" if not missing and all(r["Decision"] != "MASTER_TABLE_INCOMPLETE" for r in main_rows) else "MASTER_TABLE_INCOMPLETE"
    manifest = {"status": status, "n_components": 10, "available_handoffs": sorted(available), "missing_handoffs": missing, "resolved_handoff_paths": resolved_manifest, "main_rows": len(main_rows)}
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sums = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if status == "MASTER_COMPONENT_SWEEP_READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
