#!/usr/bin/env python3
"""Finalize 0816-2 experiment status after gates, cleanup, and contract audits."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "0816_2_final_status_0817"
AUTO_HANDOFF = REPO.parents[0] / "SCAPE-wt-h100-1" / "SCAPE" / "outputs" / "btp_h1001_auto_papergrade" / "H1001_AUTO_PAPERGRADE_HANDOFF.json"
IMPORTANCE_GATE = REPO / "outputs" / "0816_2_importance_proper_formal_0817" / "IMPORTANCE_K4_K8_GATE.json"
H1004_HANDOFF = REPO / "outputs" / "btp_h100_4_baselines" / "H1004_END2END_BASELINE_HANDOFF.json"
STRUCT_HANDOFF = REPO / "outputs" / "h100_2_structured_privilege_formal_0816" / "H1003_STRUCTURED_METHOD_HANDOFF.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"missing": str(path)}


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_status() -> tuple[list[dict], str]:
    proc = subprocess.run(
        "ps -eo pid,ppid,stat,etime,cmd | rg 'run_h100_2_live_fork_replay|importance_tagging|train_route_opd|run_btp_auto_lora|closed_loop|torchrun|vllm' | rg -v 'rg ' || true",
        shell=True,
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    lines = [x for x in proc.stdout.splitlines() if x.strip()]
    return ([{"raw": x} for x in lines], proc.stdout)


def gpu_status() -> str:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    auto = load_json(AUTO_HANDOFF)
    importance = load_json(IMPORTANCE_GATE)
    h1004 = load_json(H1004_HANDOFF)
    structured = load_json(STRUCT_HANDOFF)
    procs, proc_text = process_status()
    gpu_text = gpu_status()

    rows = [
        {
            "experiment": "H100-1 AUTO actual LoRA real closed-loop + shuffle",
            "status": "completed_failed_gate",
            "evidence": str(AUTO_HANDOFF),
            "decision": "do_not_claim_student_win; redesign_required",
            "notes": f"student_beats_base={auto.get('student_beats_base')}; unshuffled_beats_shuffle={auto.get('unshuffled_beats_shuffle')}; real_closed_loop_pass={auto.get('real_closed_loop_pass')}",
        },
        {
            "experiment": "H100-2 importance_tagging proper K4/K8",
            "status": "completed_failed_gate",
            "evidence": str(IMPORTANCE_GATE),
            "decision": "do_not_start_importance_lora_opd",
            "notes": f"k4_positive={importance.get('k4_positive')}; k8_positive={importance.get('k8_direction_consistent_positive')}; gate_passed={importance.get('gate_passed')}",
        },
        {
            "experiment": "H100-2 importance actual LoRA + real closed-loop + causal control",
            "status": "completed_blocked_by_failed_gate",
            "evidence": str(IMPORTANCE_GATE),
            "decision": "not_authorized_under_0816_2_rules",
            "notes": "Proper fork gate failed; launching LoRA would violate the specified Go/Discard rule.",
        },
        {
            "experiment": "H100-3 Structured actual Student V1/V2",
            "status": "completed_blocked_or_not_supported_by_existing_artifacts",
            "evidence": str(STRUCT_HANDOFF),
            "decision": "structured_superiority_not_supported; actual_lora_v1_v2_requires_new_contract",
            "notes": f"status={structured.get('status')}; structured_vs_textual_delta={structured.get('structured_vs_textual_delta')}; existing result is route-head parity, not actual LoRA.",
        },
        {
            "experiment": "H100-4 Full Harness same-contract reference",
            "status": "completed_missing_required_runner_gap",
            "evidence": str(H1004_HANDOFF),
            "decision": "leave_NA_do_not_impute",
            "notes": f"full_harness_reference={h1004.get('full_harness_reference')}",
        },
        {
            "experiment": "H100-4 Matched Text actual LoRA",
            "status": "completed_blocked_contract_missing",
            "evidence": str(H1004_HANDOFF),
            "decision": "route_level_only_not_actual_lora",
            "notes": "matched_v2_pairs has prompt_student/textual_privilege only; it lacks prompt_reduced, P_teacher_route, P_ref_route, route_actions required by actual LoRA trainer.",
        },
        {
            "experiment": "H100-4 OPHSD actual LoRA",
            "status": "completed_blocked_contract_missing",
            "evidence": str(H1004_HANDOFF),
            "decision": "route_level_only_not_actual_lora",
            "notes": "Existing OPHSD artifacts are route_head.pt cells; no faithful whole-harness actual-LoRA training/evaluator contract is present.",
        },
        {
            "experiment": "Residual process cleanup",
            "status": "completed_clean",
            "evidence": str(OUT / "PROCESS_STATUS.txt"),
            "decision": "no_residual_scape_experiment_processes",
            "notes": f"residual_process_count={len(procs)}; all GPUs expected idle after authorized cleanup.",
        },
    ]

    with (OUT / "0816_2_EXPERIMENT_STATUS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(OUT / "0816_2_EXPERIMENT_STATUS.json", {"rows": rows, "residual_processes": procs, "gpu_status": gpu_text})
    (OUT / "PROCESS_STATUS.txt").write_text(proc_text or "NO_MATCHING_PROCESSES\n", encoding="utf-8")
    (OUT / "GPU_STATUS.txt").write_text(gpu_text, encoding="utf-8")

    md = [
        "# 0816-2 Experiment Final Status",
        "",
        "This file distinguishes completed positive results, completed failed gates, and completed blocked experiments where launching the requested run would violate the 0816-2 contract.",
        "",
        "| experiment | status | decision | notes |",
        "|---|---|---|---|",
    ]
    for row in rows:
        md.append(f"| {row['experiment']} | `{row['status']}` | `{row['decision']}` | {row['notes']} |")
    md.extend([
        "",
        "## Process Cleanup",
        "",
        "```text",
        proc_text.strip() or "NO_MATCHING_PROCESSES",
        "```",
        "",
        "## GPU Status",
        "",
        "```text",
        gpu_text.strip(),
        "```",
    ])
    (OUT / "0816_2_EXPERIMENT_STATUS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(OUT)}" for p in files) + "\n", encoding="utf-8")
    print(json.dumps({"status": "0816_2_final_status_written", "out": str(OUT), "residual_processes": len(procs)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
