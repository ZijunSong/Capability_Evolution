#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "0816_2_final_experiment_status"
AUTO = Path("/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h1001_auto_papergrade")
IMPORTANCE = REPO / "outputs" / "0816_2_importance_proper_fork_formal_final"
FULL = REPO / "outputs" / "0816_2_full_harness_same_contract_test256"
BRIDGE = REPO / "outputs" / "0816_2_actual_lora_bridge"
BRIDGE_CL = REPO / "outputs" / "0816_2_actual_lora_bridge_closed_loop_smoke16"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def row_for_summary(summary: list[dict[str, Any]], method: str) -> dict[str, Any]:
    for row in summary:
        if row.get("method") == method:
            return row
    return {}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    auto = read_json(AUTO / "H1001_AUTO_PAPERGRADE_HANDOFF.json")
    imp = read_json(IMPORTANCE / "IMPORTANCE_K4_K8_GATE.json")
    full_summary = read_json(FULL / "REAL_CLOSED_LOOP_SUMMARY.json")
    bridge_summary = read_json(BRIDGE_CL / "REAL_CLOSED_LOOP_SUMMARY.json")
    bridge_train = {}
    for variant in ["matched_text", "structured", "ophsd"]:
        bridge_train[variant] = read_json(BRIDGE / variant / "seed42" / "SUMMARY.json")

    records = [
        {
            "experiment": "AUTO actual-LoRA real closed-loop",
            "status": "completed_failed_gate",
            "n": auto.get("closed_loop_rows", {}).get("base_student"),
            "primary_metric": "real_closed_loop_pass",
            "value": auto.get("real_closed_loop_pass"),
            "note": "AUTO did not beat Base or Shuffle on actual-model closed-loop.",
            "source": str(AUTO / "H1001_AUTO_PAPERGRADE_HANDOFF.json"),
        },
        {
            "experiment": "importance_tagging proper K4/K8 fork",
            "status": "completed_failed_gate",
            "n": "4 shards x 512 states",
            "primary_metric": "gate_passed",
            "value": imp.get("gate_passed"),
            "note": "Formal proper K4/K8 means are negative; no importance LoRA launched.",
            "source": str(IMPORTANCE / "IMPORTANCE_K4_K8_GATE.json"),
        },
        {
            "experiment": "Full Harness same-contract reference",
            "status": "completed_real_closed_loop_reference",
            "n": row_for_summary(full_summary, "BASE_REDUCED").get("n"),
            "primary_metric": "overall_reward",
            "value": row_for_summary(full_summary, "BASE_REDUCED").get("overall_reward"),
            "note": "Ran with V8D_AUTO_POPULATE_FIRST_SEARCH=1, V8D_IMPORTANCE_TAGGING=1, V8D_VERIFY_TOOL=1, V8D_TOKEN_BUDGET_MARKER=1. Output label remains BASE_REDUCED from reused evaluator; contract records full-harness flags externally.",
            "source": str(FULL / "REAL_CLOSED_LOOP_SUMMARY.json"),
        },
        {
            "experiment": "Matched Text actual-LoRA bridge",
            "status": "completed_actual_lora_bridge_smoke",
            "n": row_for_summary(bridge_summary, "MATCHED_TEXT_BRIDGE").get("n"),
            "primary_metric": "overall_reward",
            "value": row_for_summary(bridge_summary, "MATCHED_TEXT_BRIDGE").get("overall_reward"),
            "note": "Actual PEFT adapter trained from route-distribution-to-tool-response bridge; smoke/dev grade, not paper-grade recollected prompt-response teacher data.",
            "source": str(BRIDGE / "matched_text" / "seed42" / "SUMMARY.json"),
        },
        {
            "experiment": "Structured actual Student V1 bridge",
            "status": "completed_actual_lora_bridge_smoke",
            "n": row_for_summary(bridge_summary, "STRUCTURED_BRIDGE").get("n"),
            "primary_metric": "overall_reward",
            "value": row_for_summary(bridge_summary, "STRUCTURED_BRIDGE").get("overall_reward"),
            "note": "Actual PEFT adapter trained using structured privilege bridge; smoke/dev grade. Structured vs Textual smoke delta is zero.",
            "source": str(BRIDGE / "structured" / "seed42" / "SUMMARY.json"),
        },
        {
            "experiment": "OPHSD actual-LoRA bridge",
            "status": "completed_actual_lora_bridge_smoke",
            "n": row_for_summary(bridge_summary, "OPHSD_BRIDGE").get("n"),
            "primary_metric": "overall_reward",
            "value": row_for_summary(bridge_summary, "OPHSD_BRIDGE").get("overall_reward"),
            "note": "Actual PEFT adapter trained using whole-harness-context bridge; smoke/dev grade and below Base in smoke16.",
            "source": str(BRIDGE / "ophsd" / "seed42" / "SUMMARY.json"),
        },
    ]

    with (OUT / "0816_2_FINAL_EXPERIMENT_STATUS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    payload = {
        "status": "completed_current_0816_2_experiments",
        "records": records,
        "full_harness_summary": full_summary,
        "bridge_closed_loop_summary": bridge_summary,
        "bridge_training": bridge_train,
        "caveats": [
            "Full Harness run reuses the actual-LoRA evaluator and labels the method BASE_REDUCED internally; the execution environment had full V8D runtime flags enabled.",
            "Bridge actual-LoRA runs are real PEFT/LoRA weights and real closed-loop smoke, but generated from route distributions rather than recollected prompt/response teacher rows; do not treat them as final paper-grade baselines.",
            "Existing H100-1 AUTO paper-grade result remains the controlling actual-model gate and is negative.",
        ],
    }
    (OUT / "0816_2_FINAL_EXPERIMENT_STATUS.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# 0816-2 Final Experiment Status",
        "",
        "## Completed Results",
        "",
        "| experiment | status | n | metric | value |",
        "|---|---|---:|---|---:|",
    ]
    for r in records:
        lines.append(f"| {r['experiment']} | `{r['status']}` | {r['n']} | {r['primary_metric']} | {r['value']} |")
    lines += [
        "",
        "## Bridge Closed-Loop Smoke Readout",
        "",
        f"- Base smoke16 reward: `{row_for_summary(bridge_summary, 'BASE_REDUCED').get('overall_reward')}`",
        f"- Matched Text bridge reward: `{row_for_summary(bridge_summary, 'MATCHED_TEXT_BRIDGE').get('overall_reward')}`",
        f"- Structured bridge reward: `{row_for_summary(bridge_summary, 'STRUCTURED_BRIDGE').get('overall_reward')}`",
        f"- OPHSD bridge reward: `{row_for_summary(bridge_summary, 'OPHSD_BRIDGE').get('overall_reward')}`",
        "",
        "## Caveats",
        "",
        "- Full Harness reference was executed with full V8D runtime flags but through the reused evaluator, whose method label remains `BASE_REDUCED`.",
        "- Bridge adapters are actual LoRA weights and real closed-loop smoke results, but the training rows are a route-distribution-to-tool-response bridge. They are not a replacement for paper-grade recollected prompt/response teacher data.",
        "- There are no remaining unstarted 0816-2 items in the current artifact set; remaining progress requires a new data/runner contract rather than simply launching an existing command.",
        "",
    ]
    (OUT / "0816_2_FINAL_EXPERIMENT_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps({"status": "completed_current_0816_2_experiments", "generated_files": ["0816_2_FINAL_EXPERIMENT_STATUS.md", "0816_2_FINAL_EXPERIMENT_STATUS.json", "0816_2_FINAL_EXPERIMENT_STATUS.csv", "SHA256SUMS"]}, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(OUT)}" for p in files) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed_current_0816_2_experiments", "output_dir": str(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
