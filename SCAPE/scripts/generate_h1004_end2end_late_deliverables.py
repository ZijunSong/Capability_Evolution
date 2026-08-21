#!/usr/bin/env python3
"""Generate the H100-4 late end-to-end baseline deliverables.

This script is intentionally conservative: it only copies through completed
results from audited sources and marks unavailable faithful runs as blocked/NA.
It does not synthesize scores for baselines that were not actually evaluated.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "btp_h100_4_baselines"
REAL = ROOT / "outputs" / "h100_2_real_closed_loop_bm25_0816" / "REAL_CLOSED_LOOP_HANDOFF.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_text(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rewrite_sha() -> None:
    rows = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        rows.append(f"{sha256(path)}  {path.relative_to(OUT)}")
    (OUT / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def by_method(real: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in real["summary"]}


def write_protocol(real: dict[str, Any]) -> None:
    contract = ROOT / "outputs" / "0816_2_auto_lora_actual_smoke16_s44" / "AUTO_REAL_EVAL_CONTRACT.md"
    inherited = contract.read_text(encoding="utf-8") if contract.exists() else "No AUTO_REAL_EVAL_CONTRACT.md found."
    text = f"""# END2END_BASELINE_PROTOCOL

Status: `FROZEN_FROM_AVAILABLE_0816_REAL_CLOSED_LOOP_CONTRACT`.

The late H100-4 table is frozen to the completed 0816 BM25 real closed-loop
contract where available. Rows that lack the same actual-model real-interaction
contract are explicitly marked `blocked` or `not_run` and receive no synthetic
score.

## Frozen Contract

- n_queries: `{real.get('n_queries')}`
- max_steps: `{real.get('max_steps')}`
- student_inference_privilege: `{real.get('student_inference_has_privilege')}`
- source: `outputs/h100_2_real_closed_loop_bm25_0816/REAL_CLOSED_LOOP_HANDOFF.json`
- base checkpoint: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`
- executable environment: `/opt/scape-hf-scorer/bin/python`

## Inherited Evaluator Notes

{inherited}

## Baseline Policy

- Full Harness exact same-contract rerun is required but no faithful runner exists in this checkout; score is `NA`.
- Matched Text is completed for formal AUTO route-level synchronized closed loop, but not as a new actual 7B LoRA rerun in this checkout.
- OPHSD-style is completed as route-level faithful whole-harness context adaptation; OPHSD-specific real BM25 closed-loop remains `not_run`.
- Standard OPSD is non-blocking and remains blocked unless a faithful implementation is recovered.
"""
    (OUT / "END2END_BASELINE_PROTOCOL.md").write_text(text, encoding="utf-8")


def write_baseline_files(real: dict[str, Any]) -> None:
    copy_text(OUT / "MATCHED_TEXT_TRAINING_CELLS.csv", OUT / "MATCHED_TEXT_LORA_TRAINING.csv")
    copy_text(OUT / "MATCHED_TEXT_CLOSED_LOOP.csv", OUT / "MATCHED_TEXT_REAL_CLOSED_LOOP.csv")
    copy_text(OUT / "OPHSD_TRAINING_CELLS.csv", OUT / "OPHSD_LORA_TRAINING.csv")
    copy_text(OUT / "OPHSD_CLOSED_LOOP.csv", OUT / "OPHSD_REAL_CLOSED_LOOP.csv")

    fields = [
        "method", "status", "n", "overall_reward", "curated_evidence_recall",
        "trajectory_recall", "final_answer_recall", "tool_calls", "invalid_tools",
        "latency_or_token_cost", "student_inference_has_privilege", "note",
    ]
    write_csv(OUT / "FULL_HARNESS_REAL_CLOSED_LOOP.csv", [{
        "method": "FULL_HARNESS",
        "status": "not_run_no_faithful_same_contract_runner_in_checkout",
        "n": real.get("n_queries"),
        "overall_reward": "NA",
        "curated_evidence_recall": "NA",
        "trajectory_recall": "NA",
        "final_answer_recall": "NA",
        "tool_calls": "NA",
        "invalid_tools": "NA",
        "latency_or_token_cost": "NA",
        "student_inference_has_privilege": True,
        "note": "Required by protocol; no Harness-1 full runtime runner found for exact 0816 evaluator contract.",
    }], fields)

    (OUT / "STANDARD_OPSD_STATUS.md").write_text("""# STANDARD_OPSD_STATUS

Status: `BLOCKED_NON_BLOCKING`.

No faithful reusable OPSD implementation was found in this checkout that can run
standard privileged textual context, train actual Student LoRA weights, and bind
to the same no-privilege real closed-loop evaluator. Per the 0816-2 task order,
OPHSD and Matched Text take priority and OPSD must not block the main table.
""", encoding="utf-8")


def write_tables(real: dict[str, Any]) -> None:
    methods = by_method(real)
    matched = read_json(OUT / "MATCHED_TEXT_HANDOFF.json")
    ophsd = read_json(OUT / "OPHSD_HANDOFF.json")
    rows = [
        {
            "method": "Base Student",
            "status": "completed_real_closed_loop_bm25",
            "actual_model_weights": True,
            "inference_time_privilege": False,
            "privilege_scope": "none",
            "privilege_representation": "none",
            "n": methods["BASE_REDUCED"]["n"],
            "mean_overall_reward": methods["BASE_REDUCED"]["overall_reward"],
            "std_across_seeds": "NA_single_row",
            "tool_calls": methods["BASE_REDUCED"]["tool_calls"],
            "source": "outputs/h100_2_real_closed_loop_bm25_0816/REAL_CLOSED_LOOP_HANDOFF.json",
            "claim_allowed": True,
        },
        {
            "method": "Full Harness",
            "status": "not_run_same_contract_runner_missing",
            "actual_model_weights": False,
            "inference_time_privilege": True,
            "privilege_scope": "whole_harness",
            "privilege_representation": "Harness-1 runtime",
            "n": real.get("n_queries"),
            "mean_overall_reward": "NA",
            "std_across_seeds": "NA",
            "tool_calls": "NA",
            "source": "FULL_HARNESS_REAL_CLOSED_LOOP.csv",
            "claim_allowed": False,
        },
        {
            "method": "Ours AUTO Structured/Component OPD",
            "status": "completed_real_closed_loop_bm25_route_level_sync",
            "actual_model_weights": "route_level_only_not_actual_lora",
            "inference_time_privilege": False,
            "privilege_scope": "component_local",
            "privilege_representation": "typed_structured",
            "n": methods["AUTO_STRUCT_TYPED"]["n"],
            "mean_overall_reward": methods["AUTO_STRUCT_TYPED"]["overall_reward"],
            "std_across_seeds": "NA_sync_summary",
            "tool_calls": methods["AUTO_STRUCT_TYPED"]["tool_calls"],
            "source": "outputs/h100_2_real_closed_loop_bm25_0816/REAL_CLOSED_LOOP_HANDOFF.json",
            "claim_allowed": True,
        },
        {
            "method": "Matched Text OPD",
            "status": matched["status"],
            "actual_model_weights": "route_level_only_not_actual_lora",
            "inference_time_privilege": False,
            "privilege_scope": "component_local_information_matched",
            "privilege_representation": "deterministic_textualization",
            "n": methods["AUTO_MATCHED_TEXT"]["n"],
            "mean_overall_reward": methods["AUTO_MATCHED_TEXT"]["overall_reward"],
            "std_across_seeds": "NA_sync_summary",
            "tool_calls": methods["AUTO_MATCHED_TEXT"]["tool_calls"],
            "source": "MATCHED_TEXT_HANDOFF.json",
            "claim_allowed": True,
        },
        {
            "method": "OPHSD-style",
            "status": ophsd["status"],
            "actual_model_weights": "route_level_only_not_actual_lora",
            "inference_time_privilege": False,
            "privilege_scope": "whole_harness_terminal_context",
            "privilege_representation": "hashed_terminal_context_route_target",
            "n": "NA_no_real_closed_loop",
            "mean_overall_reward": "NA",
            "std_across_seeds": "NA",
            "tool_calls": "NA",
            "source": "OPHSD_HANDOFF.json",
            "claim_allowed": "proxy_only",
        },
        {
            "method": "Shuffle control",
            "status": "not_available_in_h1004_checkout",
            "actual_model_weights": "NA",
            "inference_time_privilege": False,
            "privilege_scope": "state_target_shuffle",
            "privilege_representation": "NA",
            "n": "NA",
            "mean_overall_reward": "NA",
            "std_across_seeds": "NA",
            "tool_calls": "NA",
            "source": "pending_H1001_actual_lora_handoff",
            "claim_allowed": False,
        },
    ]
    fields = list(rows[0])
    write_csv(OUT / "END2END_MAIN_TABLE.csv", rows, fields)
    lines = [
        "# END2END_MAIN_TABLE",
        "",
        "Late H100-4 end-to-end table generated from audited 0816 artifacts. Missing faithful same-contract runs are marked `NA` rather than imputed.",
        "",
        "| method | status | actual_model_weights | inference_time_privilege | privilege_scope | reward | source |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['status']} | {row['actual_model_weights']} | "
            f"{row['inference_time_privilege']} | {row['privilege_scope']} | "
            f"{row['mean_overall_reward']} | {row['source']} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "- Base Student reward: `-0.045`.",
        "- Structured and Matched Text route-level synchronized closed-loop rewards: both `-0.015`; parity, not structured superiority.",
        "- OPHSD-style route-level adaptation completed without inference privilege, but OPHSD real BM25 closed-loop was not run.",
        "- Full Harness reference remains a required gap because no exact same-contract full runtime runner was found in this checkout.",
    ]
    (OUT / "END2END_MAIN_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    deltas = [
        ("Structured-Base", methods["AUTO_STRUCT_TYPED"]["overall_reward"] - methods["BASE_REDUCED"]["overall_reward"]),
        ("MatchedText-Base", methods["AUTO_MATCHED_TEXT"]["overall_reward"] - methods["BASE_REDUCED"]["overall_reward"]),
        ("Structured-MatchedText", methods["AUTO_STRUCT_TYPED"]["overall_reward"] - methods["AUTO_MATCHED_TEXT"]["overall_reward"]),
    ]
    write_csv(OUT / "END2END_PAIRED_BOOTSTRAP.csv", [
        {"comparison": name, "delta_mean": delta, "ci95_low": "NA_query_level_bootstrap_not_in_source", "ci95_high": "NA_query_level_bootstrap_not_in_source", "note": "Source handoff has aggregate summary only."}
        for name, delta in deltas
    ], ["comparison", "delta_mean", "ci95_low", "ci95_high", "note"])

    cost_rows = []
    for row in rows:
        cost_rows.append({
            "method": row["method"],
            "tool_calls": row["tool_calls"],
            "compute_cost": "NA_not_logged" if row["status"] != "not_run_same_contract_runner_missing" else "NA_not_run",
            "token_cost": "NA_not_logged" if row["status"] != "not_run_same_contract_runner_missing" else "NA_not_run",
            "source": row["source"],
        })
    write_csv(OUT / "END2END_COMPUTE_COST.csv", cost_rows, ["method", "tool_calls", "compute_cost", "token_cost", "source"])


def write_late_docs() -> None:
    copy_text(OUT / "NOVELTY_MATRIX_20260816.md", OUT / "NOVELTY_MATRIX_20260816_LATE.md")
    copy_text(OUT / "NOVELTY_RED_LINES.md", OUT / "NOVELTY_RED_LINES_LATE.md")
    gap = (OUT / "BASELINE_GAP.md").read_text(encoding="utf-8") if (OUT / "BASELINE_GAP.md").exists() else "# BASELINE_GAP\n"
    gap += """

## Late End-to-End Gap Update

- Required `END2END_*` artifacts have been generated from audited 0816 sources.
- Full Harness exact same-contract real closed-loop remains not run because no faithful runner was found.
- Matched Text and Structured rows are route-level synchronized real BM25 closed-loop results, not actual 7B LoRA end-to-end reruns.
- OPHSD-style remains route-level faithful adaptation with no OPHSD-specific real BM25 closed-loop row.
- Standard OPSD remains non-blocking and blocked pending a faithful implementation.
"""
    (OUT / "BASELINE_GAP_LATE.md").write_text(gap, encoding="utf-8")


def write_handoff(real: dict[str, Any]) -> None:
    base_handoff = read_json(OUT / "H1004_BTP_HANDOFF.json") if (OUT / "H1004_BTP_HANDOFF.json").exists() else {}
    methods = by_method(real)
    handoff = {
        "status": "late_end2end_deliverables_generated_with_explicit_gaps",
        "source_handoff": base_handoff,
        "which_baselines_are_actual_lora_or_full_model": {
            "Base Student": "full base model inference; no LoRA",
            "Full Harness": "not_run",
            "Matched Text OPD": "route_level_only_not_actual_lora_in_this_checkout",
            "OPHSD-style": "route_level_only_not_actual_lora_in_this_checkout",
            "Ours AUTO Structured/Component OPD": "route_level_only_not_actual_lora_in_this_checkout",
        },
        "which_are_auxiliary_route_level_only": [
            "Matched Text OPD",
            "OPHSD-style",
            "Ours AUTO Structured/Component OPD",
        ],
        "same_evaluator_contract": {
            "Base vs Matched Text vs Structured": True,
            "Full Harness": False,
            "OPHSD-style": False,
        },
        "student_inference_privilege": {
            "Base Student": False,
            "Matched Text OPD": False,
            "OPHSD-style": False,
            "Ours AUTO Structured/Component OPD": False,
        },
        "ours_vs_matched_text": {
            "delta_overall_reward": methods["AUTO_STRUCT_TYPED"]["overall_reward"] - methods["AUTO_MATCHED_TEXT"]["overall_reward"],
            "conclusion": "tie_in_available_route_level_real_closed_loop",
        },
        "ours_vs_ophsd": {
            "conclusion": "not_comparable_in_real_closed_loop; OPHSD lacks same-contract BM25 rollout",
        },
        "ours_vs_base": {
            "delta_overall_reward": methods["AUTO_STRUCT_TYPED"]["overall_reward"] - methods["BASE_REDUCED"]["overall_reward"],
            "conclusion": "structured_route_level_sync_beats_base_by_0.03",
        },
        "full_harness_reference": "missing_required_gap",
        "generated_files": [
            "END2END_BASELINE_PROTOCOL.md",
            "FULL_HARNESS_REAL_CLOSED_LOOP.csv",
            "MATCHED_TEXT_LORA_TRAINING.csv",
            "MATCHED_TEXT_REAL_CLOSED_LOOP.csv",
            "OPHSD_LORA_TRAINING.csv",
            "OPHSD_REAL_CLOSED_LOOP.csv",
            "STANDARD_OPSD_STATUS.md",
            "END2END_MAIN_TABLE.csv",
            "END2END_MAIN_TABLE.md",
            "END2END_PAIRED_BOOTSTRAP.csv",
            "END2END_COMPUTE_COST.csv",
            "NOVELTY_MATRIX_20260816_LATE.md",
            "NOVELTY_RED_LINES_LATE.md",
            "BASELINE_GAP_LATE.md",
            "H1004_END2END_BASELINE_HANDOFF.json",
            "SHA256SUMS",
        ],
    }
    write_json(OUT / "H1004_END2END_BASELINE_HANDOFF.json", handoff)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    real = read_json(REAL)
    write_protocol(real)
    write_baseline_files(real)
    write_tables(real)
    write_late_docs()
    write_handoff(real)
    rewrite_sha()
    print(json.dumps({"status": "generated", "out_dir": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
