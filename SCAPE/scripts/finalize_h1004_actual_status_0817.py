#!/usr/bin/env python3
"""Finalize H100-4 end-to-end baseline status against actual LoRA evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "btp_h100_4_baselines"
H1001 = Path("/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h1001_auto_papergrade")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rewrite_sha() -> None:
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (OUT / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(OUT)}" for p in files) + "\n", encoding="utf-8")


def model_row(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    for row in rows:
        if row.get("model") == name:
            return row
    raise KeyError(name)


def copy_h1001_sources() -> None:
    dst = OUT / "h1001_actual_lora_sources"
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "CLOSED_LOOP_RESULTS.csv",
        "CLOSED_LOOP_RESULTS.md",
        "AUTO_TRAINING_CELLS.csv",
        "PAIRED_BOOTSTRAP_AUTO.csv",
        "H1001_AUTO_PAPERGRADE_HANDOFF.json",
        "BEST_AUTO_PAPERGRADE_STUDENT.json",
        "AUTO_PAPERGRADE_SPLIT_MANIFEST.json",
    ]:
        src = H1001 / name
        if src.exists():
            shutil.copyfile(src, dst / name)


def write_protocol(handoff: dict[str, Any]) -> None:
    split = read_json(H1001 / "AUTO_PAPERGRADE_SPLIT_MANIFEST.json")
    text = f"""# END2END_BASELINE_PROTOCOL

Status: `ACTUAL_STATUS_FROZEN_20260817`.

This protocol records the strongest actual-model evidence available locally.
The H100-1 paper-grade AUTO run is actual LoRA / merged model closed-loop from
initial BrowseComp-compatible BM25 state. H100-4 Matched Text and OPHSD remain
route-level unless a matching actual-LoRA training contract is implemented.

## Frozen Actual-Model Contract

- source: `{H1001}`
- base checkpoint: `/mnt/songzijun/models/pat-jj_harness-1-full/harness-1`
- evaluator: local BM25 BrowseComp-compatible closed-loop from initial state
- test split: `test256`, query-disjoint
- n_queries: `{handoff.get('closed_loop_rows', {}).get('base_student')}`
- student inference privilege: `false`
- official_chroma_parity: `{handoff.get('official_chroma_parity')}`
- local_compat_only: `{handoff.get('local_compat_only')}`
- train_update_budget: `{split.get('train_update_budget')}`
- train_unique_states: `{split.get('train_unique')}`
- valid_unique_states: `{split.get('valid_unique')}`
- test_unique_states: `{split.get('test_unique')}`
- query_disjoint: `{split.get('query_disjoint')}`

## Fairness Rule

- Do not compare route-level Matched Text / OPHSD scores as if they were actual LoRA closed-loop scores.
- Missing same-contract Full Harness, Matched Text actual LoRA, and OPHSD actual LoRA rows are `NA`, not zero.
- The actual AUTO result is negative relative to Base and Shuffle; it is not recommended for the main table as a positive result.
"""
    (OUT / "END2END_BASELINE_PROTOCOL.md").write_text(text, encoding="utf-8")


def write_tables() -> None:
    actual = read_csv(H1001 / "CLOSED_LOOP_RESULTS.csv")
    handoff = read_json(H1001 / "H1001_AUTO_PAPERGRADE_HANDOFF.json")
    base = model_row(actual, "base_student")
    auto = model_row(actual, "auto_relevant_route_kl_reverse_seed44")
    shuffled = model_row(actual, "legacy_shuffled_seed42")
    first_turn = model_row(actual, "first_turn_only_seed42")
    matched = read_json(OUT / "MATCHED_TEXT_HANDOFF.json") if (OUT / "MATCHED_TEXT_HANDOFF.json").exists() else {}

    rows = [
        {
            "method": "Base Student",
            "status": "completed_actual_full_model_real_closed_loop",
            "actual_model_weights": True,
            "inference_time_privilege": False,
            "privilege_scope": "none",
            "privilege_representation": "none",
            "n": base["n"],
            "mean_overall_reward": base["reward"],
            "trajectory_recall": base["trajectory_recall"],
            "final_answer_recall": base["final_answer_recall"],
            "tool_calls_or_turns": base["turns"],
            "source": "h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.csv",
            "claim_allowed": True,
        },
        {
            "method": "Full Harness",
            "status": "not_run_same_contract_full_runtime_runner_missing",
            "actual_model_weights": False,
            "inference_time_privilege": True,
            "privilege_scope": "whole_harness",
            "privilege_representation": "Harness-1 runtime",
            "n": base["n"],
            "mean_overall_reward": "NA",
            "trajectory_recall": "NA",
            "final_answer_recall": "NA",
            "tool_calls_or_turns": "NA",
            "source": "FULL_HARNESS_REAL_CLOSED_LOOP.csv",
            "claim_allowed": False,
        },
        {
            "method": "Ours AUTO Component OPD",
            "status": "completed_actual_lora_real_closed_loop_failed_gate",
            "actual_model_weights": True,
            "inference_time_privilege": False,
            "privilege_scope": "component_local",
            "privilege_representation": "auto_populate_first_search route target",
            "n": auto["n"],
            "mean_overall_reward": auto["reward"],
            "trajectory_recall": auto["trajectory_recall"],
            "final_answer_recall": auto["final_answer_recall"],
            "tool_calls_or_turns": auto["turns"],
            "source": "h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.csv",
            "claim_allowed": False,
        },
        {
            "method": "Matched Text OPD",
            "status": "blocked_actual_lora_training_contract_missing; route_level_available",
            "actual_model_weights": "route_level_only_not_actual_lora",
            "inference_time_privilege": False,
            "privilege_scope": "component_local_information_matched",
            "privilege_representation": "deterministic textualization",
            "n": matched.get("real_closed_loop", {}).get("n", "NA_route_level"),
            "mean_overall_reward": "NA_actual_lora",
            "trajectory_recall": "NA_actual_lora",
            "final_answer_recall": "NA_actual_lora",
            "tool_calls_or_turns": "NA_actual_lora",
            "source": "MATCHED_TEXT_HANDOFF.json",
            "claim_allowed": False,
        },
        {
            "method": "OPHSD-style",
            "status": "blocked_actual_lora_training_contract_missing; route_level_available",
            "actual_model_weights": "route_level_only_not_actual_lora",
            "inference_time_privilege": False,
            "privilege_scope": "whole_harness_terminal_context",
            "privilege_representation": "terminal harness context",
            "n": "NA_actual_lora",
            "mean_overall_reward": "NA_actual_lora",
            "trajectory_recall": "NA_actual_lora",
            "final_answer_recall": "NA_actual_lora",
            "tool_calls_or_turns": "NA_actual_lora",
            "source": "OPHSD_HANDOFF.json",
            "claim_allowed": False,
        },
        {
            "method": "Shuffle control",
            "status": "completed_actual_lora_real_closed_loop",
            "actual_model_weights": True,
            "inference_time_privilege": False,
            "privilege_scope": "state_target_shuffle",
            "privilege_representation": "marginal-preserving shuffled route target",
            "n": shuffled["n"],
            "mean_overall_reward": shuffled["reward"],
            "trajectory_recall": shuffled["trajectory_recall"],
            "final_answer_recall": shuffled["final_answer_recall"],
            "tool_calls_or_turns": shuffled["turns"],
            "source": "h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.csv",
            "claim_allowed": True,
        },
        {
            "method": "First-turn-only control",
            "status": "completed_actual_lora_real_closed_loop",
            "actual_model_weights": True,
            "inference_time_privilege": False,
            "privilege_scope": "first_turn_only_component_control",
            "privilege_representation": "first-turn action_ce target",
            "n": first_turn["n"],
            "mean_overall_reward": first_turn["reward"],
            "trajectory_recall": first_turn["trajectory_recall"],
            "final_answer_recall": first_turn["final_answer_recall"],
            "tool_calls_or_turns": first_turn["turns"],
            "source": "h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.csv",
            "claim_allowed": True,
        },
    ]
    fields = list(rows[0])
    write_csv(OUT / "END2END_MAIN_TABLE.csv", rows, fields)

    lines = [
        "# END2END_MAIN_TABLE",
        "",
        "Actual-model status table. Missing same-contract baselines are marked `NA`; route-level results are not promoted to actual LoRA results.",
        "",
        "| method | status | actual_model_weights | inference_time_privilege | reward | trajectory_recall | final_answer_recall | source |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['status']} | {row['actual_model_weights']} | {row['inference_time_privilege']} | "
            f"{row['mean_overall_reward']} | {row['trajectory_recall']} | {row['final_answer_recall']} | {row['source']} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        f"- Actual AUTO reward: `{float(auto['reward']):.6f}` vs Base `{float(base['reward']):.6f}`; AUTO does not beat Base.",
        f"- Actual AUTO reward: `{float(auto['reward']):.6f}` vs Shuffle `{float(shuffled['reward']):.6f}`; unshuffled does not beat shuffled.",
        f"- H100-1 gate: `real_closed_loop_pass={str(handoff.get('real_closed_loop_pass')).lower()}`; `recommended_for_main_table={str(handoff.get('recommended_for_main_table')).lower()}`.",
        "- Matched Text and OPHSD actual LoRA rows remain blocked by missing same-state prompt/teacher training contracts; available rows are route-level only.",
        "- Full Harness same-contract row remains missing; no score is imputed.",
    ]
    (OUT / "END2END_MAIN_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_csv(OUT / "FULL_HARNESS_REAL_CLOSED_LOOP.csv", [{
        "method": "FULL_HARNESS",
        "status": "not_run_same_contract_full_runtime_runner_missing",
        "n": base["n"],
        "overall_reward": "NA",
        "evidence_or_recall_metric": "NA",
        "tool_calls": "NA",
        "invalid_tools": "NA",
        "latency_or_token_cost": "NA",
        "student_inference_has_privilege": True,
        "note": "Launcher exists for external vLLM/server smoke, but no completed exact H100-1 paper-grade same-contract Full Harness run was found.",
    }], ["method", "status", "n", "overall_reward", "evidence_or_recall_metric", "tool_calls", "invalid_tools", "latency_or_token_cost", "student_inference_has_privilege", "note"])

    bootstrap_rows = read_csv(H1001 / "PAIRED_BOOTSTRAP_AUTO.csv") if (H1001 / "PAIRED_BOOTSTRAP_AUTO.csv").exists() else []
    write_csv(OUT / "END2END_PAIRED_BOOTSTRAP.csv", bootstrap_rows, sorted({k for r in bootstrap_rows for k in r}) if bootstrap_rows else ["status"])

    cost_rows = [{
        "method": row["method"],
        "tool_calls_or_turns": row["tool_calls_or_turns"],
        "compute_cost": "logged_as_turns_only" if not str(row["mean_overall_reward"]).startswith("NA") else "NA_not_run",
        "token_cost": "0_or_not_logged_in_source" if not str(row["mean_overall_reward"]).startswith("NA") else "NA_not_run",
        "source": row["source"],
    } for row in rows]
    write_csv(OUT / "END2END_COMPUTE_COST.csv", cost_rows, ["method", "tool_calls_or_turns", "compute_cost", "token_cost", "source"])
    write_protocol(handoff)
    write_handoff(handoff, rows, base, auto, shuffled)


def write_handoff(h1001_handoff: dict[str, Any], rows: list[dict[str, Any]], base: dict[str, str], auto: dict[str, str], shuffled: dict[str, str]) -> None:
    handoff = {
        "status": "actual_status_finalized_with_required_gaps",
        "actual_model_contract_source": str(H1001),
        "which_baselines_are_actual_lora_or_full_model": {
            "Base Student": "full base model inference",
            "Ours AUTO Component OPD": "actual LoRA merged model; failed real closed-loop gate",
            "Shuffle control": "actual LoRA merged/control model",
            "First-turn-only control": "actual LoRA merged/control model",
            "Full Harness": "not_run",
            "Matched Text OPD": "route_level_only_not_actual_lora",
            "OPHSD-style": "route_level_only_not_actual_lora",
        },
        "which_are_auxiliary_route_level_only": ["Matched Text OPD", "OPHSD-style"],
        "same_evaluator_contract": {
            "Base vs Ours AUTO vs Shuffle vs First-turn-only": True,
            "Full Harness": False,
            "Matched Text OPD actual LoRA": False,
            "OPHSD-style actual LoRA": False,
        },
        "student_inference_privilege": {row["method"]: row["inference_time_privilege"] for row in rows},
        "ours_vs_matched_text": {"conclusion": "not_comparable_as_actual_lora; matched_text_only_route_level_available"},
        "ours_vs_ophsd": {"conclusion": "not_comparable_as_actual_lora; OPHSD_only_route_level_available"},
        "ours_vs_base": {
            "actual_delta_reward": float(auto["reward"]) - float(base["reward"]),
            "student_beats_base": h1001_handoff.get("student_beats_base"),
            "conclusion": "AUTO actual LoRA does not beat Base",
        },
        "ours_vs_shuffle": {
            "actual_delta_reward": float(auto["reward"]) - float(shuffled["reward"]),
            "unshuffled_beats_shuffle": h1001_handoff.get("unshuffled_beats_shuffle"),
            "conclusion": "AUTO actual LoRA does not beat shuffled control",
        },
        "full_harness_reference": "missing_required_gap",
        "recommended_for_main_table": h1001_handoff.get("recommended_for_main_table"),
        "blocked_or_failed_reason": h1001_handoff.get("blocked_or_failed_reason"),
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
    (OUT / "H1004_END2END_BASELINE_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_gap_docs() -> None:
    status = (OUT / "BASELINE_GAP_LATE.md").read_text(encoding="utf-8") if (OUT / "BASELINE_GAP_LATE.md").exists() else "# BASELINE_GAP_LATE\n"
    status += """

## 2026-08-17 Actual-Model Status Update

- H100-1 paper-grade actual LoRA closed-loop test256 is now the controlling actual-model evidence.
- AUTO actual LoRA failed the real closed-loop gate: it does not beat Base and does not beat Shuffle.
- Matched Text actual LoRA cannot be launched from `matched_v2_pairs.jsonl` alone because it lacks the reduced prompt plus teacher/ref route distribution contract required by `train_route_opd.py`.
- OPHSD actual LoRA remains blocked by the same missing training contract; existing OPHSD artifacts are route-head only.
- Full Harness exact same-contract run remains missing; the available official launcher requires an external vLLM server path and was not completed under the paper-grade contract.
"""
    (OUT / "BASELINE_GAP_LATE.md").write_text(status, encoding="utf-8")

    (OUT / "STANDARD_OPSD_STATUS.md").write_text("""# STANDARD_OPSD_STATUS

Status: `BLOCKED_NON_BLOCKING`.

No faithful reusable OPSD implementation was found that can train actual Student
LoRA weights and bind to the same no-privilege real closed-loop evaluator. It
must not block Full Harness / Matched Text / OPHSD gap reporting.
""", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_h1001_sources()
    write_tables()
    write_gap_docs()
    rewrite_sha()
    print(json.dumps({"status": "actual_status_finalized", "out_dir": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
