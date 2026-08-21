#!/usr/bin/env python3
"""Finalize H100-4 required-gap audit after reading CLAUDE.md and H100-1 manifests.

This is deliberately conservative. It promotes only the already-completed
H100-1 full-modules vLLM 256-query rollout to the Full Harness/reference row,
and keeps Matched Text / OPHSD actual-LoRA rows blocked because their required
training contracts are absent.
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
H1001 = Path("/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-1/SCAPE/outputs/btp_h1001_auto_papergrade")
FULL_MANIFEST = H1001 / "closed_loop_eval" / "base_student" / "test256" / "harness_rollout_manifest.json"
FULL_CONFIG = H1001 / "closed_loop_eval" / "base_student" / "test256" / "harness_resolved_config.yaml"
AUTO_RESULTS = H1001 / "CLOSED_LOOP_RESULTS.csv"
AUTO_HANDOFF = H1001 / "H1001_AUTO_PAPERGRADE_HANDOFF.json"
CLAUDE_MD = Path("/mnt/songzijun/CLAUDE.md")


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def ensure_sources() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    if not CLAUDE_MD.exists():
        raise SystemExit(f"CLAUDE.md missing: {CLAUDE_MD}")
    full = read_json(FULL_MANIFEST)
    handoff = read_json(AUTO_HANDOFF)
    results = read_csv(AUTO_RESULTS)
    dst = OUT / "h1001_actual_lora_sources"
    dst.mkdir(parents=True, exist_ok=True)
    for src in [FULL_MANIFEST, FULL_CONFIG, AUTO_RESULTS, AUTO_HANDOFF, H1001 / "PAIRED_BOOTSTRAP_AUTO.csv", H1001 / "CLOSED_LOOP_RESULTS.md"]:
        if src.exists():
            shutil.copyfile(src, dst / src.name)
    return full, handoff, results


def write_full_harness(full: dict[str, Any]) -> None:
    summary = full["summary"]
    rows = [{
        "method": "FULL_HARNESS_FULL_MODULES_VLLM",
        "status": "completed_same_contract_full_modules_vllm_test256",
        "n": full["n_episodes"],
        "overall_reward": summary["reward"],
        "recall": summary["recall"],
        "trajectory_recall": summary["trajectory_recall"],
        "final_answer_recall": summary["final_answer_recall"],
        "precision": summary["precision"],
        "tool_calls_or_turns": summary["turns"],
        "errors": summary.get("errors", 0),
        "policy_backend": full["policy_backend"],
        "model_path": full["model_path"],
        "harness_config": full["harness_config"],
        "retrieval": full["retrieval"],
        "bm25_index_path": full["bm25_index_path"],
        "max_turns": full["max_turns"],
        "max_tokens": full["max_tokens"],
        "temperature": full["temperature"],
        "inference_time_privilege": True,
        "actual_model_weights": True,
        "official_chroma_parity": False,
        "source": "h1001_actual_lora_sources/harness_rollout_manifest.json",
    }]
    write_csv(OUT / "FULL_HARNESS_REAL_CLOSED_LOOP.csv", rows, list(rows[0]))


def write_actual_lora_blockers() -> None:
    matched_audit = OUT / "matched_v2" / "MATCHED_INFORMATION_AUDIT.md"
    matched_pairs = OUT / "matched_v2" / "matched_v2_pairs.jsonl"
    matched_rows = sum(1 for line in matched_pairs.read_text(encoding="utf-8").splitlines() if line.strip()) if matched_pairs.exists() else 0
    matched = {
        "status": "blocked_actual_lora_training_contract_missing",
        "prepared_information_pairs": matched_rows,
        "information_audit": str(matched_audit.relative_to(OUT)) if matched_audit.exists() else None,
        "roundtrip": "609/609" if matched_rows == 609 else "unknown",
        "actual_lora_training_started": False,
        "real_closed_loop_started": False,
        "blocker": "matched_v2_pairs.jsonl contains state-time textualization and prompt_student, but no teacher/ref route distribution or full supervised target contract consumable by an actual HF/PEFT LoRA trainer.",
        "not_substituted_with": "route_head.pt proxy or train_tool_opd V0",
    }
    write_json(OUT / "MATCHED_TEXT_HANDOFF.json", matched)
    write_csv(OUT / "MATCHED_TEXT_LORA_TRAINING.csv", [{
        "method": "Matched Text OPD",
        "status": matched["status"],
        "actual_lora_training": False,
        "prepared_pairs": matched_rows,
        "reason": matched["blocker"],
    }], ["method", "status", "actual_lora_training", "prepared_pairs", "reason"])
    write_csv(OUT / "MATCHED_TEXT_REAL_CLOSED_LOOP.csv", [{
        "method": "Matched Text OPD",
        "status": "not_started_actual_lora_contract_blocked",
        "overall_reward": "NA",
        "reason": matched["blocker"],
    }], ["method", "status", "overall_reward", "reason"])

    ophsd = {
        "status": "blocked_actual_lora_training_contract_missing_route_level_only_available",
        "route_level_cells": sorted(str(p.relative_to(OUT)) for p in (OUT / "ophsd" / "cells").glob("OPHSD_ROUTE_CONTEXT_seed*/summary.json")),
        "actual_lora_training_started": False,
        "real_closed_loop_started": False,
        "blocker": "Existing OPHSD artifacts are route_head.pt route-level whole-harness context adaptation. No actual HF/PEFT LoRA dataset exists with reduced prompts plus whole-harness teacher target and matched update budget.",
        "not_substituted_with": "route_head.pt proxy or train_tool_opd V0",
    }
    write_json(OUT / "OPHSD_HANDOFF.json", ophsd)
    write_csv(OUT / "OPHSD_LORA_TRAINING.csv", [{
        "method": "OPHSD-style",
        "status": ophsd["status"],
        "actual_lora_training": False,
        "route_level_cells": len(ophsd["route_level_cells"]),
        "reason": ophsd["blocker"],
    }], ["method", "status", "actual_lora_training", "route_level_cells", "reason"])
    write_csv(OUT / "OPHSD_REAL_CLOSED_LOOP.csv", [{
        "method": "OPHSD-style",
        "status": "not_started_actual_lora_contract_blocked",
        "overall_reward": "NA",
        "reason": ophsd["blocker"],
    }], ["method", "status", "overall_reward", "reason"])


def write_main_tables(full: dict[str, Any], handoff: dict[str, Any], results: list[dict[str, str]]) -> None:
    full_s = full["summary"]
    auto = model_row(results, "auto_relevant_route_kl_reverse_seed44")
    shuffled = model_row(results, "legacy_shuffled_seed42")
    first_turn = model_row(results, "first_turn_only_seed42")
    rows = [
        {
            "method": "Full Harness / Base full-modules Harness-1",
            "status": "completed_same_contract_full_modules_vllm_test256",
            "actual_model_weights": True,
            "inference_time_privilege": True,
            "n": full["n_episodes"],
            "mean_overall_reward": full_s["reward"],
            "trajectory_recall": full_s["trajectory_recall"],
            "final_answer_recall": full_s["final_answer_recall"],
            "source": "FULL_HARNESS_REAL_CLOSED_LOOP.csv",
            "claim_allowed": True,
        },
        {
            "method": "Ours AUTO Component OPD",
            "status": "completed_actual_lora_real_closed_loop_failed_gate",
            "actual_model_weights": True,
            "inference_time_privilege": False,
            "n": auto["n"],
            "mean_overall_reward": auto["reward"],
            "trajectory_recall": auto["trajectory_recall"],
            "final_answer_recall": auto["final_answer_recall"],
            "source": "h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.csv",
            "claim_allowed": False,
        },
        {
            "method": "Shuffle control",
            "status": "completed_actual_lora_real_closed_loop",
            "actual_model_weights": True,
            "inference_time_privilege": False,
            "n": shuffled["n"],
            "mean_overall_reward": shuffled["reward"],
            "trajectory_recall": shuffled["trajectory_recall"],
            "final_answer_recall": shuffled["final_answer_recall"],
            "source": "h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.csv",
            "claim_allowed": True,
        },
        {
            "method": "First-turn-only control",
            "status": "completed_actual_lora_real_closed_loop",
            "actual_model_weights": True,
            "inference_time_privilege": False,
            "n": first_turn["n"],
            "mean_overall_reward": first_turn["reward"],
            "trajectory_recall": first_turn["trajectory_recall"],
            "final_answer_recall": first_turn["final_answer_recall"],
            "source": "h1001_actual_lora_sources/CLOSED_LOOP_RESULTS.csv",
            "claim_allowed": True,
        },
        {
            "method": "Matched Text OPD",
            "status": "blocked_actual_lora_training_contract_missing",
            "actual_model_weights": "not_run_actual_lora",
            "inference_time_privilege": False,
            "n": "NA",
            "mean_overall_reward": "NA",
            "trajectory_recall": "NA",
            "final_answer_recall": "NA",
            "source": "MATCHED_TEXT_HANDOFF.json",
            "claim_allowed": False,
        },
        {
            "method": "OPHSD-style",
            "status": "blocked_actual_lora_training_contract_missing_route_level_only_available",
            "actual_model_weights": "route_level_only_not_actual_lora",
            "inference_time_privilege": False,
            "n": "NA",
            "mean_overall_reward": "NA",
            "trajectory_recall": "NA",
            "final_answer_recall": "NA",
            "source": "OPHSD_HANDOFF.json",
            "claim_allowed": False,
        },
    ]
    fields = list(rows[0])
    write_csv(OUT / "END2END_MAIN_TABLE.csv", rows, fields)
    lines = [
        "# END2END_MAIN_TABLE",
        "",
        "Corrected 2026-08-17 table after reading `/mnt/songzijun/CLAUDE.md` and auditing H100-1/H100-4 manifests.",
        "",
        "The 256-query `base_student/test256` H100-1 rollout used `modules_full_v2.yaml` with evidence_state, verification, and context_budget modules enabled. It is therefore recorded as the Full Harness / full-modules Harness-1 reference, not as a reduced no-privilege Base Student row.",
        "",
        "| method | status | actual_model_weights | inference_time_privilege | reward | trajectory_recall | final_answer_recall | source |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['status']} | {row['actual_model_weights']} | {row['inference_time_privilege']} | {row['mean_overall_reward']} | {row['trajectory_recall']} | {row['final_answer_recall']} | {row['source']} |")
    lines += [
        "",
        "## Readout",
        "",
        f"- Full Harness/full-modules reward: `{float(full_s['reward']):.6f}` on {full['n_episodes']} queries.",
        f"- AUTO actual LoRA reward: `{float(auto['reward']):.6f}`; it does not beat the full-modules reference and failed the AUTO>Base/AUTO>Shuffle gate.",
        f"- Shuffle reward: `{float(shuffled['reward']):.6f}`; unshuffled AUTO does not beat shuffled.",
        "- Matched Text actual-LoRA and OPHSD actual-LoRA remain contract-blocked. Existing route-head artifacts are not promoted to actual model results.",
    ]
    (OUT / "END2END_MAIN_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    handoff_out = {
        "status": "corrected_required_gap_audit_completed",
        "claude_md_read": str(CLAUDE_MD),
        "full_harness_reference": "completed_same_contract_full_modules_vllm_test256",
        "full_harness_manifest": str(FULL_MANIFEST),
        "full_harness_summary": full_s,
        "base_student_reduced_same_contract": "not_separately_available; prior Base label used full modules_full_v2 config",
        "which_baselines_are_actual_lora_or_full_model": {
            "Full Harness / Base full-modules Harness-1": "full HF model via vLLM, full modules enabled",
            "Ours AUTO Component OPD": "actual LoRA/merged model; failed real closed-loop gate",
            "Shuffle control": "actual LoRA/merged control model",
            "First-turn-only control": "actual LoRA/merged control model",
            "Matched Text OPD": "not_run_actual_lora_contract_missing",
            "OPHSD-style": "route_level_only_not_actual_lora",
        },
        "same_evaluator_contract": {
            "Full Harness vs AUTO vs Shuffle vs First-turn-only": True,
            "Matched Text actual LoRA": False,
            "OPHSD actual LoRA": False,
        },
        "ours_vs_full_harness": {
            "delta_reward": float(auto["reward"]) - float(full_s["reward"]),
            "conclusion": "AUTO actual LoRA does not beat full-modules Harness-1 reference",
        },
        "ours_vs_shuffle": {
            "delta_reward": float(auto["reward"]) - float(shuffled["reward"]),
            "unshuffled_beats_shuffle": handoff.get("unshuffled_beats_shuffle"),
        },
        "matched_text_actual_lora_blocked": True,
        "ophsd_actual_lora_blocked": True,
        "recommended_for_main_table": False,
    }
    write_json(OUT / "H1004_END2END_BASELINE_HANDOFF.json", handoff_out)


def write_docs(full: dict[str, Any]) -> None:
    text = f"""# BASELINE_GAP_LATE

Status: `CORRECTED_REQUIRED_GAP_AUDIT_20260817`.

## CLAUDE.md

Read: `{CLAUDE_MD}`.

Relevant operating lesson applied here: do not stop at file presence; close the scorer/evaluator/contract layer or record the exact contract blocker.

## Full Harness

The prior H100-4 table marked Full Harness as missing. Audit of the H100-1 paper-grade source shows the `base_student/test256` rollout is actually a full-modules Harness-1 vLLM run:

- manifest: `{FULL_MANIFEST}`
- config: `{FULL_CONFIG}`
- harness_config: `{full['harness_config']}`
- policy_backend: `{full['policy_backend']}`
- model_path: `{full['model_path']}`
- retrieval: `{full['retrieval']}`
- n: `{full['n_episodes']}`
- reward: `{full['summary']['reward']}`

Therefore Full Harness reference is completed locally under the same H100-1 paper-grade contract. It is local BM25 compatibility, not official Chroma parity.

## Remaining Actual-LoRA Baseline Gaps

Matched Text actual-LoRA remains blocked because `matched_v2_pairs.jsonl` contains deterministic textualized state-time fields and reduced prompts, but not a consumable actual-HF/PEFT training contract with teacher/ref route distributions and matched optimizer/update budget.

OPHSD actual-LoRA remains blocked because existing OPHSD files are route-level `route_head.pt` cells; no reduced-prompt plus whole-harness teacher-target HF/PEFT LoRA dataset exists.

Neither route-head proxy nor `scape/training/train_tool_opd.py` V0 is substituted for these required actual-LoRA baselines.
"""
    (OUT / "BASELINE_GAP_LATE.md").write_text(text, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    full, handoff, results = ensure_sources()
    write_full_harness(full)
    write_actual_lora_blockers()
    write_main_tables(full, handoff, results)
    write_docs(full)
    rewrite_sha()
    print(json.dumps({"status": "corrected_required_gap_audit_completed", "out_dir": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
