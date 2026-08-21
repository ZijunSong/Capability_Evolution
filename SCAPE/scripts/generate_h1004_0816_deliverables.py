#!/usr/bin/env python3
"""Generate H100-4 2026-08-16 novelty/baseline deliverables.

The generator consolidates completed H100-4 evidence and writes the required
artifact set without promoting blocked baselines to completed results.
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
MATCHED = OUT / "matched_v2"
OPHSD = OUT / "ophsd"
IMPORTANCE = OUT / "importance_mining"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for r in rows for k in r}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def sha256sums(root: Path) -> None:
    lines: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def novelty_docs() -> None:
    matrix = [
        {
            "Paper": "Training with Harnesses / OPHSD (arXiv:2605.08741)",
            "PI source": "full harness assisted rollout and terminal harness-produced context",
            "PI representation": "textual harness context used by a static teacher",
            "Harness involved?": "yes",
            "Whole harness or component?": "whole harness",
            "Student on-policy state?": "yes, on-policy trajectory before distillation",
            "Same-state intervention?": "not component-local same-xi_t intervention",
            "Teacher supervision type": "teacher distribution / OPD-style distillation",
            "Runtime structured signal?": "no; terminal context is textualized",
            "Textual/latent/visual?": "textual",
            "Component value estimation?": "no component-level value map",
            "Internalize vs externalize decision?": "no component placement decision",
            "Interactive search?": "complex reasoning with harness; not our component-local search-control protocol",
            "Inference-time privilege?": "removed at student inference",
            "Main benchmark": "paper-specific complex reasoning tasks",
            "Closest overlap with us": "harness-assisted rollout -> distill -> remove harness",
            "What we must experimentally prove to remain distinct": "component-local structured runtime privilege beats information-matched text and/or whole-harness terminal context under the same closed-loop evaluator.",
        },
        {
            "Paper": "Privileged Information Distillation / OPSD (arXiv:2602.04942)",
            "PI source": "privileged action/state information",
            "PI representation": "action-only or privileged side information",
            "Harness involved?": "not necessarily",
            "Whole harness or component?": "not harness component placement",
            "Student on-policy state?": "yes in OPD variants",
            "Same-state intervention?": "not our full/reduced harness fork",
            "Teacher supervision type": "distillation from privileged teacher",
            "Runtime structured signal?": "not harness-native typed control signal",
            "Textual/latent/visual?": "text/action depending on variant",
            "Component value estimation?": "no",
            "Internalize vs externalize decision?": "no",
            "Interactive search?": "not central",
            "Inference-time privilege?": "removed",
            "Main benchmark": "paper-specific PI distillation tasks",
            "Closest overlap with us": "privileged action signal can be distilled",
            "What we must experimentally prove to remain distinct": "action-only PI is insufficient as a contribution; show typed harness-component runtime control and placement value.",
        },
        {
            "Paper": "HASP (arXiv:2605.17734)",
            "PI source": "executable skill programs / program-function events",
            "PI representation": "program functions and execution traces",
            "Harness involved?": "agent harness-like skill layer",
            "Whole harness or component?": "skill programs, not pre-existing Harness-1 components",
            "Student on-policy state?": "post-training/internalization setting",
            "Same-state intervention?": "not the central contract",
            "Teacher supervision type": "skill/program assisted post-training",
            "Runtime structured signal?": "structured program event, but not our harness-native component mask/control",
            "Textual/latent/visual?": "programmatic/event structured",
            "Component value estimation?": "not component counterfactual value",
            "Internalize vs externalize decision?": "skill internalization rather than component placement",
            "Interactive search?": "agent tasks, not Search/Harness-1 closed loop as used here",
            "Inference-time privilege?": "program availability varies by setting",
            "Main benchmark": "skill-program agent tasks",
            "Closest overlap with us": "post-training from external executable assistance",
            "What we must experimentally prove to remain distinct": "pre-existing harness component value and placement, not learned skill-program distillation.",
        },
        {
            "Paper": "Selective OPD / state-matched OPD family (SERL, SAGE-OPD, SMRC-SD)",
            "PI source": "selected turns/states or matched on-policy states",
            "PI representation": "text/action teacher targets over selected states",
            "Harness involved?": "generally no",
            "Whole harness or component?": "not harness components",
            "Student on-policy state?": "yes",
            "Same-state intervention?": "state matched, but not full-vs-reduced component intervention",
            "Teacher supervision type": "selective OPD / filtered distillation",
            "Runtime structured signal?": "no harness-native structured privilege",
            "Textual/latent/visual?": "mostly textual/action",
            "Component value estimation?": "generic selection, not component value mining",
            "Internalize vs externalize decision?": "no",
            "Interactive search?": "not defining feature",
            "Inference-time privilege?": "removed",
            "Main benchmark": "paper-specific selective distillation tasks",
            "Closest overlap with us": "partial-state distillation",
            "What we must experimentally prove to remain distinct": "selection is for deciding which harness component to internalize and in which runtime states it has positive value.",
        },
        {
            "Paper": "Non-text privileged information family (Visual-SDPO, LOPD)",
            "PI source": "visual or latent privileged channels",
            "PI representation": "visual / latent / non-text PI",
            "Harness involved?": "no Harness-1 component contract",
            "Whole harness or component?": "not harness components",
            "Student on-policy state?": "varies",
            "Same-state intervention?": "not our runtime full/reduced fork",
            "Teacher supervision type": "preference or latent distillation",
            "Runtime structured signal?": "non-text, but not typed harness runtime control",
            "Textual/latent/visual?": "visual or latent",
            "Component value estimation?": "no",
            "Internalize vs externalize decision?": "no",
            "Interactive search?": "not central",
            "Inference-time privilege?": "removed or hidden depending on method",
            "Main benchmark": "paper-specific multimodal/latent tasks",
            "Closest overlap with us": "non-text PI can be distilled",
            "What we must experimentally prove to remain distinct": "do not claim first non-text PI; show harness-native typed high-level runtime control beats information-matched textualization.",
        },
    ]
    fields = [
        "Paper", "PI source", "PI representation", "Harness involved?", "Whole harness or component?",
        "Student on-policy state?", "Same-state intervention?", "Teacher supervision type", "Runtime structured signal?",
        "Textual/latent/visual?", "Component value estimation?", "Internalize vs externalize decision?",
        "Interactive search?", "Inference-time privilege?", "Main benchmark", "Closest overlap with us",
        "What we must experimentally prove to remain distinct",
    ]
    lines = ["# NOVELTY_MATRIX_20260816", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in matrix:
        lines.append("| " + " | ".join(str(row[f]).replace("|", "/") for f in fields) + " |")
    lines += [
        "", "## RED_LINES", "",
        "### Cannot Claim", "",
        "- First OPD internalization of a harness.",
        "- First action-only privileged information distillation.",
        "- First non-text privileged information.",
        "- Selective or state-matched OPD alone as the contribution.",
        "- Skill-program internalization as if it were the same as pre-existing Harness-1 component placement.",
        "", "### Must Add Experiments", "",
        "- Information-matched structured-vs-textual comparison on the same states, split, update budget, checkpoints, route space, and evaluator.",
        "- Faithful OPHSD-style whole-harness terminal-context adaptation or a precise blocked contract explaining why it cannot be run.",
        "- Component value mining that decides which external capability is worth internalizing versus leaving external.",
        "- No-privilege interactive closed-loop evaluation before any student-improvement claim.",
        "", "### Required Baselines", "",
        "- Base Student.", "- Full Harness.", "- Matched Text OPD.", "- OPHSD-style whole-harness distillation.", "- Our Structured Component OPD.",
    ]
    (OUT / "NOVELTY_MATRIX_20260816.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "NOVELTY_RED_LINES.md").write_text("\n".join(lines[lines.index("## RED_LINES"):]) + "\n", encoding="utf-8")


def baseline_docs() -> None:
    (OUT / "MATCHED_TEXT_PROTOCOL.md").write_text("""# MATCHED_TEXT_PROTOCOL

Status: `BLOCKED_NO_LOCAL_INPUT` for this checkout.

The 2026-08-16 task expects `outputs/btp_h100_4_baselines/matched_v2/matched_v2_pairs.jsonl` with a `609/609` structured-to-text round-trip audit. The historical run record says the adapter generated that file from H100-1 V2 positive states, but this checkout does not contain the file or the cited generator scripts (`scripts/run_btp_h1004_baselines.py`, `scripts/build_btp_matched_text_v2.py`).

Fair comparison contract for the first runnable wave:

- Same states and train/valid/test query split as the structured branch.
- Same teacher and student checkpoint.
- Same route space and reverse Route-KL objective.
- Same update budget and LoRA budget.
- Same no-privilege inference path and closed-loop evaluator.
- Seeds: 42, 43, 44, 45.
- Do not reuse the older boolean-only representation experiment as the matched-text result.

Action needed before launch: recover or regenerate `matched_v2_pairs.jsonl`, preserve the information-equivalence audit, then connect the V2 fields to the HF route optimizer.
""", encoding="utf-8")
    matched_rows = [
        {"seed": s, "method": "Matched Text OPD", "status": "blocked_no_local_input", "reason": "matched_v2_pairs.jsonl and generator scripts absent in this checkout", "objective": "reverse_route_kl", "completed": False}
        for s in (42, 43, 44, 45)
    ]
    write_csv(OUT / "MATCHED_TEXT_TRAINING_CELLS.csv", matched_rows, ["seed", "method", "status", "reason", "objective", "completed"])
    write_csv(OUT / "MATCHED_TEXT_CLOSED_LOOP.csv", [
        {"method": "Matched Text OPD", "status": "not_run", "reason": "training cells blocked before optimizer launch", "closed_loop_completed": False}
    ], ["method", "status", "reason", "closed_loop_completed"])
    (OUT / "MATCHED_TEXT_HANDOFF.json").write_text(json.dumps({
        "method": "Matched Text OPD",
        "status": "blocked_no_local_input",
        "required_missing_files": [
            "outputs/btp_h100_4_baselines/matched_v2/matched_v2_pairs.jsonl",
            "scripts/run_btp_h1004_baselines.py",
            "scripts/build_btp_matched_text_v2.py",
        ],
        "seeds": [42, 43, 44, 45],
        "must_not_use": "old boolean-only representation experiment",
    }, indent=2) + "\n", encoding="utf-8")

    (OUT / "OPHSD_SEARCH_ADAPTATION.md").write_text("""# OPHSD_SEARCH_ADAPTATION

Status: `BLOCKED_NO_FAITHFUL_CONTRACT`.

Faithful Search/Harness-1 adaptation would require:

1. Student samples on-policy Search trajectory.
2. Full harness generates or orchestrates a terminal privileged context.
3. Frozen/static teacher conditions on that harness-produced terminal context.
4. Student matches teacher with reverse KL.
5. Student is evaluated without harness privilege.

This checkout contains same-state component influence/fork tooling and completed H100-4 confirmation artifacts, but does not contain a verified whole-harness terminal-context dataset, teacher-context renderer, token-overhead accounting path, or OPHSD training launcher. Therefore this deliverable is intentionally recorded as blocked rather than replaced with dry-run metadata.

Fairness contract once implemented:

- Base checkpoint, training queries, update budget, optimizer, LoRA budget, student inference budget, and closed-loop evaluator must match the structured branch.
- OPHSD may use faithful truncation of terminal harness context if the context is too long.
- Component-local structured signals must not be inserted into the OPHSD context.
- Teacher-context token overhead and harness runtime cost must be logged.
""", encoding="utf-8")
    ophsd_rows = [{"seed": s, "method": "OPHSD-style", "status": "blocked_no_faithful_contract", "reason": "no whole-harness terminal-context renderer/training launcher", "completed": False} for s in (42, 43, 44, 45)]
    write_csv(OUT / "OPHSD_TRAINING_CELLS.csv", ophsd_rows, ["seed", "method", "status", "reason", "completed"])
    write_csv(OUT / "OPHSD_CLOSED_LOOP.csv", [{"method": "OPHSD-style", "status": "not_run", "reason": "faithful adaptation blocked before training", "closed_loop_completed": False}], ["method", "status", "reason", "closed_loop_completed"])
    (OUT / "OPHSD_HANDOFF.json").write_text(json.dumps({
        "method": "OPHSD-style",
        "status": "blocked_no_faithful_contract",
        "missing_contracts": ["terminal harness context dataset", "teacher-context renderer", "token overhead accounting", "reverse-KL training launcher", "no-privilege closed-loop evaluation binding"],
    }, indent=2) + "\n", encoding="utf-8")


def importance_docs() -> dict[str, Any]:
    src = ROOT / "outputs" / "h100_4_influence_confirm" / "shards" / "importance_tagging" / "REAL_INFLUENCE_PER_STATE.jsonl"
    rows = read_jsonl(src)
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        out_rows.append({
            "component": "importance_tagging",
            "state_id": row.get("state_id") or f"importance_tagging_{idx:04d}",
            "query_id": row.get("query_id"),
            "step": row.get("step"),
            "snapshot_hash": row.get("snapshot_hash"),
            "I_name_normalized": row.get("I_name_normalized", row.get("I_name_raw", 0.0)),
            "I_args_raw": row.get("I_args_raw", 0.0),
            "student_tool": (row.get("student_executed_tool_action") or {}).get("name"),
            "teacher_tool": (row.get("teacher_full_greedy_tool_call") or {}).get("name"),
            "VALUE_POSITIVE_STATE": float(row.get("I_name_normalized", row.get("I_name_raw", 0.0)) or 0.0) > 0.0,
            "source": str(src.relative_to(ROOT)),
        })
    (OUT / "IMPORTANCE_VALUE_PER_STATE.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    copy_if_exists(OUT / "IMPORTANCE_VALUE_PER_STATE.jsonl", IMPORTANCE / "IMPORTANCE_VALUE_PER_STATE.jsonl")
    vals = [float(r["I_name_normalized"] or 0.0) for r in out_rows]
    args_vals = [float(r["I_args_raw"] or 0.0) for r in out_rows]
    summary = read_csv(ROOT / "outputs" / "h100_4_influence_confirm" / "shards" / "importance_tagging" / "REAL_INFLUENCE_BY_COMPONENT.csv")[0]
    gate = {
        "component": "importance_tagging",
        "status": "VALUE_POSITIVE",
        "basis": "H100-4 REAL_INF_CONFIRM128 same-state full/reduced influence confirmation",
        "n_states": len(out_rows),
        "event_support": int(float(summary.get("event_support", len(out_rows)))),
        "mean_I_name_normalized": mean(vals),
        "mean_I_args_raw": mean(args_vals),
        "gate": summary.get("gate"),
        "K4_value_mining": "approximated_from_real_influence_confirm; corrective fork K4 not rerun in this checkout",
        "K8_confirmation": "not_run",
        "handoff_allowed": True,
        "caveat": "This confirms value/influence support for importance_tagging; downstream distillation still needs matched no-privilege closed-loop training/eval.",
    }
    (OUT / "IMPORTANCE_VALUE_GATE.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    copy_if_exists(OUT / "IMPORTANCE_VALUE_GATE.json", IMPORTANCE / "IMPORTANCE_VALUE_GATE.json")
    schema = """# IMPORTANCE_PRIVILEGE_SCHEMA

Component: `importance_tagging`.

Runtime source: Harness-1 component mask and rendered dual view. The full view enables `importance_tagging`; the reduced view disables only this component while keeping the same environment state `xi_t`.

Observed fields used for value mining:

- `query_id`
- `step`
- `snapshot_hash`
- `student_executed_tool_action.name`
- `teacher_full_greedy_tool_call.name`
- `I_name_normalized`
- `I_args_raw`

Privilege type: harness-native structured runtime control over importance tagging, not future reward, gold answer, generated reasoning, or external chain-of-thought.

Current result: `VALUE_POSITIVE` on H100-4 REAL_INF_CONFIRM128, with the caveat that K8 confirmation was not rerun in this checkout.
"""
    (OUT / "IMPORTANCE_PRIVILEGE_SCHEMA.md").write_text(schema, encoding="utf-8")
    copy_if_exists(OUT / "IMPORTANCE_PRIVILEGE_SCHEMA.md", IMPORTANCE / "IMPORTANCE_PRIVILEGE_SCHEMA.md")
    return gate


def comparison_docs(importance_gate: dict[str, Any]) -> None:
    verify_rows = read_csv(ROOT / "outputs" / "h100_4_verify_confirm" / "VERIFY_REAL_INF_CONFIRM128.csv")
    influence_rows = read_csv(ROOT / "outputs" / "h100_4_influence_confirm" / "REAL_INFLUENCE_CONFIRM_BY_COMPONENT.csv")
    b_utility = read_json(ROOT / "outputs" / "h100_4_b_utility_confirm" / "H1004_B_UTILITY_HANDOFF.json")
    rows = [
        {"method": "Base Student", "status": "not_completed_for_0816_main_table", "closed_loop_reward": "", "inference_privilege": "none", "source": "not rerun in H100-4 0816 checkout", "claim_allowed": False},
        {"method": "Full Harness", "status": "not_completed_for_0816_main_table", "closed_loop_reward": "", "inference_privilege": "full harness", "source": "not rerun in H100-4 0816 checkout", "claim_allowed": False},
        {"method": "Matched Text OPD", "status": "blocked_no_local_input", "closed_loop_reward": "", "inference_privilege": "none", "source": "MATCHED_TEXT_HANDOFF.json", "claim_allowed": False},
        {"method": "OPHSD-style", "status": "blocked_no_faithful_contract", "closed_loop_reward": "", "inference_privilege": "none", "source": "OPHSD_HANDOFF.json", "claim_allowed": False},
        {"method": "Our Structured Component OPD", "status": "evidence_only_no_new_training", "closed_loop_reward": "", "inference_privilege": "none at target inference", "source": "H100-4 verify/influence/B-utility confirmations", "claim_allowed": False},
    ]
    fields = ["method", "status", "closed_loop_reward", "inference_privilege", "source", "claim_allowed"]
    write_csv(OUT / "MAIN_COMPARISON_TABLE.csv", rows, fields)
    lines = ["# MAIN_COMPARISON_TABLE", "", "Only real completed results are filled. Blocked baselines are not assigned synthetic scores.", "", "| method | status | closed_loop_reward | inference_privilege | source | claim_allowed |", "|---|---|---:|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['status']} | {r['closed_loop_reward']} | {r['inference_privilege']} | {r['source']} | {str(r['claim_allowed']).lower()} |")
    lines += [
        "", "## Completed Supporting Evidence", "",
        f"- `verify_tool`: H100-4 natural `I_name_normalized={float(verify_rows[0]['I_name_normalized']):.6f}`, `I_args_raw={float(verify_rows[0]['I_args_raw']):.6f}`, gate `{verify_rows[0]['gate']}`.",
        "- H100-4 real influence confirmed `subtractive_curation`, `importance_tagging`, and `evidence_graph` as positive.",
        f"- `importance_tagging`: value gate `{importance_gate['status']}`, n_states={importance_gate['n_states']}, mean_I={importance_gate['mean_I_name_normalized']:.6f}.",
        f"- B utility ranking decision: `{b_utility['decision']}`; `importance_tagging` remained positive but ranked behind `subtractive_curation` on short-horizon utility.",
    ]
    (OUT / "MAIN_COMPARISON_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    gap = """# BASELINE_GAP

## Blocked or Missing

- Matched Text OPD: prepared in historical run record, but `matched_v2_pairs.jsonl` and generator/trainer scripts are absent in this checkout. Must recover/regenerate and run seeds 42,43,44,45.
- OPHSD-style: no faithful whole-harness terminal-context contract, renderer, token-overhead accounting, or launcher. Must implement before reporting any number.
- OPCD / SEED / OPID: not part of the required immediate H100-4 baseline wave; existing references are dry-run or missing faithful Search/Harness-1 contracts.
- Base Student / Full Harness main-table rows: not rerun under the exact 0816 closed-loop protocol in this checkout, so left unscored.

## Fairness Protocol To Freeze

- Same base checkpoint and route space.
- Same train/valid/test query manifests.
- Same update budget, optimizer, and LoRA budget.
- Same no-privilege inference path.
- Same interactive closed-loop evaluator.
- Explicit teacher-context token and runtime cost accounting for OPHSD.

## Claims Currently Not Supported

- `Structured > Matched Text`.
- `Structured > OPHSD`.
- `Student_after > Base Student` in real no-privilege closed loop.
- Retirement or removal of any component based only on current H100-4 evidence.
"""
    (OUT / "BASELINE_GAP.md").write_text(gap, encoding="utf-8")
    handoff = {
        "status": "partial_completed_supporting_evidence_baselines_blocked",
        "completed_supporting_evidence": {
            "verify_confirm": verify_rows,
            "influence_confirm": influence_rows,
            "importance_value_gate": importance_gate,
            "b_utility": b_utility,
        },
        "matched_text": read_json(OUT / "MATCHED_TEXT_HANDOFF.json"),
        "ophsd": read_json(OUT / "OPHSD_HANDOFF.json"),
        "main_claim_allowed": False,
        "reason": "required matched-text and OPHSD baselines are blocked before training/closed-loop evaluation in this checkout",
    }
    (OUT / "H1004_BTP_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps({
        "stage": "h1004_0816_novelty_baselines",
        "status": "partial_completed_supporting_evidence_baselines_blocked",
        "exit_code": 0,
        "generated_files": "see SHA256SUMS",
        "training_launched": False,
    }, indent=2) + "\n", encoding="utf-8")
    (OUT / "STATUS_LIVE.md").write_text("""# STATUS_LIVE - h1004_0816_novelty_baselines

- novelty_matrix: completed
- importance_value_mining: completed from H100-4 REAL_INF_CONFIRM128 source rows
- matched_text_training: blocked_no_local_input
- ophsd_training: blocked_no_faithful_contract
- closed_loop_new_runs: not_run
- errors: 0
""", encoding="utf-8")


def main() -> int:
    for path in (OUT, MATCHED, OPHSD, IMPORTANCE):
        path.mkdir(parents=True, exist_ok=True)
    novelty_docs()
    baseline_docs()
    importance_gate = importance_docs()
    comparison_docs(importance_gate)
    sha256sums(OUT)
    print(json.dumps({"out_dir": str(OUT), "status": "generated", "importance_gate": importance_gate["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
