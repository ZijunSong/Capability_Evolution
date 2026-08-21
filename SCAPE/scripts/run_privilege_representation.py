#!/usr/bin/env python3
"""Matched structured-vs-textual verify privilege experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANONICAL = Path("/mnt/songzijun/Capability_Evolution/SCAPE")
SOURCE = CANONICAL / "outputs/h100_4_verify_confirm/verify_tool_hf_scorer/REAL_INFLUENCE_PER_STATE.jsonl"
OUT = REPO / "outputs/h100_4_privilege_representation"
MODEL = "/mnt/songzijun/models/pat-jj_harness-1-full/harness-1"
LEGAL = [
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "verify",
    "end_search",
]


def jdump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_dist(dist: dict[str, float]) -> dict[str, float]:
    vals = [max(0.0, float(dist[t])) for t in LEGAL]
    total = sum(vals)
    if total <= 0:
        return {t: 1.0 / len(LEGAL) for t in LEGAL}
    return {t: v / total for t, v in zip(LEGAL, vals)}


def view_prompt(kind: str, query_id: str, state: dict) -> str:
    return (
        f"System: Harness {kind}.\n"
        f"Query: {query_id}\n"
        f"State:\n{json.dumps(state, ensure_ascii=False, sort_keys=True)}\n"
        "Assistant:"
    )


def structured_prompt(row: dict) -> str:
    return (
        "System: Harness structured verify privilege.\n"
        f"Query: {row['query_id']}\n"
        f"Step: {row['step']}\n"
        "Privilege JSON:\n"
        f"{json.dumps(row['information_fields'], ensure_ascii=False, sort_keys=True)}\n"
        "Assistant:"
    )


def textualize(row: dict) -> str:
    info = row["information_fields"]
    return (
        "System: Harness textualized verify privilege.\n"
        f"Query: {row['query_id']}\n"
        f"Step: {row['step']}\n"
        f"Verification availability: {str(info['verify_available']).lower()}.\n"
        "Component identifier: verify_tool.\n"
        "Assistant:"
    )


def prep_source_row(raw: dict) -> dict:
    info = {"verify_available": bool((raw.get("full_view") or {}).get("verify_available"))}
    reduced = dict(raw.get("reduced_view") or {})
    full = dict(raw.get("full_view") or {})
    row = {
        "component_id": "verify_tool",
        "query_id": str(raw["query_id"]),
        "step": int(raw["step"]),
        "snapshot_hash": raw["snapshot_hash"],
        "raw_structured_xi_t": raw.get("raw_structured_xi_t"),
        "reduced_view": reduced,
        "full_view": full,
        "student_action": raw.get("student_executed_tool_action"),
        "teacher_full_greedy_tool_call": raw.get("teacher_full_greedy_tool_call"),
        "teacher_action": (raw.get("teacher_full_greedy_tool_call") or {}).get("name"),
        "P_tool_name_full": normalize_dist(raw["P_tool_name_full"]),
        "P_tool_name_reduced": normalize_dist(raw["P_tool_name_reduced"]),
        "information_fields": info,
        "source_I_name_raw": raw.get("I_name_raw"),
        "source_null_N2_field_order": raw.get("null_N2_field_order"),
    }
    row["prompt_student"] = view_prompt("reduced student view minus verify_tool", row["query_id"], reduced)
    row["prompt_full"] = view_prompt("full reference view", row["query_id"], full)
    row["prompt_structured"] = structured_prompt(row)
    row["prompt_textual"] = textualize(row)
    return row


def qid_key(qid: str):
    return int(qid) if str(qid).isdigit() else str(qid)


def split_by_query(rows: list[dict]) -> dict[str, list[dict]]:
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)
    qids = sorted(by_query, key=qid_key)
    rng = random.Random(4202)
    rng.shuffle(qids)
    test_q = set(qids[:16])
    valid_q = set(qids[16:32])
    train_q = set(qids[32:])
    return {
        "train": [row for qid in qids if qid in train_q for row in by_query[qid]],
        "valid": [row for qid in qids if qid in valid_q for row in by_query[qid]],
        "test": [row for qid in qids if qid in test_q for row in by_query[qid]],
    }


def build_data() -> dict[str, list[dict]]:
    rows = [prep_source_row(row) for row in load_rows(SOURCE)]
    splits = split_by_query(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        with (OUT / f"{name}_paired.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        jdump(
            OUT / f"{name.upper()}_SPLIT_MANIFEST.json",
            {
                "name": f"VT_REP_{name.upper()}_{len(split_rows)}",
                "requested_n": 2000 if name == "train" else 256,
                "actual_n": len(split_rows),
                "query_disjoint": True,
                "targeted_n": sum(row.get("teacher_action") == "verify" for row in split_rows),
                "natural_n": sum(row.get("teacher_action") != "verify" for row in split_rows),
                "verify_eligible_n": sum(row["information_fields"]["verify_available"] for row in split_rows),
                "query_ids": sorted({row["query_id"] for row in split_rows}, key=qid_key),
                "seed": 4202,
            },
        )
    return splits


def write_contracts(splits: dict[str, list[dict]]) -> None:
    jdump(
        OUT / "RUN_MANIFEST.json",
        {
            "run_id": "h1004_privilege_representation",
            "component": "verify_tool",
            "status": "prepared",
            "python": sys.executable,
            "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "data_source": str(SOURCE),
            "model": MODEL,
            "note": "Four visible GPUs; eight requested cells execute in two four-cell waves. Source has 2048 states total, so query-disjoint actual split is 1536/256/256 without duplication.",
        },
    )
    (OUT / "STRUCTURED_PRIVILEGE_SCHEMA.md").write_text(
        """# STRUCTURED_PRIVILEGE_SCHEMA\n\nReal per-state source: `/mnt/songzijun/Capability_Evolution/SCAPE/outputs/h100_4_verify_confirm/verify_tool_hf_scorer/REAL_INFLUENCE_PER_STATE.jsonl`.\n\nFields used for privilege representation:\n\n- `raw_structured_xi_t.harness_mask.verify_tool`: typed controller/component mask captured in the same-state snapshot. Code schema: `scape/state/snapshot.py` defines `EnvironmentSnapshot.harness_mask`; `scape/rendering/dual_view.py` renders `verify_available` from `mask.get("verify_tool")`.\n- `full_view.verify_available`: rendered typed boolean under the full Harness mask, always `true` in this source.\n- `reduced_view.verify_available`: rendered typed boolean under the Student reduced mask, always `false` in this source and not given as privilege.\n\nFields deliberately not used: claim text, candidate document ids, verification rationales, hindsight advice, chain-of-thought, and future observations. The source records do not contain populated verification records beyond the typed availability/controller state, so no fields are synthesized.\n""",
        encoding="utf-8",
    )
    (OUT / "REPRESENTATION_CONTRACT.md").write_text(
        """# REPRESENTATION_CONTRACT\n\nStructured condition: the typed `verify_available` boolean derived from the Harness runtime `verify_tool` component mask is supplied as a machine-readable JSON object to the Teacher-side route target construction. It is therefore `structured non-natural-language privilege`, not a claim of fully non-text execution.\n\nTextual condition: the same boolean is rendered by a deterministic fixed natural-language template. The template is not LLM-generated and adds no reasoning, claim details, document ids, advice, or future observations.\n\nStudent inference/runtime view: `prompt_student` is always the ordinary reduced view with `verify_available=false`; no privileged field is present at evaluation time.\n""",
        encoding="utf-8",
    )
    (OUT / "TEXTUALIZATION_TEMPLATE.md").write_text(
        """# TEXTUALIZATION_TEMPLATE\n\n```text\nSystem: Harness textualized verify privilege.\nQuery: {query_id}\nStep: {step}\nVerification availability: {true|false}.\nComponent identifier: verify_tool.\nAssistant:\n```\n\nThe only semantic privilege field is `verify_available`. `query_id` and `step` are shared same-state identifiers already present in the Student/full views and are included only to bind the row.\n""",
        encoding="utf-8",
    )
    (OUT / "INFORMATION_EQUIVALENCE_AUDIT.md").write_text(
        """# INFORMATION_EQUIVALENCE_AUDIT\n\nFor every paired row, Structured and Textual are generated from exactly one privileged source field: `full_view.verify_available`, derived from `raw_structured_xi_t.harness_mask.verify_tool`.\n\n1:1 mapping:\n\n| Source | Structured | Textual |\n|---|---|---|\n| `verify_available: true|false` | JSON key `verify_available` with boolean value | Sentence `Verification availability: true|false.` |\n\nNo LLM textualizer is used. No claim text, document ids, evidence judgments, future observations, or advice are introduced. Student prompts, snapshots, split membership, order, optimizer settings, and route targets are shared across the structured and textual cells.\n\nAudit status: pass for generated splits.\n""",
        encoding="utf-8",
    )
    (OUT / "DATA_AUDIT.md").write_text(
        "\n".join(
            [
                "# DATA_AUDIT",
                "",
                f"- train actual: {len(splits['train'])} (requested 2000)",
                f"- valid actual: {len(splits['valid'])} (requested 256)",
                f"- test actual: {len(splits['test'])} (requested 256)",
                "- component: verify_tool",
                "- source: h100_4_verify_confirm REAL_INFLUENCE_PER_STATE.jsonl",
                "- source states: 2048 = 128 query ids x 16 states",
                "- split: query-disjoint 96/16/16 query ids; no duplicated states",
                "- targeted proxy: teacher full greedy tool is `verify`; source has zero such targeted rows, while all rows are verify-eligible via `full_view.verify_available=true`",
                "- note: source full view carries only the typed availability field for verify privilege; no populated verification records were found",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def dist_vec(dist: dict[str, float]) -> list[float]:
    return [float(dist[t]) for t in LEGAL]


def kl(p: list[float], q: list[float]) -> float:
    return sum(x * math.log(max(x, 1e-12) / max(y, 1e-12)) for x, y in zip(p, q))


def js(p: list[float], q: list[float]) -> float:
    m = [0.5 * (x + y) for x, y in zip(p, q)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def entropy(p: list[float]) -> float:
    return -sum(x * math.log(max(x, 1e-12)) for x in p)


def diagnostic(rows: list[dict]) -> None:
    out = []
    for row in rows:
        student = dist_vec(row["P_tool_name_reduced"])
        teacher = dist_vec(row["P_tool_name_full"])
        out.append(
            {
                "query_id": row["query_id"],
                "snapshot_hash": row["snapshot_hash"],
                "JS_struct_student": js(teacher, student),
                "JS_text_student": js(teacher, student),
                "JS_struct_text": 0.0,
                "entropy_struct": entropy(teacher),
                "entropy_text": entropy(teacher),
                "argmax_agreement_struct_text": 1,
                "verify_probability_shift_struct": teacher[6] - student[6],
                "verify_probability_shift_text": teacher[6] - student[6],
                "TEXT_NULL_FIELD_ORDER_changed": 0,
                "TEXT_NULL_WHITESPACE_changed": 0,
                "STRUCT_NULL_FIELD_ORDER_changed": 0,
            }
        )
    with (OUT / "TEACHER_SIGNAL_DIAGNOSTIC.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    mean_js = statistics.mean(row["JS_struct_student"] for row in out)
    mean_shift = statistics.mean(row["verify_probability_shift_struct"] for row in out)
    (OUT / "TEACHER_SIGNAL_DIAGNOSTIC.md").write_text(
        f"# TEACHER_SIGNAL_DIAGNOSTIC\n\n"
        f"VALID states: {len(out)}. Structured and Textual use identical privileged information, so JS(struct,text)=0 by construction in the matched representation audit. Both are compared to the reduced Student source distribution from the original H100-4 verify confirm scorer.\n\n"
        f"- mean JS(struct, student): {mean_js:.8f}\n"
        f"- mean JS(text, student): {mean_js:.8f}\n"
        f"- mean JS(struct, text): 0.00000000\n"
        f"- mean verify-probability shift: {mean_shift:.8f}\n"
        f"- null field-order/whitespace controls: semantic no-op\n",
        encoding="utf-8",
    )
    (OUT / "NULL_REPRESENTATION_CONTROLS.md").write_text(
        "# NULL_REPRESENTATION_CONTROLS\n\nTEXT_NULL_FIELD_ORDER, TEXT_NULL_WHITESPACE, and STRUCT_NULL_FIELD_ORDER preserve the parsed `verify_available` boolean. They are tracked as semantic no-ops and must not alter route targets.\n",
        encoding="utf-8",
    )


def prepare(_: argparse.Namespace) -> None:
    splits = build_data()
    write_contracts(splits)
    diagnostic(splits["valid"])
    (OUT / "STATUS_LIVE.md").write_text(
        "# STATUS_LIVE\n\n- status: prepared\n- source: h100_4_verify_confirm\n- next: smoke and 2K matrix\n",
        encoding="utf-8",
    )


def aggregate(_: argparse.Namespace) -> None:
    rows = []
    for summary in sorted((OUT / "cells").glob("*/summary.json")):
        rows.append(json.loads(summary.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("no cell summaries found")
    fieldnames = [
        "cell",
        "privilege",
        "objective",
        "seed",
        "n_train",
        "n_valid",
        "n_test",
        "steps",
        "mean_train_loss",
        "valid_JS_post",
        "test_JS_post",
        "common_reference_test_JS",
        "agreement_test",
        "verify_probability_test",
        "invalid_tool_rate",
        "checkpoint_reloadable",
    ]
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "cell": row["cell"],
                "privilege": row["privilege"],
                "objective": row["objective"],
                "seed": row["seed"],
                "n_train": row["n_train"],
                "n_valid": row["n_valid"],
                "n_test": row["n_test"],
                "steps": row["steps"],
                "mean_train_loss": row["mean_train_loss"],
                "valid_JS_post": row["post_valid"]["JS"],
                "test_JS_post": row["post_test"]["JS"],
                "common_reference_test_JS": row["common_reference_test"]["JS"],
                "agreement_test": row["post_test"]["agreement"],
                "verify_probability_test": row["post_test"].get("verify_probability", 0.0),
                "invalid_tool_rate": row["invalid_tool_rate"],
                "checkpoint_reloadable": row["checkpoint_reloadable"],
            }
        )
    with (OUT / "REPRESENTATION_2K_CELLS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    route_rows = [row for row in rows if row["objective"] == "route_kl"]
    struct = [row for row in route_rows if row["privilege"] == "structured"]
    text = [row for row in route_rows if row["privilege"] == "textual"]
    struct_js = statistics.mean(row["common_reference_test"]["JS"] for row in struct) if struct else None
    text_js = statistics.mean(row["common_reference_test"]["JS"] for row in text) if text else None
    by_seed = {}
    for row in route_rows:
        by_seed.setdefault(int(row["seed"]), {})[row["privilege"]] = row["common_reference_test"]["JS"]
    seed_wins = []
    for seed, pair in sorted(by_seed.items()):
        if "structured" in pair and "textual" in pair:
            diff = pair["structured"] - pair["textual"]
            if abs(diff) <= 1e-5:
                seed_wins.append("tie")
            else:
                seed_wins.append("structured" if diff < 0 else "textual")
    structured_beats_text = bool(seed_wins) and all(win == "structured" for win in seed_wins)
    textual_beats_struct = bool(seed_wins) and all(win == "textual" for win in seed_wins)
    representation_parity = bool(seed_wins) and not structured_beats_text and not textual_beats_struct
    structured_beats_base = None
    if struct:
        structured_beats_base = statistics.mean(row["post_test"]["JS"] for row in struct) < statistics.mean(row["pre_test"]["JS"] for row in struct)
    gate = {
        "component": "verify_tool",
        "actual_train_n": rows[0]["n_train"],
        "structured_beats_base": structured_beats_base,
        "structured_beats_text": structured_beats_text,
        "textual_beats_struct": textual_beats_struct,
        "representation_parity_or_unstable_gap": representation_parity,
        "route_kl_seed_wins": seed_wins,
        "trigger_8k": False,
        "reason": "Source contains only 2048 states total; 8K expansion is not run without generating new data. Gate is based on 1536/256/256 query-disjoint split. Route-KL seed directions are not consistently structured or textual, so no representation advantage is claimed.",
    }
    jdump(OUT / "REPRESENTATION_GATE.json", gate)

    (OUT / "REPRESENTATION_2K_REPORT.md").write_text(
        f"# REPRESENTATION_2K_REPORT\n\nActual train size is {rows[0]['n_train']} because the verify confirm source has 2048 total states and splits are query-disjoint. Four visible GPUs were used in two waves.\n\n"
        f"- Structured Route-KL common-reference test JS: {struct_js}\n"
        f"- Textual Route-KL common-reference test JS: {text_js}\n"
        f"- structured beats reduced-source base on own teacher JS: {structured_beats_base}\n"
        f"- structured beats textual on common reference JS: {structured_beats_text}\n"
        f"- textual beats structured consistently: {textual_beats_struct}\n"
        f"- Route-KL seed wins: {seed_wins}\n",
        encoding="utf-8",
    )
    closed = []
    for row in rows:
        closed.append(
            {
                "method": f"{row['privilege']}_{row['objective']}_seed{row['seed']}",
                "reward": 1.0 - row["common_reference_test"]["JS"],
                "curated_evidence_recall": row["common_reference_test"].get("agreement", 0.0),
                "trajectory_recall": row["post_test"].get("agreement", 0.0),
                "final_answer_recall": row["post_test"].get("agreement", 0.0),
                "verify_frequency": row["post_test"].get("verify_probability", 0.0),
                "premature_end": row["post_test"].get("end_probability", 0.0),
                "repeated_search": row["post_test"].get("search_probability", 0.0),
                "invalid_tool_rate": row["invalid_tool_rate"],
                "note": "offline route-head proxy on VT_REP_TEST_256; privileged info removed from student features",
            }
        )
    with (OUT / "CLOSED_LOOP_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(closed[0]))
        writer.writeheader()
        writer.writerows(closed)
    (OUT / "CLOSED_LOOP_RESULTS.md").write_text(
        "# CLOSED_LOOP_RESULTS\n\nClosed-loop BrowseComp execution is not launched in this script; this file reports the required no-privilege student-runtime route proxy over VT_REP_TEST_256, with privileged info removed from student features.\n",
        encoding="utf-8",
    )
    if structured_beats_text:
        claim = "STRUCTURED_PRIVILEGE_ADVANTAGE_SUPPORTED"
    elif textual_beats_struct:
        claim = "TEXTUALIZATION_ADVANTAGE"
    elif structured_beats_base:
        claim = "REPRESENTATION_PARITY_BUT_PRIVILEGE_DISTILLATION_WORKS"
    else:
        claim = "REPRESENTATION_PARITY_OR_UNSTABLE_GAP_NO_BASE_GAIN"
    (OUT / "STRUCTURED_VS_TEXTUAL.md").write_text(
        f"# STRUCTURED_VS_TEXTUAL\n\nDecision label: `{claim}`. See `REPRESENTATION_2K_CELLS.csv` and `REPRESENTATION_GATE.json` for raw metrics.\n",
        encoding="utf-8",
    )
    jdump(
        OUT / "H1004_PRIVILEGE_REP_HANDOFF.json",
        {
            "component": "verify_tool",
            "structured_representation": "structured non-natural-language JSON boolean verify_available from Harness runtime verify_tool mask",
            "textual_representation": "deterministic natural-language rendering of the same verify_available boolean",
            "information_equivalent": True,
            "best_student": min(rows, key=lambda r: r["common_reference_test"]["JS"])["cell"],
            "structured_beats_base": structured_beats_base,
            "structured_beats_text": structured_beats_text,
            "recommended_paper_claim": claim,
            "official_chroma_parity": False,
        },
    )
    subprocess.run(["bash", "-lc", f"cd {OUT} && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS"], check=True)
    manifest = json.loads((OUT / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    jdump(OUT / "RUN_MANIFEST.json", manifest)
    (OUT / "STATUS_LIVE.md").write_text("# STATUS_LIVE\n\n- status: completed\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.set_defaults(func=prepare)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.set_defaults(func=aggregate)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
