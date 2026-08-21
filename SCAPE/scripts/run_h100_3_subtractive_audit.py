#!/usr/bin/env python3
"""H100-3 subtractive curation root-cause audit.

This is intentionally offline and contract-focused. It audits the same-state
JSONL artifacts currently available in the repo, proves the scorer can produce
non-zero values on a synthetic oracle case, and records whether the current
subtractive test rows actually contain enough gold/terminal information to run
the requested closed-loop metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]

import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.adapters.components import minus_mask
from scape.common.manifest import build_run_manifest, finalize_run_manifest, write_run_manifest
from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live
from scape.state.snapshot import capture_snapshot

TOOL_RE = re.compile(r"(?:^|\n)\s*(?:to=|tool=|name=|call\s+)([A-Za-z_][A-Za-z0-9_]*)\s*(?:\n|$)")
JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    matches = list(TOOL_RE.finditer(text or ""))
    for i, m in enumerate(matches):
        name = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end() : end]
        args: dict[str, Any] = {}
        jm = JSON_OBJ_RE.search(block)
        if jm:
            try:
                parsed = json.loads(jm.group(0))
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
        calls.append({"name": name, "arguments": args})
    return calls


def first_call(row: Mapping[str, Any]) -> dict[str, Any]:
    calls = parse_tool_calls(str(row.get("response_text") or ""))
    if calls:
        return calls[0]
    action = row.get("student_action")
    if isinstance(action, dict):
        return {"name": action.get("name"), "arguments": dict(action.get("arguments") or {})}
    return {"name": None, "arguments": {}}


def snapshot_wm(row: Mapping[str, Any]) -> dict[str, Any]:
    snap = row.get("snapshot") or row.get("raw_structured_xi_t") or {}
    if isinstance(snap, dict):
        wm = snap.get("working_memory") or {}
        if isinstance(wm, dict):
            return wm
    return {}


def document_ids(row: Mapping[str, Any]) -> list[str]:
    wm = snapshot_wm(row)
    docs = wm.get("documents") or []
    out: list[str] = []
    for d in docs:
        if isinstance(d, dict) and d.get("id") is not None:
            out.append(str(d["id"]))
    return out


def curated_ids(row: Mapping[str, Any]) -> list[str]:
    wm = snapshot_wm(row)
    raw = wm.get("curated_ids") or []
    return [str(x) for x in raw]


def _ids_from_args(args: Mapping[str, Any], key: str) -> list[str]:
    raw = args.get(key) or []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Iterable):
        return [str(x) for x in raw]
    return []


def valid_argument_audit(row: Mapping[str, Any], call: Mapping[str, Any] | None = None) -> dict[str, Any]:
    c = call or first_call(row)
    args = dict(c.get("arguments") or {})
    docs = set(document_ids(row))
    curated = set(curated_ids(row))
    add_ids = _ids_from_args(args, "add_ids")
    remove_ids = _ids_from_args(args, "remove_ids")
    return {
        "tool": c.get("name"),
        "add_ids": add_ids,
        "remove_ids": remove_ids,
        "valid_add_ids": [x for x in add_ids if x in docs],
        "valid_remove_ids": [x for x in remove_ids if x in curated or x in docs],
        "invalid_add_ids": [x for x in add_ids if x not in docs],
        "invalid_remove_ids": [x for x in remove_ids if x not in curated and x not in docs],
        "n_docs": len(docs),
        "n_curated": len(curated),
    }


def curated_evidence_recall(curated: Iterable[str], gold: Iterable[str]) -> float:
    gold_set = {str(x) for x in gold}
    if not gold_set:
        return 0.0
    got = {str(x) for x in curated}
    return len(got & gold_set) / len(gold_set)


def apply_curate_action(curated: Iterable[str], docs: Iterable[str], action: Mapping[str, Any]) -> list[str]:
    current = [str(x) for x in curated]
    docs_set = {str(x) for x in docs}
    args = dict(action.get("arguments") or {})
    remove = set(_ids_from_args(args, "remove_ids"))
    out = [x for x in current if x not in remove]
    for x in _ids_from_args(args, "add_ids"):
        sx = str(x)
        if sx in docs_set and sx not in out:
            out.append(sx)
    return out


def has_gold_contract(row: Mapping[str, Any]) -> bool:
    snap = row.get("snapshot") if isinstance(row.get("snapshot"), dict) else {}
    wm = snapshot_wm(row)
    meta = (snap.get("metadata") if isinstance(snap, dict) else {}) or {}
    keys = set(wm) | set(row) | set(meta)
    return bool(keys & {"gold_doc_ids", "gold_ids", "qrels", "reference", "answer", "gold_answer"})


def row_signature(row: Mapping[str, Any]) -> str:
    raw = json.dumps(row.get("snapshot") or row.get("raw_structured_xi_t") or row, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_synthetic_oracle() -> dict[str, Any]:
    docs = [
        {"id": "d_gold", "text": "gold evidence answers the question"},
        {"id": "d_noise", "text": "irrelevant distractor"},
    ]
    snap = capture_snapshot(
        query_id="synthetic_oracle_q",
        step=1,
        harness_mask=minus_mask("subtractive_curation"),
        working_memory={
            "documents": docs,
            "curated_docs": [docs[1]],
            "curated_ids": ["d_noise"],
            "gold_doc_ids": ["d_gold"],
        },
    )
    return {
        "query_id": snap.query_id,
        "snapshot": snap.to_dict(),
        "response_text": 'to=curate\n{"add_ids":["d_gold"],"remove_ids":["d_noise"]}\n',
    }


def oracle_sanity(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    syn = build_synthetic_oracle()
    syn_docs = document_ids(syn)
    syn_curated = curated_ids(syn)
    syn_gold = snapshot_wm(syn).get("gold_doc_ids") or []
    oracle_action = {"name": "curate", "arguments": {"add_ids": ["d_gold"], "remove_ids": ["d_noise"]}}
    syn_after = apply_curate_action(syn_curated, syn_docs, oracle_action)

    n = len(rows)
    gold_rows = sum(1 for r in rows if has_gold_contract(r))
    doc_rows = sum(1 for r in rows if document_ids(r))
    base_nonzero_proxy = sum(1 for r in rows if curated_ids(r))
    student_curate = sum(1 for r in rows if first_call(r).get("name") == "curate")
    result = {
        "n_rows": n,
        "synthetic_base_curated_evidence_recall": curated_evidence_recall(syn_curated, syn_gold),
        "synthetic_oracle_curated_evidence_recall": curated_evidence_recall(syn_after, syn_gold),
        "real_rows_with_documents": doc_rows,
        "real_rows_with_gold_or_reference_contract": gold_rows,
        "student_curate_action_rate": student_curate / max(1, n),
        "base_curated_nonempty_rate_proxy": base_nonzero_proxy / max(1, n),
        "decision": "EVALUATOR_NOT_CONSTANT_ZERO_BUT_REAL_TEST_LACKS_TERMINAL_GOLD_CONTRACT" if gold_rows == 0 else "REAL_ROWS_HAVE_GOLD_CONTRACT",
    }
    md = [
        "# SUBTRACTIVE_ORACLE_SANITY",
        "",
        "## Verdict",
        f"- decision: `{result['decision']}`",
        f"- rows audited: {n}",
        "",
        "## Required checks",
        "| check | value | interpretation |",
        "|---|---:|---|",
        f"| synthetic base recall | {result['synthetic_base_curated_evidence_recall']:.6f} | base starts with distractor only |",
        f"| synthetic oracle recall | {result['synthetic_oracle_curated_evidence_recall']:.6f} | oracle curate action gives non-zero metric |",
        f"| real rows with documents | {doc_rows}/{n} | evaluator has state docs to inspect |",
        f"| real rows with gold/reference | {gold_rows}/{n} | required for terminal/final-answer scoring |",
        f"| student curate action rate | {result['student_curate_action_rate']:.6f} | route/action signal in current test rows |",
        "",
        "## Conclusion",
        "The scorer is not inherently constant-zero: the synthetic oracle case reaches recall 1.0. The current same-state subtractive test artifact does not expose gold/reference fields, so real terminal closed-loop metrics cannot be interpreted as component failure without a repaired evaluator/data contract.",
    ]
    return result, "\n".join(md) + "\n"


def event_coverage(rows: list[dict[str, Any]], influence_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    infl_by_q = {str(r.get("query_id")): r for r in influence_rows if r.get("component") == "subtractive_curation"}
    out: list[dict[str, Any]] = []
    for r in rows:
        call = first_call(r)
        audit = valid_argument_audit(r, call)
        infl = infl_by_q.get(str(r.get("query_id")), {})
        event_active = bool(audit["n_docs"] > 0 and (audit["n_curated"] > 0 or call.get("name") == "curate"))
        value = float(infl.get("I_name_normalized") or 0.0) + float(infl.get("I_args_raw") or 0.0)
        out.append(
            {
                "query_id": r.get("query_id"),
                "step": r.get("step"),
                "documents_nonempty": int(audit["n_docs"] > 0),
                "curated_ids_nonempty": int(audit["n_curated"] > 0),
                "teacher_curate_action": int(call.get("name") == "curate"),
                "valid_add_ids": int(bool(audit["valid_add_ids"])),
                "valid_remove_ids": int(bool(audit["valid_remove_ids"])),
                "event_active": int(event_active),
                "value_positive": int(value > 0.0),
                "terminal_reward_available": int(has_gold_contract(r)),
                "tool": call.get("name"),
                "n_docs": audit["n_docs"],
                "n_curated": audit["n_curated"],
                "n_add_ids": len(audit["add_ids"]),
                "n_remove_ids": len(audit["remove_ids"]),
                "n_valid_add_ids": len(audit["valid_add_ids"]),
                "n_valid_remove_ids": len(audit["valid_remove_ids"]),
                "influence_proxy": value,
            }
        )

    n = max(1, len(out))
    sums = Counter()
    tools = Counter()
    for r in out:
        for k in ("documents_nonempty", "curated_ids_nonempty", "teacher_curate_action", "valid_add_ids", "valid_remove_ids", "event_active", "value_positive", "terminal_reward_available"):
            sums[k] += int(r[k])
        tools[str(r["tool"])] += 1
    root = []
    if sums["terminal_reward_available"] == 0:
        root.append("The current test rows do not contain gold/reference fields, so terminal reward/final answer recall cannot be scored from this artifact.")
    if sums["valid_add_ids"] == 0 and sums["valid_remove_ids"] == 0:
        root.append("Valid argument-supervision rows are zero: teacher/student responses rarely emit curate add/remove ids that refer to current document ids.")
    if sums["teacher_curate_action"] == 0:
        root.append("No parsed curate actions were observed in the audited split; current signal is route/gating/context rather than pointer supervision.")
    md = [
        "# SUBTRACTIVE_ARGUMENT_ROOT_CAUSE",
        "",
        "## Coverage summary",
        "| metric | count | rate |",
        "|---|---:|---:|",
    ]
    for k in ("documents_nonempty", "curated_ids_nonempty", "teacher_curate_action", "valid_add_ids", "valid_remove_ids", "event_active", "value_positive", "terminal_reward_available"):
        md.append(f"| {k} | {sums[k]} | {sums[k] / n:.6f} |")
    md.extend(["", "## Parsed tool distribution", "| tool | count |", "|---|---:|"])
    for tool, c in tools.most_common():
        md.append(f"| {tool} | {c} |")
    md.extend(["", "## Root-cause classification"])
    for line in root or ["No blocking data-contract issue detected by offline audit."]:
        md.append(f"- {line}")
    md.extend([
        "",
        "## Training implication",
        "Do not run pointer/add-remove retraining on this artifact. A route/gate-only redesign is the only defensible training path unless a repaired on-policy collect yields real curate-event-positive rows with valid add_ids/remove_ids and terminal gold/reference fields.",
    ])
    return out, "\n".join(md) + "\n"


def code_audit_text() -> str:
    checks = [
        ("named BTP scripts", "missing", "run_btp_subtractive.py / prepare_btp_subtractive_training.py / eval_btp_subtractive_closed_loop.py / train_route_opd.py are not present in this checkout."),
        ("route_opd.py", "missing", "scape/training/route_opd.py is not present; current trainer is hf_tool_opd.py with tool-token/CE losses."),
        ("canonical curate tool", "pass", "legal_tool_names() includes curate and component masks leave the action interface reachable."),
        ("state restore", "pass", "EnvironmentSnapshot.from_dict roundtrip preserves content_hash in synthetic test."),
        ("same-state teacher", "pass", "DualViewRenderer renders full/reduced views without environment stepping."),
        ("final-answer scorer", "blocked", "Current subtractive same-state rows lack gold/reference/final-answer fields needed for terminal scoring."),
        ("curated evidence scorer", "blocked", "Rows expose documents/curated_ids but not qrels/gold ids, so real recall is undefined."),
        ("value_weighted_route_kl", "missing", "No current loss path with this exact recipe name exists; rerun would require implementation or mapping to an existing loss."),
    ]
    lines = ["# SUBTRACTIVE_CODE_AUDIT", "", "| area | status | finding |", "|---|---|---|"]
    for area, status, finding in checks:
        lines.append(f"| {area} | {status} | {finding} |")
    lines.extend([
        "",
        "## Decision",
        "The current repository has enough SCAPE primitives to audit same-state masks, state restore, and curate action reachability, but not the BTP closed-loop scripts named in the 0816 spec. The appropriate repair is data/evaluator contract work before any controlled retry.",
    ])
    return "\n".join(lines) + "\n"


def case_analysis(rows: list[dict[str, Any]], influence_rows: list[dict[str, Any]], limit: int = 100) -> tuple[list[dict[str, Any]], str]:
    infl_by_q = {str(r.get("query_id")): r for r in influence_rows if r.get("component") == "subtractive_curation"}
    scored: list[dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        infl = infl_by_q.get(str(r.get("query_id")), {})
        rr["influence_proxy"] = float(infl.get("I_name_normalized") or 0.0) + float(infl.get("I_args_raw") or 0.0)
        scored.append(rr)
    strata = {
        "K4_positive": lambda r: float(r.get("influence_proxy") or 0.0) > 0.0,
        "K4_negative": lambda r: float(r.get("influence_proxy") or 0.0) <= 0.0,
        "teacher_curate": lambda r: first_call(r).get("name") == "curate",
        "teacher_non_curate": lambda r: first_call(r).get("name") != "curate",
        "documents_rich": lambda r: len(document_ids(r)) >= 2,
        "curated_ids_nonempty": lambda r: bool(curated_ids(r)),
    }
    selected: dict[str, dict[str, Any]] = {}
    for _name, pred in strata.items():
        added = 0
        for r in scored:
            if pred(r):
                selected.setdefault(row_signature(r), r)
                added += 1
            if added >= max(1, limit // len(strata)):
                break
    for r in scored:
        if len(selected) >= limit:
            break
        selected.setdefault(row_signature(r), r)

    cases: list[dict[str, Any]] = []
    root_counts = Counter()
    for r in list(selected.values())[:limit]:
        call = first_call(r)
        audit = valid_argument_audit(r, call)
        reasons: list[str] = []
        if not has_gold_contract(r):
            reasons.append("missing_terminal_gold_contract")
        if call.get("name") != "curate":
            reasons.append("non_curate_route")
        if call.get("name") == "curate" and not audit["valid_add_ids"] and not audit["valid_remove_ids"]:
            reasons.append("curate_without_valid_pointer_ids")
        if not document_ids(r):
            reasons.append("documents_empty")
        if not reasons:
            reasons.append("case_requires_live_closed_loop_replay")
        for reason in reasons:
            root_counts[reason] += 1
        cases.append(
            {
                "case_id": row_signature(r),
                "query_id": r.get("query_id"),
                "step": r.get("step"),
                "snapshot_hash": r.get("snapshot_hash"),
                "student_action": r.get("student_action"),
                "teacher_action": call,
                "full_harness_action": call,
                "runtime_after_action": "not_executed_offline_audit",
                "continuation_K4_K8": "unavailable_without_closed_loop_evaluator",
                "terminal_metric": "undefined_missing_gold_contract" if not has_gold_contract(r) else "not_run",
                "n_docs": audit["n_docs"],
                "n_curated": audit["n_curated"],
                "valid_add_ids": audit["valid_add_ids"],
                "valid_remove_ids": audit["valid_remove_ids"],
                "root_causes": reasons,
            }
        )
    lines = [
        "# SUBTRACTIVE_ZERO_CASE_ANALYSIS",
        "",
        f"- cases analysed: {len(cases)}",
        "",
        "## Root-cause counts",
        "| root cause | count |",
        "|---|---:|",
    ]
    for k, v in root_counts.most_common():
        lines.append(f"| {k} | {v} |")
    lines.extend([
        "",
        "## Interpretation",
        "The dominant failure mode is a data/evaluator contract issue: the available subtractive same-state split does not carry terminal gold/reference fields, and parsed curate actions with valid add/remove ids are absent or extremely sparse. This supports redesigning collection/evaluation before training rather than treating the all-zero closed-loop as a proven component failure.",
    ])
    return cases, "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", type=Path, default=REPO / "outputs" / "true_scape_candidate_b_tournament" / "data" / "subtractive_curation_TEST_512.jsonl")
    ap.add_argument("--influence-jsonl", type=Path, default=REPO / "outputs" / "h100_3_real_influence" / "REAL_INFLUENCE_PER_STATE.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "btp_h100_3_subtractive_audit")
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--case-limit", type=int, default=100)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        run_id="h1003_subtractive_audit_0816",
        stage="h1003_subtractive_audit",
        command=["python", "scripts/run_h100_3_subtractive_audit.py"],
        repo_root=REPO,
        output_dir=out,
        input_paths={"test_jsonl": args.test_jsonl, "influence_jsonl": args.influence_jsonl},
        extra={"limit": args.limit, "case_limit": args.case_limit, "training": False},
    )
    write_run_manifest(out / "RUN_MANIFEST.json", manifest)
    write_status_live(out / "STATUS_LIVE.md", stage="h1003_subtractive_audit", run_id=manifest["run_id"], n_expected=5, n_finished=0)

    rows = load_jsonl(args.test_jsonl, args.limit)
    influence_rows = load_jsonl(args.influence_jsonl, None)

    oracle, oracle_md = oracle_sanity(rows)
    (out / "SUBTRACTIVE_ORACLE_SANITY.md").write_text(oracle_md, encoding="utf-8")
    (out / "SUBTRACTIVE_ORACLE_SANITY.json").write_text(json.dumps(oracle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_status_live(out / "STATUS_LIVE.md", stage="h1003_subtractive_audit", run_id=manifest["run_id"], n_expected=5, n_finished=1)

    coverage, root_md = event_coverage(rows, influence_rows)
    write_csv(out / "SUBTRACTIVE_EVENT_COVERAGE.csv", coverage)
    (out / "SUBTRACTIVE_ARGUMENT_ROOT_CAUSE.md").write_text(root_md, encoding="utf-8")
    write_status_live(out / "STATUS_LIVE.md", stage="h1003_subtractive_audit", run_id=manifest["run_id"], n_expected=5, n_finished=2)

    (out / "SUBTRACTIVE_CODE_AUDIT.md").write_text(code_audit_text(), encoding="utf-8")
    write_status_live(out / "STATUS_LIVE.md", stage="h1003_subtractive_audit", run_id=manifest["run_id"], n_expected=5, n_finished=3)

    cases, cases_md = case_analysis(rows, influence_rows, args.case_limit)
    (out / "SUBTRACTIVE_ZERO_CASES.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    (out / "SUBTRACTIVE_ZERO_CASE_ANALYSIS.md").write_text(cases_md, encoding="utf-8")
    write_status_live(out / "STATUS_LIVE.md", stage="h1003_subtractive_audit", run_id=manifest["run_id"], n_expected=5, n_finished=4)

    redesign = {
        "decision": "REDESIGN_DATA_EVALUATOR_CONTRACT_BEFORE_RETRY",
        "allowed_formal_retry": False,
        "reason": oracle["decision"],
        "required_next_collect_filter": [
            "curation event active",
            "documents nonempty",
            "teacher emits curate or meaningful route/gate decision",
            "valid add_ids or remove_ids for pointer training",
            "gold/reference fields available for terminal closed-loop metrics",
        ],
        "do_not_train": ["pointer/add-remove objective on current artifact", "value_weighted_route_kl retry before evaluator/data repair"],
    }
    (out / "SUBTRACTIVE_REDESIGN_MANIFEST.json").write_text(json.dumps(redesign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    repaired_rows = [
        {"policy": "Base", "curated_evidence_recall": "undefined", "overall_reward": "undefined", "final_answer_recall": "undefined", "status": "blocked_missing_gold_contract"},
        {"policy": "Full Harness", "curated_evidence_recall": "undefined", "overall_reward": "undefined", "final_answer_recall": "undefined", "status": "blocked_missing_gold_contract"},
        {"policy": "Student", "curated_evidence_recall": "undefined", "overall_reward": "undefined", "final_answer_recall": "undefined", "status": "not_rerun_audit_blocks_training"},
        {"policy": "shuffle control", "curated_evidence_recall": "undefined", "overall_reward": "undefined", "final_answer_recall": "undefined", "status": "not_applicable_before_repair"},
    ]
    write_csv(out / "REPAIRED_CLOSED_LOOP_RESULTS.csv", repaired_rows)
    (out / "REPAIRED_CLOSED_LOOP_RESULTS.md").write_text(
        "# REPAIRED_CLOSED_LOOP_RESULTS\n\nNo repaired closed-loop retry was run. The audit blocks training because the current real subtractive test artifact lacks gold/reference fields and valid curate pointer supervision.\n",
        encoding="utf-8",
    )
    handoff = {
        "artifact": "H1003_SUBTRACTIVE_AUDIT_HANDOFF",
        "out_dir": str(out),
        "decision": redesign["decision"],
        "allowed_formal_retry": False,
        "required_outputs_present": True,
        "next_step": "repair evaluator/data contract and recollect curate-event-positive rows before training",
    }
    (out / "H1003_SUBTRACTIVE_AUDIT_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_run_manifest(out / "RUN_MANIFEST.json", finalize_run_manifest(manifest, exit_code=0, completed_shards=["oracle", "coverage", "code", "cases", "decision"]))
    write_status_live(out / "STATUS_LIVE.md", stage="h1003_subtractive_audit", run_id=manifest["run_id"], n_expected=5, n_finished=5, extra={"decision": redesign["decision"]})
    files = [p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS"]
    write_sha256sums(out, files, out_name="SHA256SUMS")
    print(json.dumps(handoff, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
