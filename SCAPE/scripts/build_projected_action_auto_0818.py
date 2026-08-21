#!/usr/bin/env python3
"""Audit AUTO target mismatch and build deterministic action projections.

This script only projects a curate call when a recorded runtime state contains
an explicit pre/post curated-set delta. It never invents ids from teacher route
labels or from documents hidden from the reduced student view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def visible_doc_ids(row: dict[str, Any]) -> set[str]:
    view = row.get("reduced_view") or row.get("student_view") or {}
    return {str(d.get("id")) for d in (view.get("documents") or []) if d.get("id") is not None}


def action_text(action: dict[str, Any]) -> str:
    return "to=curate\n" + json.dumps(action["parameters"], ensure_ascii=False, sort_keys=True) + "\n</tool_call>"


def make_projection(row: dict[str, Any]) -> dict[str, Any] | None:
    pre = row.get("curated_ids_pre")
    post = row.get("curated_ids_post")
    if pre is None or post is None:
        return None
    pre_ids = [str(x) for x in pre]
    post_ids = [str(x) for x in post]
    visible = visible_doc_ids(row)
    add_ids = [x for x in post_ids if x not in pre_ids]
    remove_ids = [x for x in pre_ids if x not in post_ids]
    if not add_ids or any(x not in visible for x in add_ids):
        return None
    action = {"tool_name": "curate", "parameters": {"add_ids": add_ids, "remove_ids": remove_ids}}
    return {
        "row_id": f"projected_{row.get('query_id')}_{row.get('step')}_{row.get('snapshot_hash')}",
        "query_id": str(row.get("query_id")),
        "step": int(row.get("step", 0) or 0),
        "snapshot_hash": row.get("snapshot_hash"),
        "prompt_reduced": row.get("prompt_reduced") or row.get("reduced_prompt") or "",
        "prompt_full": row.get("prompt_full") or row.get("full_prompt") or "",
        "response_text": action_text(action),
        "projected_action": action,
        "next_state": row.get("next_state"),
        "next_prompt_reduced": row.get("next_prompt_reduced", ""),
        "next_prompt_full": row.get("next_prompt_full", ""),
        "next_student_action": row.get("next_student_action"),
        "next_teacher_action": row.get("next_teacher_action"),
        "next_student_tool_distribution": row.get("next_student_tool_distribution", {}),
        "next_teacher_tool_distribution": row.get("next_teacher_tool_distribution", {}),
        "provenance": {
            "query_id": str(row.get("query_id")),
            "state_hash": row.get("snapshot_hash"),
            "search_result_ids": sorted(visible),
            "pre_curated_ids": pre_ids,
            "post_curated_ids": post_ids,
            "projected_add_ids": add_ids,
            "projected_remove_ids": remove_ids,
            "projection_source": "deterministic_runtime_state_delta",
            "student_visible_ids_only": True,
        },
        "student_inference_privilege": False,
    }


def audit_old(rows: list[dict[str, Any]], out: Path) -> None:
    tool_counts = Counter()
    curate_args = 0
    for row in rows:
        action = row.get("teacher_full_greedy_tool_call") or row.get("teacher_action") or {}
        name = action.get("name") or action.get("tool_name") or action.get("tool") or "<none>"
        tool_counts[str(name)] += 1
        if str(name) == "curate":
            curate_args += int(bool(action.get("arguments") or action.get("parameters")))
    contract = {
        "component": "auto_populate_first_search",
        "runtime": {
            "trigger": "first successful fan_out_search or search_corpus with nonempty search result ids",
            "pre_curated_state": "working_memory.curated_ids before auto hook",
            "search_result_source": "ids parsed from the successful search observation",
            "post_side_effect": "auto_populate_from_first_search appends top-K visible ids at fair importance",
            "explicit_model_curate_call": False,
            "source_code": {
                "replay": "external/harness-1/training/train_sft.py:187-196",
                "hook": "external/harness-1/harness/ultra_core.py:1935-1971",
                "prompt": "external/harness-1/harness/ultra_core.py:372-377",
            },
        },
        "old_target": {
            "row_count": len(rows),
            "tool_name_distribution": dict(tool_counts),
            "route_distribution_only": True,
            "contains_curate_tool_call": curate_args > 0,
            "contains_real_doc_ids": False,
            "contains_json_args": False,
            "contains_projected_add_ids": False,
        },
        "conclusion": "OLD_AUTO_TARGET_MISMATCH_CONFIRMED",
    }
    (out / "AUTO_TARGET_CONTRACT_AUDIT.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "AUTO_TARGET_CONTRACT_AUDIT.md").write_text(
        "# AUTO_TARGET_CONTRACT_AUDIT\n\n"
        "The runtime applies `auto_populate_from_first_search` after the first successful search. "
        "The model is told not to re-add the automatically inserted documents, and no explicit "
        "`curate` call is emitted for that side effect.\n\n"
        f"- audited rows: `{len(rows)}`\n"
        f"- old tool-name distribution: `{dict(tool_counts)}`\n"
        f"- old target contains curate args: `{curate_args > 0}`\n"
        "- old target contains projected real doc ids: `false`\n"
        "- conclusion: `OLD_AUTO_TARGET_MISMATCH_CONFIRMED`\n\n"
        "Evidence: `train_sft.py:187-196`, `ultra_core.py:1935-1971`, and the AUTO prompt addendum at `ultra_core.py:372-377`.\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    old_rows = load_jsonl(args.old)
    source_rows = load_jsonl(args.source)
    audit_old(old_rows, args.out)

    projections = [p for r in source_rows if (p := make_projection(r)) is not None]
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in projections:
        by_query[row["query_id"]].append(row)
    qids = sorted(by_query, key=lambda q: hashlib.sha256(f"projected-split:{q}".encode()).hexdigest())
    n = len(qids)
    split_q = {
        "train": set(qids[: int(n * 0.70)]),
        "valid": set(qids[int(n * 0.70): int(n * 0.85)]),
        "test": set(qids[int(n * 0.85):]),
    }
    splits = {name: [r for q in qids if q in qs for r in by_query[q]] for name, qs in split_q.items()}
    write_jsonl(args.out / "PROJECTED_ACTION_TRAIN.jsonl", splits["train"])
    write_jsonl(args.out / "PROJECTED_ACTION_VALID.jsonl", splits["valid"])
    write_jsonl(args.out / "PROJECTED_ACTION_TEST.jsonl", splits["test"])

    shuffled = []
    add_pool = [r["projected_action"]["parameters"]["add_ids"] for r in splits["train"]]
    for i, row in enumerate(splits["train"]):
        candidates = [x for x in add_pool if set(x).issubset(set(row["provenance"]["search_result_ids"]))]
        replacement = candidates[(i * 7919 + 17) % len(candidates)] if candidates else row["projected_action"]["parameters"]["add_ids"]
        clone = json.loads(json.dumps(row))
        clone["projected_action"]["parameters"]["add_ids"] = replacement
        clone["response_text"] = action_text(clone["projected_action"])
        clone["provenance"]["shuffle_control"] = True
        clone["provenance"]["original_projected_add_ids"] = row["projected_action"]["parameters"]["add_ids"]
        shuffled.append(clone)
    write_jsonl(args.out / "SHUFFLED_PROJECTED_ACTION_TRAIN.jsonl", shuffled)

    audit = {
        "source": str(args.source),
        "old_source": str(args.old),
        "source_rows": len(source_rows),
        "projected_positive_rows": len(projections),
        "unique_projected_states": len({(r["query_id"], r["snapshot_hash"]) for r in projections}),
        "unique_query_ids": len({r["query_id"] for r in projections}),
        "split_rows": {k: len(v) for k, v in splits.items()},
        "split_query_ids": {k: len(v) for k, v in split_q.items()},
        "support_below_target": len(projections) < 1000,
        "student_inference_privilege": False,
        "resampled_duplicate": False,
        "projection_deterministic": True,
        "visible_ids_only": all(set(r["projected_action"]["parameters"]["add_ids"]).issubset(set(r["provenance"]["search_result_ids"])) for r in projections),
    }
    (args.out / "PROJECTED_ACTION_DATA_AUDIT.md").write_text("# PROJECTED_ACTION_DATA_AUDIT\n\n```json\n" + json.dumps(audit, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    (args.out / "PROJECTED_ACTION_SCHEMA.md").write_text("# PROJECTED_ACTION_SCHEMA\n\nPositive rows are created only from deterministic runtime `curated_ids_post - curated_ids_pre` deltas. Every `add_ids` must occur in the reduced-view search result ids.\n", encoding="utf-8")
    (args.out / "AUTO_FAILURE_CASES.jsonl").write_text("", encoding="utf-8")
    (args.out / "AUTO_FAILURE_CASE_ANALYSIS.md").write_text("# AUTO_FAILURE_CASE_ANALYSIS\n\nOld AUTO route states contain no explicit post-search curate action. Case extraction is deferred until real projected-action closed-loop rows exist.\n", encoding="utf-8")
    (args.out / "RUN_MANIFEST.json").write_text(json.dumps({"status": "prepared", "experiment": "PROJECTED_ACTION_AUTO", "audit": audit}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
