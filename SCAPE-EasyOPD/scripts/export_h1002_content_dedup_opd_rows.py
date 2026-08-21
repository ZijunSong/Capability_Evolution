#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

QWEN3_LOGICAL_MODEL_ID = os.environ.get("SCAPE_STUDENT_LOGICAL_MODEL", "Qwen3-30B-A3B-Instruct-2507")


def action_text(target: dict[str, Any] | None) -> str:
    target = target or {}
    name = target.get("name") or target.get("tool_name") or "curate"
    args = target.get("arguments") or target.get("parameters") or {}
    return f"to={name}\n{json.dumps(args, ensure_ascii=False, sort_keys=True)}\n</tool_call>"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _compact_prefix(row: dict[str, Any]) -> str:
    text = str(row.get("student_visible_prefix") or "")
    query_line = ""
    for line in text.splitlines():
        if line.startswith("Query:"):
            query_line = line
            break
    env = row.get("student_observable_env_state") or {}
    return "\n".join([
        "== Dedup OPD compact student state ==",
        query_line or f"Query id: {row.get('query_id')}",
        f"pool_size={env.get('pool_size', 0)} curated_count={env.get('curated_count', 0)}",
        "Task: choose the canonical non-redundant document ids to curate; skip suppressed redundant duplicates.",
    ])


def _compact_tool_history(row: dict[str, Any]) -> list[dict[str, Any]]:
    history = row.get("tool_history") or []
    if not history:
        return []
    last = dict(history[-1])
    ids = [str(x) for x in last.get("returned_doc_ids") or []]
    last["returned_doc_ids_head"] = ids[:4]
    last["returned_doc_ids_tail"] = ids[-4:]
    last["returned_doc_count"] = len(ids)
    last.pop("returned_doc_ids", None)
    last.pop("doc_texts", None)
    last.pop("observation", None)
    return [last]


def _compact_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("event_payload_student_visible") or {}
    target = row.get("projectable_target") or {}
    args = target.get("arguments") or target.get("parameters") or {}
    search_ids = [str(x) for x in payload.get("search_result_doc_ids") or []]
    return {
        "event_type": row.get("event_type"),
        "duplicate_suppressed_count": payload.get("duplicate_suppressed_count"),
        "search_result_doc_count": len(search_ids),
        "search_result_doc_ids_head": search_ids[:4],
        "canonical_add_ids": args.get("add_ids") or [],
        "student_native_targets": payload.get("student_native_targets") or ["CURATE_CANONICAL", "SKIP_REDUNDANT"],
    }


def convert(row: dict[str, Any], index: int) -> dict[str, Any]:
    component = "content_dedup"
    env = row.get("student_observable_env_state") or {}
    compact_env = {
        "curated_count": env.get("curated_count", 0),
        "pool_size": env.get("pool_size", 0),
        "visible_doc_count": len(env.get("visible_doc_ids") or []),
        "curated_ids": env.get("curated_ids") or [],
    }
    prompt_reduced = json.dumps({
        "task": "Choose the next legal Harness-1 tool call as JSON.",
        "component": component,
        "query_id": row.get("query_id"),
        "student_visible_prefix": _compact_prefix(row),
        "tool_history": _compact_tool_history(row),
        "student_observable_env_state": compact_env,
        "student_inference_privilege": False,
    }, ensure_ascii=False, sort_keys=True)
    prompt_full = json.dumps({
        "task": "Choose the next legal Harness-1 tool call as JSON.",
        "component": component,
        "query_id": row.get("query_id"),
        "student_visible_prefix": _compact_prefix(row),
        "tool_history": _compact_tool_history(row),
        "student_observable_env_state": compact_env,
        "teacher_component_enabled": True,
        "event_payload_student_visible": _compact_payload(row),
        "projectable_target": row.get("projectable_target"),
        "student_inference_privilege": False,
    }, ensure_ascii=False, sort_keys=True)
    return {
        "row_id": f"content_dedup_{index:06d}_{str(row.get('state_uid', ''))[:12]}",
        "component": component,
        "query_id": str(row.get("query_id")),
        "state_uid": row.get("state_uid"),
        "prompt_reduced": prompt_reduced,
        "prompt_full": prompt_full,
        "response_text": action_text(row.get("projectable_target")),
        "projectable_target": row.get("projectable_target"),
        "loss_modes": ["projected_action_ce", "next_turn_reverse_kl"],
        "loss_path": "projected_action_ce",
        "collector_mode": row.get("collector_mode"),
        "runtime_name": row.get("runtime_name", "harness1"),
        "model_id": QWEN3_LOGICAL_MODEL_ID,
        "student_inference_privilege": False,
        "projection_valid": bool(row.get("projection_valid")),
        "valid_args": bool(row.get("valid_args")),
        "duplicate_suppressed_count": (row.get("event_payload_student_visible") or {}).get("duplicate_suppressed_count"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component-dir", type=Path, required=True)
    ap.add_argument("--train-frac", type=float, default=0.9)
    args = ap.parse_args()
    states_path = args.component_dir / "TRAIN_STATES_5K.jsonl"
    rows = [convert(row, i) for i, row in enumerate(load_jsonl(states_path))]
    status = "ready"
    if len(rows) != 5000 or len({r["state_uid"] for r in rows}) != 5000:
        status = "invalid_5k"
    if any(r["collector_mode"] != "real_harness1" for r in rows):
        status = "invalid_collector_mode"
    if any(not r["projection_valid"] or not r["valid_args"] for r in rows):
        status = "invalid_projection"
    rows.sort(key=lambda r: hashlib.sha256(f"20260818:{r['state_uid']}".encode()).hexdigest())
    split = int(len(rows) * args.train_frac)
    write_jsonl(args.component_dir / "OPD_TRAIN_ROWS.jsonl", rows[:split])
    write_jsonl(args.component_dir / "OPD_VALID_ROWS.jsonl", rows[split:])
    manifest = {
        "component": "content_dedup",
        "status": status,
        "train_rows": split,
        "valid_rows": len(rows) - split,
        "unique_state_uid": len({r["state_uid"] for r in rows}),
        "loss_path": "projected_action_ce",
        "synthetic_fallback": False,
        "mean_duplicate_suppressed_count": sum(float(r.get("duplicate_suppressed_count") or 0) for r in rows) / max(1, len(rows)),
    }
    (args.component_dir / "H1002_CONTENT_DEDUP_OPD_ROWS_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if status == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
