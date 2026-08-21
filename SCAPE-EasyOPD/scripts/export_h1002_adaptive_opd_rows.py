#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

COMPONENT = "adaptive_rerank_instruction"
QWEN3_LOGICAL_MODEL_ID = "Qwen3-30B-A3B-Instruct-2507"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _search_query_from_state(row: dict[str, Any]) -> str:
    prefix = str(row.get("student_visible_prefix") or "")
    marker = 'Query: "'
    if marker in prefix:
        tail = prefix.split(marker, 1)[1]
        return tail.split('"', 1)[0]
    return str(row.get("query") or row.get("query_id") or "SCAPE evidence query")


def _search_action_text(query: str) -> str:
    return "to=search_corpus\n" + json.dumps({"query": query}, ensure_ascii=False, sort_keys=True) + "\n</tool_call>"


def convert(row: dict[str, Any], index: int) -> dict[str, Any]:
    payload = row.get("event_payload_student_visible") or {}
    env = row.get("student_observable_env_state") or {}
    tool_history = row.get("tool_history") or []
    base_query = _search_query_from_state(row)
    instruction = str(payload.get("instruction_effect") or "")
    reduced = {
        "task": "Choose the next legal Harness-1 tool call.",
        "component": COMPONENT,
        "query_id": row.get("query_id"),
        "student_visible_prefix": row.get("student_visible_prefix"),
        "tool_history": tool_history,
        "student_observable_env_state": env,
        "student_inference_privilege": False,
    }
    full = {
        **reduced,
        "teacher_component_enabled": True,
        "adaptive_rerank_instruction": instruction,
        "retrieved_doc_ids_off": payload.get("retrieved_doc_ids_off"),
        "retrieved_doc_ids_on": payload.get("retrieved_doc_ids_on"),
        "topK_overlap": payload.get("topK_overlap"),
    }
    # DIRECT component: the teacher supplies an instruction/context delta on the same
    # pre-event Student state.  Keep the response as a legal parser-visible tool call
    # so the shared tool-span audit and token losses remain meaningful.
    teacher_query = base_query if not instruction else f"{base_query} -- focus on specific entities, dates, quantities, and direct multi-constraint evidence"
    return {
        "row_id": f"{COMPONENT}_{index:06d}_{str(row.get('state_uid', ''))[:12]}",
        "component": COMPONENT,
        "query_id": str(row.get("query_id")),
        "state_uid": row.get("state_uid"),
        "prompt_reduced": json.dumps(reduced, ensure_ascii=False, sort_keys=True),
        "prompt_full": json.dumps(full, ensure_ascii=False, sort_keys=True),
        "response_text": _search_action_text(teacher_query),
        "loss_modes": ["direct_full_response_reverse_kl"],
        "collector_mode": row.get("collector_mode"),
        "runtime_name": row.get("runtime_name", "harness1"),
        "model_id": QWEN3_LOGICAL_MODEL_ID,
        "student_inference_privilege": False,
        "projection_valid": False,
        "valid_args": True,
        "adaptive_instruction_effect": instruction,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component-dir", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/components/adaptive_rerank_instruction"))
    ap.add_argument("--train-frac", type=float, default=0.9)
    args = ap.parse_args()
    states_path = args.component_dir / "TRAIN_STATES_5K.jsonl"
    states = load_jsonl(states_path)
    rows = [convert(row, i) for i, row in enumerate(states)]
    manifest: dict[str, Any] = {"component": COMPONENT, "synthetic_fallback": False}
    if len(rows) != 5000 or len({r["state_uid"] for r in rows}) != 5000:
        manifest.update({"status": "invalid_5k", "rows": len(rows), "unique_state_uid": len({r["state_uid"] for r in rows})})
        (args.component_dir / "H1002_ADAPTIVE_OPD_ROWS_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 3
    if any(r.get("collector_mode") != "real_harness1" for r in rows):
        manifest.update({"status": "invalid_collector_mode"})
        (args.component_dir / "H1002_ADAPTIVE_OPD_ROWS_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 3
    rows.sort(key=lambda r: hashlib.sha256(f"20260818:{r['state_uid']}".encode()).hexdigest())
    split = int(len(rows) * args.train_frac)
    write_jsonl(args.component_dir / "OPD_TRAIN_ROWS.jsonl", rows[:split])
    write_jsonl(args.component_dir / "OPD_VALID_ROWS.jsonl", rows[split:])
    manifest.update({"status": "ready", "train_rows": split, "valid_rows": len(rows) - split, "unique_state_uid": 5000, "loss_path": "full_response_kl"})
    (args.component_dir / "H1002_ADAPTIVE_OPD_ROWS_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
