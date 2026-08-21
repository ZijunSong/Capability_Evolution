#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def row_to_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    component = str(row.get("component", "scape_component"))
    prefix = str(row.get("student_visible_prefix", ""))
    payload = row.get("event_payload_student_visible") or {}
    instruction = ""
    if isinstance(payload, dict):
        instruction = str(payload.get("instruction_effect") or payload.get("compressed_teacher_view") or "")
    if not instruction and row.get("projectable_target"):
        instruction = json.dumps(row.get("projectable_target"), ensure_ascii=False, sort_keys=True)
    user = (
        "You are the SCAPE student at a real Harness-1 event-active state.\n"
        "Use the same Student-visible state only; do not assume privileged future observations.\n\n"
        "Student-visible state:\n"
        f"{prefix}\n\n"
        "Component event payload visible for OPD supervision:\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)[:4000]}"
    )
    return {
        "data_source": "scape_harness1_opd",
        "ability": component,
        "prompt": [
            {"role": "system", "content": "You are a retrieval agent being trained with SCAPE component-local OPD supervision."},
            {"role": "user", "content": user},
        ],
        "reward_model": {"style": "rule", "ground_truth": instruction or row.get("event_type") or component},
        "extra_info": {
            "index": index,
            "component": component,
            "query_id": row.get("query_id"),
            "rollout_id": row.get("rollout_id"),
            "rollout_seed": row.get("rollout_seed"),
            "step_id": row.get("step_id"),
            "state_uid": row.get("state_uid"),
            "event_type": row.get("event_type"),
            "collector_mode": row.get("collector_mode"),
            "projectable_target": row.get("projectable_target"),
            "event_payload_student_visible": payload,
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    records = [row_to_record(row, i) for i, row in enumerate(rows)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows": len(records)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
