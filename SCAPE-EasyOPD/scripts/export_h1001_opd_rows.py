#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

COMPONENTS = ("auto_populate_first_search", "importance_tagging", "subtractive_curation")
QWEN3_LOGICAL_MODEL_ID = os.environ.get("SCAPE_STUDENT_LOGICAL_MODEL", "Qwen3-30B-A3B-Instruct-2507")


def action_text(target: dict[str, Any] | None) -> str:
    target = target or {}
    name = target.get("name") or target.get("tool_name") or "curate"
    args = target.get("arguments") or target.get("parameters") or {}
    return f"to={name}\n{json.dumps(args, ensure_ascii=False, sort_keys=True)}\n</tool_call>"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def convert(component: str, row: dict[str, Any], index: int) -> dict[str, Any]:
    env = row.get("student_observable_env_state") or {}
    payload = row.get("event_payload_student_visible") or {}
    prompt_reduced = json.dumps({
        "task": "Choose the next legal Harness-1 tool call as JSON.",
        "component": component,
        "query_id": row.get("query_id"),
        "student_visible_prefix": row.get("student_visible_prefix"),
        "tool_history": row.get("tool_history") or [],
        "student_observable_env_state": env,
        "student_inference_privilege": False,
    }, ensure_ascii=False, sort_keys=True)
    prompt_full = json.dumps({
        "task": "Choose the next legal Harness-1 tool call as JSON.",
        "component": component,
        "query_id": row.get("query_id"),
        "student_visible_prefix": row.get("student_visible_prefix"),
        "tool_history": row.get("tool_history") or [],
        "student_observable_env_state": env,
        "teacher_component_enabled": True,
        "event_payload_student_visible": payload,
        "projectable_target": row.get("projectable_target"),
        "student_inference_privilege": False,
    }, ensure_ascii=False, sort_keys=True)
    return {
        "row_id": f"{component}_{index:06d}_{row.get('state_uid', '')[:12]}",
        "component": component,
        "query_id": str(row.get("query_id")),
        "state_uid": row.get("state_uid"),
        "prompt_reduced": prompt_reduced,
        "prompt_full": prompt_full,
        "response_text": action_text(row.get("projectable_target")),
        "loss_modes": ["projected_action_ce", "next_turn_reverse_kl"],
        "collector_mode": row.get("collector_mode"),
        "runtime_name": row.get("runtime_name", "harness1"),
        "model_id": QWEN3_LOGICAL_MODEL_ID,
        "student_inference_privilege": False,
        "projection_valid": bool(row.get("projection_valid")),
        "valid_args": bool(row.get("valid_args")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_1"))
    ap.add_argument("--train-frac", type=float, default=0.9)
    args = ap.parse_args()
    manifest: dict[str, Any] = {"status": "OPD_ROWS_EXPORTED", "components": {}, "synthetic_fallback": False}
    for component in COMPONENTS:
        comp_dir = args.root / component
        states_path = comp_dir / "TRAIN_STATES_5K.jsonl"
        if not states_path.exists():
            manifest["components"][component] = {"status": "missing_train_states"}
            continue
        states = load_jsonl(states_path)
        rows = [convert(component, row, i) for i, row in enumerate(states)]
        if len(rows) != 5000 or len({r["state_uid"] for r in rows}) != 5000:
            manifest["components"][component] = {"status": "invalid_5k", "rows": len(rows), "unique_state_uid": len({r["state_uid"] for r in rows})}
            continue
        rows.sort(key=lambda r: hashlib.sha256(f"20260818:{r['state_uid']}".encode()).hexdigest())
        split = int(len(rows) * args.train_frac)
        write_jsonl(comp_dir / "OPD_TRAIN_ROWS.jsonl", rows[:split])
        write_jsonl(comp_dir / "OPD_VALID_ROWS.jsonl", rows[split:])
        manifest["components"][component] = {"status": "ready", "train_rows": split, "valid_rows": len(rows) - split, "unique_state_uid": 5000}
    (args.root / "H1001_OPD_ROWS_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if all(v.get("status") == "ready" for v in manifest["components"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
