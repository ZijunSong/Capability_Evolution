#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    prefix = str(row.get("student_visible_prefix") or row.get("query") or row.get("query_id"))[:6000]
    component = str(row.get("component"))
    return [
        {"role": "system", "content": "You are a SCAPE retrieval agent. Use evidence from the current state and respond with the next useful assistant action or answer."},
        {"role": "user", "content": f"Component: {component}\nCurrent state:\n{prefix}"},
    ]


def convert(component: str, in_path: Path, out_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(in_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, row in enumerate(rows):
        if row.get("collector_mode") != "real_harness1" or row.get("synthetic") or row.get("synthetic_fallback"):
            raise ValueError(f"non-real row refused: {component} index={idx}")
        records.append({
            "data_source": f"scape_component_opd/{component}",
            "prompt": make_prompt(row),
            "ability": "scape_component_opd",
            "reward_model": {"style": "rule", "ground_truth": str(row.get("event_type") or component)},
            "extra_info": {
                "index": idx,
                "component": component,
                "query_id": row.get("query_id"),
                "rollout_id": row.get("rollout_id"),
                "state_uid": row.get("state_uid"),
                "event_type": row.get("event_type"),
                "collector_mode": row.get("collector_mode"),
            },
        })
    train = pd.DataFrame(records[:4500])
    valid = pd.DataFrame(records[4500:5000])
    train_path = out_dir / "OPD_TRAIN_ROWS.parquet"
    valid_path = out_dir / "OPD_VALID_ROWS.parquet"
    train.to_parquet(train_path, index=False)
    valid.to_parquet(valid_path, index=False)
    return {"component": component, "train_rows": len(train), "valid_rows": len(valid), "train_file": str(train_path), "valid_file": str(valid_path), "synthetic_fallback": False, "collector_mode": "real_harness1"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=Path("outputs/component_sweep_0818/h100_3_qwen3_faststart"))
    args = ap.parse_args()
    summaries = []
    for component in ["evidence_graph", "sentence_compress"]:
        summaries.append(convert(component, args.output_root / component / "TRAIN_STATES_5K.jsonl", args.output_root / component))
    manifest = {"status": "H1003_OPD_PARQUET_READY", "components": summaries}
    (args.output_root / "H1003_OPD_PARQUET_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
