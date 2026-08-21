#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

RUNNER = Path(__file__).with_name("eval_h1004_token_budget_adapter_conditioned.py")
spec = importlib.util.spec_from_file_location("token_eval", RUNNER)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

CELLS = ("PURE_OPD_seed42", "PURE_OPD_seed43", "RL_PLUS_OPD_seed42", "RL_PLUS_OPD_seed43")


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def save(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bootstrap(values: list[float], seed: int = 20260820, n_boot: int = 2000) -> dict[str, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n_boot))
    return {"mean": sum(values) / len(values), "ci95_low": means[int(0.025 * (n_boot - 1))], "ci95_high": means[int(0.975 * (n_boot - 1))], "n_boot": n_boot}


def replay(path: Path, output: Path) -> list[dict[str, Any]]:
    rows = load(path)
    out = []
    for idx, row in enumerate(rows):
        action = {"tool_name": row["tool_name"], "params": row["params"], "legal": row["legal"], "executable": row["executable"]}
        query = {"query_id": row["query_id"], "query": row["params"].get("query", row["query_id"])}
        live = module.run_tool(action, query, output / row["query_id"])
        row.update(live)
        row["overall_reward"] = 0.25 * float(row["legal"]) + 0.25 * float(row["executable"]) + 0.25 * float(row["executed"])
        out.append(row)
        if (idx + 1) % 64 == 0:
            print(json.dumps({"file": str(path), "replayed": idx + 1, "n": len(rows)}), flush=True)
    save(path, out)
    return out


def summarize(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    b = {x["query_id"]: x for x in before}
    a = {x["query_id"]: x for x in after}
    ids = sorted(set(b) & set(a))
    deltas = [float(a[x]["overall_reward"]) - float(b[x]["overall_reward"]) for x in ids]
    return {
        "n_pairs": len(ids),
        "student_before_reward": sum(float(b[x]["overall_reward"]) for x in ids) / len(ids),
        "student_after_reward": sum(float(a[x]["overall_reward"]) for x in ids) / len(ids),
        "paired_delta": bootstrap(deltas),
        "positive": sum(x > 0 for x in deltas),
        "negative": sum(x < 0 for x in deltas),
        "zero": sum(x == 0 for x in deltas),
        "before_invalid_tool_rate": sum(not b[x]["legal"] for x in ids) / len(ids),
        "after_invalid_tool_rate": sum(not a[x]["legal"] for x in ids) / len(ids),
        "before_live_execution_rate": sum(b[x]["executed"] for x in ids) / len(ids),
        "after_live_execution_rate": sum(a[x]["executed"] for x in ids) / len(ids),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    before = {}
    for split in ("dev", "test"):
        path = args.root / "STUDENT_BEFORE" / "before" / split / f"{split}_PER_QUERY.jsonl"
        before[split] = replay(path, args.root / "live_replay" / "STUDENT_BEFORE" / split)
    cells = []
    for cell in CELLS:
        item = {"cell_name": cell, "splits": {}}
        for split in ("dev", "test"):
            path = args.root / cell / "after" / split / f"{split}_PER_QUERY.jsonl"
            after = replay(path, args.root / "live_replay" / cell / split)
            item["splits"][split] = summarize(before[split], after)
        cells.append(item)
    payload = {
        "status": "TOKEN_BUDGET_ADAPTER_CONDITIONED_PAIRED_EVAL_READY",
        "component": "token_budget_marker",
        "student_inference_privilege": False,
        "adapter_conditioned_generation": True,
        "real_harness1_tool_execution": True,
        "reward_contract": "0.25 legal + 0.25 executable + 0.25 live Harness-1 execution",
        "cells": cells,
    }
    module.write_json(args.root / "TOKEN_BUDGET_ADAPTER_CONDITIONED_PAIRED_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
