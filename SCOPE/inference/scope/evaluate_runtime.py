#!/usr/bin/env python3
"""Runtime evaluation across bare / minimal / minimal+verifier / full harness.

This script aggregates metrics from existing eval outputs or computes placeholder
summaries for SCOPE bookkeeping. Plug into live BrowseComp runners as needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RUNTIME_CONFIGS = (
    "bare_model",
    "minimal_executor",
    "minimal_executor_plus_hard_verifier",
    "full_harness",
)

DEFAULT_METRICS = {
    "answer_accuracy": 0.0,
    "final_answer_recall": 0.0,
    "citation_precision": 0.0,
    "unsupported_answer_rate": 0.0,
    "search_calls": 0.0,
    "trajectory_length": 0.0,
}


def summarize_trajectory_file(path: Path) -> dict[str, float]:
    if not path.exists():
        return dict(DEFAULT_METRICS)
    n = 0
    acc = 0.0
    fa = 0.0
    cites = 0.0
    unsupported = 0.0
    searches = 0.0
    turns = 0.0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            acc += float(row.get("answer_accuracy", row.get("success", 0.0)))
            fa += float(row.get("final_answer_recall", 0.0))
            cites += float(row.get("citation_precision", row.get("precision", 0.0)))
            unsupported += float(row.get("unsupported_answer", 0.0))
            searches += float(row.get("search_calls", row.get("n_search", 0.0)))
            turns += float(row.get("trajectory_length", row.get("num_turns", 0.0)))
    if n == 0:
        return dict(DEFAULT_METRICS)
    return {
        "answer_accuracy": acc / n,
        "final_answer_recall": fa / n,
        "citation_precision": cites / n,
        "unsupported_answer_rate": unsupported / n,
        "search_calls": searches / n,
        "trajectory_length": turns / n,
        "n": float(n),
    }


def evaluate_runtimes(input_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in RUNTIME_CONFIGS:
        path = input_dir / f"{name}.jsonl"
        alt = input_dir / f"{name}_metrics.json"
        if alt.exists():
            out[name] = json.loads(alt.read_text(encoding="utf-8"))
        else:
            out[name] = summarize_trajectory_file(path)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="outputs/scope_runtime")
    p.add_argument("--out", type=str, default="outputs/scope_runtime/summary.json")
    args = p.parse_args(argv)
    summary = evaluate_runtimes(Path(args.input_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
