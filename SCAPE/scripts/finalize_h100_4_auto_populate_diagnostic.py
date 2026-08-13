#!/usr/bin/env python3
"""Build H100-4 auto_populate_first_search argument diagnostic from real influence states."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "h100_3_real_influence_shards" / "auto_populate_first_search" / "REAL_INFLUENCE_PER_STATE.jsonl"
OUT = ROOT / "outputs" / "h100_4_verify_confirm" / "auto_populate_argument_diagnostic"


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with SRC.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:128]


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def sha256sums(root: Path) -> None:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        lines.append(f"{h.hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    per_state = OUT / "AUTO_POPULATE_ARGUMENT_DIAGNOSTIC_PER_STATE.jsonl"
    with per_state.open("w", encoding="utf-8") as f:
        for r in rows:
            rec = {
                "component": "auto_populate_first_search",
                "query_id": r.get("query_id"),
                "step": r.get("step"),
                "snapshot_hash": r.get("snapshot_hash"),
                "I_name_normalized": r.get("I_name_normalized", r.get("I_name_raw")),
                "I_args_raw": r.get("I_args_raw"),
                "I_arg_key": r.get("I_arg_key"),
                "I_arg_value": r.get("I_arg_value"),
                "student_tool": (r.get("student_executed_tool_action") or {}).get("name"),
                "teacher_tool": (r.get("teacher_full_greedy_tool_call") or {}).get("name"),
                "args_negative": float(r.get("I_args_raw", 0.0)) < 0,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "component": "auto_populate_first_search",
        "n_states": len(rows),
        "source": str(SRC),
        "I_name_normalized_mean": mean([float(r.get("I_name_normalized", r.get("I_name_raw", 0.0))) for r in rows]),
        "I_args_raw_mean": mean([float(r.get("I_args_raw", 0.0)) for r in rows]),
        "I_arg_key_mean": mean([float(r.get("I_arg_key", 0.0)) for r in rows]),
        "I_arg_value_mean": mean([float(r.get("I_arg_value", 0.0)) for r in rows]),
        "negative_args_states": sum(1 for r in rows if float(r.get("I_args_raw", 0.0)) < 0),
        "diagnosis": "argument signal is not strongly negative on the first 128 real-influence states" if mean([float(r.get("I_args_raw", 0.0)) for r in rows]) >= 0 else "argument signal remains negative; inspect token alignment",
        "scorer": "hf_continuation_logprob",
        "gpu_rescore": "skipped: source real-influence per-state rows already contain token_logprob-derived I_args/I_arg_key/I_arg_value",
    }
    (OUT / "AUTO_POPULATE_ARGUMENT_DIAGNOSTIC.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "AUTO_POPULATE_ARGUMENT_DIAGNOSTIC.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader(); w.writerow(summary)
    (OUT / "AUTO_POPULATE_ARGUMENT_DIAGNOSTIC.md").write_text("\n".join([
        "# AUTO_POPULATE_ARGUMENT_DIAGNOSTIC",
        "",
        f"- n_states: {summary['n_states']}",
        f"- I_name_normalized_mean: {summary['I_name_normalized_mean']:.6f}",
        f"- I_args_raw_mean: {summary['I_args_raw_mean']:.6f}",
        f"- I_arg_key_mean: {summary['I_arg_key_mean']:.6f}",
        f"- I_arg_value_mean: {summary['I_arg_value_mean']:.6f}",
        f"- negative_args_states: {summary['negative_args_states']}",
        f"- diagnosis: {summary['diagnosis']}",
        f"- gpu_rescore: {summary['gpu_rescore']}",
    ]) + "\n", encoding="utf-8")
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps({"stage": "h100_4_auto_populate_argument_diagnostic", "status": "completed", "exit_code": 0, **summary}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sha256sums(OUT)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
