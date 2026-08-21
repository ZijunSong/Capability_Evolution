#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

QWEN3_STUDENT_BASE = os.environ.get("CANONICAL_STUDENT_BASE", "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
QWEN3_LOGICAL_MODEL_ID = os.environ.get("SCAPE_STUDENT_LOGICAL_MODEL", "Qwen3-30B-A3B-Instruct-2507")

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("auto_populate_first_search", "importance_tagging", "subtractive_curation")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], *, cwd: Path, log: Path) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    log.write_text("$ " + " ".join(cmd) + "\n\nSTDOUT:\n" + result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
    return result


def blocked(component: str, code: str, reason: str, out: Path) -> dict[str, Any]:
    row = {
        "component": component,
        "status": "blocked",
        "decision": code,
        "reason": reason,
        "data": {"collection_status": code, "n_train_unique_states": 0, "target_train_unique_states": 5000},
        "teacher": {},
        "student_before": {},
        "pure_opd": {"seed42": "N/A", "seed43": "N/A"},
        "rl_plus_opd": {"seed42": "N/A", "seed43": "N/A"},
    }
    write_json(out / "COMPONENT_RESULT.json", row)
    (out / "COMPONENT_RESULT.md").write_text(f"# {component}\n\n- status: `{code}`\n- reason: {reason}\n- no synthetic TRAIN_STATES_5K was written.\n", encoding="utf-8")
    return row


def component_collect(component: str, args: argparse.Namespace, pool: Path, out: Path) -> dict[str, Any]:
    result = run([
        sys.executable,
        str(ROOT / "scripts" / "scape_component_opd.py"),
        "collect",
        "--mode", "formal",
        "--component", component,
        "--runtime", "harness1",
        "--student-base", QWEN3_STUDENT_BASE,
        "--query-manifest", str(pool),
        "--output-dir", str(args.output_root),
        "--collection-output-dir", str(out),
        "--query-min", str(args.query_min),
        "--query-max", str(args.query_max),
        "--rollouts-min", str(args.rollouts_min),
        "--rollouts-max", str(args.rollouts_max),
        "--target-unique-event-states", str(args.target_unique_event_states),
        "--selection-seed", str(args.selection_seed),
    ], cwd=ROOT, log=out / "COLLECT.log")
    stats_path = out / "DATA_STATS.json"
    if not stats_path.exists():
        return blocked(component, "COLLECTOR_FAILED", (result.stderr or result.stdout)[-3000:], out)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("collection_status") != "READY_5K":
        return blocked(component, "INSUFFICIENT_5K_EVENT_SUPPORT", json.dumps(stats, sort_keys=True), out)
    row = blocked(component, "TEACHER_METRIC_REQUIRED_BEFORE_TRAINING", "5K data ready; Phase E trainer intentionally not launched until Teacher/Before utility gates are implemented and passed.", out)
    row["data"] = stats
    write_json(out / "COMPONENT_RESULT.json", row)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "component_sweep_0818" / "h100_1")
    ap.add_argument("--pool", type=Path, default=ROOT / "manifests" / "COMPONENT_SWEEP_TRAIN_POOL.json")
    ap.add_argument("--query-min", type=int, default=1000)
    ap.add_argument("--query-max", type=int, default=2000)
    ap.add_argument("--rollouts-min", type=int, default=2)
    ap.add_argument("--rollouts-max", type=int, default=4)
    ap.add_argument("--target-unique-event-states", type=int, default=5000)
    ap.add_argument("--selection-seed", type=int, default=20260818)
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    build = run([sys.executable, str(ROOT / "scripts" / "build_component_sweep_train_pool.py"), "--out-dir", str(args.pool.parent), "--target", str(args.query_max)], cwd=ROOT, log=args.output_root / "BUILD_TRAIN_POOL.log")
    if build.returncode != 0:
        reason = build.stdout[-3000:] + build.stderr[-3000:]
        rows = [blocked(c, "QUERY_POOL_INSUFFICIENT", reason, args.output_root / c) for c in COMPONENTS]
    else:
        rows = [component_collect(c, args, args.pool, args.output_root / c) for c in COMPONENTS]
    write_json(args.output_root / "H1001_COMPONENT_ROWS.json", rows)
    csv_lines = ["component,status,decision,n_unique_event_active,n_queries_selected,n_rollouts_total"]
    for row in rows:
        data = row.get("data", {}) if isinstance(row.get("data"), dict) else {}
        csv_lines.append(",".join(str(x) for x in [row.get("component"), row.get("status"), row.get("decision"), data.get("n_unique_event_active", "N/A"), data.get("n_queries_selected", "N/A"), data.get("n_rollouts_total", "N/A")]))
    (args.output_root / "H1001_COMPONENT_ROWS.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    handoff = {"status": "H1001_COMPONENT_SWEEP_BLOCKED" if any(r.get("status") == "blocked" for r in rows) else "H1001_COMPONENT_SWEEP_READY", "components": rows, "synthetic_fallback": False, "canonical_student_base": QWEN3_STUDENT_BASE, "logical_model_id": QWEN3_LOGICAL_MODEL_ID, "output_root": str(args.output_root)}
    write_json(args.output_root / "H1001_COMPONENT_HANDOFF.json", handoff)
    sums = []
    for path in sorted(p for p in args.output_root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        sums.append(f"{sha256(path)}  {path.relative_to(args.output_root)}")
    (args.output_root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps(handoff, indent=2, ensure_ascii=False))
    return 0 if handoff["status"] == "H1001_COMPONENT_SWEEP_READY" else 3


if __name__ == "__main__":
    raise SystemExit(main())
