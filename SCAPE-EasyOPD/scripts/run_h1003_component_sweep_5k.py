#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from easyopd.methods.scape_component_opd.harness1_bridge import QWEN3_LOGICAL_MODEL_ID, QWEN3_STUDENT_BASE

COMPONENTS = ("evidence_graph", "sentence_compress")
TRAIN_JOBS = [
    (0, "evidence_graph", "PURE_OPD", 42),
    (1, "evidence_graph", "PURE_OPD", 43),
    (2, "evidence_graph", "RL_PLUS_OPD", 42),
    (3, "evidence_graph", "RL_PLUS_OPD", 43),
    (4, "sentence_compress", "PURE_OPD", 42),
    (5, "sentence_compress", "PURE_OPD", 43),
    (6, "sentence_compress", "RL_PLUS_OPD", 42),
    (7, "sentence_compress", "RL_PLUS_OPD", 43),
]
COORD = Path("/mnt/songzijun/SCAPE实验协调.md")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], *, cwd: Path, log: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False, timeout=timeout)
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
    ], cwd=ROOT, log=out / "COLLECT.log", timeout=args.collect_timeout)
    stats_path = out / "DATA_STATS.json"
    if not stats_path.exists():
        return blocked(component, "COLLECTOR_FAILED", (result.stderr or result.stdout)[-3000:], out)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if result.returncode != 0 or stats.get("collection_status") != "READY_5K":
        return blocked(component, "INSUFFICIENT_5K_EVENT_SUPPORT", json.dumps(stats, sort_keys=True), out)
    row = {
        "component": component,
        "status": "data_ready",
        "decision": "TEACHER_METRIC_REQUIRED_BEFORE_TRAINING",
        "reason": "5K data ready; Phase E trainer is gated on Teacher and Student Before utility metrics.",
        "data": stats,
        "teacher": {"status": "pending"},
        "student_before": {"status": "pending"},
        "pure_opd": {"seed42": "gated", "seed43": "gated"},
        "rl_plus_opd": {"seed42": "gated", "seed43": "gated"},
    }
    write_json(out / "COMPONENT_RESULT.json", row)
    return row


def launch_training_placeholders(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready = {row["component"] for row in rows if row.get("status") == "data_ready"}
    launched: list[dict[str, Any]] = []
    if ready != set(COMPONENTS):
        return launched
    for gpu, component, loss, seed in TRAIN_JOBS:
        comp_out = args.output_root / component
        train_file = comp_out / "OPD_TRAIN_ROWS.parquet"
        if not train_file.exists():
            train_file = comp_out / "TRAIN_STATES_5K.jsonl"
        if not train_file.exists():
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "scape_component_opd.py"),
            "train",
            "--component", component,
            "--loss", "reverse_kl" if loss == "PURE_OPD" else "hybrid_rl_opd",
            "--seed", str(seed),
            "--output-dir", str(args.output_root),
            "--train-file", str(train_file),
            "--train-batch-size", "2",
            "--student-model", QWEN3_STUDENT_BASE,
            "--teacher-model", QWEN3_STUDENT_BASE,
            "--total-training-steps", str(args.total_training_steps),
        ]
        launched.append({"gpu": gpu, "component": component, "loss": loss, "seed": seed, "status": "configured", "cmd": cmd})
    return launched


def write_coord(handoff: dict[str, Any], output_root: Path) -> None:
    lines = [
        "",
        "## H100-3 component sweep update (2026-08-19)",
        "",
        f"- Handoff status: `{handoff['status']}`, synthetic_fallback={handoff['synthetic_fallback']}.",
        f"- Output root: `{output_root}`.",
    ]
    for row in handoff["components"]:
        data = row.get("data", {}) if isinstance(row.get("data"), dict) else {}
        lines.append(f"- `{row['component']}`: {row['decision']}; unique_event_active={data.get('n_unique_event_active', 'N/A')}; train_states={data.get('train_states', 'N/A')}")
    with COORD.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "component_sweep_0818" / "h100_3")
    ap.add_argument("--pool", type=Path, default=ROOT / "manifests" / "COMPONENT_SWEEP_TRAIN_POOL.json")
    ap.add_argument("--query-min", type=int, default=1000)
    ap.add_argument("--query-max", type=int, default=2000)
    ap.add_argument("--rollouts-min", type=int, default=2)
    ap.add_argument("--rollouts-max", type=int, default=4)
    ap.add_argument("--target-unique-event-states", type=int, default=5000)
    ap.add_argument("--selection-seed", type=int, default=20260818)
    ap.add_argument("--collect-timeout", type=int, default=7200)
    ap.add_argument("--total-training-steps", type=int, default=1)
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    build = run([sys.executable, str(ROOT / "scripts" / "build_component_sweep_train_pool.py"), "--out-dir", str(args.pool.parent), "--target", str(args.query_max)], cwd=ROOT, log=args.output_root / "BUILD_TRAIN_POOL.log", timeout=600)
    if build.returncode != 0:
        reason = build.stdout[-3000:] + build.stderr[-3000:]
        rows = [blocked(c, "QUERY_POOL_INSUFFICIENT", reason, args.output_root / c) for c in COMPONENTS]
    else:
        rows = [component_collect(c, args, args.pool, args.output_root / c) for c in COMPONENTS]
    training = launch_training_placeholders(args, rows)
    write_json(args.output_root / "H1003_COMPONENT_ROWS.json", rows)
    csv_lines = ["component,status,decision,n_unique_event_active,n_queries_selected,n_rollouts_total,train_states"]
    for row in rows:
        data = row.get("data", {}) if isinstance(row.get("data"), dict) else {}
        csv_lines.append(",".join(str(x) for x in [row.get("component"), row.get("status"), row.get("decision"), data.get("n_unique_event_active", "N/A"), data.get("n_queries_selected", "N/A"), data.get("n_rollouts_total", "N/A"), data.get("train_states", "N/A")]))
    (args.output_root / "H1003_COMPONENT_ROWS.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    status = "H1003_COMPONENT_SWEEP_READY_FOR_PHASE_E" if all(r.get("status") == "data_ready" for r in rows) else "H1003_COMPONENT_SWEEP_BLOCKED"
    handoff = {"status": status, "components": rows, "training_jobs": training, "synthetic_fallback": False, "canonical_student_base": QWEN3_STUDENT_BASE, "logical_model_id": QWEN3_LOGICAL_MODEL_ID, "output_root": str(args.output_root)}
    write_json(args.output_root / "H1003_COMPONENT_HANDOFF.json", handoff)
    sums = []
    for path in sorted(p for p in args.output_root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        sums.append(f"{sha256(path)}  {path.relative_to(args.output_root)}")
    (args.output_root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    write_coord(handoff, args.output_root)
    print(json.dumps(handoff, indent=2, ensure_ascii=False))
    return 0 if status == "H1003_COMPONENT_SWEEP_READY_FOR_PHASE_E" else 3


if __name__ == "__main__":
    raise SystemExit(main())
