#!/usr/bin/env python3
"""Run the H100-4 component sweep under the frozen EasyOPD contract.

This wrapper is intentionally fail-closed. Missing real rollout manifests or a
failed framework handoff produce blocked artifacts; smoke fixtures are never
promoted to paper-grade data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EASYOPD = REPO.parent / "SCAPE-EasyOPD"
HANDOFF_CANDIDATES = (
    EASYOPD / "outputs" / "scape_easyopd" / "framework" / "H1003_SCAPE_EASYOPD_HANDOFF.json",
    EASYOPD / "H1003_SCAPE_EASYOPD_HANDOFF.json",
)
QWEN3_STUDENT_BASE = "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507"
QWEN3_LOGICAL_MODEL_ID = "Qwen3-30B-A3B-Instruct-2507"
COMPONENTS = ("token_budget_marker", "verify_tool")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def blocked(component: str, code: str, reason: str, out: Path) -> dict[str, Any]:
    payload = {"component": component, "status": "blocked", "decision": code, "reason": reason, "realizability": "NON_REALIZABLE" if component == "verify_tool" else "PARTIAL", "data": {"collection_status": code, "n_train_unique_states": 0, "target_train_unique_states": 5000}, "teacher": {}, "student_before": {}, "pure_opd": {"seed42": "N/A", "seed43": "N/A"}, "rl_plus_opd": {"seed42": "N/A", "seed43": "N/A"}}
    write_json(out / "COMPONENT_RESULT.json", payload)
    (out / "COMPONENT_RESULT.md").write_text(f"# {component}\n\n- status: `{code}`\n- reason: {reason}\n- no synthetic data or score was written.\n", encoding="utf-8")
    return payload


def resolve_handoff() -> Path | None:
    for path in HANDOFF_CANDIDATES:
        if path.exists():
            return path
    return None


def run_collect(component: str, args: argparse.Namespace, out: Path) -> dict[str, Any]:
    query = args.query_manifest or (EASYOPD / "manifests" / "COMPONENT_SWEEP_TRAIN_POOL.json")
    if not query.exists():
        return blocked(component, "STOP_MISSING_QUERY_MANIFEST", f"missing query manifest: {query}", out)
    cmd = [sys.executable, str(EASYOPD / "scripts" / "scape_component_opd.py"), "collect", "--component", component, "--runtime", "harness1", "--student-base", QWEN3_STUDENT_BASE, "--query-manifest", str(query), "--output-dir", str(out.parent), "--collection-output-dir", str(out), "--query-min", "1000", "--query-max", "2000", "--rollouts-min", "2", "--rollouts-max", "4", "--target-unique-event-states", "5000"]
    if args.rollout_manifest and args.rollout_manifest.exists():
        cmd.extend(["--rollout-manifest", str(args.rollout_manifest)])
    result = subprocess.run(cmd, cwd=EASYOPD, text=True, capture_output=True, check=False)
    (out / "COLLECT_STDOUT.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    stats_path = out / "DATA_STATS.json"
    if not stats_path.exists():
        return blocked(component, "COLLECTOR_FAILED", result.stderr[-2000:], out)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("collection_status") != "READY_5K":
        return blocked(component, "INSUFFICIENT_5K_EVENT_SUPPORT", json.dumps(stats, sort_keys=True), out)
    if component == "verify_tool":
        result = blocked(component, "NON_REALIZABLE_ACTION_SPACE_MISMATCH", "Student verify interface is absent; Teacher/Before/event-support diagnostics only; Student After is N/A.", out)
        result["data"] = stats
        write_json(out / "COMPONENT_RESULT.json", result)
        return result
    result = blocked(component, "TEACHER_METRIC_REQUIRED_BEFORE_TRAINING", "5K collection is ready, but no real single-component Teacher metric is available to pass the utility gate.", out)
    result["data"] = stats
    write_json(out / "COMPONENT_RESULT.json", result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=EASYOPD / "outputs" / "component_sweep_0818" / "h100_4")
    ap.add_argument("--query-manifest", type=Path)
    ap.add_argument("--rollout-manifest", type=Path)
    args = ap.parse_args()
    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    handoff_path = resolve_handoff()
    if handoff_path is None:
        for component in COMPONENTS:
            blocked(component, "STOP_FRAMEWORK_NOT_READY", f"missing framework handoff candidates: {[str(p) for p in HANDOFF_CANDIDATES]}", out_root / component)
        return 2
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if handoff.get("status") not in {"SCAPE_EASYOPD_READY", "SCAPE_EASYOPD_PILOT_READY"}:
        for component in COMPONENTS:
            blocked(component, "STOP_FRAMEWORK_NOT_READY", f"handoff status={handoff.get('status')}", out_root / component)
        return 2
    if handoff.get("canonical_student_base") not in {None, QWEN3_STUDENT_BASE}:
        for component in COMPONENTS:
            blocked(component, "STOP_FRAMEWORK_BASE_MISMATCH", f"handoff canonical_student_base={handoff.get('canonical_student_base')}", out_root / component)
        return 2
    rows = []
    for component in COMPONENTS:
        row = run_collect(component, args, out_root / component)
        rows.append(row)
    write_json(out_root / "H1004_COMPONENT_ROWS.json", rows)
    (out_root / "H1004_COMPONENT_HANDOFF.json").write_text(json.dumps({"status": "H1004_COMPONENT_SWEEP_BLOCKED" if any(r.get("status") == "blocked" for r in rows) else "H1004_COMPONENT_SWEEP_READY", "components": rows, "framework_handoff_sha256": sha256(handoff_path), "framework_handoff_path": str(handoff_path)}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0 if all(r.get("status") != "blocked" for r in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
