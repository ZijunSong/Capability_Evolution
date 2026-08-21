#!/usr/bin/env python3
"""Resumable PROJECTED_CURATION_BUNDLE orchestrator for 2026-08-18 H100-2.

The experiment spec requires a hard sequence of gates. This script does not
promote low-support data or missing-environment attempts into formal results:
it records exactly how far the run can proceed, and resumes later from the same
output directory when support and the /opt ML runtime are available.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.common.sha256sums import write_sha256sums
from scape.common.status import write_status_live

OUT_DEFAULT = REPO / "outputs" / "0818_projected_curation_bundle"
RUN_ID = "h1002_projected_curation_bundle_0818"
TARGET_STATES = 512
TRAIN_CELLS = [
    (0, "COMBINED_PROJECTED_ACTION_CE", 42, "train_combined_ce_s42_gpu0.log"),
    (1, "COMBINED_PROJECTED_ACTION_CE", 43, "train_combined_ce_s43_gpu1.log"),
    (2, "COMBINED_PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL", 42, "train_combined_next_s42_gpu2.log"),
    (3, "COMBINED_PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL", 43, "train_combined_next_s43_gpu3.log"),
    (4, "SUBTRACTIVE_ONLY_PROJECTED_ACTION_CE", 42, "train_subonly_s42_gpu4.log"),
    (5, "SUBTRACTIVE_ONLY_PROJECTED_ACTION_CE", 43, "train_subonly_s43_gpu5.log"),
    (6, "SHUFFLED_CURATION_DELTA_CE", 42, "train_shuffle_s42_gpu6.log"),
    (7, "SHUFFLED_CURATION_DELTA_CE", 43, "train_shuffle_s43_gpu7.log"),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fields:
            return
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sh(cmd: list[str], *, cwd: Path = REPO, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def gpu_snapshot() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi_not_found"}
    q = sh(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    apps = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
    return {"available": q.returncode == 0, "gpus": q.stdout.strip().splitlines(), "compute_apps": apps.stdout.strip().splitlines() if apps.stdout.strip() else []}


def python_has_ml(py: Path) -> tuple[bool, dict[str, bool], str]:
    code = "import importlib.util, json; mods=['torch','transformers','peft']; print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}))"
    proc = sh([str(py), "-c", code])
    if proc.returncode != 0:
        return False, {}, proc.stdout.strip()
    mods = json.loads(proc.stdout.strip())
    return all(mods.values()), mods, proc.stdout.strip()


def find_ml_python(preferred: str | None) -> tuple[Path | None, dict[str, Any]]:
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.extend([Path("/opt/scape-hf-scorer/bin/python"), Path("/opt/scape/bin/python"), Path("/opt/scape-venv/bin/python")])
    seen = set()
    report = []
    for py in candidates:
        if py in seen:
            continue
        seen.add(py)
        if not py.exists():
            report.append({"python": str(py), "exists": False})
            continue
        ok, mods, raw = python_has_ml(py)
        report.append({"python": str(py), "exists": True, "ml_ready": ok, "modules": mods, "raw": raw})
        if ok:
            return py, {"checked": report, "selected": str(py)}
    return None, {"checked": report, "selected": None}


def ensure_collect(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    train = out / "CURATION_BUNDLE_TRAIN.jsonl"
    valid = out / "CURATION_BUNDLE_VALID.jsonl"
    test = out / "CURATION_BUNDLE_TEST.jsonl"
    if args.force_collect or not (train.exists() and valid.exists() and test.exists()):
        write_status_live(out / "STATUS_LIVE.md", stage="projected_curation_collect", run_id=RUN_ID, n_expected=1, n_finished=0, extra={"gpu_snapshot": gpu_snapshot()})
        proc = sh([sys.executable, "scripts/run_projected_curation_bundle_0818.py", "collect", "--out-dir", str(out), "--n-queries", str(args.n_queries), "--seed", str(args.seed)])
        (out / "logs").mkdir(exist_ok=True)
        (out / "logs" / "collect.log").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit(f"collect failed rc={proc.returncode}; see {out / 'logs' / 'collect.log'}")
    rows = load_jsonl(train) + load_jsonl(valid) + load_jsonl(test)
    audit = {
        "source_rows": len(rows),
        "unique_states": len({r.get("state_hash") for r in rows}),
        "unique_queries": len({r.get("query_id") for r in rows}),
        "split_rows": {"train": len(load_jsonl(train)), "valid": len(load_jsonl(valid)), "test": len(load_jsonl(test))},
        "valid_add_ids": sum(bool(r.get("valid_add_ids") or r.get("added_ids")) for r in rows),
        "valid_remove_ids": sum(bool(r.get("valid_remove_ids") or r.get("removed_ids")) for r in rows),
        "terminal_reward_nonzero": sum(float(r.get("qrel_terminal_reward_post") or 0.0) > 0 for r in rows),
        "remove_ids_from_curated_pre": all(set(map(str, r.get("removed_ids") or [])).issubset(set(map(str, r.get("curated_ids_pre") or []))) for r in rows),
        "add_ids_from_visible_documents": all(set(map(str, r.get("added_ids") or [])).issubset({str(d.get("id")) for d in r.get("documents") or []}) for r in rows),
        "student_inference_privilege": any(bool(r.get("student_inference_privilege")) for r in rows),
        "target_states": TARGET_STATES,
    }
    (out / "CURATION_ORACLE_SANITY.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(out / "CURATION_EVENT_COVERAGE.csv", [{"metric": k, "count": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, bool)) else v} for k, v in audit.items()], ["metric", "count"])
    (out / "CURATION_EVALUATOR_REPAIR.md").write_text(
        "# CURATION_EVALUATOR_REPAIR\n\n"
        f"- collected rows: `{audit['source_rows']}`\n"
        f"- unique states: `{audit['unique_states']}` / required `{TARGET_STATES}`\n"
        f"- valid add rows: `{audit['valid_add_ids']}`\n"
        f"- valid remove rows: `{audit['valid_remove_ids']}`\n"
        f"- terminal reward nonzero rows: `{audit['terminal_reward_nonzero']}`\n"
        f"- add ids visible: `{audit['add_ids_from_visible_documents']}`\n"
        f"- remove ids from pre-curated set: `{audit['remove_ids_from_curated_pre']}`\n"
        f"- student inference privilege: `{audit['student_inference_privilege']}`\n\n"
        "The evaluator contract is usable only when all id/terminal checks pass and support reaches the formal state target. Low-support rows are retained as evidence, not promoted to the formal LoRA gate.\n",
        encoding="utf-8",
    )
    return audit


def build_controls(out: Path) -> None:
    train = load_jsonl(out / "CURATION_BUNDLE_TRAIN.jsonl")
    shuffled = []
    remove_pool = [list(r.get("removed_ids") or []) for r in train]
    add_pool = [list(r.get("added_ids") or []) for r in train]
    for i, row in enumerate(train):
        clone = json.loads(json.dumps(row))
        visible = {str(d.get("id")) for d in clone.get("documents") or []}
        pre = set(map(str, clone.get("curated_ids_pre") or []))
        add_candidates = [x for xs in add_pool for x in xs if str(x) in visible]
        rem_candidates = [x for xs in remove_pool for x in xs if str(x) in pre]
        n_add = len(clone.get("added_ids") or [])
        n_rem = len(clone.get("removed_ids") or [])
        add = [str(add_candidates[(i + j) % len(add_candidates)]) for j in range(n_add)] if add_candidates else list(clone.get("added_ids") or [])
        rem = [str(rem_candidates[(i + j) % len(rem_candidates)]) for j in range(n_rem)] if rem_candidates else list(clone.get("removed_ids") or [])
        clone["added_ids"] = list(dict.fromkeys(add))
        clone["removed_ids"] = list(dict.fromkeys(rem))
        clone["projected_action"] = {"tool": "curate", "arguments": {"add_ids": clone["added_ids"], "remove_ids": clone["removed_ids"]}}
        clone["response_text"] = "to=curate\n" + json.dumps(clone["projected_action"]["arguments"], ensure_ascii=False, sort_keys=True) + "\n</tool_call>"
        clone["control"] = "SHUFFLED_CURATION_DELTA"
        shuffled.append(clone)
    write_jsonl(out / "SHUFFLED_CURATION_DELTA.jsonl", shuffled)
    (out / "CURATION_BUNDLE_SCHEMA.md").write_text(
        "# CURATION_BUNDLE_SCHEMA\n\n"
        "Rows supervise only native student `curate(add_ids, remove_ids)` actions. `add_ids` must be visible document ids; `remove_ids` must come from `curated_ids_pre`; no importance values are supervised.\n",
        encoding="utf-8",
    )


def gate_value(out: Path, audit: dict[str, Any]) -> dict[str, Any]:
    rows = load_jsonl(out / "CURATION_BUNDLE_TRAIN.jsonl") + load_jsonl(out / "CURATION_BUNDLE_VALID.jsonl") + load_jsonl(out / "CURATION_BUNDLE_TEST.jsonl")
    per_state = []
    for r in rows:
        pre = float(r.get("qrel_terminal_reward_pre") or 0.0)
        post = float(r.get("qrel_terminal_reward_post") or 0.0)
        delta = post - pre
        per_state.append({"row_id": r.get("row_id"), "query_id": r.get("query_id"), "state_hash": r.get("state_hash"), "projected_terminal_reward_delta": delta, "valid_add": bool(r.get("added_ids")), "valid_remove": bool(r.get("removed_ids")), "student_inference_privilege": False})
    write_jsonl(out / "CURATION_BUNDLE_VALUE_PER_STATE.jsonl", per_state)
    mean_delta = sum(float(r["projected_terminal_reward_delta"]) for r in per_state) / max(1, len(per_state))
    formal_ready = (
        audit["unique_states"] >= TARGET_STATES
        and audit["valid_add_ids"] >= TARGET_STATES
        and audit["valid_remove_ids"] >= TARGET_STATES
        and audit["terminal_reward_nonzero"] > 0
        and audit["remove_ids_from_curated_pre"]
        and audit["add_ids_from_visible_documents"]
        and not audit["student_inference_privilege"]
        and mean_delta > 0
    )
    gate = {
        "decision": "GATE_PASS_FORMAL" if formal_ready else "BLOCKED_SUPPORT_BELOW_FORMAL_TARGET" if audit["unique_states"] < TARGET_STATES else "GATE_FAIL_PROJECTED_EFFECT",
        "n_states": audit["unique_states"],
        "target_states": TARGET_STATES,
        "support_below_target": audit["unique_states"] < TARGET_STATES,
        "mean_terminal_reward_delta": mean_delta,
        "positive_delta_rows": sum(float(r["projected_terminal_reward_delta"]) > 0 for r in per_state),
        "valid_add_ids": audit["valid_add_ids"],
        "valid_remove_ids": audit["valid_remove_ids"],
        "student_inference_privilege": False,
        "allow_training": formal_ready,
        "K": [4, 8],
        "seeds": [42, 43],
        "note": "This is the executable projected-action gate over available repaired rows; formal training is blocked unless n_states >= 512.",
    }
    (out / "CURATION_BUNDLE_K4_K8_GATE.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return gate


def write_empty_downstream(out: Path, reason: str) -> None:
    metric_fields = ["method", "n", "overall_reward", "curated_evidence_recall", "trajectory_recall", "final_answer_recall", "invalid_tool_rate", "curate_event_rate", "valid_add_rate", "valid_remove_rate", "relevant_added_rate", "irrelevant_removed_rate", "curated_set_churn", "tool_calls", "status"]
    rows = [{"method": m, "n": 0, "status": reason} for m in ["BASE_STUDENT", "COMBINED_PROJECTED", "SUBTRACTIVE_ONLY_PROJECTED", "SHUFFLED_CURATION_DELTA", "FULL_HARNESS_REFERENCE"]]
    write_csv(out / "DEV_REAL_CLOSED_LOOP.csv", rows, metric_fields)
    write_csv(out / "TEST_REAL_CLOSED_LOOP.csv", rows, metric_fields)
    write_csv(out / "PAIRED_BOOTSTRAP.csv", [{"comparison": "COMBINED_PROJECTED-BASE_STUDENT", "metric": "overall_reward", "n": 0, "ci95_low": "NA", "ci95_high": "NA", "status": reason}], ["comparison", "metric", "n", "ci95_low", "ci95_high", "status"])
    write_csv(out / "CURATION_MECHANISM_METRICS.csv", [{"metric": "valid_remove_add_mechanism", "value": "NA", "status": reason}], ["metric", "value", "status"])
    (out / "CURATION_CASE_ANALYSIS.md").write_text(f"# CURATION_CASE_ANALYSIS\n\nNot run: `{reason}`. Case analysis requires formal gate pass and real closed-loop rows.\n", encoding="utf-8")


def run_training_if_ready(args: argparse.Namespace, out: Path, gate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    py, env_report = find_ml_python(args.ml_python)
    if not gate.get("allow_training"):
        write_empty_downstream(out, str(gate["decision"]))
        write_csv(out / "TRAINING_CELLS.csv", [{"gpu": g, "variant": v, "seed": s, "status": "not_started", "notes": gate["decision"]} for g, v, s, _ in TRAIN_CELLS], ["gpu", "variant", "seed", "status", "notes"])
        return "blocked_before_training", env_report
    if py is None:
        write_empty_downstream(out, "BLOCKED_ML_ENV_MISSING")
        write_csv(out / "TRAINING_CELLS.csv", [{"gpu": g, "variant": v, "seed": s, "status": "blocked", "notes": "BLOCKED_ML_ENV_MISSING"} for g, v, s, _ in TRAIN_CELLS], ["gpu", "variant", "seed", "status", "notes"])
        return "blocked_ml_env_missing", env_report

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO))
    procs = []
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    for gpu, variant, seed, log_name in TRAIN_CELLS:
        cell_dir = out / "cells" / f"{variant}_seed{seed}"
        if (cell_dir / "DONE").exists() and not args.force_train:
            continue
        log = (logs / log_name).open("w", encoding="utf-8")
        cmd = [
            str(py),
            "scripts/run_projected_curation_bundle_0818.py",
            "train-cell",
            "--out-dir",
            str(out),
            "--train",
            str(out / "CURATION_BUNDLE_TRAIN.jsonl"),
            "--valid",
            str(out / "CURATION_BUNDLE_VALID.jsonl"),
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--gpu",
            str(gpu),
            "--train-limit",
            str(args.train_limit),
            "--eval-limit",
            str(args.eval_limit),
        ]
        procs.append((gpu, variant, seed, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log, log_name))
    cell_rows = []
    while procs:
        live = []
        for item in list(procs):
            gpu, variant, seed, proc, log, log_name = item
            rc = proc.poll()
            live.append(f"gpu{gpu}:{variant}:seed{seed}:pid{proc.pid}:rc={rc}")
            if rc is not None:
                log.close()
                procs.remove(item)
                status = "completed" if rc == 0 else "failed"
                cell_rows.append({"gpu": gpu, "variant": variant, "seed": seed, "status": status, "returncode": rc, "notes": log_name})
        write_status_live(out / "STATUS_LIVE.md", stage="projected_curation_train_8gpu", run_id=RUN_ID, n_expected=len(TRAIN_CELLS), n_finished=len(cell_rows), extra={"live": live, "gpu_snapshot": gpu_snapshot()})
        if procs:
            time.sleep(args.poll_seconds)
    write_csv(out / "TRAINING_CELLS.csv", cell_rows, ["gpu", "variant", "seed", "status", "returncode", "notes"])
    return ("training_completed" if all(r["status"] == "completed" for r in cell_rows) else "training_failed"), env_report


def finalize(out: Path, audit: dict[str, Any], gate: dict[str, Any], train_status: str, env_report: dict[str, Any]) -> dict[str, Any]:
    if gate.get("allow_training") and train_status == "training_completed":
        decision = "REDESIGN_ONCE_CURATION_BUNDLE"
        status = "training_completed_needs_closed_loop_extension"
    elif audit["unique_states"] < TARGET_STATES:
        decision = "REDESIGN_ONCE_CURATION_BUNDLE"
        status = "blocked_support_below_formal_target"
    elif train_status == "blocked_ml_env_missing":
        decision = "REDESIGN_ONCE_CURATION_BUNDLE"
        status = "blocked_ml_env_missing"
    else:
        decision = "DISCARD_CURATION_BUNDLE" if gate["mean_terminal_reward_delta"] <= 0 else "REDESIGN_ONCE_CURATION_BUNDLE"
        status = str(gate["decision"])
    handoff = {
        "artifact": "H1002_PROJECTED_CURATION_BUNDLE_0818_HANDOFF",
        "status": status,
        "decision": decision,
        "out_dir": str(out),
        "audit": audit,
        "gate": gate,
        "training_status": train_status,
        "ml_environment": env_report,
        "gpu_snapshot_final": gpu_snapshot(),
        "supersedes_prior_contradictory_handoff": True,
        "student_inference_privilege": False,
        "next_step": "restore approved /opt ML runtime and recollect to >=512 formal states before actual-LoRA/closed-loop" if status.startswith("blocked") else "complete real closed-loop and case analysis",
    }
    (out / "H1002_0818_HANDOFF.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {"run_id": RUN_ID, "experiment": "PROJECTED_CURATION_BUNDLE", "status": status, "decision": decision, "audit": audit, "gate": gate, "training_status": train_status, "ml_environment": env_report, "student_inference_privilege": False}
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_status_live(out / "STATUS_LIVE.md", stage="projected_curation_finalize", run_id=RUN_ID, n_expected=6, n_finished=3 if status.startswith("blocked") else 6, extra={"status": status, "decision": decision, "training_status": train_status})
    write_sha256sums(out, [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"])
    return handoff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n-queries", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=8182)
    ap.add_argument("--ml-python")
    ap.add_argument("--train-limit", type=int, default=2000)
    ap.add_argument("--eval-limit", type=int, default=512)
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--force-collect", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    audit = ensure_collect(args, out)
    build_controls(out)
    gate = gate_value(out, audit)
    train_status, env_report = run_training_if_ready(args, out, gate)
    handoff = finalize(out, audit, gate, train_status, env_report)
    print(json.dumps(handoff, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
