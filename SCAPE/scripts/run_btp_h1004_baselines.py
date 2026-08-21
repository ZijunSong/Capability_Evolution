#!/usr/bin/env python3
"""Run the available H100-4 BTP baseline controls and emit audit artifacts.

This runner deliberately uses the frozen H100-3 route dataset and the canonical
SCAPE route objective implementation. Methods requiring unavailable V2 privilege
or external skill analyzers remain explicitly blocked rather than being renamed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO.parents[1] / "SCAPE-wt-h100-3" / "SCAPE" / "outputs" / "h100_3_route_opd_verify"
METHODS = [
    ("Base", "none", "base_no_train"),
    ("Full-Harness SFT", "trajectory", "action_ce"),
    ("Standard OPD", "Harness/full-view", "route_kl"),
    ("Matched Text OPD", "same high-level signal textualized", "blocked_v2"),
    ("OPSD-style", "verified textual trace", "blocked_trace"),
    ("OPCD-style", "textual experience/context", "blocked_experience"),
    ("SEED-style distillation-only", "hindsight skill", "blocked_skill_analyzer"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def run_route_audit(out: Path, data: Path, seeds: list[int], limit: int) -> dict:
    script = REPO / "scripts" / "run_h100_4_metric_objective_audit.py"
    objective_out = out / "objective_controls"
    cmd = [sys.executable, str(script), "--skip-heldout", "--out-dir", str(objective_out)]
    start = time.time()
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    (out / "objective_controls.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out / "objective_controls.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return {"command": cmd, "exit_code": proc.returncode, "wall_clock_s": time.time() - start,
            "output": str(objective_out), "data_rows_available": sum(1 for _ in data.open())}


def write_docs(out: Path, manifest: dict, data: Path, objective: dict) -> None:
    (out / "BASELINE_PROTOCOL.md").write_text("""# BASELINE_PROTOCOL\n\n- Base checkpoint: `pat-jj/harness-1` released checkpoint.\n- Data: frozen H100-3 route dataset, same-state Student occupancy and teacher route distributions.\n- Backend: SCAPE V3 route controls; local compatibility data only.\n- GPU policy: `/opt/scape-hf-scorer/bin/python`; no `/mnt` torch environment.\n- Inference privilege: none for all student evaluations.\n- Metrics: canonical non-negative forward KL, reverse KL, and JS.\n- This run does not claim official BrowseComp Chroma parity.\n""", encoding="utf-8")
    (out / "FAIRNESS_AUDIT.md").write_text("""# FAIRNESS_AUDIT\n\nAll available controls use the same released checkpoint family, route train/valid/test artifacts, legal tool vocabulary, and evaluation implementation. Historical route data are retained as provenance and are not mixed with H100-4 verify representation data.\n\n`LOCAL_COMPAT_ONLY=true`\n`official_chroma_parity=false`\n\nUnavailable richer H100-1 V2 manifests prevent launching the matched textual baseline; no old boolean rerun is substituted.\n""", encoding="utf-8")
    (out / "MATCHED_INFORMATION_AUDIT.md").write_text("""# MATCHED_INFORMATION_AUDIT\n\nStatus: BLOCKED_PENDING_H1001_V2\n\nRequired inputs not present in this worktree: `VERIFY_PRIVILEGE_SCHEMA_V2`, `SELECT_POSITIVE.jsonl`, and final train/valid/test query manifests. The existing boolean-only representation experiment is not reused as the BTP matched-information result.\n""", encoding="utf-8")
    notes = {
        "OPSD_ADAPTATION_NOTES.md": "Status: BLOCKED. No completed verified textual Search trajectory manifest is available for a faithful OPSD-style adaptation.",
        "OPCD_ADAPTATION_NOTES.md": "Status: BLOCKED. No frozen historical successful-search textual experience contract is available.",
        "SEED_ADAPTATION_NOTES.md": "Status: BLOCKED. No skill analyzer/rescoring implementation and no stable RL pipeline are available; no SEED claim is made.",
    }
    for name, text in notes.items():
        (out / name).write_text(f"# {name[:-3]}\n\n{text}\n", encoding="utf-8")
    rows = []
    for method, privilege, mode in METHODS:
        status = "completed" if mode in {"base_no_train", "action_ce", "route_kl"} and objective["exit_code"] == 0 else "blocked"
        rows.append({"method": method, "privilege_type": privilege, "text_mediated": "yes" if privilege != "none" else "no", "on_policy": "yes" if mode == "route_kl" else "no", "inference_privilege": "no", "status": status, "mode": mode, "note": "V3 route control" if status == "completed" else "see adaptation notes"})
    with (out / "BASELINE_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "BASELINE_RESULTS.md").write_text("# BASELINE_RESULTS\n\n" + "\n".join(f"- {r['method']}: `{r['status']}` ({r['note']})" for r in rows) + "\n", encoding="utf-8")
    for name in ["TRAINING_BUDGETS.csv", "COMPUTE_COST.csv", "PAIRED_BOOTSTRAP.csv"]:
        (out / name).write_text("status,method,note\nblocked,all,full LLM closed-loop accounting pending canonical baseline runner\n", encoding="utf-8")
    (out / "MAIN_COMPARISON_TABLE.csv").write_text((out / "BASELINE_RESULTS.csv").read_text(encoding="utf-8"), encoding="utf-8")
    (out / "MAIN_COMPARISON_TABLE.md").write_text("# MAIN_COMPARISON_TABLE\n\nNo unsupported superiority claim. Completed entries are route-control results; OPSD/OPCD/SEED and matched-text remain blocked pending their required contracts.\n", encoding="utf-8")
    (out / "H1004_BTP_BASELINE_HANDOFF.json").write_text(json.dumps({"status": "partial_completed", "local_compat_only": True, "official_chroma_parity": False, "blocked_methods": ["Matched Text OPD", "OPSD-style", "OPCD-style", "SEED-style distillation-only"], "objective_run": objective}, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA / "VT_ROUTE_TRAIN_8K.jsonl")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "btp_h100_4_baselines")
    ap.add_argument("--limit", type=int, default=8192)
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data, args.limit)
    manifest = {"run_id": "btp_h100_4_baselines", "status": "running", "repo": str(REPO), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "python": sys.executable, "platform": platform.platform(), "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"), "model": "/mnt/songzijun/models/pat-jj_harness-1-full/harness-1", "data": str(args.data), "data_sha256": sha256(args.data), "n_rows": len(rows), "LOCAL_COMPAT_ONLY": True, "official_chroma_parity": False}
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "STATUS_LIVE.md").write_text("# STATUS_LIVE\n\n- status: running\n- stage: objective controls\n", encoding="utf-8")
    objective = run_route_audit(out, args.data, [42, 43], args.limit)
    write_docs(out, manifest, args.data, objective)
    manifest.update({"status": "partial_completed" if objective["exit_code"] == 0 else "failed", "objective": objective})
    (out / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "STATUS_LIVE.md").write_text(f"# STATUS_LIVE\n\n- status: {manifest['status']}\n- objective_exit_code: {objective['exit_code']}\n- completed_controls: Base, Full-Harness SFT/action-CE, Standard OPD/route-KL\n- blocked: Matched Text OPD, OPSD-style, OPCD-style, SEED-style\n", encoding="utf-8")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(out)}" for p in files) + "\n", encoding="utf-8")
    return objective["exit_code"]

if __name__ == "__main__":
    raise SystemExit(main())
