#!/usr/bin/env python3
"""Aggregate learnability audit outputs into final reports."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/learnability_audit"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fam = row.get("family", "")
            cid = row.get("checkpoint_id", "")
            if not fam or not cid or fam == "family" or cid == "checkpoint_id":
                continue
            key = (fam, cid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def run_unit_tests(python: str) -> dict:
    proc = subprocess.run(
        [python, "-m", "pytest", "tests/test_learnability_metrics.py", "-v", "--tb=short"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return {
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
        "pass": proc.returncode == 0,
    }


def aggregate_overfit(out_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(out_dir.glob("overfit/*/results.json")):
        data = json.loads(p.read_text())
        for r in data:
            r["source"] = str(p.parent.name)
            rows.append(r)
    return rows


def decide_go_no_go(
    reeval: list[dict],
    crosscheck: dict,
    overfit: list[dict],
    unit_pass: bool,
) -> dict:
    kl_nonneg = all(
        float(r.get("forward_KL", -1)) >= -1e-7
        and float(r.get("reverse_KL", -1)) >= -1e-7
        and float(r.get("JS", -1)) >= -1e-7
        for r in reeval
    ) if reeval else False

    eval_match = crosscheck.get("pass", False)

    true_jobs = [r for r in overfit if not r.get("shuffled_teacher")]
    shuf_jobs = [r for r in overfit if r.get("shuffled_teacher")]

    def kl_drop(job_rows: list[dict]) -> float:
        if not job_rows:
            return 0.0
        by_step = {int(r["step"]): float(r["forward_KL"]) for r in job_rows}
        if 0 not in by_step:
            return 0.0
        final_step = max(by_step.keys())
        return by_step[0] - by_step[final_step]

    true_drop = max(kl_drop([r for r in true_jobs if r["job"] == j]) for j in {r["job"] for r in true_jobs}) if true_jobs else 0.0
    shuf_drop = max(kl_drop([r for r in shuf_jobs if r["job"] == j]) for j in {r["job"] for r in shuf_jobs}) if shuf_jobs else 0.0

    overfit_ok = true_drop > 0.01 and shuf_drop < true_drop * 0.5

    audit_pass = unit_pass and kl_nonneg and eval_match and overfit_ok

    root_causes = []
    if not kl_nonneg:
        root_causes.append("METRIC_SIGN_BUG")
    if reeval and any(float(r.get("legacy_div", 0)) < -0.001 for r in reeval):
        root_causes.append("METRIC_NAMING_BUG")
    if not eval_match:
        root_causes.append("REDUCTION_BUG")
    if not overfit_ok and true_drop <= 0:
        root_causes.append("TRAINER_BUG")
    if kl_nonneg and not overfit_ok and true_drop <= 0.01:
        root_causes.append("TRUE_NEGATIVE_LEARNABILITY")

    if len(root_causes) > 1:
        label = "MULTIPLE"
    elif len(root_causes) == 1:
        label = root_causes[0]
    elif audit_pass:
        label = "UNRESOLVED"
    else:
        label = "UNRESOLVED"

    return {
        "AUDIT_PASS": audit_pass,
        "kl_nonneg": kl_nonneg,
        "evaluator_trainer_match": eval_match,
        "true_teacher_kl_drop": true_drop,
        "shuffled_kl_drop": shuf_drop,
        "overfit_ok": overfit_ok,
        "unit_tests_pass": unit_pass,
        "ROOT_CAUSE": label,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--python", default="/data/ppnm/miniconda3/envs/bishop/bin/python")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    unit = {"pass": True, "skipped": True}
    if not args.skip_tests:
        unit = run_unit_tests(args.python)
        (out / "UNIT_TEST_REPORT.md").write_text(
            f"# UNIT_TEST_REPORT\n\n"
            f"pass={unit['pass']}\n\n```\n{unit.get('stdout_tail', '')}\n```\n",
            encoding="utf-8",
        )

    reeval = read_csv(out / "HISTORICAL_REEVAL.csv")
    crosscheck_path = out / "crosscheck" / "report.json"
    crosscheck = json.loads(crosscheck_path.read_text()) if crosscheck_path.exists() else {}
    overfit = aggregate_overfit(out)

    overfit_csv = out / "CONTROLLED_OVERFIT.csv"
    if overfit:
        fields = sorted({k for r in overfit for k in r.keys()})
        with overfit_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(overfit)

    verdict = decide_go_no_go(reeval, crosscheck, overfit, unit.get("pass", False))

    (out / "ROOT_CAUSE.json").write_text(json.dumps({"ROOT_CAUSE": verdict["ROOT_CAUSE"], **verdict}, indent=2) + "\n")
    (out / "GO_NO_GO.md").write_text(
        f"# GO / NO-GO\n\n"
        f"- AUDIT_PASS: **{verdict['AUDIT_PASS']}**\n"
        f"- kl_nonneg: {verdict['kl_nonneg']}\n"
        f"- evaluator_trainer_match: {verdict['evaluator_trainer_match']}\n"
        f"- true_teacher_kl_drop: {verdict['true_teacher_kl_drop']:.6f}\n"
        f"- shuffled_kl_drop: {verdict['shuffled_kl_drop']:.6f}\n"
        f"- overfit_ok: {verdict['overfit_ok']}\n"
        f"- ROOT_CAUSE: `{verdict['ROOT_CAUSE']}`\n",
        encoding="utf-8",
    )

    if overfit:
        (out / "CONTROLLED_OVERFIT_REPORT.md").write_text(
            f"# CONTROLLED_OVERFIT_REPORT\n\n"
            f"rows={len(overfit)}\n"
            f"true_drop={verdict['true_teacher_kl_drop']:.6f}\n"
            f"shuf_drop={verdict['shuffled_kl_drop']:.6f}\n",
            encoding="utf-8",
        )

    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
