#!/usr/bin/env python3
"""Aggregate Phase A outputs and produce LEARNABILITY_GATE_V2 + decision artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_controls(python: str) -> dict[str, Any]:
    proc = subprocess.run(
        [python, "-m", "pytest", "tests/test_learnability_metrics_v2.py", "-v", "--tb=short"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    return {
        "pass": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-3000:],
    }


def correlation_csv(rescore: list[dict], historical: list[dict], out: Path) -> None:
    hist_by_ck = {(r.get("family"), r.get("checkpoint_id")): r for r in historical}
    rows = []
    for r in rescore:
      key = (r.get("family"), r.get("checkpoint_id"))
      h = hist_by_ck.get(key, {})
      rows.append({
        "family": r.get("family"),
        "checkpoint_id": r.get("checkpoint_id"),
        "legacy_d_pre": r.get("legacy_d_pre") or h.get("signed_gap", ""),
        "legacy_d_post": r.get("legacy_d_post") or h.get("legacy_div", ""),
        "signed_gap_pre": r.get("signed_gap_pre"),
        "signed_gap_post": r.get("signed_gap_post"),
        "JS_name_pre": r.get("JS_name_pre"),
        "JS_name_post": r.get("JS_name_post"),
        "CE_pre": r.get("CE_T_on_S_pre"),
        "CE_post": r.get("CE_T_on_S_post"),
        "KL_name_pre": r.get("KL_name_pre"),
        "KL_name_post": r.get("KL_name_post"),
        "v2_gate_pass": r.get("v2_gate_pass"),
      })
    if not rows:
        return
    fields = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def reinterpret_old_gate(rescore: list[dict], out_md: Path) -> None:
    lines = [
        "# OLD_GATE_REINTERPRETATION",
        "",
        "Legacy `d_pre/d_post` = signed logprob gap (M_diag), NOT forward KL.",
        "Negative legacy values indicate student assigned higher prob to teacher tokens.",
        "",
        "| family | checkpoint | legacy_d_pre | legacy_d_post | JS_name Δ | CE Δ | v2_pass |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rescore:
        try:
            js_delta = float(r["JS_name_post"]) - float(r["JS_name_pre"])
            ce_delta = float(r["CE_T_on_S_post"]) - float(r["CE_T_on_S_pre"])
        except (TypeError, ValueError):
            js_delta = 0.0
            ce_delta = 0.0
        lines.append(
            f"| {r.get('family')} | {r.get('checkpoint_id')} | "
            f"{r.get('legacy_d_pre')} | {r.get('legacy_d_post')} | "
            f"{js_delta:.4f} | {ce_delta:.4f} | {r.get('v2_gate_pass')} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide_phase_a(
    controls_pass: bool,
    rescore: list[dict],
) -> dict[str, Any]:
    if not controls_pass:
        return {
            "case": "BLOCKED_BY_METRIC_BUG",
            "STOP_ALL_TRAINING": True,
            "reason": "C0-C4 controls failed",
        }

    passes = [r for r in rescore if str(r.get("v2_gate_pass")).lower() in ("true", "1")]
    # Group by family+objective for seed consistency
    by_setting: dict[str, list[dict]] = defaultdict(list)
    for r in passes:
        fam = r.get("family", "")
        loss = r.get("loss_path", "")
        ck = r.get("checkpoint_id", "")
        # extract seed from checkpoint id suffix
        by_setting[f"{fam}:{loss}:{ck.rsplit('_', 1)[0]}"].append(r)

    consistent = []
    for setting, rows in by_setting.items():
        if len(rows) >= 2:
            consistent.append({"setting": setting, "n_seeds": len(rows), "rows": rows})

    if consistent:
        best = max(consistent, key=lambda x: x["n_seeds"])
        return {
            "case": "EXISTING_CHECKPOINT_STAGE_S",
            "STOP_ALL_TRAINING": False,
            "strongest_candidate": best,
            "reason": "V2 gate flipped PASS with seed consistency",
        }

  # Check if any single pass at all
    if passes:
        return {
            "case": "EXISTING_CHECKPOINT_STAGE_S",
            "STOP_ALL_TRAINING": False,
            "strongest_candidate": passes[0],
            "reason": "single V2 PASS — seed consistency weak",
        }

    return {
        "case": "GRAPH_HYBRID_8K",
        "STOP_FULL_COMPONENT_MIGRATION": True,
        "STOP_ALL_TRAINING": False,
        "reason": "V2 confirms old FAIL — proceed Graph-Hybrid",
    }


def write_metric_contract(out: Path) -> None:
    text = """# METRIC_CONTRACT_V2

| ID | Name | Formula | Gate? | Range |
|----|------|---------|-------|-------|
| M1 | JS_name | JS on tool-name spans | Yes | >= 0 |
| M2 | CE_T_on_S | -mean log p_S(teacher token) on reduced view | Yes | >= 0 |
| M3 | KL_name/args | forward KL(T||S) on spans | Optional | >= 0 |
| M4 | action_agreement | name/exact/arg agreement | Guard | [0,1] |
| M_diag | signed_logprob_gap | mean(log p_T - log p_S) | **No** | signed |

**Gate L V2**: JS_name_post < JS_name_pre AND CE_post < CE_pre AND invalid_tool_rate not worse.

Legacy `d_pre/d_post` / `div` = M_diag. Do NOT use as divergence.
"""
    out.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/0813_next_h20")
    ap.add_argument("--python", default="/data/ppnm/miniconda3/envs/bishop/bin/python")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    controls = run_controls(args.python)
    (out / "CONTROLS_REPORT.json").write_text(json.dumps(controls, indent=2) + "\n")

    rescore = read_csv(out / "phase_a" / "RESCORE_V2.csv")
    historical = read_csv(out / "phase_a" / "HISTORICAL_REEVAL.csv")
    if not historical:
        audit_hist = REPO / "outputs/learnability_audit/HISTORICAL_REEVAL.csv"
        if audit_hist.exists():
            historical = read_csv(audit_hist)

    correlation_csv(rescore, historical, out / "OLD_NEW_METRIC_CORRELATION.csv")
    reinterpret_old_gate(rescore, out / "OLD_GATE_REINTERPRETATION.md")
    write_metric_contract(out / "METRIC_CONTRACT_V2.md")

    decision = decide_phase_a(controls["pass"], rescore)
    gate = {
        "metric_v2_controls_pass": controls["pass"],
        "n_rescored": len(rescore),
        "n_v2_pass": sum(1 for r in rescore if str(r.get("v2_gate_pass")).lower() in ("true", "1")),
        "phase_a_decision": decision,
    }
    (out / "LEARNABILITY_GATE_V2.json").write_text(json.dumps(gate, indent=2) + "\n")
    (out / "LEARNABILITY_GATE_V2.md").write_text(
        f"# LEARNABILITY_GATE_V2\n\n"
        f"- controls_pass: {controls['pass']}\n"
        f"- n_rescored: {gate['n_rescored']}\n"
        f"- n_v2_pass: {gate['n_v2_pass']}\n"
        f"- decision: `{decision['case']}`\n"
        f"- reason: {decision.get('reason')}\n",
        encoding="utf-8",
    )

    next_decision = {"NEXT_DECISION": decision["case"]}
    (out / "NEXT_DECISION.json").write_text(json.dumps(next_decision, indent=2) + "\n")
    (out / "DECISION_STATE.json").write_text(json.dumps({**gate, **next_decision}, indent=2) + "\n")

    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
