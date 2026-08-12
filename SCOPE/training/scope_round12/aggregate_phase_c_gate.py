#!/usr/bin/env python3
"""Aggregate Phase C 3-seed operation + Stage2 gates → FROZEN_LIVE_GATE.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "outputs" / "scope_round12"
PC = OUT / "phase_c"


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    op_seeds = ["full_stage1_seed42", "full_stage1_seed43", "full_stage1_seed44"]
    ck_seeds = [
        "ckpt_canonical_listwise_seed42",
        "ckpt_canonical_listwise_seed43",
        "ckpt_canonical_listwise_seed44",
    ]
    op_reports = []
    for s in op_seeds:
        r = load(PC / s / "TRAIN_AND_EVAL_REPORT.json")
        tau = load(PC / s / "SCALAR_THRESHOLD.json")
        op_reports.append({"seed": s, "report": r, "tau": tau})

    def split_metrics(rep: dict, split: str) -> dict:
        block = (rep.get("report") or {}).get(split) or {}
        # prefer calibrated live if present
        return block.get("learned_stage1_learned_stage2") or {}

    live_bals = []
    op_pass = True
    for item in op_reports:
        off = split_metrics(item, "offline_valid")
        live = item["tau"].get("base_live") or split_metrics(item, "holdout")
        if not off or not live:
            op_pass = False
            continue
        live_bals.append(float(live.get("balanced_accuracy") or 0))
        if not (
            float(off.get("balanced_accuracy") or 0) >= 0.80
            and float(off.get("ContinueRecall") or 0) >= 0.70
            and float(off.get("RollbackRecall") or 0) >= 0.70
            and float(live.get("balanced_accuracy") or 0) >= 0.70
            and float(live.get("ContinueRecall") or 0) >= 0.70
            and float(live.get("RollbackRecall") or 0) >= 0.70
        ):
            op_pass = False
        parity = (live.get("canonical_parity") or {}).get("pass")
        if parity is False:
            op_pass = False
    span = (max(live_bals) - min(live_bals)) if len(live_bals) >= 2 else 999
    if span > 0.05:
        op_pass = False

    ck_pass = True
    ck_top1s = []
    ck_details = []
    for s in ck_seeds:
        r = load(PC / s / "TRAIN_AND_EVAL_REPORT.json")
        if not r:
            ck_pass = False
            ck_details.append({"seed": s, "missing": True})
            continue
        off = ((r.get("offline_valid") or {}).get("oracle_stage1_learned_stage2") or {})
        live = ((r.get("holdout") or {}).get("oracle_stage1_learned_stage2") or {})
        ck_top1s.append(float(live.get("checkpoint_top1") or 0))
        ok = (
            float(off.get("checkpoint_top1") or 0) >= 0.70
            and float(off.get("checkpoint_mrr") or 0) >= 0.85
            and float(live.get("checkpoint_top1") or 0) >= 0.70
            and float(live.get("checkpoint_mrr") or 0) >= 0.85
            and float(live.get("gold_candidate_coverage") or 0) >= 0.99
            and float(live.get("invalid_checkpoint_rate") or 1) < 0.01
        )
        if not ok:
            ck_pass = False
        ck_details.append({"seed": s, "offline": off, "holdout": live, "ok": ok})
    if ck_top1s and (max(ck_top1s) - min(ck_top1s) > 0.05):
        ck_pass = False
    if not any((PC / s / "DONE").exists() for s in ck_seeds):
        # Stage2 not run
        ck_pass = False

    gate = {
        "ran": True,
        "OPERATION_3SEED_GATE_PASS": op_pass,
        "STAGE2_GATE_PASS": ck_pass,
        "pass": bool(op_pass and ck_pass),
        "seed_span_live_bal": span if live_bals else None,
        "operation_seeds": op_reports,
        "checkpoint_seeds": ck_details,
        "STOP_AFTER_FROZEN_LIVE": not bool(op_pass and ck_pass),
    }
    (OUT / "FROZEN_LIVE_GATE.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    # refresh report
    from training.scope_round12.write_report import main as write_report

    write_report()
    print(json.dumps({"pass": gate["pass"], "op": op_pass, "ck": ck_pass}, indent=2))


if __name__ == "__main__":
    main()
