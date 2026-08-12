#!/usr/bin/env python3
"""Stage1 VALID gate (+ conditional TEST) for Round13."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round13/phase_b_stage1"
TRAIN = OUT / "training"

MAIN_SEEDS = [
    "r13_onpolicy_querynorm_seed42",
    "r13_onpolicy_querynorm_seed43",
    "r13_onpolicy_querynorm_seed44",
]
ABLATIONS = [
    "r13_onpolicy_querynorm_nohard_seed42",
    "r13_onpolicy_eventuniform_seed42",
]


def load_metrics(variant: str, split: str) -> dict | None:
    p = TRAIN / variant / f"eval_{split}" / "METRICS.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["metrics"]


def collapse(prior: dict) -> bool:
    # single-class collapse if any class >= 0.98
    return max(float(prior.get("CONTINUE", 0)), float(prior.get("ROLLBACK_TO", 0))) >= 0.98


def seed_pass(m: dict) -> tuple[bool, list[str]]:
    reasons = []
    bal = float(m.get("balanced_accuracy") or 0)
    cr = float(m.get("ContinueRecall") or 0)
    rr = float(m.get("RollbackRecall") or 0)
    prior = m.get("prediction_prior") or {}
    parity = float((m.get("canonical_parity") or {}).get("operation_agreement") or 0)
    if bal < 0.75:
        reasons.append(f"bal={bal:.3f}<0.75")
    if cr < 0.70:
        reasons.append(f"CR={cr:.3f}<0.70")
    if rr < 0.70:
        reasons.append(f"RR={rr:.3f}<0.70")
    if collapse(prior):
        reasons.append(f"collapse prior={prior}")
    if parity < 1.0 - 1e-9:
        reasons.append(f"parity={parity}")
    return (len(reasons) == 0), reasons


def test_pass(m: dict) -> tuple[bool, list[str]]:
    reasons = []
    bal = float(m.get("balanced_accuracy") or 0)
    cr = float(m.get("ContinueRecall") or 0)
    rr = float(m.get("RollbackRecall") or 0)
    if bal < 0.72:
        reasons.append(f"bal={bal:.3f}<0.72")
    if cr < 0.68:
        reasons.append(f"CR={cr:.3f}<0.68")
    if rr < 0.68:
        reasons.append(f"RR={rr:.3f}<0.68")
    return (len(reasons) == 0), reasons


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per = {}
    bals = []
    n_pass = 0
    for v in MAIN_SEEDS:
        m = load_metrics(v, "valid")
        if m is None:
            per[v] = {"missing": True, "pass": False}
            continue
        ok, reasons = seed_pass(m)
        per[v] = {"metrics": m, "pass": ok, "fail_reasons": reasons}
        bals.append(float(m.get("balanced_accuracy") or 0))
        n_pass += int(ok)

    span = (max(bals) - min(bals)) if bals else 999.0
    span_ok = span <= 0.05
    gate_pass = n_pass == 3 and span_ok

    ablations = {}
    for v in ABLATIONS:
        m = load_metrics(v, "valid")
        ablations[v] = m

    valid_gate = {
        "n_main_seeds_pass": n_pass,
        "seed_span_balanced_accuracy": span,
        "span_ok": span_ok,
        "STAGE1_VALID_GATE_PASS": gate_pass,
        "STOP_AFTER_STAGE1_VALID": (not gate_pass),
        "per_seed": per,
        "ablations_valid": ablations,
        "tau": 0.0,
    }
    (OUT / "STAGE1_VALID_GATE.json").write_text(json.dumps(valid_gate, indent=2) + "\n")
    print(json.dumps(valid_gate, indent=2))

    if not gate_pass:
        print("STOP_AFTER_STAGE1_VALID=true")
        return

    # Build TEST100 events if needed — for now TEST uses sealed TEST queries collected? 
    # Todo: first and only eval on R13_TEST100. We need test events.
    # If test.jsonl missing, attempt build from onpolicy if collected later; else skip with note.
    test_path = _REPO / "artifacts/datasets/scope_round13/operation_sdi/test.jsonl"
    if not test_path.exists():
        # Collect TEST100 on free GPUs is a separate step; mark pending
        test_gate = {
            "STAGE1_FRESH_GENERALIZATION_PASS": False,
            "pending_test_collection": True,
            "note": "test.jsonl missing; run collect on R13_TEST100 then rebuild",
        }
        (OUT / "STAGE1_TEST_GATE.json").write_text(json.dumps(test_gate, indent=2) + "\n")
        print(json.dumps(test_gate, indent=2))
        return

    # Eval each main seed on TEST if missing
    for i, v in enumerate(MAIN_SEEDS):
        m = load_metrics(v, "test")
        if m is not None:
            continue
        vdir = TRAIN / v
        port = 18710 + i
        subprocess.run(
            [
                sys.executable,
                str(_REPO / "training/scope_round13/eval_stage1_split.py"),
                "--variant-dir",
                str(vdir),
                "--split",
                "test",
                "--gpu",
                str(i),
                "--port",
                str(port),
            ],
            check=False,
            cwd=_REPO,
        )

    test_per = {}
    test_bals = []
    test_n_pass = 0
    for v in MAIN_SEEDS:
        m = load_metrics(v, "test")
        if m is None:
            test_per[v] = {"missing": True, "pass": False}
            continue
        ok, reasons = test_pass(m)
        test_per[v] = {"metrics": m, "pass": ok, "fail_reasons": reasons}
        test_bals.append(float(m.get("balanced_accuracy") or 0))
        test_n_pass += int(ok)
    test_span = (max(test_bals) - min(test_bals)) if test_bals else 999.0
    gen_pass = test_n_pass == 3 and test_span <= 0.05
    test_gate = {
        "n_main_seeds_pass": test_n_pass,
        "seed_span_balanced_accuracy": test_span,
        "STAGE1_FRESH_GENERALIZATION_PASS": gen_pass,
        "STOP_AFTER_STAGE1_TEST": (not gen_pass),
        "per_seed": test_per,
    }
    (OUT / "STAGE1_TEST_GATE.json").write_text(json.dumps(test_gate, indent=2) + "\n")
    print(json.dumps(test_gate, indent=2))


if __name__ == "__main__":
    main()
