#!/usr/bin/env python3
"""Stage2 VALID/TEST gates for Round13 pointer scorers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "outputs/scope_round13/stage2_targeted"
TRAIN = OUT / "training"
SEEDS = [
    "r13_ckpt_pointer_seed42",
    "r13_ckpt_pointer_seed43",
    "r13_ckpt_pointer_seed44",
]


def load_metrics(variant: str, split: str) -> dict | None:
    p = TRAIN / variant / f"eval_{split}" / "METRICS.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["metrics"]


def valid_ok(m: dict) -> tuple[bool, list[str]]:
    reasons = []
    if float(m.get("top1") or 0) < 0.75:
        reasons.append(f"top1={m.get('top1')}<0.75")
    if float(m.get("MRR") or 0) < 0.88:
        reasons.append(f"MRR={m.get('MRR')}<0.88")
    if float(m.get("coverage") or 0) < 0.99:
        reasons.append(f"coverage={m.get('coverage')}<0.99")
    if float(m.get("invalid_checkpoint") or 0) >= 0.01:
        reasons.append(f"invalid={m.get('invalid_checkpoint')}")
    return len(reasons) == 0, reasons


def test_ok(m: dict) -> tuple[bool, list[str]]:
    reasons = []
    if float(m.get("top1") or 0) < 0.70:
        reasons.append(f"top1={m.get('top1')}<0.70")
    if float(m.get("MRR") or 0) < 0.85:
        reasons.append(f"MRR={m.get('MRR')}<0.85")
    if float(m.get("coverage") or 0) < 0.99:
        reasons.append(f"coverage={m.get('coverage')}<0.99")
    return len(reasons) == 0, reasons


def ensure_test_split() -> None:
    test_path = _REPO / "artifacts/datasets/scope_round13/checkpoint_targeted/test.jsonl"
    if test_path.exists():
        return
    from training.scope_round13.build_natural_stage2 import build_split

    sdi = _REPO / "artifacts/datasets/scope_round13/operation_sdi/test.jsonl"
    if not sdi.exists():
        return
    rows = [json.loads(l) for l in sdi.open() if l.strip()]
    test = build_split(rows)
    with test_path.open("w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_eval(variant: str, split: str, gpu: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(_REPO)
    subprocess.run(
        [
            sys.executable,
            str(_REPO / "training/scope_round13/eval_stage2_pointer.py"),
            "--variant-dir",
            str(TRAIN / variant),
            "--split",
            split,
            "--gpu",
            "cuda:0",
        ],
        check=False,
        cwd=_REPO,
        env=env,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_test_split()

    per = {}
    tops = []
    n_pass = 0
    for i, v in enumerate(SEEDS):
        m = load_metrics(v, "valid")
        if m is None:
            vdir = TRAIN / v
            if (vdir / "merged" / "config.json").exists() or (vdir / "lora").exists():
                run_eval(v, "valid", i)
            m = load_metrics(v, "valid")
        if m is None:
            per[v] = {"missing": True, "pass": False}
            continue
        ok, reasons = valid_ok(m)
        per[v] = {"metrics": m, "pass": ok, "fail_reasons": reasons}
        tops.append(float(m.get("top1") or 0))
        n_pass += int(ok)

    span = (max(tops) - min(tops)) if tops else 999.0
    gate = {
        "n_seeds_pass": n_pass,
        "top1_span": span,
        "STAGE2_VALID_GATE_PASS": n_pass == 3 and span <= 0.05,
        "per_seed": per,
        "historical_listwise_baseline": {"top1": 0.627, "MRR": 0.808},
    }
    (OUT / "STAGE2_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))
    if not gate["STAGE2_VALID_GATE_PASS"]:
        return

    test_per = {}
    test_n = 0
    for i, v in enumerate(SEEDS):
        m = load_metrics(v, "test")
        if m is None:
            run_eval(v, "test", i)
            m = load_metrics(v, "test")
        if m is None:
            test_per[v] = {"missing": True, "pass": False}
            continue
        ok, reasons = test_ok(m)
        test_per[v] = {"metrics": m, "pass": ok, "fail_reasons": reasons}
        test_n += int(ok)
    out = {
        "STAGE2_NONTRIVIAL_GENERALIZATION_PASS": test_n == 3,
        "n_seeds_pass": test_n,
        "per_seed": test_per,
    }
    gate.update(out)
    (OUT / "STAGE2_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
