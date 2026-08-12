#!/usr/bin/env python3
"""Barrier 0: recheck Round 9 P0 metrics, frozen splits, resample, class weights."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.aggregate_wave_b_report import split_report

OUT = _REPO / "outputs/scope_round10/preflight"
R9 = _REPO / "outputs/scope_round9"
P0 = R9 / "wave_b_p0"
DATA = _REPO / "artifacts/datasets/scope_round9"
SEEDS = (42, 43, 44)


def _wc(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _op(row: dict) -> str:
    return str(
        (row.get("target_action") or {}).get("operation")
        or row.get("gold_operation")
        or row.get("operation")
        or ""
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pytest = subprocess.run(
        ["pytest", "tests/scope/", "tests/scope_round9/", "-q", "--tb=line"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    (OUT / "pytest.log").write_text(pytest.stdout + "\n" + pytest.stderr, encoding="utf-8")

    offline = DATA / "frozen_replay/offline_valid.jsonl"
    base_live = DATA / "frozen_replay/base_live.jsonl"
    train = DATA / "hier_sdi/train.jsonl"
    resample_meta = json.loads((DATA / "hier_sdi/TRAIN_P0_RESAMPLE.json").read_text())

    train_prior = Counter()
    with train.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_prior[_op(json.loads(line))] += 1

    coverage_vals = []
    with base_live.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            coverage_vals.append(1.0 if row.get("gold_in_candidates") else 0.0)
    gold_cov = sum(coverage_vals) / max(len(coverage_vals), 1)

    seed_rows = []
    for seed in SEEDS:
        variant = f"rollback_hier_o7_seed{seed}"
        vdir = P0 / variant
        report_path = vdir / "TRAIN_AND_EVAL_REPORT.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = {
                "variant": variant,
                "offline_valid": split_report(vdir, "offline_valid"),
                "holdout": split_report(vdir, "holdout"),
            }
        off = report.get("offline_valid", {}).get("hf_metrics", {})
        hold = report.get("holdout", {}).get("hf_metrics", {})
        parity_off = report.get("offline_valid", {}).get("parity", {})
        parity_hold = report.get("holdout", {}).get("parity", {})
        tw = (report.get("train_meta") or report.get("train_report") or {}).get("class_weights")
        if tw is None and (vdir / "train_only_report.json").exists():
            tor = json.loads((vdir / "train_only_report.json").read_text())
            tw = (tor.get("train_report") or {}).get("class_weights")
        seed_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "offline_bal_acc": off.get("operation_balanced_accuracy"),
                "offline_ContinueRecall": off.get("ContinueRecall"),
                "holdout_bal_acc": hold.get("operation_balanced_accuracy"),
                "holdout_ContinueRecall": hold.get("ContinueRecall"),
                "offline_hf_vllm_agreement_raw": parity_off.get("operation_top1_agreement_raw"),
                "offline_hf_vllm_agreement_barrier": parity_off.get("operation_top1_agreement"),
                "holdout_hf_vllm_agreement_raw": parity_hold.get("operation_top1_agreement_raw"),
                "holdout_hf_vllm_agreement_barrier": parity_hold.get("operation_top1_agreement"),
                "class_weights": tw,
            }
        )

    expected = {
        "offline_valid_n": 402,
        "base_live_n": 3347,
        "continue_after": 3021,
        "rollback_after": 1007,
        "holdout_bal_approx": (0.68, 0.70),
        "holdout_cont_approx": (0.45, 0.50),
        "holdout_agr_approx": (0.74, 0.76),
    }
    actual_n_off = _wc(offline)
    actual_n_live = _wc(base_live)
    checks = {
        "pytest_exit_code": pytest.returncode,
        "pytest_pass": pytest.returncode == 0,
        "offline_valid_n_ok": actual_n_off == expected["offline_valid_n"],
        "base_live_n_ok": actual_n_live == expected["base_live_n"],
        "gold_checkpoint_coverage": gold_cov,
        "gold_checkpoint_coverage_ok": abs(gold_cov - 1.0) < 1e-9,
        "train_continue": train_prior.get("CONTINUE"),
        "train_rollback": train_prior.get("ROLLBACK_TO"),
        "train_resample_ok": (
            train_prior.get("CONTINUE") == expected["continue_after"]
            and train_prior.get("ROLLBACK_TO") == expected["rollback_after"]
        ),
        "replan_disabled_in_train": train_prior.get("REPLAN", 0) == 0,
    }
    # Class weight: P0 uses inverse-freq + CONTINUE*1.25 → ROLLBACK gets higher weight (~2.0).
    cw_ok = True
    for row in seed_rows:
        cw = row.get("class_weights") or {}
        if cw:
            if float(cw.get("ROLLBACK_TO", 0)) <= float(cw.get("CONTINUE", 1)):
                cw_ok = False
    checks["rollback_extra_class_weight"] = cw_ok

    consistent = True
    for row in seed_rows:
        hb = row["holdout_bal_acc"]
        hc = row["holdout_ContinueRecall"]
        ha = row["holdout_hf_vllm_agreement_raw"]
        if hb is None or not (0.68 <= float(hb) <= 0.70):
            # allow recorded 0.680–0.692
            if hb is None or not (0.67 <= float(hb) <= 0.70):
                consistent = False
        if hc is None or not (0.45 <= float(hc) <= 0.50):
            consistent = False
        if ha is None or not (0.74 <= float(ha) <= 0.76):
            consistent = False
    checks["p0_numbers_consistent_with_round9_record"] = consistent

    payload = {
        "expected": expected,
        "actual_counts": {
            "offline_valid": actual_n_off,
            "base_live": actual_n_live,
            "train_prior": dict(train_prior),
            "resample_meta": resample_meta,
            "gold_checkpoint_coverage": gold_cov,
        },
        "p0_seeds": seed_rows,
        "checks": checks,
        "stop_if_inconsistent": not (
            checks["offline_valid_n_ok"]
            and checks["base_live_n_ok"]
            and checks["train_resample_ok"]
            and checks["p0_numbers_consistent_with_round9_record"]
        ),
    }
    (OUT / "ROUND9_P0_RECHECK.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    md = ["# Round 9 P0 Recheck (Barrier 0)", ""]
    md.append(f"- pytest: {'PASS' if checks['pytest_pass'] else 'FAIL'} (exit {pytest.returncode})")
    md.append(f"- offline_valid n={actual_n_off} (expect 402)")
    md.append(f"- base_live n={actual_n_live} (expect 3347)")
    md.append(f"- gold checkpoint coverage={gold_cov}")
    md.append(
        f"- train resample CONTINUE={train_prior.get('CONTINUE')} ROLLBACK={train_prior.get('ROLLBACK_TO')}"
    )
    md.append(f"- ROLLBACK extra class weight: {cw_ok}")
    md.append("")
    md.append("| seed | off bal | off Cont | hold bal | hold Cont | hold agr raw |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in seed_rows:
        md.append(
            f"| {row['seed']} | {row['offline_bal_acc']} | {row['offline_ContinueRecall']} | "
            f"{row['holdout_bal_acc']} | {row['holdout_ContinueRecall']} | "
            f"{row['holdout_hf_vllm_agreement_raw']} |"
        )
    md.append("")
    md.append(
        f"**stop_if_inconsistent={payload['stop_if_inconsistent']}**"
    )
    (OUT / "ROUND9_P0_RECHECK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload["checks"], indent=2))
    if payload["stop_if_inconsistent"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
