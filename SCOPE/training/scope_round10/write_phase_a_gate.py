#!/usr/bin/env python3
"""Write final Phase A PARITY_GATE + root-cause report (may FAIL)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.decide_rollback_operation import decide_rollback_operation
from training.scope_round10.build_parity_ledger import build_split
from training.scope_round10.phase_a_root_cause import main as write_root_cause

OUT = _REPO / "outputs/scope_round10"
PHASE_A = OUT / "phase_a"


def dec(scores: dict) -> str:
    d = decide_rollback_operation(
        score_continue=float(scores.get("CONTINUE", -1e9)),
        score_replan=float(scores.get("REPLAN", -1e9)),
        score_rollback=float(scores.get("ROLLBACK_TO", -1e9)),
        threshold=0.0,
        disable_replan=True,
    )
    return d.predicted_operation.value


def agr(hf_path: Path, vl_path: Path, hf_key: str, vl_key: str) -> dict:
    mism = n = 0
    with hf_path.open() as hf, vl_path.open() as vl:
        for a, b in zip(hf, vl):
            a = json.loads(a)
            b = json.loads(b)
            n += 1
            if dec(a.get(hf_key) or {}) != dec(b.get(vl_key) or {}):
                mism += 1
    return {"n": n, "mismatch": mism, "agreement": 1.0 - mism / max(n, 1)}


def main() -> None:
    summaries = []
    for seed in (42, 43, 44):
        for split in ("offline_valid", "base_live"):
            summaries.append(build_split(seed, split))
    (PHASE_A / "LEDGER_INDEX.json").write_text(json.dumps(summaries, indent=2) + "\n")

    fresh = []
    for seed in (42, 43, 44):
        for split, default_hf in (
            ("offline_valid", "hf_float32_replay.jsonl"),
            ("base_live", "hf_float32_replay.jsonl"),
        ):
            hf = None
            for name in (default_hf, "hf_bf16_replay.jsonl", "hf_float32_replay.jsonl"):
                p = PHASE_A / f"seed{seed}" / split / name
                if p.exists():
                    hf = p
                    break
            vl = PHASE_A / f"seed{seed}" / split / "vllm_fixed_replay.jsonl"
            if hf is None or not vl.exists():
                # fallback: P0 HF vs new vLLM for holdout
                if split == "base_live":
                    hf = (
                        _REPO
                        / f"outputs/scope_round9/wave_b_p0/rollback_hier_o7_seed{seed}/eval_holdout/hf_replay.jsonl"
                    )
                else:
                    fresh.append({"seed": seed, "split": split, "ready": False})
                    continue
            row = agr(hf, vl, "hf_logits", "vllm_logits")
            row.update({"seed": seed, "split": split, "ready": True, "hf_path": str(hf), "vl_path": str(vl)})
            fresh.append(row)

    (PHASE_A / "FRESH_FLOAT32_VLLM_AGREEMENT.json").write_text(json.dumps(fresh, indent=2) + "\n")

    # Remove ineffective deadzone stabilize claim if present
    stab = PHASE_A / "STABLE_TIE_RULE.json"
    if stab.exists():
        stab.unlink()

    write_root_cause()

    all_ready = all(r.get("ready") for r in fresh)
    all_one = all_ready and all(abs(float(r["agreement"]) - 1.0) < 1e-12 for r in fresh)

    gate = {
        "pass": False,
        "STOP_AFTER_PHASE_A": True,
        "note": (
            "PRIMARY FIX APPLIED: vLLM replay now uses disable_replan=True (R10-P9). "
            "This lifts holdout raw agreement from ~0.75 to ~0.98–0.997. "
            "Residual numerical C↔R flips remain (~0.2–1.7%); float32 HF rescoring does not "
            "eliminate them; local CONTINUE deadzone cannot guarantee 1.0 without creating "
            "straddle disagreements. Per 0807 Gate A (raw agreement must be 1.0), "
            "STOP_AFTER_PHASE_A — no Phase B training."
        ),
        "primary_root_cause": "R10-P9_vllm_replay_missing_disable_replan",
        "secondary_root_cause": "R10-P8_dtype_numerical_score_disagreement",
        "fix_applied": {
            "replay_frozen_vllm_disable_replan": True,
            "redecide_replay_logits_disable_replan": True,
            "contract_unit_test": "tests/scope/test_decide_disable_replan_parity.py",
        },
        "observed": {
            "ledger_after_disable_replan": summaries,
            "fresh_hf_vs_vllm_fixed": fresh,
        },
        "gate_pass_strict_raw_1_0": all_one,
        "all_fresh_ready": all_ready,
    }
    (OUT / "PARITY_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")

    # Strengthen markdown report
    report = (OUT / "PARITY_ROOT_CAUSE_REPORT.md").read_text(encoding="utf-8")
    report += "\n## Gate A decision\n\n"
    report += "**FAIL** — `STOP_AFTER_PHASE_A=true`\n\n"
    report += gate["note"] + "\n"
    (OUT / "PARITY_ROOT_CAUSE_REPORT.md").write_text(report, encoding="utf-8")

    decision = {
        "PARITY_REGRESSION_ROOT_CAUSE": "R10-P9_vllm_replay_missing_disable_replan",
        "PARITY_FIXED": False,
        "PARITY_PRIMARY_BUG_FIXED": True,
        "PARITY_RESIDUAL_NUMERICAL_FLIPS_REMAIN": True,
        "CONTINUE_COLLAPSE_AFTER_PARITY_FIX": None,
        "CLASS_WEIGHT_PRIOR_IS_CAUSAL": None,
        "STAGE1_CANDIDATE_SUMMARY_IS_CAUSAL": None,
        "FROZEN_LIVE_GATE_PASS": False,
        "SMOKE20_STARTED": False,
        "FINAL100_STARTED": False,
        "HARD_CAPABILITY_POSITIVE_SIGNAL": False,
        "RECOMMEND_ROLLBACK_830": False,
        "STOP_AFTER_PHASE_A": True,
    }
    (OUT / "ROOT_CAUSE_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps({"pass": False, "fresh": fresh}, indent=2))


if __name__ == "__main__":
    main()
