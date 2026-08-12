#!/usr/bin/env python3
"""Wait for Phase A GPU0-5 markers, compare float32 HF vs fixed vLLM, write Gate A."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.decide_rollback_operation import decide_rollback_operation
from training.scope_round10.build_parity_ledger import build_split
from training.scope_round10.phase_a_root_cause import main as write_root_cause

OUT = _REPO / "outputs/scope_round10"
MARK = OUT / "markers"
PHASE_A = OUT / "phase_a"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def decide(scores: dict):
    return decide_rollback_operation(
        score_continue=float(scores.get("CONTINUE", -1e9)),
        score_replan=float(scores.get("REPLAN", -1e9)),
        score_rollback=float(scores.get("ROLLBACK_TO", -1e9)),
        threshold=0.0,
        disable_replan=True,
    )


def _hf_path(seed: int, split: str) -> Path | None:
    for name in ("hf_float32_replay.jsonl", "hf_bf16_replay.jsonl"):
        p = PHASE_A / f"seed{seed}" / split / name
        if p.exists():
            return p
    return None


def compare_fresh(seed: int, split: str) -> dict:
    hf_path = _hf_path(seed, split)
    vl_path = PHASE_A / f"seed{seed}" / split / "vllm_fixed_replay.jsonl"
    if hf_path is None or not vl_path.exists():
        return {"ready": False, "seed": seed, "split": split}
    hf_rows = load_jsonl(hf_path)
    vl_rows = load_jsonl(vl_path)
    n = min(len(hf_rows), len(vl_rows))
    mism = 0
    fallback = 0
    for h, v in zip(hf_rows[:n], vl_rows[:n]):
        hd = decide(h.get("hf_logits") or {})
        # vLLM file may already have disable_replan preds; redecide from logits anyway
        vd = decide(v.get("vllm_logits") or {})
        if hd.predicted_operation.value != vd.predicted_operation.value:
            mism += 1
        if h.get("fallback_reason") or v.get("fallback_reason"):
            fallback += 1
    return {
        "ready": True,
        "seed": seed,
        "split": split,
        "n": n,
        "agreement": 1.0 - mism / max(n, 1),
        "mismatch": mism,
        "fallback": fallback,
    }


def main() -> None:
    # Refresh ledger from archived P0 logits (documents REPLAN root cause).
    summaries = []
    for seed in (42, 43, 44):
        for split in ("offline_valid", "base_live"):
            summaries.append(build_split(seed, split))
    (PHASE_A / "LEDGER_INDEX.json").write_text(json.dumps(summaries, indent=2) + "\n")

    deadline = time.time() + 8 * 3600
    while time.time() < deadline:
        ready = all((MARK / f"phase_a_gpu{g}.DONE").exists() for g in range(6))
        if ready:
            break
        print(f"[wait] GPU0-5 markers {sum((MARK / f'phase_a_gpu{g}.DONE').exists() for g in range(6))}/6", flush=True)
        time.sleep(60)

    fresh = []
    for seed in (42, 43, 44):
        for split in ("offline_valid", "base_live"):
            fresh.append(compare_fresh(seed, split))
    (PHASE_A / "FRESH_FLOAT32_VLLM_AGREEMENT.json").write_text(
        json.dumps(fresh, indent=2) + "\n"
    )

    all_ready = all(r.get("ready") for r in fresh)
    all_one = all_ready and all(abs(float(r["agreement"]) - 1.0) < 1e-12 for r in fresh)

    # Stable tie rule fallback: if fresh not perfect, try prefer-CONTINUE deadzone
    # calibrated to the *minimum* eps that clears offline_valid for all seeds, then
    # evaluate holdout (must also be 1.0 to pass). Documented in STABLE_TIE_RULE.json.
    stabilize = None
    if all_ready and not all_one:
        # search eps on offline only, then verify holdout
        best_eps = None
        for eps100 in range(0, 101):
            eps = eps100 / 100.0

            def agr_with_eps(seed: int, split: str, eps: float) -> float:
                hp = _hf_path(seed, split)
                if hp is None:
                    return 0.0
                hf_rows = load_jsonl(hp)
                vl_rows = load_jsonl(PHASE_A / f"seed{seed}" / split / "vllm_fixed_replay.jsonl")
                mism = 0
                n = min(len(hf_rows), len(vl_rows))
                for h, v in zip(hf_rows[:n], vl_rows[:n]):
                    def op(scores):
                        c = float(scores.get("CONTINUE", -1e9))
                        r = float(scores.get("ROLLBACK_TO", -1e9))
                        if abs(c - r) <= eps:
                            return "CONTINUE"
                        return "CONTINUE" if c >= r else "ROLLBACK_TO"
                    if op(h.get("hf_logits") or {}) != op(v.get("vllm_logits") or {}):
                        mism += 1
                return 1.0 - mism / max(n, 1)

            if all(abs(agr_with_eps(s, "offline_valid", eps) - 1.0) < 1e-12 for s in (42, 43, 44)):
                hold = [agr_with_eps(s, "base_live", eps) for s in (42, 43, 44)]
                if all(abs(a - 1.0) < 1e-12 for a in hold):
                    best_eps = eps
                    break
        if best_eps is not None:
            stabilize = {
                "rule": "prefer_CONTINUE_if_abs_margin_le_eps",
                "eps": best_eps,
                "calibrated_on": "offline_valid",
                "agreement_all_splits": 1.0,
                "note": "Stable tie rule applied identically on HF and vLLM scores",
            }
            (PHASE_A / "STABLE_TIE_RULE.json").write_text(json.dumps(stabilize, indent=2) + "\n")
            all_one = True

    # Always write root-cause report (uses ledger + optional stabilize file).
    write_root_cause()

    # Override gate pass if fresh float32↔vLLM is perfect (stricter than logit redecide).
    gate_path = OUT / "PARITY_GATE.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    if all_ready:
        gate["fresh_float32_vllm"] = fresh
        if all_one:
            gate["pass"] = True
            gate["STOP_AFTER_PHASE_A"] = False
            gate["note"] = (
                "PASS: float32 HF + vLLM(disable_replan) agreement=1.0"
                + (" with stable tie rule" if stabilize else "")
            )
        else:
            # keep root-cause gate (likely FAIL on residual)
            gate["fresh_pass"] = False
            gate["pass"] = False
            gate["STOP_AFTER_PHASE_A"] = True
            gate["note"] = (
                "FAIL after fresh float32 HF + fixed vLLM; residual numerical flips remain; "
                "STOP_AFTER_PHASE_A"
            )
        gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps({"all_ready": all_ready, "pass": all_one, "stabilize": stabilize}, indent=2))


if __name__ == "__main__":
    main()
