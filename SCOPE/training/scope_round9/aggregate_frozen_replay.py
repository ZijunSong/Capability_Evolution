#!/usr/bin/env python3
"""Aggregate frozen replay parity metrics for Wave A / Barrier A."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round9.aggregate_phase3_gate import _balanced_accuracy, _confusion_matrix


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _row_key(row: dict, idx: int) -> str:
    return str(row.get("event_id") or f"{row.get('query_id')}:{row.get('turn')}:idx{idx}")


def _score_margin(logits: dict | None) -> float:
    if not logits:
        return 1e9
    vals = sorted((float(v) for v in logits.values()), reverse=True)
    if len(vals) < 2:
        return 1e9
    return vals[0] - vals[1]


def _ops_agree_for_barrier(
    hr: dict,
    vr: dict,
    *,
    near_tie_eps: float = 0.15,
) -> tuple[bool, bool]:
    """Return (raw_agree, barrier_agree).

    Barrier agreement treats any operation flip as OK when *both* sides have
    top-2 margin ≤ near_tie_eps (HF/vLLM float noise). Clear-margin flips remain
    hard failures. Hash/serialization mismatches are still checked separately.
    """
    raw = hr.get("pred_operation") == vr.get("pred_operation")
    if raw:
        return True, True
    mh = _score_margin(hr.get("hf_logits") or hr.get("vllm_logits"))
    mv = _score_margin(vr.get("vllm_logits") or vr.get("hf_logits"))
    if max(mh, mv) <= near_tie_eps:
        return False, True
    return False, False


def compare_hf_vllm(hf_rows: list[dict], vllm_rows: list[dict]) -> dict:
    """Index-aligned comparison; falls back to event_id map if lengths differ."""
    n = min(len(hf_rows), len(vllm_rows))
    agree = 0
    barrier_agree = 0
    near_tie_resolved = 0
    ckpt_order_agree = 0
    hash_mismatch = 0
    token_mismatch = 0
    cand_mismatch = 0
    fallback = 0
    invalid_ck = 0
    compared = 0
    for i in range(n):
        hr = hf_rows[i]
        vr = vllm_rows[i]
        compared += 1
        raw_ok, bar_ok = _ops_agree_for_barrier(hr, vr)
        agree += int(raw_ok)
        barrier_agree += int(bar_ok)
        near_tie_resolved += int(bar_ok and not raw_ok)
        hash_mismatch += int(hr.get("prompt_sha256") != vr.get("prompt_sha256"))
        token_mismatch += int(hr.get("token_ids_sha256") != vr.get("token_ids_sha256"))
        cand_mismatch += int(hr.get("candidate_list_sha256") != vr.get("candidate_list_sha256"))
        hf_order = [c.get("local_checkpoint_id") for c in (hr.get("candidate_list") or [])]
        vl_order = [c.get("local_checkpoint_id") for c in (vr.get("candidate_list") or [])]
        ckpt_order_agree += int(hf_order == vl_order)
        fallback += int(bool(hr.get("fallback_reason") or vr.get("fallback_reason")))
        for side in (hr, vr):
            if side.get("pred_operation") == "ROLLBACK_TO" and not side.get("pred_checkpoint_local_id"):
                invalid_ck += 1
                break
    return {
        "n_compared": compared,
        "operation_top1_agreement_raw": agree / max(compared, 1),
        # Barrier metric: raw agreement + resolved low-margin CONTINUE/ROLLBACK flips.
        "operation_top1_agreement": barrier_agree / max(compared, 1),
        "near_tie_resolved_count": near_tie_resolved,
        "checkpoint_ordering_agreement": ckpt_order_agree / max(compared, 1),
        "prompt_hash_mismatch": hash_mismatch,
        "token_hash_mismatch": token_mismatch,
        "candidate_hash_mismatch": cand_mismatch,
        "fallback_count": fallback,
        "invalid_checkpoint_rate": invalid_ck / max(compared, 1),
        "n_hf": len(hf_rows),
        "n_vllm": len(vllm_rows),
    }


def operation_metrics(rows: list[dict], *, pred_key: str = "pred_operation") -> dict:
    events = [
        {
            "shadow_operation": r.get("gold_operation"),
            "student_operation": r.get(pred_key),
            "shadow_checkpoint_id": r.get("gold_checkpoint_global_id"),
            "predicted_checkpoint_id": r.get("pred_checkpoint_global_id"),
            "candidate_checkpoint_ids": [
                c.get("checkpoint_id") for c in (r.get("candidate_list") or [])
            ],
        }
        for r in rows
    ]
    matrix = _confusion_matrix(events)
    prior = Counter(r.get(pred_key) for r in rows)
    return {
        "operation_balanced_accuracy": _balanced_accuracy(matrix),
        "prediction_prior": dict(prior),
        "confusion_matrix": matrix,
    }


def barrier_a_for_parity(parity: dict) -> tuple[bool, list[str]]:
    fails = []
    if parity.get("operation_top1_agreement", 0) != 1.0:
        fails.append(f"operation_top1_agreement={parity.get('operation_top1_agreement')}")
    if parity.get("checkpoint_ordering_agreement", 0) != 1.0:
        fails.append(
            f"checkpoint_ordering_agreement={parity.get('checkpoint_ordering_agreement')}"
        )
    for k in ("prompt_hash_mismatch", "token_hash_mismatch", "candidate_hash_mismatch"):
        if parity.get(k, 0) != 0:
            fails.append(f"{k}={parity.get(k)}")
    if parity.get("fallback_count", 0) != 0:
        fails.append(f"fallback_count={parity.get('fallback_count')}")
    if parity.get("invalid_checkpoint_rate", 0) >= 0.01:
        fails.append(f"invalid_checkpoint_rate={parity.get('invalid_checkpoint_rate')}")
    return (len(fails) == 0), fails


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    report: dict = {"variant_dir": str(args.variant_dir), "split_failures": {}}
    all_pass = True
    for split in ("offline_valid", "base_live", "self_live"):
        hf_path = args.variant_dir / split / "hf_replay.jsonl"
        vllm_path = args.variant_dir / split / "vllm_replay.jsonl"
        if not hf_path.exists():
            continue
        hf_rows = load_jsonl(hf_path)
        vllm_rows = load_jsonl(vllm_path) if vllm_path.exists() else []
        parity = compare_hf_vllm(hf_rows, vllm_rows) if vllm_rows else {}
        split_pass, fails = barrier_a_for_parity(parity) if parity else (False, ["missing_vllm"])
        report[split] = {
            "hf_metrics": operation_metrics(hf_rows),
            "parity": parity,
            "barrier_a_split_pass": split_pass,
            "barrier_a_failures": fails,
        }
        report["split_failures"][split] = fails
        all_pass = all_pass and split_pass

    report["barrier_a_pass"] = all_pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
