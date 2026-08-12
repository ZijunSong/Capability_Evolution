#!/usr/bin/env python3
"""Build event-level HF↔vLLM parity ledger from P0 replays (with disable_replan redecide)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.decide_rollback_operation import decide_rollback_operation

P0 = _REPO / "outputs/scope_round9/wave_b_p0"
OUT = _REPO / "outputs/scope_round10/phase_a"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decide(scores: dict, *, disable_replan: bool = True):
    d = decide_rollback_operation(
        score_continue=float(scores.get("CONTINUE", -1e9)),
        score_replan=float(scores.get("REPLAN", -1e9)),
        score_rollback=float(scores.get("ROLLBACK_TO", -1e9)),
        threshold=0.0,
        candidate_checkpoint_id=None,
        disable_replan=disable_replan,
    )
    return d


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_split(seed: int, split: str) -> dict:
    variant = f"rollback_hier_o7_seed{seed}"
    eval_name = "eval_offline_valid" if split == "offline_valid" else "eval_holdout"
    hf_rows = load_jsonl(P0 / variant / eval_name / "hf_replay.jsonl")
    vllm_rows = load_jsonl(P0 / variant / eval_name / "vllm_replay.jsonl")
    assert len(hf_rows) == len(vllm_rows), (seed, split, len(hf_rows), len(vllm_rows))

    out_dir = OUT / f"seed{seed}" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "all_events.jsonl"
    mism_path = out_dir / "mismatch_events.jsonl"
    summary_path = out_dir / "mismatch_summary.json"

    n = 0
    raw_mism = 0
    fixed_mism = 0
    replan_only_mism = 0
    near_boundary = 0
    events = []
    mismatches = []

    for hf, vl in zip(hf_rows, vllm_rows):
        n += 1
        hf_scores = hf.get("hf_logits") or {}
        vl_scores = vl.get("vllm_logits") or {}
        hf_raw = hf.get("pred_operation")
        vl_raw = vl.get("pred_operation")
        hf_d = _decide(hf_scores, disable_replan=True)
        vl_d = _decide(vl_scores, disable_replan=True)
        hf_op = hf_d.predicted_operation.value
        vl_op = vl_d.predicted_operation.value
        prompt = hf.get("effective_input_text") or ""
        prompt_sha = hf.get("prompt_sha256") or _sha(prompt)
        token_sha = hf.get("token_ids_sha256") or ""
        cand = hf.get("candidate_list") or []
        cand_ids = [c.get("local_checkpoint_id") or c.get("checkpoint_id") for c in cand]
        cand_order_sha = hf.get("candidate_list_sha256") or _sha("|".join(map(str, cand_ids)))
        state_sha = hf.get("serialized_state_sha256") or hf.get("state_sha256") or ""

        raw_agree = hf_raw == vl_raw
        fixed_agree = hf_op == vl_op
        if not raw_agree:
            raw_mism += 1
            if vl_raw == "REPLAN" and hf_raw != "REPLAN":
                replan_only_mism += 1
        if not fixed_agree:
            fixed_mism += 1
            if abs(hf_d.score_continue - hf_d.score_rollback) < 0.25 or abs(
                vl_d.score_continue - vl_d.score_rollback
            ) < 0.25:
                near_boundary += 1

        event = {
            "event_id": hf.get("event_id"),
            "query_id": hf.get("query_id"),
            "gold_operation": hf.get("gold_operation"),
            "serialized_state_sha256": state_sha,
            "rendered_prompt_sha256": prompt_sha,
            "rendered_prompt_text": prompt,
            "token_ids_sha256": token_sha,
            "n_input_tokens": hf.get("token_length_after") or hf.get("n_input_tokens"),
            "truncation_applied": bool(hf.get("truncated")),
            "truncation_start_end": hf.get("truncation_start_end"),
            "candidate_checkpoint_ids": cand_ids,
            "candidate_order_sha256": cand_order_sha,
            "verbalizer_tokens": ["CONTINUE", "REPLAN", "ROLLBACK_TO"],
            "hf_score_continue": hf_d.score_continue,
            "hf_score_rollback": hf_d.score_rollback,
            "hf_score_replan": hf_d.score_replan,
            "hf_margin": hf_d.margin,
            "hf_operation_raw": hf_raw,
            "hf_operation": hf_op,
            "vllm_score_continue": vl_d.score_continue,
            "vllm_score_rollback": vl_d.score_rollback,
            "vllm_score_replan": vl_d.score_replan,
            "vllm_margin": vl_d.margin,
            "vllm_operation_raw": vl_raw,
            "vllm_operation": vl_op,
            "operation_agreement_raw": raw_agree,
            "operation_agreement": fixed_agree,
            "fallback_used": bool(hf.get("fallback_reason") or vl.get("fallback_reason")),
            "hf_fallback_reason": hf.get("fallback_reason"),
            "vllm_fallback_reason": vl.get("fallback_reason"),
        }
        events.append(event)
        if not fixed_agree or not raw_agree:
            mismatches.append(event)

    with all_path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with mism_path.open("w", encoding="utf-8") as f:
        for e in mismatches:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    summary = {
        "seed": seed,
        "split": split,
        "n": n,
        "raw_agreement": 1.0 - raw_mism / max(n, 1),
        "raw_mismatch": raw_mism,
        "replan_path_mismatch": replan_only_mism,
        "disable_replan_agreement": 1.0 - fixed_mism / max(n, 1),
        "disable_replan_mismatch": fixed_mism,
        "near_boundary_residual": near_boundary,
        "prompt_hash_mismatch": sum(
            1
            for h, v in zip(hf_rows, vllm_rows)
            if (h.get("prompt_sha256") or "") != (v.get("prompt_sha256") or "")
        ),
        "token_hash_mismatch": sum(
            1
            for h, v in zip(hf_rows, vllm_rows)
            if (h.get("token_ids_sha256") or "") != (v.get("token_ids_sha256") or "")
        ),
        "candidate_hash_mismatch": sum(
            1
            for h, v in zip(hf_rows, vllm_rows)
            if (h.get("candidate_list_sha256") or "") != (v.get("candidate_list_sha256") or "")
        ),
        "fallback_count": sum(1 for e in events if e["fallback_used"]),
        "all_events": str(all_path),
        "mismatch_events": str(mism_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, nargs="+", default=[42, 43, 44])
    args = p.parse_args()
    summaries = []
    for seed in args.seed:
        for split in ("offline_valid", "base_live"):
            summaries.append(build_split(seed, split))
    out = OUT / "LEDGER_INDEX.json"
    out.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
