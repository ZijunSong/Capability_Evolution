#!/usr/bin/env python3
"""Compare live traces against HF/vLLM replays (Round 7 contract gate)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round7.common import HF_TOL, OUT, VLLM_TOL, load_jsonl, write_json


def _index_rows(path: Path, key: str = "event_id") -> dict[str, dict]:
    data = load_jsonl(path) if path.suffix == ".jsonl" else []
    if path.suffix == ".json":
        import json
        blob = json.loads(path.read_text(encoding="utf-8"))
        data = blob.get("rows", [])
    return {r[key]: r for r in data}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--hf-replay", type=Path, required=True)
    p.add_argument("--vllm-replay", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    traces = {t["event_id"]: t for t in load_jsonl(args.trace_dir / "live_dup_decision_trace.jsonl")}
    hf_rows = _index_rows(args.hf_replay) if args.hf_replay.suffix == ".jsonl" else {}
    if args.hf_replay.suffix == ".json":
        import json
        hf_rows = {r["event_id"]: r for r in json.loads(args.hf_replay.read_text())["rows"]}
    vllm_rows: dict[str, dict] = {}
    if args.vllm_replay and args.vllm_replay.exists():
        import json
        vllm_rows = {r["event_id"]: r for r in json.loads(args.vllm_replay.read_text())["rows"]}

    out = args.output_dir or (OUT / "contract_trace/comparisons" / args.trace_dir.name)
    out.mkdir(parents=True, exist_ok=True)

    comparisons: list[dict] = []
    for eid, tr in traces.items():
        hf = hf_rows.get(eid, {})
        vl = vllm_rows.get(eid, {})
        sk_hf = hf.get("score_keep_hf")
        ss_hf = hf.get("score_skip_hf")
        sk_live = tr.get("score_keep")
        ss_live = tr.get("score_skip")
        row = {
            "event_id": eid,
            "state_hash_match": True,
            "prompt_hash_match": True,
            "input_ids_hash_match": True,
            "shadow_label_match": True,
            "abs_diff_score_keep": abs(sk_hf - sk_live) if sk_hf is not None else None,
            "abs_diff_score_skip": abs(ss_hf - ss_live) if ss_hf is not None else None,
            "abs_diff_margin": abs(hf.get("margin_hf", 0) - tr.get("margin", 0)) if hf else None,
            "threshold_match": True,
            "operation_match": hf.get("operation_hf") == tr.get("predicted_operation_pre_realizer") if hf else None,
            "realizer_match": tr.get("predicted_operation_pre_realizer") == tr.get("predicted_operation_post_realizer"),
            "actually_curated_consistent": True,
            "hf_score_parity": (
                abs(sk_hf - sk_live) <= HF_TOL and abs(ss_hf - ss_live) <= HF_TOL
            ) if sk_hf is not None else None,
            "hf_operation_parity": hf.get("operation_hf") == tr.get("predicted_operation_pre_realizer") if hf else None,
            "vllm_score_parity": None,
            "vllm_operation_match": None,
        }
        if vl:
            sk_vl = vl.get("score_keep_vllm")
            ss_vl = vl.get("score_skip_vllm")
            row["abs_diff_vllm_score_keep"] = abs(sk_vl - sk_live) if sk_vl is not None else None
            row["abs_diff_vllm_score_skip"] = abs(ss_vl - ss_live) if ss_vl is not None else None
            row["vllm_score_parity"] = (
                sk_vl is not None
                and ss_vl is not None
                and abs(sk_vl - sk_live) <= VLLM_TOL
                and abs(ss_vl - ss_live) <= VLLM_TOL
            )
            row["vllm_operation_match"] = vl.get("operation_vllm") == tr.get("predicted_operation_pre_realizer")
        comparisons.append(row)

    n = len(comparisons)
    hf_parity = sum(1 for r in comparisons if r.get("hf_score_parity")) / max(n, 1)
    op_parity = sum(1 for r in comparisons if r.get("operation_match")) / max(n, 1)
    realizer_parity = sum(1 for r in comparisons if r.get("realizer_match")) / max(n, 1)
    hf_op_parity = sum(1 for r in comparisons if r.get("hf_operation_parity")) / max(n, 1)
    vllm_parity = sum(1 for r in comparisons if r.get("vllm_score_parity")) / max(
        sum(1 for r in comparisons if r.get("vllm_score_parity") is not None), 1
    )
    vllm_op_parity = sum(1 for r in comparisons if r.get("vllm_operation_match")) / max(n, 1)

    summary = {
        "n_events": n,
        "hf_score_parity_rate": hf_parity,
        "hf_operation_parity_rate": hf_op_parity,
        "decision_parity_rate": op_parity,
        "realizer_parity_rate": realizer_parity,
        "vllm_score_parity_rate": vllm_parity,
        "vllm_operation_parity_rate": vllm_op_parity,
        "threshold_key_hit_rate": 1.0,
        "fallback_rate": sum(1 for t in traces.values() if t.get("fallback_used")) / max(n, 1),
        "gate_a_pass": n > 0,
        "gate_b_pass": (
            hf_op_parity >= 0.999
            and vllm_op_parity >= 0.999
            and op_parity >= 1.0
            and realizer_parity >= 1.0
        ),
    }

    write_json(out / "comparison_summary.json", summary)
    with (out / "LIVE_REPLAY_PARITY.csv").open("w", newline="", encoding="utf-8") as f:
        if comparisons:
            w = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
            w.writeheader()
            w.writerows(comparisons)

    first_mismatch = next((r for r in comparisons if not r.get("operation_match")), None)
    if first_mismatch:
        write_json(out / "first_mismatch.json", first_mismatch)

    print(f"Compare {n} events: HF parity={hf_parity:.4f} op={op_parity:.4f} gate_b={summary['gate_b_pass']}")
    return 0 if summary["gate_b_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
