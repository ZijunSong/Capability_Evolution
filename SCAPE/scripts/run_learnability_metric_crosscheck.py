#!/usr/bin/env python3
"""Cross-check trainer metric vs evaluator canonical metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.same_state import load_same_state_jsonl
from scape.training.hf_tool_opd import ScapeHFToolOPD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model-path", default="/data/ppnm/models/harness-1")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_same_state_jsonl(Path(args.jsonl))[:args.n]
    backend = ScapeHFToolOPD(
        model_path=args.model_path,
        device_map=f"cuda:{args.gpu}",
        use_lora=False,
    )

    diffs = []
    for row in rows:
        legacy = backend.score_divergence(
            prompt_reduced=row["prompt_reduced"],
            prompt_full=row["prompt_full"],
            response_text=row["response_text"],
            loss_path="tool_token_kl",
        )
        canon = backend.score_canonical_metrics(
            prompt_reduced=row["prompt_reduced"],
            prompt_full=row["prompt_full"],
            response_text=row["response_text"],
            loss_path="tool_token_kl",
        )
        diffs.append({
            "legacy_div": legacy["div"],
            "signed_gap": canon["signed_gap"],
            "forward_KL": canon["forward_KL"],
            "gap_match": abs(legacy["div"] - canon["signed_gap"]) < 1e-4,
            "forward_nonneg": canon["forward_KL"] >= -1e-7,
        })

    n_match = sum(1 for d in diffs if d["gap_match"])
    n_nonneg = sum(1 for d in diffs if d["forward_nonneg"])
    report = {
        "n_rows": len(rows),
        "trainer_evaluator_gap_match_rate": n_match / max(1, len(rows)),
        "forward_kl_nonneg_rate": n_nonneg / max(1, len(rows)),
        "mean_legacy_div": sum(d["legacy_div"] for d in diffs) / max(1, len(diffs)),
        "mean_signed_gap": sum(d["signed_gap"] for d in diffs) / max(1, len(diffs)),
        "mean_forward_KL": sum(d["forward_KL"] for d in diffs) / max(1, len(diffs)),
        "details_head": diffs[:16],
        "pass": n_match == len(rows) and n_nonneg == len(rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
