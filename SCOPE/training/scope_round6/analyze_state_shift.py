#!/usr/bin/env python3
"""Turn-wise state shift analysis (Round 6)."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round6.common import OUT, write_json
from training.scope_round6.metrics import aggregate_scored_rows, ScoredRow
from training.scope_round6.scorer_utils import load_merged_model, score_sample_hf, scorer_paths_for_tag
from training.scope_round6.state_sources import load_b6_admission_states


TURN_BUCKETS = [
    ("turn_0_4", 0, 4),
    ("turn_5_9", 5, 9),
    ("turn_10_19", 10, 19),
    ("turn_20_29", 20, 29),
    ("turn_30_plus", 30, 9999),
]


def _bucket(turn: int) -> str:
    for name, lo, hi in TURN_BUCKETS:
        if lo <= turn <= hi:
            return name
    return "turn_30_plus"


def _ks_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    na, nb = len(a_sorted), len(b_sorted)
    i, j = 0, 0
    d = 0.0
    while i < na and j < nb:
        if a_sorted[i] <= b_sorted[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / na - j / nb))
    return d


def _feature_row(sample: dict[str, Any]) -> dict[str, float]:
    ds = sample.get("decision_state") or {}
    pool = ds.get("pool_document_ids") or []
    curated = ds.get("curated_document_ids") or []
    turn = int(sample.get("turn_id", ds.get("turn_id", 0)))
    rendered = str(ds.get("rendered_context") or "")
    return {
        "turn": float(turn),
        "rendered_token_length": float(len(rendered.split())),
        "n_pool": float(len(pool)),
        "n_curated": float(len(curated)),
        "remaining_turns": float(max(35 - turn, 0)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorer", default="o7_42")
    p.add_argument("--state-source", default="o7_42")
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=OUT / "phase_b")
    args = p.parse_args()

    samples = load_b6_admission_states(args.state_source)
    paths = scorer_paths_for_tag(args.scorer)
    model, tokenizer, dev = load_merged_model(paths["merged"], args.gpu)

    by_bucket: dict[str, list[ScoredRow]] = defaultdict(list)
    by_bucket_samples: dict[str, list[dict]] = defaultdict(list)
    scored_pairs: list[tuple[dict, ScoredRow]] = []

    for s in samples:
        row = score_sample_hf(model, tokenizer, s, dev)
        scored_pairs.append((s, row))
        b = _bucket(int(s.get("turn_id", 0)))
        by_bucket[b].append(row)
        by_bucket_samples[b].append(s)

    turn_metrics = {}
    for name, rows in by_bucket.items():
        turn_metrics[name] = aggregate_scored_rows(rows)

    # Compare early vs late feature drift
    early_feats = [_feature_row(s) for s, _ in scored_pairs if int(s.get("turn_id", 0)) <= 9]
    late_feats = [_feature_row(s) for s, _ in scored_pairs if int(s.get("turn_id", 0)) >= 20]
    drift: dict[str, float] = {}
    if early_feats and late_feats:
        for key in early_feats[0]:
            early_vals = [f[key] for f in early_feats]
            late_vals = [f[key] for f in late_feats]
            drift[f"KS_{key}"] = _ks_distance(early_vals, late_vals)

    report = {
        "scorer": args.scorer,
        "state_source": args.state_source,
        "turn_wise": turn_metrics,
        "feature_drift_early_vs_late": drift,
        "early_turn_better": (
            turn_metrics.get("turn_0_4", {}).get("BalancedAcc", 0)
            > turn_metrics.get("turn_30_plus", {}).get("BalancedAcc", 0)
        ),
    }
    write_json(args.output_dir / f"STATE_SHIFT_{args.scorer}_{args.state_source}.json", report)
    md = args.output_dir / "STATE_SHIFT_REPORT.md"
    lines = ["# State Shift Report", "", f"scorer={args.scorer} source={args.state_source}", ""]
    for name, m in turn_metrics.items():
        lines.append(
            f"- {name}: AUROC={m.get('AUROC', 0):.3f} DupRejectRecall={m.get('DupRejectRecall', 0):.3f} "
            f"FSR={m.get('FalseSkipRate', 0):.3f}"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
