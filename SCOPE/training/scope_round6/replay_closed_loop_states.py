#!/usr/bin/env python3
"""Replay closed-loop states with shadow labeling (Round 6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from harness.capability.dup_operation import DupOperation
from training.scope_round6.common import OUT, write_json
from training.scope_round6.metrics import aggregate_scored_rows
from training.scope_round6.scorer_utils import load_merged_model, score_samples_hf, scorer_paths_for_tag
from training.scope_round6.state_sources import load_b6_admission_states


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--checkpoint", type=str, required=True, help="scorer tag e.g. o7_42")
    p.add_argument("--state-source", type=str, required=True, help="base|o7_42|o7_43|o7_44")
    p.add_argument("--output-dir", type=Path, default=OUT / "phase_b/replay")
    args = p.parse_args()

    paths = scorer_paths_for_tag(args.checkpoint)
    samples = load_b6_admission_states(args.state_source)
    model, tokenizer, dev = load_merged_model(paths["merged"], args.gpu)
    scored = score_samples_hf(model, tokenizer, samples, dev)
    metrics = aggregate_scored_rows(scored)

    out = args.output_dir / f"{args.checkpoint}_on_{args.state_source}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", metrics)
    write_json(out / "margin_report.json", {
        "margin_duplicate": metrics.get("margin_duplicate"),
        "margin_unique": metrics.get("margin_unique"),
        "label_counts": {
            "duplicate": metrics.get("n_duplicate"),
            "unique": metrics.get("n_unique"),
        },
    })
    print(f"Replay {args.checkpoint} on {args.state_source}: n={metrics['n']} AUROC={metrics['AUROC']:.4f}")


if __name__ == "__main__":
    main()
