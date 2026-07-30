#!/usr/bin/env python3
"""Build cross-score matrix: scorer × state source (Round 6 core experiment)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope_round6.common import OUT, SCORER_TAGS, STATE_SOURCES, write_json
from training.scope_round6.metrics import aggregate_scored_rows
from training.scope_round6.scorer_utils import load_merged_model, score_samples_hf, scorer_paths_for_tag
from training.scope_round6.state_sources import load_state_source


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=OUT / "phase_b")
    p.add_argument("--scorer", choices=SCORER_TAGS, default=None)
    p.add_argument("--state-source", choices=STATE_SOURCES, default=None)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    scorers = [args.scorer] if args.scorer else list(SCORER_TAGS)
    sources = [args.state_source] if args.state_source else list(STATE_SOURCES)

    matrix: dict[str, dict[str, dict]] = {}
    csv_rows: list[dict] = []

    for tag in scorers:
        paths = scorer_paths_for_tag(tag)
        model, tokenizer, dev = load_merged_model(paths["merged"], args.gpu)
        matrix[tag] = {}
        for src in sources:
            samples = load_state_source(src)
            if args.max_samples > 0:
                samples = samples[:args.max_samples]
            scored = score_samples_hf(model, tokenizer, samples, dev)
            metrics = aggregate_scored_rows(scored, threshold=args.threshold)
            matrix[tag][src] = metrics
            csv_rows.append({
                "scorer": tag,
                "state_source": src,
                "n": metrics.get("n", 0),
                "AUROC": metrics.get("AUROC"),
                "AUPRC": metrics.get("AUPRC"),
                "DupRejectRecall": metrics.get("DupRejectRecall"),
                "UniqueKeepRecall": metrics.get("UniqueKeepRecall"),
                "FalseSkipRate": metrics.get("FalseSkipRate"),
                "BalancedAcc": metrics.get("BalancedAcc"),
                "predicted_SKIP_prior": metrics.get("predicted_SKIP_prior"),
            })
            print(f"{tag} x {src}: AUROC={metrics['AUROC']:.4f} BalAcc={metrics['BalancedAcc']:.4f}")
        del model

    write_json(out / "cross_score_matrix.json", matrix)
    csv_path = out / "CROSS_SCORE_MATRIX.csv"
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
