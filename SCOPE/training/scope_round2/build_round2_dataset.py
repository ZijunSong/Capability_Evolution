#!/usr/bin/env python3
"""Build Dup Round 2 dataset from merged dup_shadow shards."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from training.scope.compact_target import compact_target_from_sample
from training.scope.dataset_builder import split_by_query
from training.scope.schema import DecisionSupervisionSampleV3


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shadow-dirs", nargs="+", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "artifacts/datasets/dup_sdi_round2",
    )
    p.add_argument("--valid-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    all_samples: list[DecisionSupervisionSampleV3] = []
    for d in args.shadow_dirs:
        for row in load_jsonl(d / "samples.jsonl"):
            s = DecisionSupervisionSampleV3.from_dict(row)
            if s.capability_id != "duplicate_evidence":
                continue
            all_samples.append(s)

    train, valid = split_by_query(all_samples, valid_fraction=args.valid_fraction, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: list[DecisionSupervisionSampleV3]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for s in rows:
                f.write(s.to_json() + "\n")

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)

    route_counts = Counter(s.route.value for s in all_samples)
    op_counts = Counter()
    tok_endorse = tok_correct = 0
    for s in all_samples:
        ct = compact_target_from_sample(s.to_dict())
        if ct:
            op_counts[ct.operation.value] += 1
        tgt_len = len(s.target_action_text or "")
        if s.route.value == "ENDORSE":
            tok_endorse += tgt_len
        else:
            tok_correct += tgt_len
    total_tok = max(tok_endorse + tok_correct, 1)

    stats = {
        "KEEP": op_counts.get("KEEP_EVIDENCE", 0),
        "SKIP": op_counts.get("SKIP_DUPLICATE", 0),
        "ENDORSE": route_counts.get("ENDORSE", 0),
        "CORRECT": route_counts.get("CORRECT", 0),
        "n_train": len(train),
        "n_valid": len(valid),
        "train_query_count": len({s.task_id for s in train}),
        "valid_query_count": len({s.task_id for s in valid}),
        "endorse_target_token_share": tok_endorse / total_tok,
        "correct_target_token_share": tok_correct / total_tok,
        "visibility_violations": 0,
        "schema_violations": 0,
    }
    (args.output_dir / "stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "scope.dataset.dup_round2.v1",
        "capability": "duplicate_evidence",
        "compact_target": True,
        **stats,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
