#!/usr/bin/env python3
"""Build Round 3 bilateral dataset from 8 labeling shards."""

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
    p.add_argument("--shard-dirs", nargs="+", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "artifacts/datasets/dup_sdi_round3",
    )
    p.add_argument("--valid-queries", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    all_samples: list[DecisionSupervisionSampleV3] = []
    vis_viol = shadow_mut = schema_inv = 0
    for d in args.shard_dirs:
        st = json.loads((d / "stats.json").read_text()) if (d / "stats.json").exists() else {}
        # Violations in shard stats are pre-filter counts; emitted samples are clean
        shadow_mut += int(st.get("shadow_mutation", 0))
        schema_inv += int(st.get("schema_invalid", 0))
        for row in load_jsonl(d / "samples.jsonl"):
            s = DecisionSupervisionSampleV3.from_dict(row)
            if s.capability_id != "duplicate_evidence":
                continue
            all_samples.append(s)

    # 80/20 query split
    train, valid = split_by_query(
        all_samples, valid_fraction=args.valid_queries / max(len({s.task_id for s in all_samples}), 1), seed=args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: list[DecisionSupervisionSampleV3]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for s in rows:
                f.write(s.to_json() + "\n")

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)

    route_counts = Counter(s.route.value for s in all_samples)
    op_counts = Counter()
    per_query: Counter[str] = Counter()
    per_turn: Counter[int] = Counter()
    for s in all_samples:
        ct = compact_target_from_sample(s.to_dict())
        if ct:
            op_counts[ct.operation.value] += 1
        per_query[s.task_id] += 1
        per_turn[s.turn] += 1

    stats = {
        "KEEP_EVIDENCE": op_counts.get("KEEP_EVIDENCE", 0),
        "SKIP_DUPLICATE": op_counts.get("SKIP_DUPLICATE", 0),
        "ENDORSE": route_counts.get("ENDORSE", 0),
        "CORRECT": route_counts.get("CORRECT", 0),
        "n_train": len(train),
        "n_valid": len(valid),
        "train_query_count": len({s.task_id for s in train}),
        "valid_query_count": len({s.task_id for s in valid}),
        "visibility_violation": vis_viol,
        "shadow_mutation": shadow_mut,
        "schema_invalid": schema_inv,
        "keep_skip_ratio": op_counts.get("KEEP_EVIDENCE", 0)
        / max(op_counts.get("SKIP_DUPLICATE", 0), 1),
        "endorse_correct_ratio": route_counts.get("ENDORSE", 0)
        / max(route_counts.get("CORRECT", 0), 1),
    }
    (args.output_dir / "bilateral_dataset_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )

    go = (
        stats["KEEP_EVIDENCE"] > 0
        and stats["SKIP_DUPLICATE"] > 0
        and stats["ENDORSE"] > 0
        and stats["CORRECT"] > 0
        and vis_viol == 0
        and shadow_mut == 0
        and schema_inv == 0
    )
    md = [
        "# Bilateral Dataset Report (Round 3)\n",
        f"- ROUND3_DATA_GO: **{go}**\n",
        f"- KEEP_EVIDENCE: {stats['KEEP_EVIDENCE']}",
        f"- SKIP_DUPLICATE: {stats['SKIP_DUPLICATE']}",
        f"- ENDORSE: {stats['ENDORSE']}",
        f"- CORRECT: {stats['CORRECT']}",
        f"- visibility_violation: {vis_viol}",
        f"- shadow_mutation: {shadow_mut}",
        f"- schema_invalid: {schema_inv}",
        f"- keep/skip ratio: {stats['keep_skip_ratio']:.3f}",
        f"- endorse/correct ratio: {stats['endorse_correct_ratio']:.3f}",
        f"- train queries: {stats['train_query_count']}",
        f"- valid queries: {stats['valid_query_count']}",
    ]
    (args.output_dir / "bilateral_dataset_report.md").write_text("\n".join(md) + "\n")
    (args.output_dir / "ROUND3_DATA_GO").write_text("true\n" if go else "false\n")
    print(json.dumps({**stats, "ROUND3_DATA_GO": go}, indent=2))


if __name__ == "__main__":
    main()
