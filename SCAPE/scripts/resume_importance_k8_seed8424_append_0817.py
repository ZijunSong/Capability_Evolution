#!/usr/bin/env python3
"""Append-resume the interrupted 0816-2 importance K8 seed8424 formal stream shard.

The original stream runner opens its shard with mode "w", so rerunning it would
truncate completed rows. This script preserves the existing shard, rebuilds the
same frozen query/candidate sequence, skips already-written candidate indices,
and appends only missing rows until the shard reaches the requested count.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.run_h100_2_live_fork_replay as base
import scripts.run_h100_2_live_fork_replay_stream as stream
from pyserini.search.lucene import LuceneSearcher
from scape.common.status import write_status_live
from scape.rendering.dual_view import DualViewRenderer


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    bcp = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "0816_2_importance_proper_fork_formal_stream" / "K8_seed8424")
    ap.add_argument("--component", default="importance_tagging")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=8424)
    ap.add_argument("--n-states", type=int, default=512)
    ap.add_argument("--n-queries-pool", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default="/mnt/songzijun/models/harness-1")
    ap.add_argument("--browsecomp-root", type=Path, default=bcp)
    ap.add_argument("--index-path", type=Path, default=bcp / "indexes" / "bm25")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = ap.parse_args()

    os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-21-openjdk-amd64")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "shards").mkdir(exist_ok=True)
    shard_path = args.out_dir / "shards" / f"{args.component}_K{args.K}.jsonl"
    status_path = args.out_dir / f"STATUS_{args.component}_K{args.K}.md"

    existing_rows = load_existing(shard_path)
    if len(existing_rows) >= args.n_states:
        print(json.dumps({"status": "already_complete", "rows": len(existing_rows), "path": str(shard_path)}), flush=True)
        return 0

    queries = base._load_queries(args.browsecomp_root / "topics-qrels" / "queries.tsv")
    qrels = base._load_qrels(args.browsecomp_root / "topics-qrels" / "qrel_evidence.txt")
    qids = base.freeze_qids(args, queries, qrels, args.out_dir)
    searcher = LuceneSearcher(str(args.index_path))
    scorer = stream.BatchedHFContinuationScorer(args.model, device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens)
    renderer = DualViewRenderer()

    n_written = len(existing_rows)
    with shard_path.open("a", encoding="utf-8") as f:
        for idx, item in enumerate(stream.candidate_items(args.component, qids, queries, qrels, searcher, scorer, renderer)):
            if idx < len(existing_rows):
                continue
            row = stream.row_for_item(args, item, queries, qrels, searcher, scorer, renderer, idx)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_written += 1
            if n_written % 4 == 0:
                write_status_live(
                    status_path,
                    stage="h100_2_candidate_b_live_utility",
                    run_id="h1002_true_live_stream_append_resume",
                    n_expected=args.n_states,
                    n_finished=n_written,
                    errors=[],
                    extra={"component": args.component, "K": args.K, "runner": "append_resume_batched_stream", "start_existing_rows": len(existing_rows)},
                )
                print(json.dumps({"component": args.component, "K": args.K, "n": n_written, "mode": "append_resume"}), flush=True)
            if n_written >= args.n_states:
                break

    print(json.dumps({"component": args.component, "K": args.K, "n": n_written, "path": str(shard_path)}), flush=True)
    return 0 if n_written >= args.n_states else 2


if __name__ == "__main__":
    raise SystemExit(main())
