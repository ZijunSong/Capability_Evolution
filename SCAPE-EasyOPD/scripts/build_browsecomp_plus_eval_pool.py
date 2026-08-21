#!/usr/bin/env python3
"""Build and audit a deterministic BrowseComp-Plus evaluation pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(text: str) -> str:
    return " ".join(text.lower().strip().split())


def read_queries(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"malformed queries.tsv line {lineno}")
            if parts[0] in rows:
                raise ValueError(f"duplicate query id {parts[0]}")
            rows[parts[0]] = parts[1]
    return rows


def read_qrels(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"malformed qrel line {lineno}: {line.rstrip()}")
            qid, _, docid, relevance = parts
            if float(relevance) <= 0:
                continue
            if docid not in result[qid]:
                result[qid].append(docid)
    return dict(result)


def load_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("query_ids", "train_query_ids", "test_query_ids"):
            if key in payload:
                return {str(x) for x in payload[key]}
        rows = payload.get("queries", [])
    else:
        rows = payload
    return {str(row.get("query_id", row.get("id"))) if isinstance(row, dict) else str(row) for row in rows}


def quantile_edges(values: list[int], bins: int) -> list[int]:
    ordered = sorted(values)
    edges = []
    for index in range(bins + 1):
        pos = min(len(ordered) - 1, math.floor(index * (len(ordered) - 1) / bins))
        edges.append(ordered[pos])
    return edges


def bin_index(value: int, edges: list[int]) -> int:
    for index in range(len(edges) - 1):
        if value <= edges[index + 1]:
            return index
    return len(edges) - 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-tsv", type=Path, required=True)
    parser.add_argument("--qrel-evidence", type=Path, required=True)
    parser.add_argument("--qrel-golds", type=Path, required=True)
    parser.add_argument("--training-pool", type=Path, required=True)
    parser.add_argument("--official-splits", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--exclude-training", action="store_true")
    args = parser.parse_args()

    queries = read_queries(args.queries_tsv)
    evidence = read_qrels(args.qrel_evidence)
    golds = read_qrels(args.qrel_golds)
    eligible = sorted(set(queries) & set(evidence) & set(golds), key=lambda x: (int(x) if x.isdigit() else x))
    if len(eligible) < args.target:
        raise ValueError(f"only {len(eligible)} eligible queries, target is {args.target}")

    training = json.loads(args.training_pool.read_text(encoding="utf-8"))
    training_rows = training.get("queries", training) if isinstance(training, dict) else training
    training_ids = {str(row.get("query_id", row.get("id"))) if isinstance(row, dict) else str(row) for row in training_rows}
    training_norm = {norm(str(row.get("query", row.get("question", "")))) for row in training_rows if isinstance(row, dict) and row.get("query")}
    if args.exclude_training:
        eligible = [qid for qid in eligible if qid not in training_ids and norm(queries[qid]) not in training_norm]
    if len(eligible) < args.target:
        raise ValueError(f"only {len(eligible)} eligible queries after exclusions, target is {args.target}")

    rows = [{"query_id": qid, "query": queries[qid], "evidence_docids": evidence[qid], "gold_docids": golds[qid]} for qid in eligible]
    e_edges = quantile_edges([len(row["evidence_docids"]) for row in rows], args.bins)
    g_edges = quantile_edges([len(row["gold_docids"]) for row in rows], args.bins)
    strata: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (bin_index(len(row["evidence_docids"]), e_edges), bin_index(len(row["gold_docids"]), g_edges))
        strata[key].append(row)

    # Proportional allocation with largest-remainder correction, then seeded sampling.
    allocation = {key: min(len(value), args.target * len(value) // len(rows)) for key, value in strata.items()}
    remainder = args.target - sum(allocation.values())
    ranking = sorted(strata, key=lambda key: (-(args.target * len(strata[key]) / len(rows) - allocation[key]), key))
    for key in ranking[:remainder]:
        allocation[key] += 1
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda row: row["query_id"])
        rng.shuffle(candidates)
        selected.extend(candidates[: allocation[key]])
    selected.sort(key=lambda row: (row["query_id"]))

    selected_ids = {row["query_id"] for row in selected}
    selected_norm = {norm(row["query"]) for row in selected}
    splits = json.loads(args.official_splits.read_text(encoding="utf-8"))
    official_train = {str(x) for x in splits["train_query_ids"]}
    official_test = {str(x) for x in splits["test_query_ids"]}
    overlap_id = selected_ids & training_ids
    overlap_text = selected_norm & training_norm

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {str(path): sha256_file(path) for path in (args.queries_tsv, args.qrel_evidence, args.qrel_golds, args.training_pool, args.official_splits)}
    for row in selected:
        row["stratum"] = f"e{bin_index(len(row['evidence_docids']), e_edges)}_g{bin_index(len(row['gold_docids']), g_edges)}"
    manifest = {
        "schema_version": "browsecomp_plus_eval_pool_v1",
        "status": "CANDIDATE_REJECTED_TRAIN_OVERLAP" if overlap_id or overlap_text else "FROZEN_VALID",
        "pool_contract": f"{args.target} unique official queries present in both qrels; deterministic stratified sampling; exclude_training={args.exclude_training}",
        "seed": args.seed,
        "target": args.target,
        "query_count": len(selected),
        "queries": selected,
        "stratification": {"bins": args.bins, "evidence_edges": e_edges, "gold_edges": g_edges, "allocation": {f"{k[0]}_{k[1]}": v for k, v in allocation.items()}},
        "input_sha256": input_hashes,
        "training_overlap": {"query_id_count": len(overlap_id), "query_ids": sorted(overlap_id), "normalized_text_count": len(overlap_text)},
        "official_split_counts": {"eligible_total": len(eligible), "selected_official_train": len(selected_ids & official_train), "selected_official_test": len(selected_ids & official_test)},
    }
    (args.out_dir / "query_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out_dir / "source_pool.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    audit = {"status": manifest["status"], "selected_count": len(selected), "unique_query_ids": len(selected_ids), "all_have_evidence_qrel": all(row["evidence_docids"] for row in selected), "all_have_gold_qrel": all(row["gold_docids"] for row in selected), "training_overlap_query_ids": sorted(overlap_id), "training_overlap_normalized_text_count": len(overlap_text), "synthetic_query_ids": [row["query_id"] for row in selected if row["query_id"].startswith("synth_doc_")], "input_sha256": input_hashes}
    (args.out_dir / "overlap_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "overlap_audit.md").write_text(f"# BrowseComp-Plus {args.target}-query pool audit\n\n" + "\n".join(f"- **{key}**: `{value}`" for key, value in audit.items()) + "\n", encoding="utf-8")
    run_manifest = {"schema_version": "browsecomp_plus_eval_pool_run_v1", "seed": args.seed, "builder": str(Path(__file__).resolve()), "inputs": input_hashes, "status": manifest["status"]}
    (args.out_dir / "RUN_MANIFEST.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = ["query_manifest.json", "source_pool.jsonl", "overlap_audit.json", "overlap_audit.md", "RUN_MANIFEST.json"]
    (args.out_dir / "SHA256SUMS").write_text("\n".join(f"{sha256_file(args.out_dir / name)}  {name}" for name in files) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "query_count": len(selected), "training_overlap_ids": len(overlap_id), "strata": Counter(row["stratum"] for row in selected)}, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "FROZEN_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
