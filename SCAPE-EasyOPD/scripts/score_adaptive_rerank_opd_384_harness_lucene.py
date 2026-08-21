#!/usr/bin/env python3
"""Strict Harness-contract and Lucene Recall@5 scoring for adaptive rerank OPD."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ["V8D_ADAPTIVE_RERANK_INSTRUCTION"] = "1"

SETTINGS = ("TEACHER", "STUDENT_BEFORE_OPD", "STUDENT_AFTER_PURE_OPD", "STUDENT_AFTER_RL_PLUS_OPD")
BASE_MODEL = Path("/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
ADAPTER_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/component_sweep_0818/h100_2/formal_hf_adaptive_8gpu")
INDEX = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25")
HARNESS = Path("/mnt/songzijun/Capability_Evolution/SCAPE/external/harness-1")
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from harness.ultra_core import FAN_OUT_MAX_QUERIES
try:
    from pyserini.search.lucene import LuceneSearcher
except ModuleNotFoundError as exc:
    raise RuntimeError("official Pyserini/Lucene backend is required for formal scoring; do not use local fallback") from exc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(docid: str) -> str:
    return str(docid).split("_", 1)[0]


def recall(ranked: list[str], relevant: set[str], k: int) -> float:
    return len({norm(x) for x in ranked[:k]} & relevant) / max(1, len(relevant))


def ndcg(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    gains = [1.0 if norm(docid) in relevant else 0.0 for docid in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg else 0.0


def action_contract(row: dict[str, Any]) -> tuple[bool, list[str], str | None]:
    name, params = row.get("tool_name"), row.get("params") or {}
    if name == "search_corpus":
        query = params.get("query") or params.get("q")
        valid = isinstance(query, str) and bool(query.strip())
        return valid, [query.strip()] if valid else [], None if valid else "search_corpus requires nonempty query"
    if name == "fan_out_search":
        queries = params.get("queries")
        valid = (
            isinstance(queries, list)
            and 1 <= len(queries) <= FAN_OUT_MAX_QUERIES
            and all(isinstance(q, str) and bool(q.strip()) for q in queries)
        )
        valid_queries = [q.strip() for q in queries] if valid else []
        return valid, valid_queries, None if valid else f"fan_out_search requires 1-{FAN_OUT_MAX_QUERIES} nonempty string queries"
    if name == "grep_corpus":
        pattern = params.get("pattern")
        valid = isinstance(pattern, str) and bool(pattern.strip())
        return valid, [], None if valid else "grep_corpus requires nonempty pattern"
    if name == "read_document":
        valid = isinstance(params.get("doc_id"), str) and bool(params["doc_id"].strip())
        return valid, [], None if valid else "read_document requires doc_id"
    if name == "review_docs":
        ids = params.get("doc_ids")
        valid = isinstance(ids, list) and any(isinstance(x, str) and x for x in ids)
        return valid, [], None if valid else "review_docs requires doc_ids"
    if name == "curate":
        add, remove = params.get("add_ids"), params.get("remove_ids")
        valid = (isinstance(add, list) and bool(add)) or (isinstance(remove, list) and bool(remove))
        return valid, [], None if valid else "curate requires add_ids or remove_ids"
    if name in {"verify", "end_search"}:
        return True, [], None
    return False, [], "unknown tool"


def execute_search(searcher: LuceneSearcher, queries: list[str], depth: int) -> list[str]:
    # fan_out_search executes all queries in parallel. Fuse their ordered runs
    # rank-wise so every subquery can contribute before applying the depth cap.
    runs = [[str(hit.docid) for hit in searcher.search(query, depth)] for query in queries]
    ordered = []
    seen = set()
    for rank in range(depth):
        for run in runs:
            if rank >= len(run):
                continue
            docid = run[rank]
            if docid not in seen:
                seen.add(docid)
                ordered.append(docid)
                if len(ordered) >= depth:
                    return ordered
    return ordered


def summarize(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    subset = rows if split == "all_pool" else [r for r in rows if r["official_split"] == "test"]
    n = len(subset)
    metrics = ("evidence_recall_at_5",)
    result = {
        "split": split, "n_queries": n,
        "legal_action_rate": sum(bool(r["legal"]) for r in subset) / n,
        "executable_action_rate": sum(bool(r["executable"]) for r in subset) / n,
        "successful_search_rate": sum(bool(r["retrieved_docids"]) for r in subset) / n,
    }
    result.update({key: sum(float(r[key]) for r in subset) / n for key in metrics})
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=1000)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests = [args.shards / setting / "384_QUERY_MANIFEST.json" for setting in SETTINGS]
    manifest_hashes = [sha256(path) for path in manifests]
    if len(set(manifest_hashes)) != 1:
        raise RuntimeError(f"shard manifest mismatch: {manifest_hashes}")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    qrels = {row["query_id"]: row for row in manifest["queries"]}
    manifest_out = args.output_dir / "384_QUERY_MANIFEST.json"
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    searcher = LuceneSearcher(str(INDEX))
    summaries = []
    ordered_ids = None
    source_hashes = {}
    for setting in SETTINGS:
        source = args.shards / setting / setting / "PER_QUERY.jsonl"
        source_hashes[setting] = sha256(source)
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [row["query_id"] for row in rows]
        if len(rows) != 384 or len(set(ids)) != 384:
            raise RuntimeError(f"{setting}: expected 384 unique rows")
        if ordered_ids is None:
            ordered_ids = ids
        elif ids != ordered_ids:
            raise RuntimeError(f"{setting}: ordered query IDs differ")

        for i, row in enumerate(rows, 1):
            executable, queries, error = action_contract(row)
            retrieved = execute_search(searcher, queries, args.depth) if executable and queries else []
            evidence = {norm(x) for x in qrels[row["query_id"]]["evidence_docids"]}
            row.update({
                "legal": executable,
                "executable": executable,
                "execution_error": error,
                "executed_queries": queries,
                "retrieval_backend": "pyserini_lucene",
                "retrieved_docids": retrieved,
                "evidence_recall_at_5": recall(retrieved, evidence, 5),
            })
            if i % 32 == 0:
                print(json.dumps({"setting": setting, "completed": i, "n": len(rows)}), flush=True)

        out = args.output_dir / setting
        out.mkdir(parents=True, exist_ok=True)
        with (out / "PER_QUERY.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        old_summary = json.loads((args.shards / setting / setting / "SUMMARY.json").read_text(encoding="utf-8"))
        summary = {
            "setting": setting,
            "adapter_reload_path": old_summary["adapter_reload_path"],
            "all_pool": summarize(rows, "all_pool"),
            "official_test": summarize(rows, "official_test"),
        }
        (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(summary)

    adapter_paths = {
        "STUDENT_AFTER_PURE_OPD": ADAPTER_ROOT / "PURE_OPD_seed42" / "lora_checkpoint",
        "STUDENT_AFTER_RL_PLUS_OPD": ADAPTER_ROOT / "RL_PLUS_OPD_seed42" / "lora_checkpoint",
    }
    script_path = Path(__file__).resolve()
    provenance = {
        "base_model_path": str(BASE_MODEL),
        "adapter_artifacts": {
            setting: {
                "path": str(path),
                "adapter_config_sha256": sha256(path / "adapter_config.json"),
                "adapter_model_sha256": sha256(path / "adapter_model.safetensors"),
            } for setting, path in adapter_paths.items()
        },
        "source_generation_per_query_sha256": source_hashes,
        "runner_path": "/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/scripts/eval_adaptive_rerank_opd_384.py",
        "runner_sha256": sha256(Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/scripts/eval_adaptive_rerank_opd_384.py")),
        "scorer_path": str(script_path),
        "scorer_sha256": sha256(script_path),
        "lucene_index_file_sha256": {p.name: sha256(p) for p in sorted(INDEX.iterdir()) if p.is_file()},
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "java_home": os.environ.get("JAVA_HOME"),
            "java_version": "21.0.11",
            "torch": "2.10.0+cu128",
            "transformers": "5.14.1",
            "peft": "0.19.1",
        },
        "harness_contract": {
            "ultra_core_sha256": sha256(HARNESS / "harness" / "ultra_core.py"),
            "train_rl_sha256": sha256(HARNESS / "training" / "train_rl.py"),
            "fan_out_max_queries": FAN_OUT_MAX_QUERIES,
            "adaptive_rerank_instruction_enabled_for_teacher": True,
        },
    }
    payload = {
        "status": "ADAPTIVE_RERANK_INSTRUCTION_OPD_384_HARNESS_LUCENE_COMPLETE",
        "component": "adaptive_rerank_instruction",
        "query_count": 384,
        "test_query_count": 76,
        "manifest_sha256": sha256(manifest_out),
        "retrieval_backend": "pyserini_lucene",
        "retrieval_index": str(INDEX),
        "retrieval_depth": args.depth,
        "fan_out_fusion": "rankwise_round_robin_deduplicated_then_depth_cap",
        "action_contract": "Harness-1 strict: fan_out_search requires 1-5 nonempty string queries; search_corpus requires query:str",
        "component_contract": "Teacher prompt includes adaptive rerank instruction; Student Before/After prompts do not expose it.",
        "settings": summaries,
        "provenance": provenance,
        "student_inference_privilege": False,
    }
    summary_path = args.output_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in args.output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(f"{sha256(p)}  {p.relative_to(args.output_dir)}" for p in sorted(files)) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
