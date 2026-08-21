#!/usr/bin/env python3
"""Strict Harness-schema scorer for token_budget_marker OPD actions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

FAN_OUT_MAX_QUERIES = 5
from pyserini.search.lucene import LuceneSearcher

SETTINGS = ("TEACHER", "STUDENT_BEFORE_OPD", "STUDENT_AFTER_PURE_OPD", "STUDENT_AFTER_RL_PLUS_OPD")
INDEX = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25")
HARNESS_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCAPE/external/harness-1")
AUTO_MANIFEST = Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/0821_auto_populate_opd_384_formal_v2/shards/TEACHER/384_QUERY_MANIFEST.json")
TOKEN_ROOT = Path("/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/outputs/0821_token_budget_marker_opd_384")
RUNNER = Path(__file__).with_name("eval_token_budget_marker_opd_384.py")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(doc: str) -> str:
    return str(doc).split("_", 1)[0]


def extract_action(text: str) -> tuple[str | None, dict[str, Any]]:
    """Extract the first complete compact Harness tool object."""
    for match in re.finditer(r"\{", text or ""):
        try:
            obj, _ = json.JSONDecoder().raw_decode((text or "")[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("name")
        if not isinstance(name, str):
            continue
        nested = obj.get("args")
        if nested is None:
            nested = obj.get("arguments")
        params = dict(nested) if isinstance(nested, dict) else {
            k: v for k, v in obj.items() if k not in {"tool", "name", "args", "arguments"}
        }
        return name, params
    return None, {}


def contract(name: str | None, params: dict[str, Any]) -> tuple[bool, list[str]]:
    if name == "fan_out_search":
        queries = params.get("queries")
        valid = (
            isinstance(queries, list)
            and 1 <= len(queries) <= FAN_OUT_MAX_QUERIES
            and all(isinstance(q, str) and bool(q.strip()) for q in queries)
        )
        return valid, [q.strip() for q in queries] if valid else []
    if name == "search_corpus":
        query = params.get("query") or params.get("q")
        valid = isinstance(query, str) and bool(query.strip())
        return valid, [query.strip()] if valid else []
    if name == "grep_corpus":
        pattern = params.get("pattern")
        valid = isinstance(pattern, str) and bool(pattern.strip())
        return valid, [pattern.strip()] if valid else []
    return False, []


def fused_top5(searcher: Any, queries: list[str]) -> list[str]:
    runs = [[str(hit.docid) for hit in searcher.search(query, 5)] for query in queries]
    output: list[str] = []
    seen: set[str] = set()
    for rank in range(5):
        for run in runs:
            if rank < len(run) and run[rank] not in seen:
                seen.add(run[rank])
                output.append(run[rank])
                if len(output) == 5:
                    return output
    return output


def summary(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    selected = rows if split == "all_pool" else [r for r in rows if r["official_split"] == "test"]
    if not selected:
        raise RuntimeError(f"empty split: {split}")
    return {
        "split": split,
        "n_queries": len(selected),
        "legal_action_rate": sum(bool(r["legal"]) for r in selected) / len(selected),
        "evidence_recall_at_5": sum(r["evidence_recall_at_5"] for r in selected) / len(selected),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, default=TOKEN_ROOT)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(AUTO_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("query_count") != 384 or manifest.get("test_query_count") != 76:
        raise RuntimeError("authoritative 384-query manifest gate failed")
    queries = manifest["queries"]
    if len({r["query_id"] for r in queries}) != 384:
        raise RuntimeError("manifest query uniqueness gate failed")
    source_by_qid = {r["query_id"]: r for r in queries}
    manifest_copy = args.output_dir / "384_QUERY_MANIFEST.json"
    manifest_copy.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    searcher = LuceneSearcher(str(INDEX))
    summaries: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    ordered_ids: list[str] | None = None
    for setting in SETTINGS:
        source_path = args.source_root / setting / "PER_QUERY.jsonl"
        source_hashes[setting] = sha(source_path)
        rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [str(r.get("query_id")) for r in rows]
        if len(rows) != 384 or len(set(ids)) != 384:
            raise RuntimeError(f"{setting}: expected 384 unique rows")
        if ordered_ids is None:
            ordered_ids = ids
        elif ids != ordered_ids:
            raise RuntimeError(f"{setting}: ordered query IDs mismatch")

        scored: list[dict[str, Any]] = []
        for old in rows:
            qid = str(old["query_id"])
            source = source_by_qid.get(qid)
            if source is None:
                raise RuntimeError(f"{setting}: query ID not in authoritative manifest: {qid}")
            name, params = extract_action(old.get("generated_text", ""))
            legal, executed = contract(name, params)
            docs = fused_top5(searcher, executed) if legal else []
            gold = {norm(x) for x in source["evidence_docids"]}
            scored.append({
                **old,
                "tool_name": name,
                "params": params,
                "official_split": source["official_split"],
                "legal": legal,
                "executable": legal,
                "executed_queries": executed,
                "retrieval_backend": "pyserini_lucene",
                "fan_out_fusion": "rankwise_round_robin_top5",
                "retrieved_docids_at_5": docs,
                "evidence_recall_at_5": len({norm(x) for x in docs} & gold) / max(1, len(gold)),
                "evidence_qrel_count": len(gold),
                "parser_version": "strict_harness_token_marker_r5_v1",
            })
        out_dir = args.output_dir / setting
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "PER_QUERY.jsonl").open("w", encoding="utf-8") as f:
            for row in scored:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        setting_summary = {
            "setting": setting,
            "adapter_reload_path": json.loads((args.source_root / setting / "SUMMARY.json").read_text(encoding="utf-8")).get("adapter_reload_path"),
            "all_pool": summary(scored, "all_pool"),
            "official_test": summary(scored, "official_test"),
        }
        (out_dir / "SUMMARY.json").write_text(json.dumps(setting_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(setting_summary)

    payload = {
        "status": "TOKEN_BUDGET_MARKER_OPD_384_R5_COMPLETE",
        "component": "token_budget_marker",
        "query_count": 384,
        "test_query_count": 76,
        "manifest_sha256": sha(manifest_copy),
        "metrics": ["legal_action_rate", "evidence_recall_at_5"],
        "explicitly_not_computed": ["recall_at_100", "recall_at_1000"],
        "settings": summaries,
        "provenance": {
            "source_generation_sha256": source_hashes,
            "runner_sha256": sha(RUNNER),
            "scorer_sha256": sha(Path(__file__)),
            "java_home": os.environ.get("JAVA_HOME"),
            "fan_out_max_queries": FAN_OUT_MAX_QUERIES,
            "index_sha256": {p.name: sha(p) for p in sorted(INDEX.iterdir()) if p.is_file()},
            "authoritative_manifest": str(AUTO_MANIFEST),
        },
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in args.output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(f"{sha(p)}  {p.relative_to(args.output_dir)}" for p in sorted(files)) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
