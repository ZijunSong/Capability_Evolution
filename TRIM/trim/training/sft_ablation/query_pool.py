"""Fixed SEC train queries. Gold labels are not a retrieval shortcut."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from trim.training.sft_ablation.contracts import fingerprint
from trim.training.sft_ablation.io_util import atomic_write_json, sha256_tree


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, str):
            rows.append({"query_id": item, "query": item})
        else:
            rows.append(dict(item))
    return rows


def load_train_queries(args: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = str(getattr(args, "query_manifest", "") or "")
    if manifest:
        rows = _read_manifest(Path(manifest))
        meta = {"source": manifest, "pool_contract": "explicit_manifest"}
    elif str(getattr(args, "train_data", "sec")) == "sec":
        from trim.eval.sec_corpus import load_sec_rl_queries

        source = Path(args.rl_data) if str(getattr(args, "rl_data", "") or "") else None
        corpus = Path(args.sec_corpus_root) if str(getattr(args, "sec_corpus_root", "") or "") else None
        rows, meta = load_sec_rl_queries(source, corpus_root=corpus)
    else:
        raise ValueError(f"unsupported train-data {args.train_data}; v1 uses SEC")
    cleaned = []
    seen: set[str] = set()
    for row in rows:
        qid = str(row.get("query_id") or "")
        if not qid or qid in seen:
            continue
        seen.add(qid)
        cleaned.append(
            {
                "query_id": qid,
                "query": str(row.get("query") or row.get("question") or ""),
            }
        )
    if not cleaned:
        raise ValueError("train query pool is empty")
    order = [row["query_id"] for row in cleaned]
    info = {
        "count": len(cleaned),
        "order_digest": fingerprint(order),
        "source": meta.get("source") or meta.get("path") or manifest,
        "pool_contract": meta.get("pool_contract", "sec"),
    }
    eval_path = str(getattr(args, "eval_manifest", "") or "")
    if eval_path:
        eval_ids = {str(row.get("query_id") or row) for row in _read_manifest(Path(eval_path))}
        overlap = sorted(set(order) & eval_ids)
        if overlap:
            raise ValueError(f"train queries overlap eval manifest: {overlap[:8]}")
        info["eval_boundary"] = "disjoint"
    else:
        info["eval_boundary"] = "not_provided"
    return cleaned, info


def corpus_digest(args: Any) -> str:
    root = str(getattr(args, "sec_corpus_root", "") or getattr(args, "corpus", "") or "")
    if not root:
        return "unspecified"
    return sha256_tree(Path(root)) if Path(root).exists() else "missing"


def retrieval_digest(args: Any) -> str:
    index = str(getattr(args, "bm25_index", "") or "")
    if not index:
        return "upstream_default"
    return sha256_tree(Path(index)) if Path(index).exists() else "missing"


def write_query_manifest(out: Path, rows: Sequence[dict[str, Any]], info: dict[str, Any]) -> None:
    atomic_write_json(Path(out) / "query_manifest.json", {"info": info, "queries": list(rows)})
