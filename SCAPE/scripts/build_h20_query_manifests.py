#!/usr/bin/env python3
"""Freeze BASE_EVAL_128 queries + FORMAT_REPAIR splits from public SFT only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.eval.harmony_runtime import CANONICAL_TOOLS  # noqa: E402
from scape.training.clean_sft import write_jsonl  # noqa: E402

RAW_DEFAULT = REPO / "outputs/0814_clean_mechanism/data/hf_raw/sft_trajectories.jsonl"
TRAIN_DEFAULT = REPO / "outputs/0814_clean_mechanism/data/CLEAN_SFT_TRAIN.jsonl"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unwrap(rec: dict[str, Any]) -> dict[str, Any]:
    payload = rec.get("payload_json")
    if isinstance(payload, str) and payload.strip().startswith("{"):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict):
        merged = {**rec, **payload}
        merged.pop("payload_json", None)
        merged.setdefault("query_text", rec.get("query") or payload.get("query_text") or "")
        return merged
    rec.setdefault("query_text", rec.get("query") or "")
    return rec


def load_unique_queries(raw_path: Path) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    with raw_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = _unwrap(json.loads(line))
            if str(rec.get("stage", "sft")).lower() == "rl":
                continue
            qid = str(rec.get("query_id") or rec.get("qid") or f"sft_{i}")
            if qid in seen:
                continue
            qtext = str(rec.get("query_text") or rec.get("query") or "").strip()
            if not qtext:
                continue
            gids = rec.get("ground_truth_ids") or rec.get("document_ids_json") or []
            if isinstance(gids, str):
                try:
                    gids = json.loads(gids)
                except json.JSONDecodeError:
                    gids = []
            seen[qid] = {
                "query_id": qid,
                "query_text": qtext,
                "dataset": str(rec.get("dataset") or rec.get("dataset_name") or "unknown"),
                "n_turns": len(rec.get("turn_history") or []),
                "has_doc_store": bool(rec.get("doc_store")),
                "n_ground_truth_ids": len(gids) if isinstance(gids, list) else 0,
                "source": "pat-jj/harness-1-train-data stage=sft",
                "is_rl": False,
            }
    return [seen[k] for k in sorted(seen)]


def split_queries(rows: list[dict[str, Any]], seed: int = 817) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    order = list(rows)
    rng.shuffle(order)
    base = order[:128]
    rest = order[128:]
    auto = rest[:512]
    rest2 = rest[512:]
    dev = rest2[:128]
    test = rest2[128:]  # remaining, no duplicates
    return {"base_eval": base, "auto": auto, "real_dev": dev, "real_test": test}


def balanced_sample(
    examples: list[dict[str, Any]],
    *,
    n: int,
    seed: int,
    end_search_upweight: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        name = str(ex.get("tool_name") or "")
        if name in CANONICAL_TOOLS:
            by_tool[name].append(ex)
    for name in by_tool:
        rng.shuffle(by_tool[name])
    if end_search_upweight:
        n_end = max(len(by_tool.get("end_search") or []), int(0.18 * n))
        n_end = min(n_end, len(by_tool.get("end_search") or []))
        other = n - n_end
        tools = [t for t in CANONICAL_TOOLS if t != "end_search"]
        per = other // max(1, len(tools))
        out: list[dict[str, Any]] = []
        for t in tools:
            pool = by_tool.get(t) or []
            take = pool[:per]
            for row in take:
                row = dict(row)
                row["resampled_duplicate"] = False
                out.append(row)
        end_pool = list(by_tool.get("end_search") or [])
        i = 0
        while len([r for r in out if r.get("tool_name") == "end_search"]) < n_end and end_pool:
            row = dict(end_pool[i % len(end_pool)])
            row["resampled_duplicate"] = i >= len(end_pool)
            out.append(row)
            i += 1
        rng.shuffle(out)
        return out[:n]
    # round-robin unique
    idxs = {t: 0 for t in CANONICAL_TOOLS}
    out = []
    while len(out) < n:
        progressed = False
        for t in CANONICAL_TOOLS:
            pool = by_tool.get(t) or []
            i = idxs[t]
            if i < len(pool):
                row = dict(pool[i])
                row["resampled_duplicate"] = False
                out.append(row)
                idxs[t] = i + 1
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    rng.shuffle(out)
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raw-jsonl", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--train-jsonl", type=Path, default=TRAIN_DEFAULT)
    ap.add_argument("--seed", type=int, default=817)
    args = ap.parse_args()
    out: Path = args.out
    br = out / "base_recovery"
    data = out / "auto_data"
    br.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    queries = load_unique_queries(args.raw_jsonl)
    splits = split_queries(queries, seed=args.seed)
    master = {
        "seed": args.seed,
        "source": str(args.raw_jsonl),
        "source_sha256": _sha256_file(args.raw_jsonl) if args.raw_jsonl.is_file() else None,
        "n_unique_queries": len(queries),
        "n_base_eval": len(splits["base_eval"]),
        "n_auto": len(splits["auto"]),
        "n_real_dev": len(splits["real_dev"]),
        "n_real_test": len(splits["real_test"]),
        "disjoint": True,
        "duplicate_queries_allowed": False,
        "note": "TEST uses all remaining unique queries; no resampling.",
        "used_rl": False,
        "LOCAL_COMPAT_ONLY": True,
    }
    (br / "BASE_QUERY_MANIFEST.json").write_text(
        json.dumps({"queries": splits["base_eval"], "meta": master}, indent=2) + "\n"
    )
    (data / "AUTO_CLEAN_SPLIT_MANIFEST.json").write_text(
        json.dumps(
            {
                "meta": master,
                "auto_query_ids": [r["query_id"] for r in splits["auto"]],
                "real_dev_query_ids": [r["query_id"] for r in splits["real_dev"]],
                "real_test_query_ids": [r["query_id"] for r in splits["real_test"]],
                "base_eval_query_ids": [r["query_id"] for r in splits["base_eval"]],
            },
            indent=2,
        )
        + "\n"
    )
    (out / "QUERY_SPLITS.json").write_text(
        json.dumps({k: [r["query_id"] for r in v] for k, v in splits.items()} | {"meta": master}, indent=2)
        + "\n"
    )

    # FORMAT_REPAIR from public SFT examples only
    examples: list[dict[str, Any]] = []
    with args.train_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("is_rl"):
                continue
            if str(ex.get("tool_name") or "") not in CANONICAL_TOOLS:
                continue
            if not ex.get("input_ids") or not ex.get("n_context"):
                continue
            examples.append(ex)
    tools_all = Counter(str(e.get("tool_name")) for e in examples)
    n_end = sum(1 for e in examples if e.get("tool_name") == "end_search")
    train_a = balanced_sample(examples, n=4096, seed=args.seed, end_search_upweight=False)
    valid = balanced_sample(
        [e for e in examples if (e.get("query_id"), e.get("turn_idx")) not in {(r.get("query_id"), r.get("turn_idx")) for r in train_a}],
        n=256,
        seed=args.seed + 1,
        end_search_upweight=False,
    )
    train_c = balanced_sample(examples, n=4096, seed=args.seed + 2, end_search_upweight=True)
    fr_dir = br / "format_repair_data"
    fr_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(fr_dir / "FORMAT_REPAIR_TRAIN.jsonl", train_a)
    write_jsonl(fr_dir / "FORMAT_REPAIR_VALID.jsonl", valid)
    write_jsonl(fr_dir / "FORMAT_REPAIR_TRAIN_ENDUP.jsonl", train_c)
    audit = {
        "n_source_tool_turns": len(examples),
        "tool_counts_source": dict(tools_all),
        "n_end_search_source": n_end,
        "n_train": len(train_a),
        "n_valid": len(valid),
        "n_train_endup": len(train_c),
        "train_tool_counts": dict(Counter(r.get("tool_name") for r in train_a)),
        "endup_tool_counts": dict(Counter(r.get("tool_name") for r in train_c)),
        "n_resampled_train": sum(1 for r in train_a if r.get("resampled_duplicate")),
        "n_resampled_endup": sum(1 for r in train_c if r.get("resampled_duplicate")),
        "used_rl": False,
        "used_synthetic_mock": False,
        "used_future_reward": False,
    }
    (fr_dir / "FORMAT_REPAIR_DATA_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"queries": master, "format_repair": audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
