#!/usr/bin/env python3
"""Replace a few rerun queries in an existing upstream_api eval directory."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_TRIM_ROOT = Path(__file__).resolve().parents[1]
if str(_TRIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRIM_ROOT))

from trim.eval.eval_parallel import load_json, load_jsonl, write_json, write_jsonl
from trim.eval.harness1_api_eval import summarize_api_traces
from trim.eval.sr_opd_four_cell_eval import write_upstream_api_eval_outputs


def _backup(path: Path, dest_dir: Path) -> None:
    if path.is_file():
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest_dir / path.name)


def merge_rerun(
    original_out: Path,
    rerun_out: Path,
    query_ids: Sequence[str],
    *,
    backup_name: str = "_pre_ellipsis_rerun",
) -> dict[str, Any]:
    want = [str(q) for q in query_ids if str(q).strip()]
    want_set = set(want)
    api = original_out / "upstream_api"
    rerun_api = rerun_out / "upstream_api"
    old_pq_path = api / "PER_QUERY.jsonl"
    new_pq_path = rerun_api / "PER_QUERY.jsonl"
    if not old_pq_path.is_file():
        raise FileNotFoundError(old_pq_path)
    if not new_pq_path.is_file():
        raise FileNotFoundError(new_pq_path)

    old_pq = load_jsonl(old_pq_path)
    new_pq = load_jsonl(new_pq_path)
    new_by_id = {str(row.get("query_id")): row for row in new_pq if row.get("query_id")}
    missing = [qid for qid in want if qid not in new_by_id]
    if missing:
        raise RuntimeError(f"rerun missing query_ids: {missing}")
    still_infra = [
        qid
        for qid in want
        if str((new_by_id[qid] or {}).get("finish_reason") or "") == "infra_error"
    ]
    if still_infra:
        raise RuntimeError(f"rerun still infra_error: {still_infra}")

    merged_pq: list[dict[str, Any]] = []
    replaced = 0
    for row in old_pq:
        qid = str(row.get("query_id") or "")
        if qid in want_set:
            merged_pq.append(dict(new_by_id[qid]))
            replaced += 1
        else:
            merged_pq.append(row)
    if replaced != len(want_set):
        raise RuntimeError(f"expected to replace {len(want_set)} rows, replaced {replaced}")

    old_turns_path = api / "TURNS.jsonl"
    new_turns_path = rerun_api / "TURNS.jsonl"
    old_turns = load_jsonl(old_turns_path) if old_turns_path.is_file() else []
    new_turns = load_jsonl(new_turns_path) if new_turns_path.is_file() else []
    merged_turns = [row for row in old_turns if str(row.get("query_id") or "") not in want_set]
    merged_turns.extend(row for row in new_turns if str(row.get("query_id") or "") in want_set)

    backup_dir = api / backup_name
    for path in (
        old_pq_path,
        old_turns_path,
        api / "SUMMARY.json",
        api / "DONE.json",
        original_out / "FOUR_CELL_OFFICIAL_SUMMARY.json",
    ):
        _backup(path, backup_dir)

    write_jsonl(old_pq_path, merged_pq)
    if merged_turns or old_turns_path.is_file():
        write_jsonl(old_turns_path, merged_turns)

    for shard_dir in sorted((api / "shards").glob("rank*")) if (api / "shards").is_dir() else []:
        shard_pq = shard_dir / "PER_QUERY.jsonl"
        if not shard_pq.is_file():
            continue
        rows = load_jsonl(shard_pq)
        if not any(str(row.get("query_id") or "") in want_set for row in rows):
            continue
        _backup(shard_pq, backup_dir / shard_dir.name)
        shard_merged = []
        for row in rows:
            qid = str(row.get("query_id") or "")
            shard_merged.append(dict(new_by_id[qid]) if qid in want_set else row)
        write_jsonl(shard_pq, shard_merged)
        done_path = shard_dir / "DONE.json"
        if done_path.is_file():
            _backup(done_path, backup_dir / shard_dir.name)
            done = load_json(done_path)
            n_infra = sum(1 for row in shard_merged if row.get("finish_reason") == "infra_error")
            done["n_infra_error"] = n_infra
            done["infra_clean"] = n_infra == 0
            done["partial"] = n_infra > 0
            done["ok"] = n_infra == 0
            write_json(done_path, done)

    old_summary = load_json(api / "SUMMARY.json") if (api / "SUMMARY.json").is_file() else {}
    official = load_json(original_out / "FOUR_CELL_OFFICIAL_SUMMARY.json") if (
        original_out / "FOUR_CELL_OFFICIAL_SUMMARY.json"
    ).is_file() else {}
    pool_meta = dict(official.get("pool") or {})
    component = str(official.get("component") or old_summary.get("component") or "all")
    summary = summarize_api_traces(merged_pq, n_planned=len(merged_pq))
    summary["eval_profile"] = old_summary.get("eval_profile") or "upstream_core_local_bm25"
    summary["eval_replicas"] = old_summary.get("eval_replicas")
    summary["execution_complete"] = True
    summary["setting"] = "upstream_api"
    summary["eval_mode"] = old_summary.get("eval_mode") or "harness"
    if int(summary.get("n_infra_error") or 0) == 0:
        summary.pop("shard_failures", None)
        summary.pop("shard_partials", None)
        summary["partial"] = False
    if old_summary.get("shard_status"):
        summary["shard_status"] = old_summary["shard_status"]
    write_json(api / "SUMMARY.json", summary)
    write_json(
        api / "DONE.json",
        {
            "ok": int(summary.get("n_infra_error") or 0) == 0,
            "partial": bool(summary.get("partial")),
            "n_queries": len(merged_pq),
            "n_planned": len(merged_pq),
            "n_infra_error": int(summary.get("n_infra_error") or 0),
            "execution_complete": True,
            "infra_clean": int(summary.get("n_infra_error") or 0) == 0,
            "replaced_query_ids": want,
        },
    )
    payload = write_upstream_api_eval_outputs(
        original_out,
        component_id=component,
        summaries=[summary],
        pool_meta=pool_meta,
    )
    report = {
        "original_out": str(original_out),
        "rerun_out": str(rerun_out),
        "replaced_query_ids": want,
        "formal_eligible": bool(payload.get("formal_eligible")),
        "claim_status": payload.get("claim_status"),
        "n_infra_error": payload.get("settings", [{}])[0].get("n_infra_error")
        if payload.get("settings")
        else summary.get("n_infra_error"),
        "recall": summary.get("recall"),
        "backup": str(backup_dir),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-out", type=Path, required=True)
    parser.add_argument("--rerun-out", type=Path, required=True)
    parser.add_argument("--query-ids", required=True, help="Comma-separated query ids")
    args = parser.parse_args(argv)
    qids = [part.strip() for part in str(args.query_ids).split(",") if part.strip()]
    report = merge_rerun(args.original_out, args.rerun_out, qids)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("formal_eligible") else 1


if __name__ == "__main__":
    raise SystemExit(main())
