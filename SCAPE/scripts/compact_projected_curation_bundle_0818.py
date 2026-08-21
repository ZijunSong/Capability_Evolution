#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact_row(r: dict[str, Any], max_prompt_chars: int) -> dict[str, Any]:
    docs = r.get("documents") or []
    curated = list(map(str, r.get("curated_ids_pre") or []))
    keep_ids = set(curated) | set(map(str, r.get("added_ids") or [])) | set(map(str, r.get("removed_ids") or [])) | set(map(str, r.get("incoming_ids") or []))
    slim_docs = []
    for d in docs:
        did = str(d.get("id"))
        if did in keep_ids or len(slim_docs) < 24:
            slim_docs.append({"id": did, "text": str(d.get("text") or "")[:600]})
        if len(slim_docs) >= 40:
            break
    state = {
        "query_id": r.get("query_id"),
        "documents": slim_docs,
        "curated_ids": curated,
        "remaining_budget": r.get("remaining_budget", 8192),
        "student_inference_privilege": False,
    }
    prompt_reduced = "Choose one legal Harness-1 tool and JSON arguments.\nTOOLS: fan_out_search, search_corpus, grep_corpus, read_document, review_docs, curate, verify, end_search\nSTATE:\n" + json.dumps(state, ensure_ascii=False, sort_keys=True)
    if len(prompt_reduced) > max_prompt_chars:
        prompt_reduced = prompt_reduced[:max_prompt_chars]
    out = {
        "row_id": r.get("row_id"),
        "query_id": str(r.get("query_id")),
        "state_hash": r.get("state_hash"),
        "snapshot_hash": r.get("state_hash"),
        "prompt_reduced": prompt_reduced,
        "prompt_full": prompt_reduced,
        "response_text": r.get("response_text"),
        "projected_action": r.get("projected_action"),
        "added_ids": r.get("added_ids") or [],
        "removed_ids": r.get("removed_ids") or [],
        "curated_ids_pre": curated,
        "incoming_ids": r.get("incoming_ids") or [],
        "gold_evidence_ids": r.get("gold_evidence_ids") or [],
        "qrel_terminal_reward_pre": r.get("qrel_terminal_reward_pre", 0.0),
        "qrel_terminal_reward_post": r.get("qrel_terminal_reward_post", 0.0),
        "sampling_mode": r.get("sampling_mode"),
        "oversampled_capacity_event": bool(r.get("oversampled_capacity_event")),
        "student_inference_privilege": False,
    }
    return out


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-prompt-chars", type=int, default=12000)
    args = ap.parse_args()
    summary = {}
    for split in ["TRAIN", "VALID", "TEST"]:
        src = args.out_dir / f"CURATION_BUNDLE_{split}.jsonl"
        dst = args.out_dir / f"CURATION_BUNDLE_{split}_COMPACT.jsonl"
        n = add = rem = term = over = 0
        with dst.open("w", encoding="utf-8") as f:
            for r in iter_jsonl(src):
                c = compact_row(r, args.max_prompt_chars)
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n += 1
                add += int(bool(c["added_ids"]))
                rem += int(bool(c["removed_ids"]))
                term += int(float(c["qrel_terminal_reward_post"] or 0.0) > 0)
                over += int(bool(c["oversampled_capacity_event"]))
        summary[split.lower()] = {"rows": n, "valid_add": add, "valid_remove": rem, "terminal_nonzero": term, "oversampled": over, "path": str(dst)}
    (args.out_dir / "COMPACT_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
