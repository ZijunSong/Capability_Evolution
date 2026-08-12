#!/usr/bin/env python3
"""Build A0–A4 frozen view prompts for offline_valid and base_live."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transformers import AutoTokenizer

from training.scope_round11.stage1_views import VIEW_NAMES, build_stage1_view

BASE_MODEL = "/data/ppnm/models/Qwen2.5-7B-Instruct"
R8_VALID = _REPO / "artifacts/datasets/scope_round8/rollback_sdi/valid.jsonl"
HOLD = _REPO / "artifacts/datasets/scope_round9/hier_sdi/frozen_live_holdout.jsonl"
OUT = _REPO / "artifacts/datasets/scope_round11/phase_a_views"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def enrich_from_source(sample: dict, state_source: str) -> dict:
    ds = sample.get("decision_state") or {}
    return {
        **sample,
        "query_id": str(sample.get("query_id") or ds.get("task_id") or ""),
        "turn": int(ds.get("turn_id", sample.get("turn", sample.get("turn_id", 0))) or 0),
        "event_id": sample.get("event_id") or f"{sample.get('query_id')}:{ds.get('turn_id', 0)}",
        "state_source": state_source,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--max-length", type=int, default=8100)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    sources = {
        "offline_valid": (R8_VALID, "offline_valid"),
        "base_live": (HOLD, "base_live"),
    }
    report: dict = {"views": {}, "n": {}}
    for split, (src, state_source) in sources.items():
        rows = [enrich_from_source(r, state_source) for r in load_jsonl(src)]
        report["n"][split] = len(rows)
        for view in VIEW_NAMES:
            out_rows = []
            for r in rows:
                vr = build_stage1_view(r, tok, view, max_length=args.max_length)
                out_rows.append(
                    {
                        "query_id": r["query_id"],
                        "turn": r["turn"],
                        "event_id": r["event_id"],
                        "state_source": state_source,
                        "view": view,
                        "effective_input_text": vr.effective_input_text,
                        "prompt_sha256": vr.prompt_sha256,
                        "candidate_list": vr.candidate_list,
                        "gold_operation": vr.gold_operation,
                        "gold_checkpoint_local_id": vr.gold_checkpoint_local_id,
                        "gold_checkpoint_global_id": vr.gold_checkpoint_global_id,
                        "gold_in_candidates": vr.gold_in_candidates,
                        "truncated": vr.truncated,
                        "token_length_before": vr.token_length_before,
                        "token_length_after": vr.token_length_after,
                    }
                )
            out_path = args.out / split / f"{view}.jsonl"
            write_jsonl(out_path, out_rows)
            report["views"][f"{split}/{view}"] = str(out_path)
            print(f"wrote {out_path} n={len(out_rows)}")
    (args.out / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
