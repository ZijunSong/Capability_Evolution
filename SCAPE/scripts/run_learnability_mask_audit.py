#!/usr/bin/env python3
"""Mask/tokenization audit on 512 sampled states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.collection.same_state import load_same_state_jsonl
from scape.training.hf_tool_opd import ScapeHFToolOPD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--model-path", default="/data/ppnm/models/harness-1")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_same_state_jsonl(Path(args.jsonl))[:args.n]
    backend = ScapeHFToolOPD(
        model_path=args.model_path,
        device_map=f"cuda:{args.gpu}",
        use_lora=False,
    )
    texts = [r["response_text"] for r in rows]
    audit = backend.audit_tool_spans(texts)

    span_stats = {"parsable": 0, "empty_mask": 0, "name_only": 0, "full_tool": 0}
    details = []
    for row in rows:
        resp = row["response_text"]
        resp_ids = backend.encode(resp)
        spans = backend.span_token_masks(resp, len(resp_ids))
        n_tool = sum(spans["tool"])
        n_name = sum(spans["name"])
        n_key = sum(spans["key"])
        n_val = sum(spans["value"])
        ok = n_tool > 0 and n_name > 0
        if ok:
            span_stats["parsable"] += 1
        if n_tool == 0:
            span_stats["empty_mask"] += 1
        if n_name > 0 and n_key == 0 and n_val == 0:
            span_stats["name_only"] += 1
        if n_tool > 0:
            span_stats["full_tool"] += 1
        details.append({
            "n_tool_tokens": n_tool,
            "n_name": n_name,
            "n_key": n_key,
            "n_val": n_val,
            "response_len": len(resp_ids),
        })

    report = {
        "n_sampled": len(rows),
        "span_stats": span_stats,
        "parsable_rate": span_stats["parsable"] / max(1, len(rows)),
        "backend_audit": audit,
        "details_head": details[:32],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = out.parent / "MASK_AUDIT.md"
    md.write_text(
        f"# MASK_AUDIT\n\n"
        f"- n_sampled: {report['n_sampled']}\n"
        f"- parsable_rate: {report['parsable_rate']:.4f}\n"
        f"- span_stats: {json.dumps(span_stats)}\n"
        f"- backend_parsable_rate: {audit['parsable_rate']:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
