#!/usr/bin/env python3
"""n=128 first-turn Harmony tool-channel eval for a clean gpt-oss checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.eval.harmony_runtime import (  # noqa: E402
    CANONICAL_TOOLS,
    build_first_turn_prompt_ids,
    generate_tool_turn,
    load_harmony_enc,
)
from scape.training.clean_sft import load_causal_lm  # noqa: E402


SEARCH_TOOLS = {"fan_out_search", "search_corpus", "grep_corpus"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model", default="/data/ppnm/models/gpt-oss-20b")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads(args.manifest.read_text(encoding="utf-8"))
    queries = man["queries"] if isinstance(man, dict) and "queries" in man else man
    if args.limit and args.limit > 0:
        queries = queries[: args.limit]

    enc = load_harmony_enc()
    tok, model = load_causal_lm(
        args.model_path,
        device_map=f"cuda:{args.gpu}",
        base_model=args.base_model,
    )
    del tok  # Harmony IDs are the SFT contract

    rows = []
    t0 = time.time()
    jsonl_path = out / "generations.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    for i, q in enumerate(queries):
        qid = q.get("query_id") or f"q{i}"
        qtext = q["query_text"]
        prompt_ids = build_first_turn_prompt_ids(qtext, enc=enc)
        gen = generate_tool_turn(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            enc=enc,
        )
        p = gen["parsed"]
        name = p.get("tool_name")
        row = {
            "idx": i,
            "query_id": qid,
            "dataset": q.get("dataset"),
            "n_prompt_tokens": len(prompt_ids),
            "n_generated_tokens": gen["n_tokens"],
            "termination": gen["termination"],
            "tool_name": name,
            "parsed": bool(p.get("parsed")),
            "legal": bool(p.get("legal") and name in CANONICAL_TOOLS),
            "schema_legal": bool(p.get("legal")),
            "arguments": p.get("arguments"),
            "parse_method": p.get("parse_method"),
            "parse_error": p.get("error"),
            "text_head": (gen["text"] or "")[:1200],
        }
        rows.append(row)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(queries) - i - 1)
        (out / "progress.json").write_text(
            json.dumps(
                {
                    "tag": args.tag,
                    "i": i + 1,
                    "n": len(queries),
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                    "last_tool": name,
                    "last_parsed": row["parsed"],
                },
                indent=2,
            )
            + "\n"
        )

    n = max(1, len(rows))
    n_parse = sum(1 for r in rows if r["parsed"])
    n_legal = sum(1 for r in rows if r["legal"])
    hist = Counter(r["tool_name"] or "NONE" for r in rows)
    first = Counter(r["tool_name"] or "NONE" for r in rows)
    term = Counter(r["termination"] for r in rows)
    coverage = {t: hist.get(t, 0) for t in CANONICAL_TOOLS}
    summary = {
        "tag": args.tag,
        "model_path": args.model_path,
        "n": len(rows),
        "tool_parse_rate": n_parse / n,
        "legal_tool_rate": n_legal / n,
        "invalid_tool_rate": 1.0 - (n_legal / n),
        "tool_name_histogram": dict(hist),
        "first_action_distribution": dict(first),
        "search_read_curate_verify_end_coverage": {
            "search": sum(hist[t] for t in SEARCH_TOOLS),
            "read_document": hist.get("read_document", 0),
            "review_docs": hist.get("review_docs", 0),
            "curate": hist.get("curate", 0),
            "verify": hist.get("verify", 0),
            "end_search": hist.get("end_search", 0),
        },
        "tool_coverage": coverage,
        "non_degenerate_tool_coverage": (
            sum(hist[t] for t in SEARCH_TOOLS) >= 1
            and len([t for t in CANONICAL_TOOLS if hist.get(t, 0) > 0]) >= 2
        ),
        "mean_generated_tokens": sum(r["n_generated_tokens"] for r in rows) / n,
        "termination_reason": dict(term),
        "gate": {
            "parse_rate": n_parse / n,
            "legal_tool_rate": n_legal / n,
            "invalid_tool_rate": 1.0 - (n_legal / n),
            "pass": (n_parse / n) >= 0.99
            and (n_legal / n) >= 0.99
            and (1.0 - n_legal / n) <= 0.01,
        },
        "elapsed_s": time.time() - t0,
        "LOCAL_COMPAT_ONLY": True,
        "legacy_scope_path_used": False,
        "student_inference_privilege": False,
        "harmony_contract": "build_context+render_conversation_for_completion",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "DONE").write_text("ok\n")
    print(json.dumps({k: summary[k] for k in summary if k != "tool_coverage"}, indent=2))
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
