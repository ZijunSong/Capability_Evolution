#!/usr/bin/env python3
"""Actual-model real multi-step closed-loop eval (LOCAL_COMPAT_ONLY doc_store executor)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.eval.harmony_runtime import (
    build_continuation_prompt_ids,
    build_first_turn_prompt_ids,
    generate_tool_turn,
    load_harmony_enc,
    make_action,
    make_observation,
)
from scape.eval.local_search_env import curated_recall, execute_tool, new_state, wm_text
from scape.training.clean_sft import load_causal_lm


def _unwrap(rec: dict[str, Any]) -> dict[str, Any]:
    payload = rec.get("payload_json")
    if isinstance(payload, str) and payload.strip().startswith("{"):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict):
        return {**rec, **payload}
    return rec


def load_queries(raw: Path, ids: list[str]) -> list[dict[str, Any]]:
    want = set(ids)
    found: dict[str, dict[str, Any]] = {}
    with raw.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = _unwrap(json.loads(line))
            qid = str(rec.get("query_id") or "")
            if qid in want and qid not in found:
                found[qid] = rec
            if len(found) >= len(want):
                break
    return [found[i] for i in ids if i in found]


def gold_ids(rec: dict[str, Any]) -> list[str]:
    g = rec.get("ground_truth_ids") or rec.get("document_ids_json") or []
    if isinstance(g, str):
        try:
            g = json.loads(g)
        except json.JSONDecodeError:
            g = []
    return [str(x) for x in g] if isinstance(g, list) else []


def run_episode(model, enc, rec: dict[str, Any], *, max_steps: int, max_new_tokens: int) -> dict[str, Any]:
    qtext = str(rec.get("query_text") or rec.get("query") or "")
    store = rec.get("doc_store") or {}
    gold = gold_ids(rec)
    st = new_state(qtext, store)
    acts: list[tuple[Any, Any]] = []
    tools = []
    t0 = time.time()
    for step in range(max_steps):
        if step == 0:
            ids = build_first_turn_prompt_ids(qtext, enc=enc)
        else:
            ids = build_continuation_prompt_ids(
                qtext, actions_obs=acts, wm_text=wm_text(st, auto_on=False), enc=enc
            )
        gen = generate_tool_turn(model, ids, max_new_tokens=max_new_tokens, enc=enc)
        p = gen["parsed"]
        name = p.get("tool_name")
        args = p.get("arguments") or {}
        tools.append(
            {
                "step": step,
                "name": name,
                "legal": bool(p.get("legal")),
                "args": args,
                "n_tokens": gen.get("n_tokens"),
                "termination": gen.get("termination"),
                "text_head": (gen["text"] or "")[:400],
            }
        )
        st, obs, _ok = execute_tool(st, name, args)
        try:
            acts.append((make_action(name or "search_corpus", args or {}), make_observation(obs)))
        except Exception:
            break
        if st.get("ended"):
            break
    rec_v = curated_recall(st, gold)
    n_invalid = int(st.get("invalid_tools") or 0)
    return {
        "query_id": rec.get("query_id"),
        "n_steps": len(tools),
        "tools": tools,
        "tool_names": [t["name"] for t in tools],
        "n_tool_calls": int(st.get("n_tool_calls") or 0),
        "n_search_calls": int(st.get("n_search_calls") or 0),
        "invalid_tool_rate": n_invalid / max(1, int(st.get("n_tool_calls") or 1)),
        "termination": "end_search" if st.get("ended") else "max_steps",
        "end_reason": st.get("end_reason"),
        "n_curated": len(st.get("curated") or {}),
        "evidence_qrel_recall": rec_v,
        "external_task_reward": rec_v if rec_v is not None else "N/A",
        "final_answer": "N/A",
        "final_answer_gold_available": bool(rec.get("answer")),
        "student_inference_privilege": False,
        "wall_s": time.time() - t0,
        "state_mutated": bool(st.get("pool") or st.get("curated") or st.get("step", 0) > 0),
        "multi_step": len(tools) > 1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-model", default="/data/ppnm/models/gpt-oss-20b")
    ap.add_argument("--raw-jsonl", type=Path, required=True)
    ap.add_argument("--query-ids-json", type=Path, required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--parent-adapter", default=None)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    blob = json.loads(args.query_ids_json.read_text())
    if isinstance(blob, dict):
        ids = blob.get("query_ids") or blob.get("ids") or blob.get("real_dev_query_ids") or blob.get("real_test_query_ids") or []
    else:
        ids = blob
    if args.limit:
        ids = ids[: args.limit]
    recs = load_queries(args.raw_jsonl, ids)
    enc = load_harmony_enc()
    _, model = load_causal_lm(
        args.model_path,
        device_map=f"cuda:{args.gpu}",
        base_model=args.base_model,
        parent_adapter=args.parent_adapter,
    )
    cases = out / "cases.jsonl"
    if cases.exists():
        cases.unlink()
    rows = []
    t0 = time.time()
    for i, rec in enumerate(recs):
        ep = run_episode(model, enc, rec, max_steps=args.max_steps, max_new_tokens=args.max_new_tokens)
        ep["tag"] = args.tag
        rows.append(ep)
        with cases.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        (out / "progress.json").write_text(
            json.dumps({"i": i + 1, "n": len(recs), "elapsed_s": time.time() - t0}, indent=2) + "\n"
        )
    n = max(1, len(rows))
    recalls = [r["evidence_qrel_recall"] for r in rows if isinstance(r["evidence_qrel_recall"], float)]
    summary = {
        "tag": args.tag,
        "model_path": args.model_path,
        "n": len(rows),
        "max_steps": args.max_steps,
        "mean_evidence_qrel_recall": (sum(recalls) / len(recalls)) if recalls else "N/A",
        "n_with_qrel": len(recalls),
        "mean_tool_calls": sum(r["n_tool_calls"] for r in rows) / n,
        "mean_search_calls": sum(r["n_search_calls"] for r in rows) / n,
        "mean_invalid_tool_rate": sum(r["invalid_tool_rate"] for r in rows) / n,
        "termination": dict(Counter(r["termination"] for r in rows)),
        "mean_wall_s": sum(r["wall_s"] for r in rows) / n,
        "final_answer": "N/A",
        "student_inference_privilege": False,
        "LOCAL_COMPAT_ONLY": True,
        "parent_adapter": args.parent_adapter,
        "max_new_tokens": args.max_new_tokens,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "DONE").write_text("ok\n")
    print(json.dumps(summary, indent=2))
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
