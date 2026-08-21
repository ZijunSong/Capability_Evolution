#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scape.training.hf_tool_opd import ScapeHFToolOPD, run_tool_opd_train

DEFAULT_SRC = REPO / "outputs" / "h100_2_structured_privilege_formal_0816"
TOOLS = ["fan_out_search", "search_corpus", "grep_corpus", "read_document", "review_docs", "curate", "verify", "end_search"]


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def teacher_tool(row: dict[str, Any], *, variant: str = "matched_text") -> str:
    full = {t: max(float((row.get("P_tool_name_full") or {}).get(t, 0.0)), 1e-12) for t in TOOLS}
    if variant == "structured_residual":
        reduced = {t: max(float((row.get("P_tool_name_reduced") or {}).get(t, 0.0)), 1e-12) for t in TOOLS}
        residual = {t: math.log(full[t]) - math.log(reduced[t]) for t in TOOLS}
        return max(TOOLS, key=lambda t: residual[t])
    return max(TOOLS, key=lambda t: full[t])


def action_for(tool: str, row: dict[str, Any]) -> dict[str, Any]:
    qid = str(row.get("query_id"))
    q = f"BrowseComp query {qid} evidence"
    if tool == "fan_out_search":
        return {"tool_name": tool, "parameters": {"queries": [q, f"{q} source", f"{q} answer"]}}
    if tool == "search_corpus":
        return {"tool_name": tool, "parameters": {"query": q}}
    if tool == "grep_corpus":
        return {"tool_name": tool, "parameters": {"pattern": "evidence"}}
    if tool == "read_document":
        return {"tool_name": tool, "parameters": {"doc_id": "DUMMY_DOC_0"}}
    if tool == "review_docs":
        return {"tool_name": tool, "parameters": {"doc_ids": ["DUMMY_DOC_0", "DUMMY_DOC_1"]}}
    if tool == "curate":
        return {"tool_name": tool, "parameters": {"add_ids": ["DUMMY_DOC_0"], "remove_ids": []}}
    if tool == "verify":
        return {"tool_name": tool, "parameters": {"doc_ids": ["DUMMY_DOC_0"], "claim": q[:120]}}
    return {"tool_name": "end_search", "parameters": {}}


def make_prompt(row: dict[str, Any], *, variant: str, full: bool) -> str:
    base = {
        "task": "Choose the next BrowseComp tool call as JSON.",
        "query_id": str(row.get("query_id")),
        "step": row.get("step", 0),
        "available_tools": TOOLS,
        "reduced_runtime": {
            "snapshot_hash": row.get("snapshot_hash"),
            "component_id": row.get("component_id"),
        },
    }
    if full:
        if variant == "matched_text":
            base["matched_text_privilege"] = row.get("textual_privilege")
        elif variant in {"structured", "structured_typed_v2"}:
            base["structured_privilege"] = row.get("information_fields")
            base["typed_adapter_schema"] = {
                "boolean": ["auto_seed_present", "first_search_pending", "component_enabled_full", "component_enabled_student"],
                "categorical": ["component", "teacher_tool"],
                "scalar": ["step", "prior_search_count", "tool_history_len", "document_count", "importance_high_count"],
            }
        elif variant == "structured_residual":
            full = {t: max(float((row.get("P_tool_name_full") or {}).get(t, 0.0)), 1e-12) for t in TOOLS}
            reduced = {t: max(float((row.get("P_tool_name_reduced") or {}).get(t, 0.0)), 1e-12) for t in TOOLS}
            base["structured_privilege"] = row.get("information_fields")
            base["control_residual_log_full_minus_log_reduced"] = {t: math.log(full[t]) - math.log(reduced[t]) for t in TOOLS}
        elif variant == "ophsd":
            base["whole_harness_context"] = {
                "information_fields": row.get("information_fields"),
                "P_tool_name_full": row.get("P_tool_name_full"),
                "P_tool_name_reduced": row.get("P_tool_name_reduced"),
            }
        else:
            raise ValueError(variant)
    return json.dumps(base, ensure_ascii=False, sort_keys=True)


def convert(rows: list[dict[str, Any]], *, variant: str) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        tool = teacher_tool(row, variant=variant)
        action = action_for(tool, row)
        response = f"to={action['tool_name']}\n{json.dumps(action.get('parameters') or {}, ensure_ascii=False, sort_keys=True)}\n</tool_call>"
        out.append({
            "row_id": f"{variant}_{i:06d}",
            "query_id": str(row.get("query_id")),
            "snapshot_hash": row.get("snapshot_hash"),
            "variant": variant,
            "teacher_tool": tool,
            "prompt_reduced": make_prompt(row, variant=variant, full=False),
            "prompt_full": make_prompt(row, variant=variant, full=True),
            "response_text": response,
            "source_component_id": row.get("component_id"),
            "source_contract": "route_distribution_to_tool_response_bridge_0816_2",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["matched_text", "structured", "structured_residual", "structured_typed_v2", "ophsd"], required=True)
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "0816_2_actual_lora_bridge")
    ap.add_argument("--model", default="/mnt/songzijun/models/pat-jj_harness-1-full/harness-1")
    ap.add_argument("--train-limit", type=int, default=64)
    ap.add_argument("--eval-limit", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--loss-path", default="tool_token_kl")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = args.out_dir / args.variant / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    train_src = load_jsonl(args.src_dir / "train_auto_paired.jsonl", args.train_limit)
    eval_src = load_jsonl(args.src_dir / "valid_auto_paired.jsonl", args.eval_limit)
    train_rows = convert(train_src, variant=args.variant)
    eval_rows = convert(eval_src, variant=args.variant)
    write_jsonl(out / "TRAIN_ROWS.jsonl", train_rows)
    write_jsonl(out / "EVAL_ROWS.jsonl", eval_rows)

    backend = ScapeHFToolOPD(model_path=args.model, device_map=args.device_map, learning_rate=1e-5, anchor_weight=0.05, use_lora=True, lora_r=8, lora_alpha=16)
    span_audit = backend.audit_tool_spans([r["response_text"] for r in train_rows[: min(32, len(train_rows))]])
    result = run_tool_opd_train(backend, train_rows, eval_rows, loss_path=args.loss_path, epochs=args.epochs, batch_size=args.batch_size)
    adapter = out / "checkpoint"
    merged = out / "merged"
    backend.save_pretrained(str(adapter))
    try:
        backend.merge_and_save(str(merged))
        merged_ok = True
    except Exception as exc:
        merged_ok = False
        (out / "MERGE_ERROR.txt").write_text(str(exc), encoding="utf-8")
    summary = {
        "status": "completed_actual_lora_bridge_smoke",
        "variant": args.variant,
        "seed": args.seed,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "loss_path": args.loss_path,
        "span_audit": span_audit,
        "training_result": result,
        "adapter_path": str(adapter),
        "merged_path": str(merged) if merged_ok else None,
        "actual_model_weights": True,
        "student_inference_privilege": False,
        "contract_caveat": "Bridge converts route-distribution teacher argmax into canonical executable tool-call text; use as actual-LoRA smoke/dev, not final paper-grade route distribution claim without recollected prompt/response teacher rows.",
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "SUMMARY.md").write_text("\n".join([
        f"# {args.variant} actual LoRA bridge smoke",
        "",
        f"- status: `{summary['status']}`",
        f"- train_rows: {len(train_rows)}",
        f"- eval_rows: {len(eval_rows)}",
        f"- D_pre: {result['D_pre']}",
        f"- D_post: {result['D_post']}",
        f"- L_m: {result['L_m']}",
        f"- adapter_path: `{adapter}`",
        f"- merged_path: `{summary['merged_path']}`",
        f"- caveat: {summary['contract_caveat']}",
    ]) + "\n", encoding="utf-8")
    with (out / "TRAINING.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["variant", "seed", "train_rows", "eval_rows", "loss_path", "D_pre", "D_post", "L_m", "adapter_path", "merged_path"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({
            "variant": args.variant,
            "seed": args.seed,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "loss_path": args.loss_path,
            "D_pre": result["D_pre"],
            "D_post": result["D_post"],
            "L_m": result["L_m"],
            "adapter_path": str(adapter),
            "merged_path": summary["merged_path"],
        })
    subprocess.run("find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS", cwd=out, shell=True, check=True)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
