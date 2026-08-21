#!/usr/bin/env python3
"""Adapter-conditioned paired evaluation for token_budget_marker.

The base and adapter-conditioned models receive the same reduced Student-visible
query prompt. Each generated action is parsed and executed through the live
Harness-1 ToolSet. Results are paired by query_id for an explicit After-Before
reward delta.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("CANONICAL_STUDENT_BASE", "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
SOURCE_QUERIES = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl")
DEV_CONTRACT = Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_DEV.json")
TEST_CONTRACT = Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_TEST.json")
TOOLS = ("fan_out_search", "search_corpus", "grep_corpus", "read_document", "review_docs", "curate", "end_search")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def remap_state(raw: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in raw.items():
        out[key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")] = value
    return out


def load_model(adapter_dir: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": 0})
    reload_path = "base_no_adapter"
    if adapter_dir is not None:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        lora_cfg = LoraConfig(task_type=cfg.get("task_type", "CAUSAL_LM"), r=int(cfg.get("r", 8)), lora_alpha=int(cfg.get("lora_alpha", 16)), lora_dropout=float(cfg.get("lora_dropout", 0.05)), target_modules=list(cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]), bias=cfg.get("bias", "none"))
        try:
            model = PeftModel.from_pretrained(model, adapter_dir)
            reload_path = "peft_model_from_pretrained"
        except Exception:
            model = get_peft_model(model, lora_cfg)
            missing, unexpected = model.load_state_dict(remap_state(load_file(str(adapter_dir / "adapter_model.safetensors"))), strict=False)
            bad_missing = [x for x in missing if "lora_" in x]
            bad_unexpected = [x for x in unexpected if "lora_" in x]
            if bad_missing or bad_unexpected:
                raise RuntimeError(f"adapter reload mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
            reload_path = "manual_safetensors_state_dict"
    model.eval()
    return tokenizer, model, reload_path


def load_queries(contract: Path, source: Path, limit: int | None) -> list[dict[str, str]]:
    ids = [str(x) for x in json.loads(contract.read_text(encoding="utf-8"))["query_ids"]]
    wanted = set(ids)
    found = {}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                qid = str(row.get("query_id", ""))
                if qid in wanted:
                    found[qid] = {"query_id": qid, "query": str(row.get("query") or row.get("question") or qid)}
    rows = [found[x] for x in ids if x in found]
    return rows[:limit] if limit is not None else rows


def prompt(query_id: str, query: str) -> str:
    payload = {
        "component": "token_budget_marker",
        "query_id": query_id,
        "student_inference_privilege": False,
        "student_observable_env_state": {"curated_count": 0, "curated_ids": [], "pool_size": 0, "search_history": [], "visible_doc_ids": []},
        "task": "Choose the next legal Harness-1 tool call. Return exactly one tool call.",
        "query": query,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def generate(tokenizer, model, text: str, max_new_tokens: int) -> str:
    messages = [{"role": "system", "content": "You are a SCAPE research agent. Return exactly one legal Harness-1 tool call."}, {"role": "user", "content": text}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, dict) or hasattr(encoded, "__getitem__"):
        ids = encoded["input_ids"]
    else:
        ids = encoded
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=False)


def parse_action(text: str) -> dict[str, Any]:
    raw = text or ""
    name = next((tool for tool in TOOLS if re.search(rf"(?:to=|<tool_call>|\\b){re.escape(tool)}\\b", raw)), None)
    params = {}
    match = re.search(r"\{.*?\}(?:<\\|im_end\\|>)?", raw, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0).replace("<|im_end|>", ""))
            if isinstance(obj, dict):
                params = obj
        except json.JSONDecodeError:
            pass
    # Qwen3 may emit the compact action schema used by the native chat prompt.
    compact = params.get("tool") if isinstance(params, dict) else None
    if name is None and compact:
        name = {"search": "search_corpus", "grep": "grep_corpus", "read": "read_document", "review": "review_docs", "curate": "curate", "end": "end_search"}.get(str(compact))
    if name == "search_corpus" and "query" not in params and isinstance(params.get("q"), str):
        params["query"] = params["q"]
    if name is not None:
        params.pop("tool", None)
    executable = False
    if name in {"search_corpus", "fan_out_search", "grep_corpus"}:
        executable = isinstance(params.get("query"), str) and bool(params["query"].strip())
    elif name in {"read_document", "review_docs"}:
        executable = bool(params.get("doc_id") or params.get("doc_ids"))
    elif name == "curate":
        executable = bool(params.get("add_ids") or params.get("remove_ids") or params.get("doc_ids"))
    elif name == "end_search":
        executable = True
    return {"tool_name": name, "params": params, "legal": name is not None, "executable": executable}


def run_tool(action: dict[str, Any], row: dict[str, str], out: Path) -> dict[str, Any]:
    try:
        import sys
        sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/SCAPE")
        from harness.config import get_config
        from harness.tools import ToolSet
        out.mkdir(parents=True, exist_ok=True)
        corpus = out / "eval_corpus.jsonl"
        if not corpus.exists():
            corpus.write_text("\n".join(json.dumps({"id": f"doc{i}", "source": f"doc{i}", "text": f"Evidence for query {row['query_id']}: {row['query']}"}, ensure_ascii=False) for i in range(3)) + "\n", encoding="utf-8")
        os.environ["SCAPE_RETRIEVAL_CORPUS"] = str(corpus)
        os.environ["SCAPE_CHROMA_PATH"] = str(out / "empty_chroma")
        os.environ["SCAPE_LOCAL_OPENAI_EMBEDDINGS"] = "1"
        os.environ["SCAPE_FORCE_LOCAL_HARMONY"] = "1"
        toolset = ToolSet.from_config(get_config(), chroma_collection_name="scape_token_budget_adapter_eval", search_display_limit=3, search_limit=3, search_knn_limit=3, snippet_max_chars=160)
        tool = toolset.get_tool(action["tool_name"]) if action.get("tool_name") else None
        if tool is None or not action["executable"]:
            return {"executed": False, "error": None}
        tool(action["params"])
        return {"executed": True, "error": None}
    except Exception as exc:
        return {"executed": False, "error": repr(exc)}


def evaluate(model, tokenizer, rows: list[dict[str, str]], split: str, out: Path, max_new_tokens: int) -> dict[str, Any]:
    records = []
    for idx, row in enumerate(rows):
        generated = generate(tokenizer, model, prompt(row["query_id"], row["query"]), max_new_tokens)
        action = parse_action(generated)
        live = run_tool(action, row, out / "live" / row["query_id"])
        reward = 0.25 * float(action["legal"]) + 0.25 * float(action["executable"]) + 0.25 * float(live["executed"])
        records.append({"query_id": row["query_id"], "split": split, "generated_text": generated, **action, **live, "overall_reward": reward, "student_inference_has_privilege": False, "adapter_conditioned_generation": True, "route_proxy": False})
        if (idx + 1) % 32 == 0:
            print(json.dumps({"split": split, "completed": idx + 1, "n": len(rows)}), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{split}_PER_QUERY.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    rewards = [x["overall_reward"] for x in records]
    summary = {"split": split, "n_queries": len(records), "overall_reward": sum(rewards) / max(1, len(rewards)), "legal_tool_rate": sum(x["legal"] for x in records) / max(1, len(records)), "executable_tool_rate": sum(x["executable"] for x in records) / max(1, len(records)), "live_tool_execution_rate": sum(x["executed"] for x in records) / max(1, len(records)), "invalid_tool_rate": sum(not x["legal"] for x in records) / max(1, len(records))}
    write_json(out / f"{split}_SUMMARY.json", summary)
    return summary


def paired(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    b = {x["query_id"]: x for x in before["records"]}
    a = {x["query_id"]: x for x in after["records"]}
    ids = sorted(set(b) & set(a))
    deltas = [a[x]["overall_reward"] - b[x]["overall_reward"] for x in ids]
    return {"n_pairs": len(ids), "delta_mean": sum(deltas) / max(1, len(deltas)), "positive": sum(x > 0 for x in deltas), "negative": sum(x < 0 for x in deltas), "zero": sum(x == 0 for x in deltas)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-name", required=True)
    ap.add_argument("--adapter-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--condition", choices=["before", "after"], required=True)
    ap.add_argument("--limit-dev", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()
    if args.condition == "after" and args.adapter_dir is None:
        ap.error("--adapter-dir is required for after")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    tokenizer, model, reload_path = load_model(args.adapter_dir if args.condition == "after" else None)
    dev = load_queries(DEV_CONTRACT, SOURCE_QUERIES, args.limit_dev)
    test = load_queries(TEST_CONTRACT, SOURCE_QUERIES, args.limit_test)
    root = args.output_dir / args.cell_name / args.condition
    dev_summary = evaluate(model, tokenizer, dev, "dev", root / "dev", args.max_new_tokens)
    test_summary = evaluate(model, tokenizer, test, "test", root / "test", args.max_new_tokens)
    payload = {"status": "ADAPTER_CONDITIONED_EVAL_READY", "cell_name": args.cell_name, "condition": args.condition, "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None, "reload_path": reload_path, "canonical_student_base": BASE_MODEL, "student_inference_privilege": False, "adapter_conditioned_generation": True, "dev": dev_summary, "test": test_summary}
    write_json(root / "ADAPTER_CONDITIONED_EVAL.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
