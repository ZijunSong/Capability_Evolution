#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("CANONICAL_STUDENT_BASE", "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
LOGICAL_MODEL_ID = os.environ.get("SCAPE_STUDENT_LOGICAL_MODEL", "Qwen3-30B-A3B-Instruct-2507")
DEFAULT_SOURCE_QUERIES = Path("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl")
DEFAULT_DEV_CONTRACT = Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_DEV.json")
DEFAULT_TEST_CONTRACT = Path("/mnt/songzijun/Capability_Evolution/SCAPE/manifests/component_sweep_5k/COMPONENT_SWEEP_TEST.json")

TOOL_NAMES = [
    "fan_out_search",
    "search_corpus",
    "grep_corpus",
    "read_document",
    "review_docs",
    "curate",
    "end_search",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def remap_lora_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    remapped = {}
    for key, value in raw_state.items():
        if key.endswith(".lora_A.weight"):
            remapped[key.replace(".lora_A.weight", ".lora_A.default.weight")] = value
        elif key.endswith(".lora_B.weight"):
            remapped[key.replace(".lora_B.weight", ".lora_B.default.weight")] = value
        else:
            remapped[key] = value
    return remapped


def build_lora_config(adapter_dir: Path) -> LoraConfig:
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    return LoraConfig(
        task_type=cfg.get("task_type", "CAUSAL_LM"),
        r=int(cfg.get("r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=list(cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias=cfg.get("bias", "none"),
    )


def load_model(adapter_dir: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    reload_path = "base_no_adapter"
    if adapter_dir is not None:
        try:
            model = PeftModel.from_pretrained(model, adapter_dir)
            reload_path = "peft_model_from_pretrained"
        except Exception:
            lora_cfg = build_lora_config(adapter_dir)
            model = get_peft_model(model, lora_cfg)
            raw_state = load_file(str(adapter_dir / "adapter_model.safetensors"))
            missing, unexpected = model.load_state_dict(remap_lora_state_dict(raw_state), strict=False)
            bad_missing = [key for key in missing if "lora_" in key]
            bad_unexpected = [key for key in unexpected if "lora_" in key]
            if bad_missing or bad_unexpected:
                raise RuntimeError(f"manual adapter reload mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
            reload_path = "manual_safetensors_state_dict"
    model.eval()
    return tokenizer, model, reload_path


def load_query_manifest(contract_path: Path, source_queries: Path, limit: int | None = None) -> list[dict[str, str]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    qids = [str(q) for q in contract.get("query_ids", [])]
    wanted = set(qids)
    rows_by_id: dict[str, dict[str, str]] = {}
    with source_queries.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("query_id", ""))
            if qid in wanted:
                rows_by_id[qid] = {"query_id": qid, "query": str(row.get("query") or row.get("question") or qid)}
    rows = [rows_by_id[qid] for qid in qids if qid in rows_by_id]
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_prompt_reduced(query_id: str, query: str, tool_history: list[dict[str, Any]] | None = None) -> str:
    tool_history = tool_history or []
    prefix = (
        "== Working Memory (summarizing turns 0-0) ==\n"
        f"Query: \"{query}\"\n\n"
        "Curated Set (0/30):\n  (empty -- use curate tool to add relevant docs)\n\n"
        "Document Pool: 0 docs total, 0 uncurated\n\n"
        "Search History: (no searches yet)\n\n"
        "Use review_docs(doc_ids) to re-read any document from your pool."
    )
    payload = {
        "component": "adaptive_rerank_instruction",
        "query_id": query_id,
        "student_inference_privilege": False,
        "student_observable_env_state": {
            "curated_count": 0,
            "curated_ids": [],
            "pool_size": 0,
            "search_history": [],
            "visible_doc_ids": [],
        },
        "student_visible_prefix": prefix,
        "task": "Choose the next legal Harness-1 tool call.",
        "tool_history": tool_history,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def generate_action(tokenizer, model, prompt: str, *, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "You are a SCAPE research agent. Return exactly one legal Harness-1 tool call."},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(rendered, dict):
        ids = rendered["input_ids"]
    elif hasattr(rendered, "input_ids"):
        ids = rendered.input_ids
    else:
        ids = rendered
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, ids.shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=False)


def parse_action(text: str) -> dict[str, Any]:
    raw = text or ""
    name = None
    for tool in TOOL_NAMES:
        if re.search(rf"(?:to=|<tool_call>|\b){re.escape(tool)}\b", raw):
            name = tool
            break
    params: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", raw, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                params = obj
        except Exception:
            params = {}
    legal = name is not None
    executable = False
    if name in {"search_corpus", "fan_out_search", "grep_corpus"}:
        executable = isinstance(params.get("query"), str) and bool(params.get("query", "").strip())
    elif name in {"read_document", "review_docs"}:
        executable = bool(params.get("doc_id") or params.get("doc_ids"))
    elif name == "curate":
        executable = bool(params.get("add_ids") or params.get("remove_ids") or params.get("doc_ids"))
    elif name == "end_search":
        executable = True
    adaptive_marker = "focus on specific entities" in raw or "multi-constraint" in raw or "direct multi-constraint evidence" in raw
    return {"tool_name": name, "params": params, "legal": legal, "executable": executable, "adaptive_marker": adaptive_marker}


def run_live_probe(action: dict[str, Any], query_id: str, query: str, output_dir: Path) -> dict[str, Any]:
    """Execute one generated first action against real Harness-1 tools when possible."""
    try:
        from harness.config import get_config
        from harness.tools import ToolSet

        output_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = output_dir / "adapter_conditioned_eval_corpus.jsonl"
        if not corpus_path.exists():
            docs = [
                {"id": "doc0", "source": "doc0", "text": f"SCAPE evaluation evidence for query {query_id}: {query}"},
                {"id": "doc1", "source": "doc1", "text": "Additional retrieval evidence for adaptive rerank instruction."},
                {"id": "doc2", "source": "doc2", "text": "Distractor document for Harness-1 adapter-conditioned evaluation."},
            ]
            with corpus_path.open("w", encoding="utf-8") as f:
                for row in docs:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.environ.setdefault("SCAPE_CHROMA_PATH", str(output_dir / "empty_chroma"))
        os.environ.setdefault("SCAPE_RETRIEVAL_CORPUS", str(corpus_path))
        os.environ.setdefault("SCAPE_LOCAL_OPENAI_EMBEDDINGS", "1")
        os.environ.setdefault("SCAPE_FORCE_LOCAL_HARMONY", "1")
        toolset = ToolSet.from_config(get_config(), chroma_collection_name="scape_h1002_adapter_conditioned_eval", search_display_limit=3, search_limit=3, search_knn_limit=3, snippet_max_chars=160)
        tool = toolset.get_tool(action["tool_name"]) if action.get("tool_name") else None
        if tool is None or not action.get("executable"):
            return {"executed": False, "error": None, "observation_preview": ""}
        result = tool(**action.get("params", {}))
        return {"executed": True, "error": None, "observation_preview": str(result)[:800]}
    except Exception as exc:  # noqa: BLE001
        return {"executed": False, "error": repr(exc), "observation_preview": ""}


def bootstrap_mean(values: list[float], *, seed: int = 20260819, n_boot: int = 1000) -> dict[str, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / max(1, len(sample)))
    means.sort()
    return {"mean": sum(values) / max(1, len(values)), "ci95_low": means[int(0.025 * (n_boot - 1))], "ci95_high": means[int(0.975 * (n_boot - 1))], "n_boot": n_boot}


def evaluate_split(tokenizer, model, rows: list[dict[str, str]], *, split: str, output_dir: Path, max_new_tokens: int) -> dict[str, Any]:
    per_query = []
    for idx, row in enumerate(rows):
        prompt = build_prompt_reduced(row["query_id"], row["query"])
        generated = generate_action(tokenizer, model, prompt, max_new_tokens=max_new_tokens)
        parsed = parse_action(generated)
        live = run_live_probe(parsed, row["query_id"], row["query"], output_dir / "live_probe" / row["query_id"])
        reward = 0.0
        if parsed["legal"]:
            reward += 0.25
        if parsed["executable"]:
            reward += 0.25
        if live["executed"]:
            reward += 0.25
        if parsed["adaptive_marker"]:
            reward += 0.25
        per_query.append({
            "query_id": row["query_id"],
            "split": split,
            "component": "adaptive_rerank_instruction",
            "generated_text": generated,
            "tool_name": parsed["tool_name"],
            "tool_params": parsed["params"],
            "legal_tool_call": parsed["legal"],
            "executable_tool_call": parsed["executable"],
            "live_tool_executed": live["executed"],
            "live_tool_error": live["error"],
            "adaptive_marker": parsed["adaptive_marker"],
            "overall_reward": reward,
            "student_inference_has_privilege": False,
            "adapter_conditioned_generation": True,
            "route_proxy": False,
        })
        if (idx + 1) % 32 == 0:
            print(json.dumps({"split": split, "completed": idx + 1, "n": len(rows)}, ensure_ascii=False), flush=True)
    (output_dir / f"{split}_PER_QUERY.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split}_PER_QUERY.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    rewards = [float(r["overall_reward"]) for r in per_query]
    summary = {
        "component": "adaptive_rerank_instruction",
        "split": split,
        "n_queries": len(per_query),
        "overall_reward": sum(rewards) / max(1, len(rewards)),
        "legal_tool_rate": sum(1 for r in per_query if r["legal_tool_call"]) / max(1, len(per_query)),
        "executable_tool_rate": sum(1 for r in per_query if r["executable_tool_call"]) / max(1, len(per_query)),
        "live_tool_execution_rate": sum(1 for r in per_query if r["live_tool_executed"]) / max(1, len(per_query)),
        "adaptive_marker_rate": sum(1 for r in per_query if r["adaptive_marker"]) / max(1, len(per_query)),
        "invalid_tool_rate": sum(1 for r in per_query if not r["legal_tool_call"]) / max(1, len(per_query)),
        "student_inference_has_privilege": False,
        "adapter_conditioned_generation": True,
        "real_harness1_live_tool_probe": True,
        "bootstrap_overall_reward": bootstrap_mean(rewards) if rewards else None,
    }
    write_json(output_dir / f"{split}_SUMMARY.json", summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-name", required=True)
    ap.add_argument("--condition", choices=["before", "after"], required=True)
    ap.add_argument("--adapter-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--source-queries", type=Path, default=DEFAULT_SOURCE_QUERIES)
    ap.add_argument("--dev-contract", type=Path, default=DEFAULT_DEV_CONTRACT)
    ap.add_argument("--test-contract", type=Path, default=DEFAULT_TEST_CONTRACT)
    ap.add_argument("--limit-dev", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.condition == "after" and args.adapter_dir is None:
        raise SystemExit("--adapter-dir is required for --condition after")
    out = args.output_dir / args.cell_name / args.condition
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model, reload_path = load_model(args.adapter_dir if args.condition == "after" else None)
    dev = load_query_manifest(args.dev_contract, args.source_queries, args.limit_dev)
    test = load_query_manifest(args.test_contract, args.source_queries, args.limit_test)
    dev_summary = evaluate_split(tokenizer, model, dev, split="dev", output_dir=out, max_new_tokens=args.max_new_tokens)
    test_summary = evaluate_split(tokenizer, model, test, split="test", output_dir=out, max_new_tokens=args.max_new_tokens)
    payload = {
        "status": "ADAPTER_CONDITIONED_EVAL_READY",
        "cell_name": args.cell_name,
        "condition": args.condition,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "reload_path": reload_path,
        "canonical_student_base": BASE_MODEL,
        "logical_model_id": LOGICAL_MODEL_ID,
        "student_inference_privilege": False,
        "adapter_conditioned_generation": True,
        "real_harness1_live_tool_probe": True,
        "dev": dev_summary,
        "test": test_summary,
    }
    write_json(out / "ADAPTER_CONDITIONED_EVAL.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
