#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOLS = {"fan_out_search", "search_corpus", "grep_corpus", "read_document", "review_docs", "curate", "verify", "end_search"}
ALIASES = {"search": "search_corpus", "fanout_search": "fan_out_search", "grep": "grep_corpus", "read": "read_document", "review": "review_docs", "finish": "end_search", "end": "end_search"}


def remap_lora_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
    return {
        key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): value
        for key, value in raw_state.items()
    }


def load_adapter(model, adapter_dir: Path):
    try:
        return PeftModel.from_pretrained(model, adapter_dir), "peft_model_from_pretrained"
    except Exception:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        lora_cfg = LoraConfig(
            task_type=cfg.get("task_type", "CAUSAL_LM"),
            r=int(cfg.get("r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=list(cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=cfg.get("bias", "none"),
        )
        model = get_peft_model(model, lora_cfg)
        raw_state = load_file(str(adapter_dir / "adapter_model.safetensors"))
        missing, unexpected = model.load_state_dict(remap_lora_state_dict(raw_state), strict=False)
        bad_missing = [key for key in missing if "lora_" in key]
        bad_unexpected = [key for key in unexpected if "lora_" in key]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"adapter reload mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
        return model, "manual_safetensors_state_dict"


def parse_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects = []
    for match in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def parse_action(text: str) -> dict[str, Any]:
    name = None
    for tool in TOOLS:
        if re.search(rf"(?:to=|<tool_call>|\b){re.escape(tool)}\b", text or ""):
            name = tool
            break
    objects = parse_json_objects(text)
    arguments = objects[-1] if objects else {}
    compact_tool = arguments.pop("tool", None) if isinstance(arguments, dict) else None
    if compact_tool:
        name = ALIASES.get(str(compact_tool), str(compact_tool))
    return normalize_action(name, arguments)


def normalize_action(name: str | None, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "curate":
        arguments = {
            "add_ids": sorted(str(value) for value in arguments.get("add_ids") or arguments.get("doc_ids") or []),
            "remove_ids": sorted(str(value) for value in arguments.get("remove_ids") or []),
        }
    return {"name": name, "arguments": arguments}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", choices=["teacher", "before", "after"], required=True)
    parser.add_argument("--cell-name", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.condition == "after" and args.adapter is None:
        parser.error("--adapter is required for condition=after")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    rows = []
    with args.rows.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    reload_path = "base_no_adapter"
    if args.condition == "after":
        model, reload_path = load_adapter(model, args.adapter)
    model.eval()

    records = []
    for index, row in enumerate(rows):
        prompt = row["prompt_full"] if args.condition == "teacher" else row["prompt_reduced"]
        messages = [
            {"role": "system", "content": "You are a SCAPE research agent. Return exactly one legal Harness-1 tool call as JSON."},
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(generated_ids[0, ids.shape[-1]:], skip_special_tokens=False)
        predicted = parse_action(generated)
        target = row.get("projectable_target") or {}
        normalized_target = normalize_action(target.get("name"), target.get("arguments") or {})
        records.append({
            "row_id": row.get("row_id"),
            "state_uid": row.get("state_uid"),
            "query_id": row.get("query_id"),
            "condition": args.condition,
            "cell_name": args.cell_name,
            "generated_text": generated,
            "predicted_action": predicted,
            "target_action": normalized_target,
            "legal_action": predicted["name"] in TOOLS,
            "exact_projected_target": predicted == normalized_target,
            "student_inference_privilege": False,
        })
        if (index + 1) % 25 == 0:
            print(json.dumps({"cell": args.cell_name, "done": index + 1, "n": len(rows)}), flush=True)

    summary = {
        "status": "completed",
        "component": "content_dedup",
        "cell_name": args.cell_name,
        "condition": args.condition,
        "n_rows": len(records),
        "legal_action_rate": sum(record["legal_action"] for record in records) / len(records),
        "exact_projected_target_rate": sum(record["exact_projected_target"] for record in records) / len(records),
        "adapter_dir": str(args.adapter) if args.adapter else None,
        "reload_path": reload_path,
        "student_inference_privilege": False,
        "metric_scope": "frozen_opd_valid_rows_action_level_internalization_diagnostic",
        "terminal_task_reward": None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
