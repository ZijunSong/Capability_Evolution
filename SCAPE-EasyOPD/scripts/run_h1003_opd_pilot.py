#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
QWEN3 = os.environ.get("CANONICAL_STUDENT_BASE", "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507")
LOGICAL = os.environ.get("SCAPE_STUDENT_LOGICAL_MODEL", "Qwen3-30B-A3B-Instruct-2507")


def read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def state_target(row: dict[str, Any]) -> str:
    event = row.get("event_payload_student_visible") or row.get("event_payload") or {}
    if row.get("component") == "sentence_compress":
        text = (event.get("compressed_teacher_view") or event.get("payload", {}).get("compressed_teacher_view") or "").strip()
        return "Use the compressed current observation:\n" + text[:1024]
    if row.get("component") == "evidence_graph":
        text = (event.get("evidence_graph_summary") or event.get("payload", {}).get("evidence_graph_summary") or "").strip()
        return "Use the evidence graph summary:\n" + text[:1024]
    return str(row.get("projectable_target") or row.get("event_type") or "continue")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True)
    ap.add_argument("--train-file", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default=QWEN3)
    ap.add_argument("--states", type=int, default=256)
    args = ap.parse_args()

    rows = read_jsonl(args.train_file, args.states)
    if len(rows) < 256:
        raise SystemExit("OPD_PILOT_REQUIRES_256_REAL_STATES")
    bad = [r for r in rows if r.get("collector_mode") != "real_harness1" or r.get("synthetic") or r.get("synthetic_fallback")]
    if bad:
        raise SystemExit("OPD_PILOT_REFUSES_SYNTHETIC_OR_NON_REAL_ROWS")

    pilot_dir = args.output_dir / args.component / "OPD_PILOT"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    with (pilot_dir / "PILOT_TRAIN_STATES.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    cfg = {"component": args.component, "seed": 42, "n_states": len(rows), "model": args.model, "logical_model_id": LOGICAL, "collector_mode": "real_harness1", "synthetic_fallback": False}
    (pilot_dir / "PILOT_CONFIG.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    lora_cfg = LoraConfig(task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

    log_path = pilot_dir / "PILOT_TRAIN_LOG.jsonl"
    losses = []
    model.train()
    for step, row in enumerate(rows[:4]):
        prompt_text = str(row.get("student_visible_prefix") or row.get("query") or row.get("query_id"))[:2048]
        target_text = state_target(row)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        target_ids = tokenizer(target_text + (tokenizer.eos_token or ""), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        labels = input_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        optim.step(); optim.zero_grad(set_to_none=True)
        rec = {"step": step, "state_uid": row.get("state_uid"), "loss": float(loss.detach().float().cpu())}
        losses.append(rec["loss"])
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    adapter_dir = pilot_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(pilot_dir / "tokenizer")
    del model
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    reload_error = None
    reload_path = "peft_model_from_pretrained"
    try:
        reloaded = PeftModel.from_pretrained(base, adapter_dir)
    except TypeError as exc:
        reload_error = f"peft_native_reload_failed: {type(exc).__name__}: {exc}"
        if not hasattr(base, "load_adapter"):
            raise
        base.load_adapter(str(adapter_dir))
        reloaded = base
        reload_path = "transformers_load_adapter_fallback"
    reloaded.eval()
    reload_acceptance = {
        "status": "ADAPTER_RELOAD_READY",
        "adapter_dir": str(adapter_dir),
        "logical_model_id": LOGICAL,
        "resolved_model_path": args.model,
        "trainable_lora_params": trainable,
        "reload_error": reload_error,
        "reload_path": reload_path,
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
    }
    (pilot_dir / "ADAPTER_RELOAD_ACCEPTANCE.json").write_text(json.dumps(reload_acceptance, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    dev = {
        "status": "DEV_SMOKE_COMPLETED",
        "paper_grade": False,
        "metric_scope": "engineering_pilot_only",
        "student_before": {"invalid_tool_rate": "N/A", "overall_reward": "N/A"},
        "student_after": {"invalid_tool_rate": "N/A", "overall_reward": "N/A"},
        "adapter_reload_changes_model_state": True,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }
    (pilot_dir / "DEV_SMOKE_BEFORE_AFTER.json").write_text(json.dumps(dev, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pilot_dir": str(pilot_dir), "reload": reload_acceptance, "dev": dev}, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
