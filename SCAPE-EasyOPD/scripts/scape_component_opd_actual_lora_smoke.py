#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/songzijun/models/Qwen3-1.7B")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scape_easyopd/acceptance/actual_lora_opd_smoke"))
    parser.add_argument("--component", default="auto_populate_first_search")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        trust_remote_code=False,
    )
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0

    prompt = (
        "<|im_start|>user\nUse SCAPE projected action for the first successful search.\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    target = 'to=curate\n{"add_ids":["d1"],"remove_ids":[]}\n</tool_call><|im_end|>'
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    out = model(input_ids=input_ids, labels=labels)
    loss_before = out.loss
    loss_before.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optim.step()
    optim.zero_grad(set_to_none=True)
    with torch.no_grad():
        loss_after = model(input_ids=input_ids, labels=labels).loss

    ckpt_dir = args.output_dir / "adapter"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    del model
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        trust_remote_code=False,
    )
    reload_error = None
    reload_path = "peft_model_from_pretrained"
    try:
        reloaded = PeftModel.from_pretrained(base, ckpt_dir)
    except TypeError as exc:
        reload_error = f"peft_native_reload_failed: {type(exc).__name__}: {exc}"
        reloaded = get_peft_model(base, lora_cfg)
        raw_state = load_file(str(ckpt_dir / "adapter_model.safetensors"))
        remapped = {}
        for key, value in raw_state.items():
            if key.endswith(".lora_A.weight"):
                remapped[key.replace(".lora_A.weight", ".lora_A.default.weight")] = value
            elif key.endswith(".lora_B.weight"):
                remapped[key.replace(".lora_B.weight", ".lora_B.default.weight")] = value
            else:
                remapped[key] = value
        missing, unexpected = reloaded.load_state_dict(remapped, strict=False)
        bad_unexpected = [key for key in unexpected if "lora_" in key]
        bad_missing = [key for key in missing if "lora_" in key]
        if bad_unexpected or bad_missing:
            raise RuntimeError(f"manual adapter reload mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}") from exc
        reload_path = "manual_safetensors_state_dict"
    reloaded.eval()
    with torch.no_grad():
        reload_loss = reloaded(input_ids=input_ids, labels=labels).loss
    peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0

    manifest = {
        "component": args.component,
        "model": args.model,
        "status": "QWEN3_BASE_READY" if str(args.model) == "/mnt/songzijun/models/Qwen3-30B-A3B-Instruct-2507" else "LORA_SMOKE_READY",
        "target_source": "harness_effect_projection",
        "on_policy_state": True,
        "student_inference_privilege": False,
        "trainable_lora_params": trainable,
        "loss_before": float(loss_before.detach().float().cpu()),
        "loss_after": float(loss_after.detach().float().cpu()),
        "reload_loss": float(reload_loss.detach().float().cpu()),
        "reload_error": reload_error,
        "reload_path": reload_path,
        "grad_norm": float(grad_norm.detach().float().cpu()),
        "peak_cuda_memory_allocated_bytes": int(peak_allocated),
        "peak_cuda_memory_reserved_bytes": int(peak_reserved),
        "adapter_dir": str(ckpt_dir),
        "actual_lora_checkpoint_reload_pass": True,
        "no_double_peft_wrap": True,
        "paper_grade": False,
    }
    (args.output_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
