#!/usr/bin/env python3
"""One-time: dequantize GPT-OSS-20B mxfp4 -> bf16 checkpoint, save to JuiceFS.

Loads the mxfp4-quantized model via HF Transformers (which dequantizes mxfp4
weights to bf16 on load), then save_pretrained to a clean bf16 directory that
vLLM can load WITHOUT the mxfp4 Marlin LoRA path (avoiding the gm402 PTX crash).

Usage: python dequant_gpt_oss_mxfp4_to_bf16.py <mxfp4_src> <bf16_dst>
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    src = sys.argv[1]
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[dequant] loading mxfp4 model from {src} (dtype=bfloat16, device_map=cuda:0)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        src,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    # Force-drop any residual quantization_config so vLLM treats this as plain bf16.
    if getattr(model.config, "quantization_config", None):
        print(f"[dequant] dropping quantization_config: {model.config.quantization_config}", flush=True)
        model.config.quantization_config = None
        # Some configs store it under a different attr; clear all quant hints.
        for attr in ("quantization", "quant_method"):
            if hasattr(model.config, attr):
                setattr(model.config, attr, None)

    print(f"[dequant] saving bf16 checkpoint to {dst}", flush=True)
    model.save_pretrained(str(dst), safe_serialization=True)
    del model

    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    tokenizer.save_pretrained(str(dst))

    # Verify saved config has no quantization_config.
    saved_cfg = json.loads((dst / "config.json").read_text())
    q = saved_cfg.get("quantization_config")
    print(f"[dequant] saved config quantization_config: {q}", flush=True)
    assert q is None or q == {}, f"FAIL: saved config still has quantization_config={q}"

    # List saved files + total size.
    files = sorted(dst.glob("*.safetensors"))
    total = sum(f.stat().st_size for f in files)
    print(f"[dequant] OK: {len(files)} safetensors, {total / 1e9:.1f} GB, dst={dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
