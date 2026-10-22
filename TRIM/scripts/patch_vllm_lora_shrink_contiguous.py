# Runtime patch injected after dependency installation. This converts the exact
# vLLM 0.19.1 LoRA kernel precondition failure into its required contiguous input.
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/lora/ops/triton_ops/lora_shrink_op.py")
source = path.read_text(encoding="utf-8")
old = "assert inputs.is_contiguous()"
new = "inputs = inputs.contiguous()"
if old not in source:
    raise SystemExit(f"expected vLLM LoRA assertion missing: {path}")
path.write_text(source.replace(old, new, 1), encoding="utf-8")
print("PATCHED_VLLM_LORA_SHRINK_CONTIGUOUS", path, flush=True)
