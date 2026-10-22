#!/usr/bin/env bash
# Diag: verify triton_kernels path for GPT-OSS-20B mxfp4 + LoRA on gm402 sm_90.
# Goal: confirm code-level fix (pip install triton_kernels + VLLM_MXFP4_USE_MARLIN=0)
# makes vLLM 0.19.1 select TRITON_UNFUSED (not MARLIN) for the mxfp4+LoRA path,
# bypassing the cudaErrorUnsupportedPtxVersion crash at marlin_utils_fp4.py:574
# (RECOVERY_LOG recovery2/cmcxl confirmed the crash on MARLIN default).
#
# v0.19.1 oracle/mxfp4.py:197 select_mxfp4_moe_backend:
#   204: triton_kernels_supported = has_triton_kernels() and (9.0<=cap<11.0)
#   210: if config.is_lora_enabled:
#   217:   if envs.VLLM_MXFP4_USE_MARLIN is False and triton_kernels_supported:
#   219:     return TRITON_UNFUSED, ...   <- target (bypasses Marlin PTX)
#   223:   return MARLIN, ...             <- default (crashes PTX on gm402)
#
# has_triton_kernels lives in vllm.utils.import_utils (line 30 import).
# envs.VLLM_MXFP4_USE_MARLIN is read once at process start, so we export BEFORE python.
#
# 1 GPU, exits in ~8-12 min. No training, no rollout — pure backend-selection probe.
set -euxo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

nvidia-smi -L
test -f /mnt/songzijun/models/openai/gpt-oss-20b/config.json

# --- 1. Install four-pack runtime + triton_kernels via Xiaomi PyPI ---
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" \
  "vllm==0.19.1" peft structlog chromadb pyserini \
  tqdm regex sentencepiece openai tenacity anthropic

echo "=== attempt triton_kernels install (--no-deps to preserve four-pack lock) ==="
# triton_kernels 0.1.0 depends on torch>=2.1 (no upper bound) which would
# upgrade torch 2.10.0->2.14.0+cu130 and break the four-pack lock + vllm 0.19.1
# (requires torch==2.10.0). Use --no-deps: triton_kernels only needs triton
# (already installed 3.6.0 as a torch dep) + numpy (already present).
if python3 -m pip install --no-cache-dir --no-deps --ignore-installed triton_kernels; then
  echo "TRITON_KERNELS_INSTALL=OK"
else
  echo "TRITON_KERNELS_INSTALL=FAILED — triton path NOT viable on this pod (no network/package)"
  exit 3
fi

# --- 2a. Probe DEFAULT (VLLM_MXFP4_USE_MARLIN unset) -> should pick MARLIN ---
echo "=== probe DEFAULT backend (expect MARLIN) ==="
python3 - <<'PY'
import importlib.metadata, torch, vllm, transformers
print("VERSIONS", {
    "torch": torch.__version__, "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "pyserini": importlib.metadata.version("pyserini"),
    "gpu_count": torch.cuda.device_count(),
    "device_cap": torch.cuda.get_device_capability(),
    "device_name": torch.cuda.get_device_name(0),
}, flush=True)
assert torch.__version__.startswith("2.10.0")
assert vllm.__version__ == "0.19.1"
assert transformers.__version__ == "4.57.6"
cap = torch.cuda.get_device_capability()
assert cap[0] == 9, f"expected sm_90, got {cap}"

# triton_kernels import
try:
    import triton_kernels  # noqa
    print("TRITON_KERNELS_IMPORT=OK", flush=True)
except Exception as e:
    print(f"TRITON_KERNELS_IMPORT=FAILED {type(e).__name__}: {e}", flush=True)
    raise SystemExit(2)

from vllm.utils.import_utils import has_triton_kernels
print("HAS_TRITON_KERNELS", has_triton_kernels(), flush=True)

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    select_mxfp4_moe_backend, Mxfp4MoeBackend,
)

class FakeQuantConfig:
    is_lora_enabled = True
    def is_mxfp4_quant(self, *a, **k):
        return True

backend, kernel_cls = select_mxfp4_moe_backend(FakeQuantConfig())
print("BACKEND_DEFAULT", str(backend), flush=True)  # expect MARLIN (the PTX-crashing path)
PY

# --- 2b. Probe VLLM_MXFP4_USE_MARLIN=0 -> expect TRITON_UNFUSED ---
echo "=== probe VLLM_MXFP4_USE_MARLIN=0 backend (expect TRITON_UNFUSED) ==="
VLLM_MXFP4_USE_MARLIN=0 python3 - <<'PY'
from vllm.utils.import_utils import has_triton_kernels
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import select_mxfp4_moe_backend
import os
print("VLLM_MXFP4_USE_MARLIN_env", os.environ.get("VLLM_MXFP4_USE_MARLIN"), flush=True)
print("HAS_TRITON_KERNELS", has_triton_kernels(), flush=True)

class FakeQuantConfig:
    is_lora_enabled = True
    def is_mxfp4_quant(self, *a, **k):
        return True

backend, kernel_cls = select_mxfp4_moe_backend(FakeQuantConfig())
print("BACKEND_MARLIN0", str(backend), flush=True)  # expect TRITON_UNFUSED
# Verdict: if TRITON_UNFUSED, the code-level fix works (bypasses Marlin PTX).
if "TRITON" in str(backend):
    print("VERDICT=TRITON_PATH_VIABLE — proceed to 8gpu RL with mxfp4+triton", flush=True)
else:
    print(f"VERDICT=TRITON_PATH_NOT_VIABLE — still {backend}, fallback to bf16-dequant", flush=True)
PY

echo "=== diag complete ==="
