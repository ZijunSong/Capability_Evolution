#!/usr/bin/env bash
# Diag r3: verify triton_kernels v3.6.0 (from triton-lang/triton tag v3.6.0) fixes the
# ModuleNotFoundError: triton_kernels.matmul_ogs that crashed recov1 t-znyvp at
# mxfp4.py:722 convert_to_mxfp4_moe_kernel_format (step0 vLLM init).
#
# vLLM 0.19.1 cmake/external_projects/triton_kernels.cmake declares
# DEFAULT_TRITON_KERNELS_TAG="v3.6.0" -> needs triton-lang/triton v3.6.0's
# python/triton_kernels (has matmul_ogs.py + swiglu.py). PyPI triton_kernels 0.1.0
# is a DIFFERENT old package without matmul_ogs.
#
# This diag ACTUALLY loads the mxfp4 model into vLLM (VLLM_MXFP4_USE_MARLIN=0 ->
# TRITON_UNFUSED backend) to confirm step0 init passes (recov1's exact failure point).
# 1 GPU, ~10-15min (pip + model load).
set -euxo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/third_party/triton_kernels_v3.6.0/triton_kernels/matmul_ogs.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b/config.json

# --- 1. four-pack (skip if torch already 2.10.0 to save time) ---
python3 - <<'PYCHK'
try:
    import torch, vllm, transformers
    ok = torch.__version__.startswith("2.10.0") and vllm.__version__=="0.19.1" and transformers.__version__=="4.57.6"
    print("FOURPACK_CHECK", torch.__version__, vllm.__version__, transformers.__version__, "ok="+(str(ok)))
except Exception:
    print("FOURPACK_CHECK missing")
PYCHK

python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" \
  "vllm==0.19.1" peft structlog chromadb pyserini \
  tqdm regex sentencepiece openai tenacity anthropic

# --- 2. triton_kernels v3.6.0 from JuiceFS (NOT PyPI 0.1.0) ---
python3 -m pip install --no-cache-dir --no-deps --ignore-installed \
  /mnt/songzijun/Capability_Evolution/TRIM/third_party/triton_kernels_v3.6.0
python3 -c "from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig; from triton_kernels import swiglu; print('TRITON_KERNELS_V360_IMPORT_OK')"
echo "TRITON_KERNELS_V360_INSTALL=OK"

export VLLM_MXFP4_USE_MARLIN=0

# --- 3. ACTUALLY load mxfp4 model into vLLM (recov1's failure point) ---
python3 - <<'PY'
import os, torch, vllm
print("VERSIONS", torch.__version__, vllm.__version__, "cap", torch.cuda.get_device_capability(), flush=True)
assert torch.__version__.startswith("2.10.0")
assert vllm.__version__ == "0.19.1"
from vllm.utils.import_utils import has_triton_kernels
print("HAS_TRITON_KERNELS", has_triton_kernels(), flush=True)
assert has_triton_kernels()

# Backend probe
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import select_mxfp4_moe_backend
class _QC:
    is_lora_enabled = True
    def is_mxfp4_quant(self, *a, **k): return True
b, _ = select_mxfp4_moe_backend(_QC())
print("MXFP4_LORA_BACKEND", str(b), flush=True)
assert "TRITON" in str(b)

# THE recov1 failure point: actually construct vLLM LLM on the mxfp4 model.
# convert_to_mxfp4_moe_kernel_format (mxfp4.py:722) does
# `from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig` during
# process_weights_after_loading. If this import fails -> EngineCore init fails.
from vllm import LLM
print("=== loading mxfp4 model into vLLM (TRITON_UNFUSED backend) ===", flush=True)
llm = LLM(
    model="/mnt/songzijun/models/openai/gpt-oss-20b",
    tensor_parallel_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.70,
    enforce_eager=True,          # eager for faster/safer init on 1gpu diag
    trust_remote_code=True,
    max_num_seqs=8,
)
print("VLLM_MXFP4_MODEL_LOADED_OK", flush=True)
# tiny generate to confirm forward works (no NaN crash)
out = llm.generate(["Hello, my name is"], vllm.SamplingParams(max_tokens=8, temperature=0))
print("GEN_OK", repr(out[0].outputs[0].text), flush=True)
print("VERDICT=MXFP4_TRITON_V360_STEP0_VIABLE", flush=True)
PY

echo "=== diag r3 complete ==="
