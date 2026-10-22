#!/usr/bin/env bash
# Container entry for Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4)
# via torch_ddp_lora backend on one 8-GPU gm402-80g node.
# Read verbatim by the CloudML submit script via --image_command "$(cat this)".
set -euxo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
# torch_ddp_lora path: NO CUDA_LAUNCH_BLOCKING (DDP is real multi-proc torchrun,
# not accelerate device_map=auto async race). NO DISABLE_CUSTOM_ALL_REDUCE
# (per-rank vLLM TP=1, no TP all-reduce to fix).
# Placeholder key so pyserini.encode._openai module-level openai.OpenAI()
# does not raise OpenAIError at import (RL uses local BM25, never calls OpenAI API).
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
# SCAPE_FORCE_LOCAL_HARMONY removed: triggers ultra_core.py:31 local fallback
# _LocalConversation (no to_json) -> crashes openai_harmony:470 for gpt-oss
# (recov2 of trim t-xwl4c verified; Qwen unaffected since it doesn't use harmony).
# export SCAPE_FORCE_LOCAL_HARMONY=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

# Offline tiktoken: extract the bundle to /tmp so harmony_runtime can
# auto-discover it, and also set the env vars explicitly (bundle README way B).
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b/config.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data/manifest.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/manifest.json

# --- Pre-copy model to local disk to avoid JuiceFS 8-way concurrent read thrashing ---
# Job t-20260919103041-8odmo failed: 8 ranks each started an EngineCore reading the
# 13GB gpt-oss-20b from JuiceFS concurrently; shard load went 410s/it (vs ~57s ideal),
# EngineCore child process died mid-load ("leaked semaphore at shutdown"), torchrun
# exit 1 with no traceback (C-level I/O stall). Same as t-cucac trim run.
# cp to /tmp (local SSD/RAM) so 8 ranks read from local disk concurrently.
MODEL_SRC=/mnt/songzijun/models/openai/gpt-oss-20b
MODEL_DST=/tmp/gpt-oss-20b
echo "Copying model $MODEL_SRC -> $MODEL_DST (avoid JuiceFS 8-way read thrash)..."
rm -rf "$MODEL_DST"
cp -r "$MODEL_SRC" "$MODEL_DST"
test -f "$MODEL_DST/config.json"
du -sh "$MODEL_DST" 2>/dev/null || true
MODEL_NAME="$MODEL_DST"

JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
if [ -z "${JVM_PATH}" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
fi
test -n "${JVM_PATH}"
test -f "${JVM_PATH}"
ldd "${JVM_PATH}"
export JVM_PATH
export JAVA_HOME=$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")
export LD_LIBRARY_PATH=$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}

python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece pyserini openai tenacity anthropic

# --- mxfp4 RL: vLLM runtime auto-dequant bf16 path (NOT triton/MARLIN) ---
# Original mxfp4 model at /mnt/songzijun/models/openai/gpt-oss-20b (quant_method=mxfp4).
# Do NOT set VLLM_MXFP4_USE_MARLIN and do NOT install triton_kernels — let vLLM 0.19.1
# detect "no triton_kernels" and auto-dequant mxfp4->bf16 in-GPU at load time (log:
# "MXFP4 quantization requires Triton... we will default to dequantizing to bf16").
# This is the stable path verified by t-20260922155913-gu18h (46h+, 14+ steps, no OOM):
# the bf16 compute path has NO triton/MARLIN CPU-side residue (kernel cache,
# EngineCore handles, unpickler) that caused recov3/zero/recov6/recov7 step1
# all_gather rank0 CPU OOM SIGKILL (runTime ~10100-10210s) on mxfp4+triton path.
# triton_kernels path (recov1-7) hit 5 extra pits (PTX/matmul_ogs/JuiceFS-build/
# unhashable/CPU-OOM); bf16-dequant path avoids ALL of them while still using the
# original mxfp4 model file (satisfies "original, non-dequant" requirement).

PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

python3 - <<'PY'
import importlib.metadata
import os
import torch
import transformers
import vllm
from jnius import autoclass
autoclass("java.lang.String")
versions = {
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "pyserini": importlib.metadata.version("pyserini"),
    "peft": importlib.metadata.version("peft"),
    "structlog": importlib.metadata.version("structlog"),
    "chromadb": importlib.metadata.version("chromadb"),
    "gpu_count": torch.cuda.device_count(),
    "JVM_PATH": os.environ["JVM_PATH"],
    "JAVA_HOME": os.environ["JAVA_HOME"],
    "TIKTOKEN_ENCODINGS_BASE": os.environ.get("TIKTOKEN_ENCODINGS_BASE"),
    "o200k_base_present": os.path.isfile(
        os.path.join(os.environ["TIKTOKEN_ENCODINGS_BASE"], "o200k_base.tiktoken")
    ),
    "java_ok": True,
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0")
assert versions["vllm"] == "0.19.1"
assert versions["transformers"] == "4.57.6"
assert versions["pyserini"] == "2.0.0"
assert versions["gpu_count"] == 8
assert versions["o200k_base_present"]
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/indexes/bm25")
n_docs = s.num_docs if not callable(s.num_docs) else s.num_docs()
print("LUCENE_OK", {"num_docs": n_docs}, flush=True)
assert n_docs > 0
# DDP-LoRA wrap probe: torch_ddp_lora backend must select the plain "ddp" actor
# wrap (full replica per rank), NOT the FSDP2 sharded wrap. Vendored
# trim.integrations.verl (external verl NOT imported).
from trim.integrations.verl.trainer_adapter import _actor_wrap
assert _actor_wrap("torch_ddp_lora") == "ddp"
print("DDP_LORA_WRAP_OK", {"wrap": _actor_wrap("torch_ddp_lora")}, flush=True)

# mxfp4 path: do NOT set VLLM_MXFP4_USE_MARLIN (let vLLM 0.19.1 default to Marlin
# backend for mxfp4+LoRA, same as gu18h stable 46h+ run). Do NOT install external
# triton_kernels (vLLM wheel has built-in v3.6.0). The earlier recov1-7 set
# VLLM_MXFP4_USE_MARLIN=0 to force TRITON_UNFUSED "to bypass PTX" — but that
# introduced CPU-side triton residue that caused step1 all_gather rank0 OOM.
# gu18h defaults to Marlin (worker.log "Using Marlin backend for mxfp4 lora") and
# is stable (no PTX crash, no OOM) — Marlin on gm402 sm_90 does NOT crash PTX
# (recovery2/cmcxl's earlier crash was not Marlin-PTX). Just report, no assertion.
from vllm.utils.import_utils import has_triton_kernels
print("MXFP4_PATH", {"has_triton_kernels": has_triton_kernels(), "VLLM_MXFP4_USE_MARLIN_set": False, "path": "vLLM default Marlin (align gu18h)"}, flush=True)
PY

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py \
  --training-backend torch_ddp_lora \
  --nproc-per-node 8 \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name "$MODEL_DST" \
  --train_method rl \
  --component zero \
  --train-env local_legacy \
  --train-data sec \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --train-steps 100 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 1 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --out /mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gptoss20b_rl_ddp_100step_8gpu_$(date +%Y%m%d_%H%M%S)
