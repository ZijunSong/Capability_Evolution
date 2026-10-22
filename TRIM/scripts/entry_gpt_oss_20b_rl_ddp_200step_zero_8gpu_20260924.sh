#!/usr/bin/env bash
# Container entry for Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4,
# non-dequant per user instruction) via torch_ddp_lora backend on one 8-GPU
# gm402-80g node (200 steps). world_size=8 (single node, 8 ranks via
# torchrun --nproc_per_node 8).
#
# This is the ZERO-component variant (H_zero baseline, --component zero = all
# advanced components OFF, student_mask=zero_mask), vs j7bje which used
# --component all (H_full). Same config otherwise: mxfp4 original (dequant bf16
# at runtime by vLLM 0.19.1, no VLLM_MXFP4_USE_MARLIN), torch_ddp_lora nproc=8,
# train_method rl, 200 steps, mb=1, max-model-len 8192, gpu-mem 0.90.
#
# All known gpt-oss mxfp4 + torch_ddp_lora pitfalls are already fixed in the
# mounted /mnt/songzijun code tree, and re-asserted in-pod by PATCH_CHECK below:
#   * trainer_adapter.py:478 startup_timeout_s=3600.0 (FIX for 900s worker-start
#     timeout when 8 DDP ranks each read 13GB mxfp4 from JuiceFS concurrently).
#   * trainer_adapter.py resolve_queries 4-tuple unpack (FIX for the
#     724d9fa ValueError: too many values to unpack).
#   * trainer_adapter.py:501 client.start() EADDRINUSE retry x3 (FIX for the
#     per-rank vLLM TP=1 TCPStore port race that killed all-RL ddp jobs at
#     step>=1; the killed 0xzst already PROVED this path safe at step1+).
#   * SCAPE_FORCE_LOCAL_HARMONY is INTENTIONALLY NOT SET (gpt-oss would crash
#     on _LocalConversation.to_json via openai_harmony Rust encoder; only
#     gpt-oss hits the Harmony path, Qwen3-4B uses hf_chat path unscathed).
#   * Model is copied to /tmp BEFORE exec (FIX2: 8-rank concurrent JuiceFS read
#     of 13.8GB exceeded the 900s worker-start window even with startup_timeout
#     patched; local /tmp read is ~2min/engine and runs outside that window).
#     Verified necessary on t-cucac/t-8odmo (JuiceFS read hang) and baked into
#     the 9/24 trim ddp entry. Still the original mxfp4 weights, no dequant.
#
# User's requested command used --model_name pointing at the JuiceFS path
# /mnt/songzijun/models/openai/gpt-oss-20b directly. We override it to the local
# /tmp copy after the cp step below (same weights, original mxfp4, no dequant)
# because the direct JuiceFS path triggers the 8-rank concurrent-read hang.
# This is the only deviation from the literal requested command and is required
# for the torch_ddp_lora path to actually start on 8 ranks.
#
# Post-submit provenance audit (per CLAUDE.md): verify world_size is truly 8 by
# the three-tuple — (1) stdout has torchrun --nproc_per_node 8; (2)
# checkpoints/<cell>/step_000001/ has 8 optimizer.rank0000..0007.pt; (3)
# vllm_sessions/ has 8 _r0.._r7 sessions. Any one missing => actually
# single-GPU run, result is NOT a valid "8-GPU DDP" artifact.
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# torch_ddp_lora path: NO CUDA_LAUNCH_BLOCKING (DDP is real multi-proc torchrun,
# not the accelerate device_map=auto single-proc that needs the sync crutch).
# NO DISABLE_CUSTOM_ALL_REDUCE (per-rank vLLM TP=1, no TP all-reduce to fix).
# mb1 (NOT 4): mb>=2 backward OOMs on 20B mxfp4 (gpt-oss MoE backward cannot
# enable gradient checkpointing, mb1 is the primary OOM defense; peak_mem
# 74.7GB/80GB constant, verified across steps on the stable RL ddp runs).

# FIX1 (to_json bug): SCAPE_FORCE_LOCAL_HARMONY is INTENTIONALLY NOT SET.
# Setting it=1 binds Conversation=_LocalConversation stub (no to_json) in
# harness/ultra_core.py, but enc is real openai_harmony.HarmonyEncoding which
# calls conv.to_json() -> AttributeError. gpt-oss triggers the Harmony path so
# this env MUST stay unset.
# Placeholder key so pyserini.encode._openai module-level openai.OpenAI()
# does not raise OpenAIError at import (RL uses local BM25, never calls OpenAI API).
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

# tiktoken bundle: harness import chain (openai_harmony) still needs o200k_base.
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

# --- Presence checks on the mounted JuiceFS tree ---
nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b/config.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data/manifest.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/manifest.json

# --- JVM discovery for Pyserini/Lucene (local BM25 retrieval) ---
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

# --- Install four-pack + harness import chain deps (internal Xiaomi PyPI) ---
# mb1 / gpu-mem 0.90 path; original mxfp4 (13GB) not dequant.
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece pyserini openai tenacity anthropic

# --- ANSERINI_CLASSPATH for PyJNI/Lucene ---
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Preflight: assert four-pack versions, 8 GPUs, JVM, import chain, Lucene ---
python3 - <<'PY'
import importlib.metadata
import json
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
# GPT-OSS-20B model probe: original mxfp4 checkpoint (config has quantization_config
# quant_method=mxfp4). Confirm config is readable and report quant method so the
# step1+ vLLM Marlin-LoRA PTX risk is visible in the preflight log.
model_cfg_path = "/mnt/songzijun/models/openai/gpt-oss-20b/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "quant_method": qcfg.get("quant_method"),
}, flush=True)
assert model_cfg.get("architectures") == ["GptOssForCausalLM"]
assert qcfg.get("quant_method") == "mxfp4"
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
# Verify the three code-tree patches are in place (so we don't silently regress).
import pathlib, re
ta = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/trainer_adapter.py").read_text()
assert "startup_timeout_s=3600.0" in ta, "startup_timeout_s=3600 patch missing"
assert re.search(r"train_rows,\s*eval_rows,\s*pool_meta,\s*_?frozen_points\s*=\s*resolve_queries", ta), "4-tuple unpack patch missing"
assert "EADDRINUSE" in ta, "EADDRINUSE retry patch missing"
print("PATCH_CHECK_OK", {"startup_timeout_s": True, "resolve_queries_4tuple": True, "eaddrinuse_retry": True}, flush=True)
PY

# --- FIX2: copy original mxfp4 model weights to local /tmp before exec ---
# Single read from JuiceFS (~13.8GB, a few minutes), then all 8 ranks load from
# fast local disk (~2min/engine). Runs BEFORE vLLM worker spawn so it does NOT
# count against the 3600s worker-start window.
MODEL_SRC=/mnt/songzijun/models/openai/gpt-oss-20b
MODEL_LOCAL=/tmp/gpt-oss-20b
rm -rf "$MODEL_LOCAL"
mkdir -p "$MODEL_LOCAL"
cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
test -f "$MODEL_LOCAL/config.json"
test -f "$MODEL_LOCAL/model.safetensors.index.json" || true
echo "MODEL_LOCAL_READY $MODEL_LOCAL"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py \
  --training-backend torch_ddp_lora \
  --nproc-per-node 8 \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/gpt-oss-20b \
  --train_method rl \
  --component zero \
  --train-env local_legacy \
  --train-data sec \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --train-steps 200 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 1 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --out /mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_200step_zero_8gpu_20260924_$(date +%Y%m%d_%H%M%S)
