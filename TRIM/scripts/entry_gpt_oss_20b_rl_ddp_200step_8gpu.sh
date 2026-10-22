#!/usr/bin/env bash
# Container entry for Harness-1 BC+ full RL (CISPO) GPT-OSS-20B (original mxfp4,
# non-dequant per user instruction) via torch_ddp_lora backend on one 8-GPU
# gm402-80g node (200 steps).
# Read verbatim by the CloudML submit script via --image_command "$(cat this)".
set -euxo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD
# torch_ddp_lora path: NO CUDA_LAUNCH_BLOCKING (DDP is real multi-proc torchrun,
# not accelerate device_map=auto async race). NO DISABLE_CUSTOM_ALL_REDUCE
# (per-rank vLLM TP=1, no TP all-reduce to fix; user chose to keep original mxfp4
# without this env, accepting the step1+ Marlin-LoRA PTX risk — if it crashes at
# step1 vLLM+LoRA reload, switch to the bf16-dequant checkpoint and resubmit).
# gpu_memory_utilization=0.90 fully uses 8x80GB for vLLM KV cache during rollout
# (per-rank TP=1 engine); backward runs AFTER close_vllm()/_release_cuda frees
# vLLM memory (Scheme A exclusive), so high util does NOT squeeze backward —
# proven on Qwen3-4B DDP (eixmm/sdvgw, 62GiB KV/卡, 18.9GiB backward peak, no
# OOM). gpt-oss-20b is MoE/heavier; DDP path opens gradient_checkpointing_enable
# (use_reentrant=False) unlike local_legacy which disables it, lowering backward
# peak. If backward still OOMs at 0.90, lower to 0.55 (and/or max-model-len 4096).
# RECOVERY1 fix: removed `export SCAPE_FORCE_LOCAL_HARMONY=1`. That env forced
# ultra_core to use _LocalConversation (no to_json) while load_harmony_enc still
# returned the openai_harmony Rust encoder whose render_conversation_for_completion
# hard-calls conversation.to_json() → AttributeError at step0 first-turn prompt
# (only gpt-oss hits this Harmony path; Qwen3-4B uses hf_chat path, unscathed).
# aemum (9/14, same gpt-oss mxfp4, local_legacy) never set this env and ran 8
# rollout sessions fine. Without it, ultra_core uses openai_harmony's standard
# Conversation (has to_json), matching the Rust encoder.
# RECOVERY2 fix: patched trim/integrations/verl/trainer_adapter.py:389 to pass
# startup_timeout_s=3600.0 to VLLMGenerateClient (was default 900s). gpt-oss-20b
# mxfp4 has 3 safetensors shards; 8 DDP ranks each spin a TP=1 vLLM engine that
# reads the 13GB model from JuiceFS concurrently → ~7min/shard × 3 = ~20min per
# engine load, exceeding the 900s default. 3600s (matches four_cell_runtime
# open_vllm) lets step0 vLLM load finish. py_compile of trainer_adapter.py added
# to preflight to verify the patch loads cleanly in-pod.
# Placeholder key so pyserini.encode._openai module-level openai.OpenAI()
# does not raise OpenAIError at import (RL uses local BM25, never calls OpenAI API).
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

# Offline tiktoken: extract the bundle to /tmp so harmony_runtime can
# auto-discover it, and also set the env vars explicitly (bundle README 方式 B).
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

PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

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
PY

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py /mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/trainer_adapter.py
cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py \
  --training-backend torch_ddp_lora \
  --nproc-per-node 8 \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /mnt/songzijun/models/openai/gpt-oss-20b \
  --train_method rl \
  --component all \
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
  --out /mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_200step_8gpu_$(date +%Y%m%d_%H%M%S)
