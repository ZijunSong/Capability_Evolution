#!/usr/bin/env bash
# === Shared environment for the three accelerate_fsdp2 SFT entries (2026-10-22) ===
#
# Sourced by entry_gpt_oss_20b_{teacher,projected_teacher,student_onpolicy}_sft_
# accelerate_fsdp2_32gpu_4node.sh. Provides, in order:
#   1. env guards (NODE_RANK/MASTER_ADDR/OUT), PYTHONPATH, harmony/tiktoken
#   2. four-pack + accelerate + PEFT (train stack lock), harness import chain
#   3. preflight: versions, 8 GPUs, bf16-dequant base, FSDP2 engine import
#   4. local /tmp model copy + RUN-TAG-SCOPED rendezvous barrier
#
# The barrier directory is per run tag (default: OUT basename) and rank 0
# recreates it before any node writes a signal, so a stale node_ready_* from a
# previous attempt can never satisfy a later barrier (fqsbi r27min failure).
# Gradient checkpointing is an explicit contract field; default off (smoke),
# export SFT_CKPT_MODE=non_reentrant only after the off path is validated.

set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1/tinker-cookbook:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set. Each pod needs its own rank id (0..3)." >&2
  exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set." >&2
  exit 1
fi
: "${MASTER_PORT:=29631}"
: "${NNODES:=4}"
export MASTER_ADDR NODE_RANK MASTER_PORT

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

# tiktoken bundle: harness import chain (openai_harmony) needs o200k_base.
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_teacher_sft_multinode.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/model.safetensors.index.json

MODEL_SRC=/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant

# --- Train stack lock (torch 2.10.0 / transformers 4.57.6 / accelerate 1.13.0) ---
# PEFT pinned to the version resolved in the first FSDP2 smoke preflight
# (t-20261022110759-szfhz, peft 0.21.2). Override with PEFT_VERSION only for an
# explicit experiment change; the run contract records the resolved version.
PEFT_SPEC="peft==${PEFT_VERSION:-0.21.2}"
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "${PEFT_SPEC}" \
  structlog chromadb \
  tqdm regex sentencepiece openai tenacity anthropic \
  chz tinker \
  tensorboard

python3 -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  json-repair langchain-text-splitters wikipedia-api modal baseten-performance-client \
  fastembed pysqlite3-binary wandb trl chz tinker rich

# .pth: external/harness-1 + tinker-cookbook win the `training` package name.
HARNESS1=/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1
TINKER_CB=$HARNESS1/tinker-cookbook
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PKGS/zzz_harness1.pth" <<EOF
import sys; sys.path.insert(0, "$HARNESS1"); sys.path.insert(0, "$TINKER_CB")
EOF
test -f "$SITE_PKGS/zzz_harness1.pth"
python3 -E -c "import training; assert 'harness-1/training' in list(training.__path__)[0], list(training.__path__); import training.train_rl, harness.ultra_core; print('HARNESS1_SUBPROCESS_IMPORT_OK', list(training.__path__))"

# --- Preflight: versions, GPUs, base model, FSDP2 engine import ---
python3 - <<'PY'
import importlib.metadata
import json
import os
import socket
import torch
import transformers
import accelerate
versions = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "peft": importlib.metadata.version("peft"),
    "structlog": importlib.metadata.version("structlog"),
    "gpu_count": torch.cuda.device_count(),
    "host": socket.gethostname(),
    "master_addr": os.environ.get("MASTER_ADDR"),
    "node_rank": os.environ.get("NODE_RANK"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["accelerate"] == "1.13.0", f"accelerate={versions['accelerate']} != 1.13.0"
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"

model_cfg_path = "/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "quant_method": qcfg.get("quant_method", "NONE"),
}, flush=True)
assert model_cfg.get("architectures") == ["GptOssForCausalLM"], model_cfg.get("architectures")
assert model_cfg.get("model_type") == "gpt_oss", model_cfg.get("model_type")
assert not qcfg.get("quant_method"), f"bf16-dequant must have NO quant_method, got {qcfg.get('quant_method')}"

# FSDP2 engine import (fail-fast before cp/barrier).
from trim.training.sft_engine.backend import (  # noqa: F401
    ACCELERATED_BACKENDS,
    create_training_context,
    export_policy_adapter,
    make_training_accelerator,
    save_training_checkpoint,
)
from trim.training.sft_engine.model import (  # noqa: F401
    assert_fsdp2_ready,
    load_fsdp2_model,
    make_fsdp2_accelerator,
)
from trim.training.sft_engine.sharded_checkpoint import (  # noqa: F401
    export_lora_fsdp2,
    publish_sharded_checkpoint,
)
from trim.training.sft_engine.fixed_runtime import run_fixed_sft  # noqa: F401
from trim.training.sft_ablation.dist_launch import launch_sft  # noqa: F401

assert "accelerate_fsdp2" in ACCELERATED_BACKENDS
print("FSDP2_IMPORT_OK", {"backend": "accelerate_fsdp2", "wrap_cls": "GptOssDecoderLayer"}, flush=True)
PY

# --- Local model copy (per node) ---
MODEL_LOCAL=/tmp/gpt-oss-20b-bf16-dequant
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"
test -f "$MODEL_LOCAL/model.safetensors.index.json"
echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

# --- Run-tag-scoped rendezvous barrier (stale signals from old runs cannot match) ---
SFT_RUN_TAG="${SFT_RUN_TAG:-$(basename "$OUT")}"
BARRIER_DIR="${OUT}/.rendezvous_barrier_${SFT_RUN_TAG}"
if [ "${NODE_RANK}" = "0" ]; then
  rm -rf "$BARRIER_DIR"
  mkdir -p "$BARRIER_DIR"
  : > "$BARRIER_DIR/barrier_open"
  echo "BARRIER_OPEN $BARRIER_DIR host=$(hostname)"
fi
echo "BARRIER_WAIT_OPEN polling barrier_open host=$(hostname) node_rank=${NODE_RANK}"
WAITED=0
while [ ! -f "$BARRIER_DIR/barrier_open" ]; do
  sleep 5
  WAITED=$((WAITED + 5))
  if [ "$WAITED" -ge 1800 ]; then
    echo "FATAL: timed out waiting for barrier_open $BARRIER_DIR" >&2
    exit 1
  fi
done
: > "$BARRIER_DIR/node_ready_${NODE_RANK}"
echo "BARRIER_NODE_READY wrote node_ready_${NODE_RANK} host=$(hostname)"
if [ "${NODE_RANK}" = "0" ]; then
  WAITED=0
  ALL_READY=false
  while [ "$ALL_READY" = "false" ]; do
    ALL_READY=true
    for r in $(seq 0 $((NNODES - 1))); do
      [ -f "$BARRIER_DIR/node_ready_${r}" ] || ALL_READY=false
    done
    if [ "$ALL_READY" = "false" ]; then
      sleep 10
      WAITED=$((WAITED + 10))
      if [ "$WAITED" -ge 3000 ]; then
        SEEN=""
        for r in $(seq 0 $((NNODES - 1))); do [ -f "$BARRIER_DIR/node_ready_${r}" ] && SEEN="${SEEN} ${r}"; done
        echo "FATAL: master timed out after ${WAITED}s waiting for all node_ready. Seen:${SEEN:- none}" >&2
        exit 1
      fi
    fi
  done
  : > "$BARRIER_DIR/all_ready"
  echo "BARRIER_ALL_READY host=$(hostname)"
else
  WAITED=0
  while [ ! -f "$BARRIER_DIR/all_ready" ]; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ "$WAITED" -ge 3000 ]; then
      echo "FATAL: worker node_rank=${NODE_RANK} timed out waiting for all_ready" >&2
      exit 1
    fi
  done
  echo "BARRIER_WORKER_SEEN_ALL after ${WAITED}s host=$(hostname) node_rank=${NODE_RANK}"
fi

echo "SFT_FSDP2_COMMON_READY host=$(hostname) node_rank=${NODE_RANK}"
