#!/usr/bin/env bash
# === GPT-OSS-20B Teacher SFT — accelerate_ddp train-only (2026-10-21) ===
# Bridges the gap between the already-collected Teacher SFT rows (produced by
# t-20261020164236-7rk9c after the _action_of codec fix) and the new
# accelerate_ddp CE backend (trim.training.sft_engine.fixed_runtime.run_fixed_sft).
#
# WHY THIS ENTRY (vs the collect entry entry_..._true.sh):
#   * The legacy ce_update (ce.py, mb1 single-rank serial 128 micro-batches) had
#     near-zero GPU utilization and got SIGTERM-killed by the platform's GPU-util
#     watchdog mid first CE step (7rk9c 06:30, 14ztx 01:15). The user added a new
#     `accelerate_ddp` CE backend: DDP 32 ranks step together, each rank handles
#     global_batch/32=4 micro-batches, GPU util high.
#   * accelerate_ddp's run_fixed_sft calls load_sealed, which requires a SEALED
#     dataset (DATASET.json + READY.json + SOURCE_CONTRACT.json + dataset.jsonl,
#     with sha256 checks). The legacy build_teacher_sft_rows only wrote
#     rows.jsonl+REJECTED.json. scripts/seal_teacher_sft_dataset.py bridges that
#     (one-shot) into _true_fix1/sft_sealed/.
#   * accelerate_ddp REQUIRES a common initial adapter (load_ddp_model: "Main
#     ablation requires the common initial adapter"). We reuse the fresh LoRA
#     the Student on-policy leg saved at optimizer_step=0 (true_p5/adapters/current,
#     r=32 alpha=32 k/o/q/v_proj) — that is an untrained LoRA init, so it keeps the
#     ablation's fresh-LoRA-shared-seed-42 contract (only variable = supervision
#     signal source).
#
# This entry is TRAIN-ONLY (--stage train, --ce-backend accelerate_ddp). It does
# NOT collect, so it drops the JVM/Pyserini/Lucene/SEC-BM25 blocks the collect
# entry needs. It KEEPS: four-pack + accelerate + harness-1 transitive deps +
# .pth (harness-1 import-shading fix) + tiktoken + cp model to /tmp + the
# bidirectional rendezvous barrier (so all 4 nodes finish cp before torchrun).
# dist_launch.apply_v2_train_defaults sets nnodes=4, nproc-per-node=8 (V2),
# rollout_replicas=1 — 32 DDP gradient ranks. master_port 29612.

set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1/tinker-cookbook:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# --- Multi-node distributed env guard ---
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set. Each pod needs its own rank id (0..3)." >&2
  exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set." >&2
  exit 1
fi
: "${MASTER_PORT:=29612}"
export MASTER_ADDR NODE_RANK MASTER_PORT

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

# --- Shared output directory (MUST be identical on every node) ---
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# FIX1 (to_json bug): SCAPE_FORCE_LOCAL_HARMONY is INTENTIONALLY NOT SET.
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

# --- Presence checks on the mounted JuiceFS tree ---
nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_teacher_sft_multinode.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/model.safetensors.index.json

# --- Sealed dataset path (produced below by seal-if-missing, after pip+.pth) ---
SEALED_DIR=/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-bf16-32gpu_20261019_true_fix1/sft_sealed
SEALED_ROWS_JSONL=/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-bf16-32gpu_20261019_true_fix1/sft_rows/rows.jsonl
SEALED_CONTRACT=/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-bf16-32gpu_20261019_true_fix1/CONTRACT.json
test -f "$SEALED_ROWS_JSONL"   # the collected legacy rows must exist
test -f "$SEALED_CONTRACT"

# --- Common initial adapter (fresh LoRA saved by Student on-policy at step 0) ---
INIT_ADAPTER=/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/adapters/current
test -f "$INIT_ADAPTER/adapter_config.json"
test -f "$INIT_ADAPTER/adapter_model.safetensors"
echo "INIT_ADAPTER_OK $INIT_ADAPTER"

# --- Install four-pack + accelerate + SFT/harness import chain deps (internal Xiaomi PyPI) ---
# accelerate_ddp needs: torch/transformers/peft/accelerate (DDP), NOT vllm/pyserini
# (train-only, no rollout/SEC retrieval). Keep structlog/chromadb/openai_harmony
# (harness import chain) + structlog (harness.ultra_core) + chz/tinker (training.train_rl).
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" peft structlog chromadb \
  tqdm regex sentencepiece openai tenacity anthropic \
  chz tinker \
  tensorboard

# --- harness-1 TRANSITIVE deps (NOT harness-1 itself; it is on PYTHONPATH) ---
python3 -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  json-repair langchain-text-splitters wikipedia-api modal baseten-performance-client \
  fastembed pysqlite3-binary wandb trl chz tinker rich

# --- .pth file: external/harness-1 + tinker-cookbook win the `training` package
# name over TRIM/trim/training (r11 root cause: real package TRIM/trim/training
# shadows the namespace package external/harness-1/training which has train_rl.py).
HARNESS1=/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1
TINKER_CB=$HARNESS1/tinker-cookbook
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PKGS/zzz_harness1.pth" <<EOF
import sys; sys.path.insert(0, "$HARNESS1"); sys.path.insert(0, "$TINKER_CB")
EOF
test -f "$SITE_PKGS/zzz_harness1.pth"
echo "HARNESS1_PTH_WRITTEN $SITE_PKGS/zzz_harness1.pth"
python3 -E -c "import training; assert 'harness-1/training' in list(training.__path__)[0], list(training.__path__); import training.train_rl, harness.ultra_core; print('HARNESS1_SUBPROCESS_IMPORT_OK', list(training.__path__))"

# --- Seal the collected rows into a v2 sealed dataset if not already sealed ---
# The seal is all-in-memory (seal_dataset sorts+checks all rows); the dev-box
# background shell kept getting SIGKILL'd at ~160k rows (harness mem cap), so we
# seal here on the training pod where memory is plentiful. rank0 seals; other
# ranks wait on a shared OUT-dir signal so nobody proceeds until the sealed
# dataset is ready.
SEALED_DONE="$SEALED_DIR/READY.json"
if [ ! -f "$SEALED_DONE" ]; then
  if [ "${NODE_RANK}" = "0" ]; then
    echo "SEAL_DATASET_START rank0 sealing rows.jsonl -> $SEALED_DIR host=$(hostname)"
    PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/seal_teacher_sft_dataset.py
    test -f "$SEALED_DIR/DATASET.json"
    test -f "$SEALED_DIR/READY.json"
    test -f "$SEALED_DIR/dataset.jsonl"
    echo "SEAL_DATASET_DONE rank0 host=$(hostname)"
  else
    echo "SEAL_DATASET_WAIT node_rank=${NODE_RANK} polling $SEALED_DONE host=$(hostname)"
    WAITED=0
    while [ ! -f "$SEALED_DONE" ]; do
      sleep 15
      WAITED=$((WAITED + 15))
      if [ "$WAITED" -ge 3600 ]; then
        echo "FATAL: node_rank=${NODE_RANK} timed out after ${WAITED}s waiting for sealed dataset" >&2
        exit 1
      fi
      if [ $((WAITED % 60)) -eq 0 ]; then
        echo "SEAL_DATASET_WAIT still waiting ${WAITED}s host=$(hostname) node_rank=${NODE_RANK}"
      fi
    done
    echo "SEAL_DATASET_SEEN after ${WAITED}s host=$(hostname) node_rank=${NODE_RANK}"
  fi
fi
test -f "$SEALED_DIR/DATASET.json"
test -f "$SEALED_DIR/READY.json"
test -f "$SEALED_DIR/dataset.jsonl"
test -f "$SEALED_DIR/SOURCE_CONTRACT.json"
echo "SEALED_DATASET_OK $SEALED_DIR"

# --- Preflight: four-pack versions, 8 GPUs/node, bf16 base, accelerate_ddp import, sealed dataset ---
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
    "TIKTOKEN_ENCODINGS_BASE": os.environ.get("TIKTOKEN_ENCODINGS_BASE"),
    "o200k_base_present": os.path.isfile(
        os.path.join(os.environ["TIKTOKEN_ENCODINGS_BASE"], "o200k_base.tiktoken")
    ),
    "host": socket.gethostname(),
    "master_addr": os.environ.get("MASTER_ADDR"),
    "node_rank": os.environ.get("NODE_RANK"),
    "rank_env": os.environ.get("RANK"),
    "world_size_env": os.environ.get("WORLD_SIZE"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
assert versions["o200k_base_present"]

# GPT-OSS-20B model probe: bf16-dequant (NO quant_method, NOT mxfp4).
model_cfg_path = "/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "quant_method": qcfg.get("quant_method", "NONE"),
    "vocab_size": model_cfg.get("vocab_size"),
}, flush=True)
assert model_cfg.get("architectures") == ["GptOssForCausalLM"], f"arch={model_cfg.get('architectures')}"
assert not qcfg.get("quant_method"), f"bf16-dequant must have NO quant_method, got {qcfg.get('quant_method')}"

# accelerate_ddp engine import (fail-fast before cp/barrier waste ~1h).
from trim.training.sft_engine.fixed_runtime import run_fixed_sft  # noqa: F401
from trim.training.sft_engine.model import make_accelerator, load_ddp_model  # noqa: F401
from trim.training.sft_engine.data import load_sealed  # noqa: F401
from trim.training.sft_ablation.dist_launch import launch_sft  # noqa: F401
from trim.training.sft_ablation.contracts import PROFILES, SFTBatchRow  # noqa: F401
from trim.training.teacher_sft.runtime import run as teacher_run  # noqa: F401
assert "teacher_sft" in PROFILES
print("ACCELERATE_DDP_IMPORT_OK", {"fixed_runtime": True, "make_accelerator": True, "load_sealed": True}, flush=True)

# Verify the sealed dataset load_seals cleanly (this is exactly what run_fixed_sft does).
SEALED_DIR = "/mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-bf16-32gpu_20261019_true_fix1/sft_sealed"
ds = load_sealed(
    manifest_path=os.path.join(SEALED_DIR, "DATASET.json"),
    rows_path=os.path.join(SEALED_DIR, "dataset.jsonl"),
    expected_experiment="teacher_sft",
)
train_pop = ds.train_positions()
from trim.training.sft_ablation.identity import tokenizer_digest
live_tok = tokenizer_digest("/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant")
print("SEALED_LOAD_OK", {
    "experiment": ds.experiment,
    "n_train_positions": len(train_pop),
    "package_digest": ds.package_digest[:16],
    "tokenizer_digest_match": live_tok == ds.tokenizer_digest,
}, flush=True)
assert train_pop, "sealed dataset has no training rows"
assert live_tok == ds.tokenizer_digest, "tokenizer digest mismatch — seal/train tokenizer diverged"

# init adapter present + is a PEFT LoRA.
import json as _json
with open("/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/adapters/current/adapter_config.json") as fh:
    acfg = _json.load(fh)
print("INIT_ADAPTER_PROBE", {"r": acfg.get("r"), "alpha": acfg.get("lora_alpha"),
                              "target": acfg.get("target_modules")}, flush=True)
assert acfg.get("r") == 32 and acfg.get("lora_alpha") == 32

print("SFT_PREFLIGHT_OK", {"four_pack": True, "accelerate": True, "bf16_dequant": True,
                            "sealed_dataset": True, "init_adapter": True}, flush=True)
PY

# --- FIX2: copy bf16-dequant model weights to local /tmp before exec ---
MODEL_SRC=/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant
MODEL_LOCAL=/tmp/gpt-oss-20b-bf16-dequant
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"
test -f "$MODEL_LOCAL/model.safetensors.index.json"
echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

# --- Bidirectional rendezvous barrier (all 4 nodes finish cp before torchrun) ---
BARRIER_DIR="${OUT}/.rendezvous_barrier"
mkdir -p "$BARRIER_DIR"
MY_RANK="${NODE_RANK:-0}"
MY_SIGNAL="$BARRIER_DIR/node_ready_${MY_RANK}"
: > "$MY_SIGNAL"
echo "BARRIER_NODE_READY wrote $MY_SIGNAL host=$(hostname) node_rank=${MY_RANK}"
if [ "${MY_RANK}" = "0" ]; then
  echo "BARRIER_MASTER_WAIT_ALL waiting for all node_ready_0..3 host=$(hostname)"
  WAITED=0
  ALL_READY=false
  while [ "$ALL_READY" = "false" ]; do
    ALL_READY=true
    for r in 0 1 2 3; do
      [ -f "$BARRIER_DIR/node_ready_${r}" ] || ALL_READY=false
    done
    if [ "$ALL_READY" = "false" ]; then
      sleep 10
      WAITED=$((WAITED + 10))
      if [ "$WAITED" -ge 3000 ]; then
        SEEN=""
        for r in 0 1 2 3; do [ -f "$BARRIER_DIR/node_ready_${r}" ] && SEEN="${SEEN} ${r}"; done
        echo "FATAL: master timed out after ${WAITED}s waiting for all node_ready. Seen:${SEEN:- none}" >&2
        exit 1
      fi
      if [ $((WAITED % 60)) -eq 0 ]; then
        SEEN=""
        for r in 0 1 2 3; do [ -f "$BARRIER_DIR/node_ready_${r}" ] && SEEN="${SEEN} ${r}"; done
        echo "BARRIER_MASTER_WAIT_ALL still waiting ${WAITED}s ready:${SEEN:- none} host=$(hostname)"
      fi
    fi
  done
  : > "$BARRIER_DIR/all_ready_0"
  echo "BARRIER_ALL_READY wrote $BARRIER_DIR/all_ready_0 host=$(hostname) node_rank=0 (all 4 cp done)"
else
  ALL_SIGNAL="$BARRIER_DIR/all_ready_0"
  echo "BARRIER_WORKER_WAIT_ALL polling $ALL_SIGNAL host=$(hostname) node_rank=${MY_RANK}"
  WAITED=0
  while [ ! -f "$ALL_SIGNAL" ]; do
    sleep 10
    WAITED=$((WAITED + 10))
    if [ "$WAITED" -ge 3000 ]; then
      echo "FATAL: worker node_rank=${MY_RANK} timed out after ${WAITED}s waiting for all_ready signal $ALL_SIGNAL" >&2
      exit 1
    fi
    if [ $((WAITED % 60)) -eq 0 ]; then
      echo "BARRIER_WORKER_WAIT_ALL still waiting ${WAITED}s host=$(hostname) node_rank=${MY_RANK}"
    fi
  done
  echo "BARRIER_WORKER_SEEN_ALL after ${WAITED}s host=$(hostname) node_rank=${MY_RANK}"
fi

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_teacher_sft_multinode.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_ablation/dist_launch.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/teacher_sft/runtime.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/fixed_runtime.py
cd /mnt/songzijun/Capability_Evolution
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_teacher_sft_multinode.py \
  --nnodes 4 \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --rollout-replicas 1 \
  --tensor-parallel-size 1 \
  --rdzv-id teacher-sft-accelerate-ddp-20261021 \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --init-adapter /mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/adapters/current \
  --ce-backend accelerate_ddp \
  --stage train \
  --sft-data /mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-bf16-32gpu_20261019_true_fix1/sft_sealed/dataset.jsonl \
  --dataset-manifest /mnt/songzijun/outputs/gpt-oss-20b-teacher-sft-bf16-32gpu_20261019_true_fix1/sft_sealed/DATASET.json \
  --num-updates 200 \
  --batch-size 32 \
  --global-batch-size 32 \
  --micro-batch-size 1 \
  --save-every 10 \
  --learning-rate 5e-6 \
  --lora-rank 32 \
  --max-model-len 4096 \
  --gradient-checkpointing non_reentrant \
  --attn-implementation eager \
  --max-new-tokens 2048 \
  --seed 42 \
  --out "${OUT}"
