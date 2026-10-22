#!/usr/bin/env bash
# === GPT-OSS-20B Student on-policy SFT v2 — DDP train-phase (2026-10-21) ===
#
# Recovery/validation run for the Accelerate DDP backend (commit c8640e4:
# "Add an Accelerate DDP backend so the three SFT runs update on every rank").
# This entry consumes ONE sealed round collected by the prior single-rank
# collect job (14ztx) and applies a SINGLE global action-position CE update on
# EVERY training rank (32-card DDP, all_reduce gradient sync), then publishes
# adapter/optimizer/scheduler/selection and exits.
#
# Purpose (per 2026-10-21 plan):
#   1. Recover from the sealed r000000 (validate sampling + init-adapter digest).
#   2. Complete exactly ONE optimizer update (global batch=128, per-card mb=1).
#   3. Profile: model load time, CE update time, peak VRAM, backward GPU set.
#   4. Confirm "CE ~1-2min on 32 cards" estimate with a real measurement.
# This is NOT the resident online loop yet — it is one update + profile, the
# unit for both first-round validation and each future step once the loop ships.
#
# Distinct from the 14ztx entry (run_student_onpolicy_sft_multinode.py):
#   * Launch: `accelerate launch --multi_gpu --num_machines 4 --num_processes 32`
#     (NOT torchrun). The Accelerator in sft_engine/model.py:make_accelerator
#     reads its process group from accelerate's launcher env. The entry unsets
#     CloudML's injected RANK/WORLD_SIZE that would otherwise make Accelerate
#     think this is single-process (CLAUDE.md PyTorchJob env-injection 铁律).
#   * Stack: sft_engine v2 (accelerate_engine.update_global_batch +
#     online_runtime.run_phase_job), NOT sft_ablation ce_update single rank0.
#   * Data: --resume-round points at a SEALED targets directory (14ztx's
#     rounds/r000000, 255 rows, policy digest 23e4175f...). --init-adapter
#     points at 14ztx's adapters/current (fresh LoRA, digest MATCHES the sealed
#     policy — inspect_sealed_round refuses if they differ).
#   * No collect/rollout in this job: phase_jobs layout, train-only. No
#     --train-data/--rl-data/--sec-corpus-root/--rollout-replicas.
#
# GPT-OSS-20B model-specific constraints (carried from only-opd/14ztx, verified):
#   1. micro-batch=1: accelerate_ddp refuses mb!=1 (cli.py), and GPT-OSS MoE
#      backward cannot grad-ckpt (k6v0p 铁律, peak ~74.7GB/80GB at mb1). DDP
#      splits the 128-row global batch across 32 ranks → ~4 rows/rank, so each
#      rank runs ~4 forward/backward slots (vs 14ztx's rank0 running 128).
#   2. bf16-dequant base (NOT mxfp4): Marlin PTX crash (v0dhh) + NaN fix.
#   3. SCAPE_FORCE_LOCAL_HARMONY INTENTIONALLY NOT SET (gpt-oss to_json crash,
#      xwl4c 铁证). No rollout here so no SCAPE_VLLM_GPU_MEM_UTIL either.
#   4. gradient_checkpointing off (cli.py accelerate_ddp refuses otherwise).
#   5. logits_mode=action_positions (cli.py accelerate_ddp refuses full).
#
# Ablation alignment preserved: same base (gpt-oss-20b-bf16-dequant), same
# init-adapter (14ztx's fresh lora-rank=32 seed=42 current), same sealed data,
# same eval as only-opd/no-gate/no-project. The ONLY change vs 14ztx is the
# CE backend: single rank0 → 32-rank DDP.

set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1:/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1/tinker-cookbook:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# --- Multi-node distributed env guard (accelerate reads MASTER_ADDR/NODE_RANK) ---
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set. Each pod needs its own rank id (0..3)." >&2
  exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set. CloudML custom_train multi-node must inject" \
       "node0's IP as MASTER_ADDR. Aborting to avoid 4 independent single-node runs." >&2
  exit 1
fi
: "${MASTER_PORT:=29623}"
export MASTER_ADDR NODE_RANK MASTER_PORT

# UNSET CloudML-injected RANK/WORLD_SIZE/LOCAL_* so accelerate computes the
# real world (32 processes across 4 machines) instead of W=1. CLAUDE.md
# PyTorchJob env-injection 铁律. accelerate launch sets these itself.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

# --- Shared output directory (MUST be identical on every node) ---
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node requires a shared --out path injected via --env OUT=... (identical on every pod). Aborting." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# Sealed round to recover from (14ztx collect output) + init-adapter (its fresh LoRA).
SEALED_ROUND="${SEALED_ROUND:-/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/rounds/r000000}"
INIT_ADAPTER="${INIT_ADAPTER:-/mnt/songzijun/outputs/gpt-oss-20b-student-onpolicy-sft-bf16-32gpu_20261019_true_p5/adapters/current}"
export SEALED_ROUND INIT_ADAPTER

export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

# tiktoken bundle: harness import chain (openai_harmony) still needs o200k_base
# (sft_engine still imports trim.training.sft_ablation.contracts which pulls harness).
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

# --- Presence checks on the mounted JuiceFS tree ---
nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_student_v2_train_phase.py
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/model.safetensors.index.json
# Sealed round + init-adapter MUST exist (train phase has no collect).
test -f "${SEALED_ROUND}/targets.json"
test -f "${INIT_ADAPTER}/adapter_config.json"
test -f "${INIT_ADAPTER}/adapter_model.safetensors"

# --- Install four-pack + accelerate + SFT/harness import chain deps (internal PyPI) ---
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece "pyserini==2.0.0" openai tenacity anthropic \
  chz tinker \
  tensorboard

# --- harness-1 transitive deps (sft_ablation.contracts → harness import chain) ---
python3 -m pip install --no-cache-dir --upgrade-strategy only-if-needed \
  json-repair langchain-text-splitters wikipedia-api modal baseten-performance-client \
  fastembed pysqlite3-binary wandb trl chz tinker rich

# --- .pth: prepend harness-1 + tinker-cookbook (same as 14ztx, r11 ROOT-CAUSE fix) ---
HARNESS1=/mnt/songzijun/Capability_Evolution/TRIM/external/harness-1
TINKER_CB=$HARNESS1/tinker-cookbook
SITE_PKGS=$(python3 -c "import site; print(site.getsitepackages()[0])")
cat > "$SITE_PKGS/zzz_harness1.pth" <<EOF
import sys; sys.path.insert(0, "$HARNESS1"); sys.path.insert(0, "$TINKER_CB")
EOF
test -f "$SITE_PKGS/zzz_harness1.pth"
echo "HARNESS1_PTH_WRITTEN $SITE_PKGS/zzz_harness1.pth"
python3 -E -c "import training; assert 'harness-1/training' in list(training.__path__)[0]; import training.train_rl, harness.ultra_core; print('HARNESS1_SUBPROCESS_IMPORT_OK', list(training.__path__))"

# --- ANSERINI_CLASSPATH for PyJNI/Lucene (sft_ablation.contracts may touch) ---
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Preflight: four-pack versions, 8 GPUs/node, SFT v2 engine import, sealed round + init-adapter digest match ---
python3 - <<'PY'
import importlib.metadata
import json
import os
import socket
import sys
import torch
import transformers
import vllm
import accelerate
from pathlib import Path

versions = {
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "pyserini": importlib.metadata.version("pyserini"),
    "peft": importlib.metadata.version("peft"),
    "structlog": importlib.metadata.version("structlog"),
    "chromadb": importlib.metadata.version("chromadb"),
    "gpu_count": torch.cuda.device_count(),
    "host": socket.gethostname(),
    "master_addr": os.environ.get("MASTER_ADDR"),
    "node_rank": os.environ.get("NODE_RANK"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["vllm"] == "0.19.1", f"vllm={versions['vllm']} != 0.19.1"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["accelerate"] == "1.13.0", f"accelerate={versions['accelerate']} != 1.13.0"
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"

# GPT-OSS-20B model probe: bf16-dequant (NO quant_method, NOT mxfp4).
model_cfg_path = "/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "quant_method": qcfg.get("quant_method", "NONE"),
}, flush=True)
assert model_cfg.get("architectures") == ["GptOssForCausalLM"], f"arch={model_cfg.get('architectures')}"
assert not qcfg.get("quant_method"), f"bf16-dequant must have NO quant_method, got {qcfg.get('quant_method')}"

# SFT v2 engine import chain (the new DDP backend).
from trim.training.sft_engine.online_runtime import run_student_v2, run_phase_job
from trim.training.sft_engine.accelerate_engine import update_global_batch
from trim.training.sft_engine.model import make_accelerator, load_ddp_model, build_optimizer, probe_action_logits
from trim.training.sft_ablation.cli import parse_sft_args
print("SFT_V2_ENGINE_IMPORT_OK", flush=True)

# Sealed round + init-adapter digest match (inspect_sealed_round refuses otherwise).
sys.path.insert(0, "/mnt/songzijun/Capability_Evolution/TRIM")
from trim.training.sft_ablation.io_util import sha256_tree
sealed = Path(os.environ["SEALED_ROUND"])
init_adapter = Path(os.environ["INIT_ADAPTER"])
targets = json.loads((sealed / "targets.json").read_text())
policy_digest = str(dict(targets.get("policy") or {}).get("digest") or "")
init_digest = sha256_tree(init_adapter)
print("SEALED_ROUND_PROBE", {
    "round_dir": str(sealed),
    "n_rows": len(targets.get("rows") or []),
    "policy_digest": policy_digest[:24],
    "init_adapter": str(init_adapter),
    "init_adapter_digest": init_digest[:24],
    "digest_match": policy_digest == init_digest,
}, flush=True)
assert policy_digest, "sealed round has no policy digest"
assert policy_digest == init_digest, f"init-adapter digest {init_digest[:16]} != sealed policy digest {policy_digest[:16]}"

print("SFT_V2_PREFLIGHT_OK", {"four_pack": True, "sft_v2_engine": True, "sealed_round": True,
                               "init_adapter_digest_match": True, "bf16_dequant": True}, flush=True)
PY

# --- copy bf16-dequant model weights to local /tmp before exec (same as 14ztx) ---
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

# --- Bidirectional rendezvous barrier (same as 14ztx; accelerate launch also needs all 4 cp done) ---
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

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_student_v2_train_phase.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/online_runtime.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/accelerate_engine.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/trim/training/sft_engine/model.py

cd /mnt/songzijun/Capability_Evolution

# accelerate launch: 4 machines x 8 processes = 32 DDP ranks. Accelerate sets
# RANK/WORLD_SIZE/LOCAL_RANK from its launcher; the entry already unset the
# CloudML-injected ones above. --mixed_precision bf16 matches make_accelerator.
exec accelerate launch \
  --multi_gpu \
  --num_machines 4 \
  --num_processes 32 \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port 29623 \
  --mixed_precision bf16 \
  /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_student_v2_train_phase.py \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --init-adapter "${INIT_ADAPTER}" \
  --resume-round "${SEALED_ROUND}" \
  --ce-backend accelerate_ddp \
  --online-layout phase_jobs \
  --train-env local_legacy \
  --num-updates 200 \
  --global-batch-size 32 \
  --batch-size 32 \
  --micro-batch-size 1 \
  --group-size 2 \
  --learning-rate 5e-6 \
  --lora-rank 32 \
  --max-model-len 4096 \
  --gradient-checkpointing non_reentrant \
  --max-new-tokens 2048 \
  --attn-implementation eager \
  --seed 42 \
  --save-every 1 \
  --log-every-microbatches 1 \
  --out "${OUT}"
