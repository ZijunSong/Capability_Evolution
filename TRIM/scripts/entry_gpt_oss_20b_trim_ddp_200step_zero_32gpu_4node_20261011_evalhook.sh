#!/usr/bin/env bash
# Container entry for Harness-1 BC+ full TRIM (SR-OPD/CISPO distill) GPT-OSS-20B (original
# mxfp4, non-dequant per user instruction) via torch_ddp_lora backend on FOUR 8-GPU
# gm402-80g nodes (4 nodes x 8 cards = 32 GPUs, --component zero = H_zero baseline).
#
# === EVALHOOK VARIANT (2026-10-11 zero-trim) ===
# Fork of entry_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261003_evalhook.sh.
# 只改 train_method rl→trim + rdzv-id 区分（gptoss20b-trim-zero-32）。其余参数
#（mxfp4 原版、默认 reward、mb1、gpu-mem0.90、200step、evalhook gate）与 zero-rl retrain1 一致。
# trim = train_method=trim（SR-OPD + CISPO 蒸馏，而非 RL outcome reward）。
#
# This is the MULTI-NODE variant of entry_gpt_oss_20b_rl_ddp_200step_zero_8gpu_20260924.sh
# (single-node 8-GPU). The single-node entry execs run_train.py directly; this
# entry execs run_train_multinode.py instead, which is the real multi-node
# launcher: each node runs ONE Scheme A process that itself torchruns
# --nnodes 4 --nproc_per_node 8 (world_size=32), shards on-policy query groups
# across the 32 ranks, gathers rollout trajectories onto rank 0 via shared disk
# (<out>/tmp/rollout_shards), rank 0 runs the DDP-LoRA CISPO optimizer step,
# then every rank barriers and reloads the fresh adapter for the next rollout.
#
# === Multi-node CloudML env injection (UNVERIFIED for custom_train, see below) ===
# CLAUDE.md (2026-09-15 probe) established that CloudML `framework: pytorch`
# PyTorchJob CRD injects RANK=0 / WORLD_SIZE=1 / PET_* / MASTER_ADDR / MASTER_PORT
# on every pod via init-pytorch, and that ImageCommand runs once PER POD (so each
# of the 4 nodes runs this entry independently). Whether custom_train
# (CustomTrainJob CRD, NOT PyTorchJob CRD) injects the same env on nodeNumber=4
# is UNVERIFIED — the 2026-09-15 probe was killed at deploying on a CPU-only
# 2-node job (starfs-user-pv FailedMount) before any pod ran the probe script.
#
# This entry is written to work under EITHER behavior:
#   * If CloudML injects MASTER_ADDR + NODE_RANK (+ RANK/WORLD_SIZE) -> we pass
#     them through to run_train_multinode.py, which handles them via parse_dist_argv
#     (env fallback for --master-addr / --node-rank) and under_torchrun().
#   * If CloudML injects nothing -> the job fails fast at the MASTER_ADDR guard
#     below with a clear message (rather than silently running 4 independent
#     single-node trains that corrupt --out).
#
# The REAL correctness gate is post-submit provenance (see CLAUDE.md): stdout
# must show torchrun --nnodes 4 --nproc_per_node 8, checkpoints/<cell>/step_000001/
# must have 32 optimizer.rank0000..0031.pt, vllm_sessions/ must have 32
# _r0.._r31 sessions. Any one missing => actually single-node or partial run,
# result is NOT a valid "32-GPU 4-node DDP" artifact.
#
# === All known gpt-oss mxfp4 + torch_ddp_lora pitfalls (carried from single-node) ===
#   * trainer_adapter.py:478 startup_timeout_s=3600.0 (FIX for 900s worker-start
#     timeout when DDP ranks each read 13GB mxfp4). Even with the per-node /tmp
#     cp below, 8 ranks/node still load from local /tmp concurrently (~2min/
#     engine); 3600s is headroom.
#   * trainer_adapter.py resolve_queries 4-tuple unpack (FIX for 724d9fa ValueError).
#   * trainer_adapter.py:501 client.start() EADDRINUSE retry x3 (FIX for the
#     per-rank vLLM TP=1 TCPStore port race at step>=1).
#   * SCAPE_FORCE_LOCAL_HARMONY INTENTIONALLY NOT SET (gpt-oss _LocalConversation
#     no-to_json crash via openai_harmony Rust encoder; only gpt-oss hits Harmony).
#   * Model cp to /tmp BEFORE exec (8-rank JuiceFS concurrent-read hang fix).
#     In multi-node, EACH NODE does its own cp from JuiceFS to local /tmp
#     (4 concurrent JuiceFS reads of 13.8GB across nodes, vs 32 if we let every
#     rank read JuiceFS directly). Local /tmp read is ~2min/engine.
#   * curate-clipping tolerant-skip patch (ethqp / axztt path: action_is_visible()
#     returns bool, tinker_opd_datum skips unprojectable step instead of abort).
#
# mb1 (NOT 4): mb>=2 backward OOMs on 20B mxfp4 (gpt-oss MoE backward cannot
# enable gradient checkpointing, mb1 is the primary OOM defense; peak_mem
# 74.7GB/80GB constant per rank on the single-node 8-GPU runs). With 4 nodes
# x 8 ranks = 32 replicas, per-rank memory budget is UNCHANGED (each rank is
# still TP=1 single-GPU), so mb1 + gpu-mem 0.90 still fits 80GB.
#
# torch_ddp_lora path: NO CUDA_LAUNCH_BLOCKING (DDP is real multi-proc torchrun,
# not the accelerate device_map=auto single-proc that needs the sync crutch).
# NO DISABLE_CUSTOM_ALL_REDUCE (per-rank vLLM TP=1, no TP all-reduce to fix).
# trainer_adapter.py:181 hardcodes init_dist_if_needed(backend="nccl"); NCCL
# backend supports broadcast_object_list via internal tensor serialization, so
# the object collectives (broadcast_object, all_gather_object) work under NCCL
# (verified by the successful single-node world_size=8 DDP runs).
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# --- Multi-node distributed env guard ---
# CloudML custom_train framework=pytorch nodeNumber=4 must give each pod a
# distinct NODE_RANK and a shared MASTER_ADDR (node0's IP). If these are absent
# the launcher cannot form a 4-node rendezvous.
#
# CloudML MIGHT inject RANK / WORLD_SIZE / PET_* (the 2026-09-15 PyTorchJob CRD
# probe saw RANK=0 / WORLD_SIZE=1 / PET_NPROC_PER_NODE=auto injected even on a
# single-node job). If custom_train does the same, under_torchrun() in
# run_train_multinode.py could short-circuit main() straight into _worker
# WITHOUT launching torchrun, so the 4 nodes would never rendezvous into a
# world_size=32 group. To make the launch deterministic regardless of what
# CloudML injects, we:
#   1. Capture MASTER_ADDR + NODE_RANK (the two things we actually need from the
#      scheduler) BEFORE clearing injected rank env.
#   2. unset RANK / WORLD_SIZE / LOCAL_RANK / LOCAL_WORLD_SIZE / PET_* / the
#      torchelastic markers, so under_torchrun() returns False and main() takes
#      the needs_torchrun() -> _exec_torchrun branch (self-launched torchrun).
# We then pass --node-rank / --master-addr explicitly, so parse_dist_argv does
# not rely on any injected env.
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set. Each pod needs its own rank id (0..3)." >&2
  exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set. CloudML custom_train multi-node must inject" \
       "node0's IP as MASTER_ADDR. If this pod has no MASTER_ADDR, the job will" \
       "fail to rendezvous. Aborting to avoid 4 independent single-node runs." >&2
  exit 1
fi
: "${MASTER_PORT:=29500}"
export MASTER_ADDR NODE_RANK MASTER_PORT

# Strip any CloudML-injected torchrun-style rank env so run_train_multinode.py's
# under_torchrun() (which keys on WORLD_SIZE>1 or non-auto PET_NPROC_PER_NODE)
# returns False and main() self-launches torchrun via _exec_torchrun. This makes
# the 4-node rendezvous deterministic instead of dependent on injection flavor.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

# --- Shared output directory (MUST be identical on every node) ---
# `run_train_multinode.py` writes per-rank rollout shards to <out>/tmp/rollout_shards
# and rank 0 interleaves them via all_gather_via_disk. If each node uses a different
# <out> (e.g. a per-pod $(date) timestamp), the shards are invisible across nodes and
# step 0 backward crashes with FileNotFoundError. OUT is injected via submit --env
# so all 4 pods share the exact same path. Never use $(date) here.
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node requires a shared --out path injected via --env OUT=... (identical on every pod). Aborting." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# FIX1 (to_json bug): SCAPE_FORCE_LOCAL_HARMONY is INTENTIONALLY NOT SET.
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
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py
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
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece pyserini openai tenacity anthropic

# --- ANSERINI_CLASSPATH for PyJNI/Lucene ---
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Preflight: assert four-pack versions, 8 GPUs (per node), JVM, import chain, Lucene ---
python3 - <<'PY'
import importlib.metadata
import json
import os
import socket
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
    "host": socket.gethostname(),
    "master_addr": os.environ.get("MASTER_ADDR"),
    "node_rank": os.environ.get("NODE_RANK"),
    "rank_env": os.environ.get("RANK"),
    "world_size_env": os.environ.get("WORLD_SIZE"),
    "pet_nproc_per_node": os.environ.get("PET_NPROC_PER_NODE"),
    "pet_node_rank": os.environ.get("PET_NODE_RANK"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["vllm"] == "0.19.1", f"vllm={versions['vllm']} != 0.19.1"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["pyserini"] == "2.0.0", f"pyserini={versions['pyserini']} != 2.0.0"
# Per-node: each pod must see 8 GPUs (pinned 1-per-rank later by pin_local_cuda_device).
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
assert versions["o200k_base_present"]
# GPT-OSS-20B model probe: original mxfp4 checkpoint.
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
# wrap (full replica per rank), NOT the FSDP2 sharded wrap.
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
# Each node does its own cp (4 concurrent JuiceFS reads of 13.8GB across nodes),
# then all 8 local ranks load from fast local disk. Runs BEFORE vLLM worker
# spawn so it does NOT count against the 3600s worker-start window.
MODEL_SRC=/mnt/songzijun/models/openai/gpt-oss-20b
MODEL_LOCAL=/tmp/gpt-oss-20b
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"
test -f "$MODEL_LOCAL/model.safetensors.index.json" || true
echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
cd /mnt/songzijun/Capability_Evolution

# --- Evalhook gate: 按 EVAL_TARGET_STEP 限制 --train-steps，按 RESUME 加 --resume ---
# orchestrator 每段注入 EVAL_TARGET_STEP（50/100/150/200）+ RESUME（0 首段/1 续段）。
# 首段 RESUME=0：fresh start，--train-steps 50，跑完 step50 落盘 STEP_COMPLETE 后 job 退出。
# 续段 RESUME=1：--resume 从 latest(=上一段终点) 续训，--train-steps 下一个目标，--rdzv-id 加 -resume 防 rdzv 冲突。
: "${EVAL_TARGET_STEP:?EVAL_TARGET_STEP is required (50/100/150/200)}"
: "${RESUME:=0}"
TRAIN_STEPS_ARG="--train-steps ${EVAL_TARGET_STEP}"
RESUME_ARG=""
RDZV_ID="gptoss20b-trim-zero-32"
if [ "${RESUME}" = "1" ]; then
  RESUME_ARG="--resume"
  RDZV_ID="gptoss20b-trim-zero-32-resume"
  # resume preflight: latest symlink + STEP_COMPLETE marker（避 resume 一个未写完的 ckpt）
  test -d "${OUT}/checkpoints/rl/latest"
  test -f "${OUT}/checkpoints/rl/latest/STEP_COMPLETE"
  echo "RESUME_TARGET latest=$(readlink -f "${OUT}/checkpoints/rl/latest") target_step=${EVAL_TARGET_STEP}"
fi
echo "EVALHOOK_GATE eval_target_step=${EVAL_TARGET_STEP} resume=${RESUME} rdzv_id=${RDZV_ID} train_steps_arg=${TRAIN_STEPS_ARG}"

exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py \
  --nnodes 4 \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dist-backend nccl \
  --rdzv-backend static \
  --rdzv-id "${RDZV_ID}" \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/gpt-oss-20b \
  --train_method trim \
  --component zero \
  --train-env local_legacy \
  --train-data sec \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  ${TRAIN_STEPS_ARG} \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 1 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  ${RESUME_ARG} \
  --out "${OUT}"
