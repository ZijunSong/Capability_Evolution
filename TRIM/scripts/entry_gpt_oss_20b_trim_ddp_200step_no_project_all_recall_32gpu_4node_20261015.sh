#!/usr/bin/env bash
# Container entry for Harness-1 BC+ full TRIM (CISPO + projected-gap, no-project
# method) GPT-OSS-20B (bf16-dequant) via torch_ddp_lora backend on FOUR 8-GPU
# gm402-80g nodes (4 nodes x 8 cards = 32 GPUs, 200 steps, --component all =
# Teacher Full / Student Zero).
#
# Fork of entry_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261002_recall_only_bf16.sh
# (same gpt-oss-20B bf16-dequant + 32gpu 4node + final-answer-recall reward),
# changing the train axis from (rl, zero) to (no-project, all):
#   * --train_method no-project : CISPO + projected-gap on native anchors only
#     (SR-OPD/CISPO protocol, see CLAUDE.md "Capability Evolution / SCAPE").
#     Uses Student-realizable Teacher projection, no Teacher-prefix leakage,
#     sr_opd_ce + CISPO, unified joint optimizer contract; no legacy tool-token KL.
#   * --component all           : Teacher Full / Student Zero (the other baseline
#     axis vs the fork's --component zero). Omitting --component on no-project
#     defaults to all; we pass it explicitly.
#   * --rl-reward final-answer-recall : reward = final-answer recall ONLY, gated
#     by TRIM_REWARD_MODE=recall env (read inside the reward fn, not the caller
#     signature). This is the 4-term-reward-free recall-only variant.
#
# === Model choice: bf16-dequant (NOT mxfp4) ===
# CLAUDE.md铁律: GPT-OSS-20B 训练禁用原版 mxfp4. mxfp4+LoRA 在 torch_ddp_lora
# RL/TRIM 路径必崩 (Marlin 硬编码 PTX, cudaErrorUnsupportedPtxVersion); 且 recall
# reward 路径 mxfp4 有 sampled-logprob NaN (vllm_hybrid.py:206 token_id=0) bug,
# 已致 txgli/zxkvl FAILED@step0. bf16-dequant 同时规避 PTX 崩溃 + NaN, 是
# 200step 训练唯一稳妥配置. 镜像 micr...songzijun-cto 已含 triton_kernels 等依赖,
# 但模型权重仍须用 JuiceFS 上的 bf16-dequant 目录, 不是原版 mxfp4 目录.
#
# === Multi-node CloudML env injection (verified on 7nzzn/iowty/9ixnl/qqf4x...) ===
# CloudML custom_train framework=pytorch nodeNumber=4 creates a PyTorchJob CRD with
# master-0/worker pods + init-pytorch that injects MASTER_ADDR / NODE_RANK / (RANK /
# WORLD_SIZE / PET_*) on every pod. ImageCommand runs once PER POD (so each of the 4
# nodes runs this entry independently). The entry unsets the injected rank env so
# run_train_multinode.py's under_torchrun() returns False and main() self-launches
# torchrun --nnodes 4 --nproc_per_node 8 (world_size=32) deterministically.
#
# The REAL correctness gate is post-submit provenance (CLAUDE.md): stdout must show
# torchrun --nnodes 4 --nproc_per_node 8; checkpoints/<cell>/step_000001/ must have
# 32 optimizer.rank0000..0031.pt; vllm_sessions/ must have 32 _r0.._r31 sessions.
# Any one missing => actually single-node or partial run, result is NOT a valid
# "32-GPU 4-node DDP" artifact.
#
# === All known gpt-oss bf16 + torch_ddp_lora pitfalls (carried from fork) ===
#   * trainer_adapter.py:478 startup_timeout_s=3600.0 (FIX for 900s worker-start
#     timeout when DDP ranks each read the 40GB bf16 model). Even with the per-node
#     /tmp cp below, 8 ranks/node still load from local /tmp concurrently (~2-3min/
#     engine); 3600s is headroom.
#   * trainer_adapter.py resolve_queries 4-tuple unpack (FIX for 724d9fa ValueError).
#   * trainer_adapter.py:501 client.start() EADDRINUSE retry x3 (FIX for the
#     per-rank vLLM TP=1 TCPStore port race at step>=1).
#   * SCAPE_FORCE_LOCAL_HARMONY INTENTIONALLY NOT SET (gpt-oss _LocalConversation
#     no-to_json crash via openai_harmony Rust encoder; only gpt-oss hits Harmony).
#   * Model cp to /tmp BEFORE exec (8-rank JuiceFS concurrent-read hang fix).
#     In multi-node, EACH NODE does its own cp from JuiceFS to local /tmp (4
#     concurrent JuiceFS reads of 40GB bf16 across nodes, vs 32 if we let every
#     rank read JuiceFS directly). Local /tmp read is ~3min/engine.
#   * curate-clipping tolerant-skip patch (ethqp/axztt path: action_is_visible()
#     returns bool, tinker_opd_datum skips unprojectable step instead of abort).
#
# mb1 (NOT 4): CLAUDE.md mb 决策铁律 — gpt-oss-20B MoE backward 无法 enable
# gradient checkpointing, mb>=2 会撞 backward OOM (peak_mem ~74.7GB/80GB 恒定).
# 多节点 4node x 8 rank = 32 replica 不改 per-rank 显存预算 (每 rank 仍是 TP=1
# 单卡), 所以 mb1 + gpu-mem 0.85 仍 fit 80GB. 多节点用 N rank 数据并行分摊
# backward, 不靠 mb 杠杆. (Qwen3-4B dense 才用 mb4, gpt-oss 必须 mb1.)
#
# torch_ddp_lora path: NO CUDA_LAUNCH_BLOCKING (DDP is real multi-proc torchrun,
# not the accelerate device_map=auto single-proc that needs the sync crutch).
# NO DISABLE_CUSTOM_ALL_REDUCE (per-rank vLLM TP=1, no TP all-reduce to fix).
# trainer_adapter.py:181 hardcodes init_dist_if_needed(backend="nccl"); NCCL
# backend supports broadcast_object_list via internal tensor serialization, so
# the object collectives (broadcast_object, all_gather_object) work under NCCL
# (verified by the successful world_size=8/32 DDP runs on 7nzzn/iowty/9ixnl).
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# --- Multi-node distributed env guard ---
# CloudML custom_train framework=pytorch nodeNumber=4 must give each pod a distinct
# NODE_RANK and a shared MASTER_ADDR (node0's IP). If these are absent the launcher
# cannot form a 4-node rendezvous.
#
# CloudML MIGHT inject RANK / WORLD_SIZE / PET_* (the PyTorchJob CRD probe saw
# RANK=0 / WORLD_SIZE=1 / PET_NPROC_PER_NODE=auto injected even on a single-node
# job). If custom_train does the same, under_torchrun() in run_train_multinode.py
# could short-circuit main() straight into _worker WITHOUT launching torchrun, so
# the 4 nodes would never rendezvous into a world_size=32 group. To make the launch
# deterministic regardless of what CloudML injects, we:
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
echo "REWARD_ENV reward_mode=${TRIM_REWARD_MODE:-recall} turn_penalty=${TRIM_TURN_PENALTY:-unset} turn_norm=${TRIM_TURN_NORM:-unset}"

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
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json
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
    "reward_mode": os.environ.get("TRIM_REWARD_MODE"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["vllm"] == "0.19.1", f"vllm={versions['vllm']} != 0.19.1"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["pyserini"] == "2.0.0", f"pyserini={versions['pyserini']} != 2.0.0"
# Per-node: each pod must see 8 GPUs (pinned 1-per-rank later by pin_local_cuda_device).
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
assert versions["o200k_base_present"]
assert versions["reward_mode"] == "recall", f"TRIM_REWARD_MODE={versions['reward_mode']} != recall (final-answer-recall reward requires TRIM_REWARD_MODE=recall)"
# GPT-OSS-20B model probe: bf16-dequant checkpoint (quant_method must be absent/NONE,
# NOT mxfp4 — bf16 avoids the mxfp4 sampled-logprob NaN crash at vllm_hybrid.py:206
# that failed txgli/zxkvl step0 rollout, and the mxfp4+LoRA Marlin PTX crash).
model_cfg_path = "/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "quant_method": qcfg.get("quant_method", "NONE"),
}, flush=True)
assert model_cfg.get("architectures") == ["GptOssForCausalLM"]
assert not qcfg.get("quant_method"), f"bf16-dequant must have NO quant_method, got {qcfg.get('quant_method')}"
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

# --- resume: assert latest ckpt + STEP_COMPLETE present (resume target) ---
# akvlv FAILED @ step1 backward done~128/136 (runTime 8057s) on 2026-10-15:
# master-0 pid=486 (rank0 worker) JVM (pyjnius/Pyserini LuceneSearcher) crashed
# with SIGSEGV at pthread_mutex_lock+0x4 (native JNI thread, JRE 21.0.11 g1gc,
# SEGV_ACCERR). torchrun agent got death signal -> SIGTERM all 8 workers ->
# SignalException. NOT OOM (peak 74714MB < 80GB), NOT NaN, NOT code bug — an
# INFRASTRUCTURE-level pyjnius/JVM native-thread SEGV, OCCASIONAL (step0
# backward done=0->146 did NOT crash at the same point; JVM elapsed 7355s =
# long-resident g1-gc/detach SEGV). step_000001 is COMPLETE, so --resume
# auto-finds <OUT>/checkpoints/no_project/latest (=step_000001) and continues
# from step1's rollout (step1 ckpt was NOT written before the crash, so step1
# reruns cleanly — no datum duplication: load_training_resume reads
# sampler/rng state, assert_resume_identity rejects config drift). rdzv-id gets
# a -resume suffix so the rendezvous endpoint is namespaced apart from the
# original akvlv run (stale peers from the killed run cannot join).
test -d "$OUT/checkpoints/no_project/latest"
test -f "$OUT/checkpoints/no_project/latest/STEP_COMPLETE"
test -f "$OUT/checkpoints/no_project/latest/optimizer.pt"
test -f "$OUT/checkpoints/no_project/latest/STEP_MANIFEST.json"
echo "RESUME_TARGET latest=$(readlink -f "$OUT/checkpoints/no_project/latest") cell=no_project"

# --- resume: purge residual step1 (b2) vllm_sessions from the failed attempt ---
# akvlv's step1 rollout created no_project_b2_r0..r31 session dirs before the
# JVM SIGSEGV killed the job mid-backward. --resume reruns step1 with the SAME
# batch id (global_rollout_batch increments 1->2, session = no_project_b2_rN),
# so stale b2 dirs + READY/RESULT files from the killed run would make vllm_wait
# falsely see READY and skip vLLM init (the eval "fresh_eval_dir" failure mode,
# training-side). Purge ONLY the failed step1's b2 session dirs (keep b1 = step0
# reference, keep checkpoints/ entirely). Rollout shards are under tmp/ empty.
if [ -d "$OUT/vllm_sessions" ]; then
  rm -rf "$OUT"/vllm_sessions/no_project_b2_r*
fi
echo "RESUME_PURGE b2 sessions purged; remaining b-sessions: $(ls "$OUT/vllm_sessions" 2>/dev/null | grep -c 'no_project_b' )"


# --- FIX2: copy bf16-dequant model weights to local /tmp before exec ---
# Each node does its own cp (4 concurrent JuiceFS reads of 40GB across nodes),
# then all 8 local ranks load from fast local disk. Runs BEFORE vLLM worker
# spawn so it does NOT count against the 3600s worker-start window.
MODEL_SRC=/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant
MODEL_LOCAL=/tmp/gpt-oss-20b-bf16-dequant
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
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py \
  --nnodes 4 \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dist-backend nccl \
  --rdzv-backend static \
  --rdzv-id gptoss20b-trim-no-project-all-recall-resume-32 \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/gpt-oss-20b-bf16-dequant \
  --train_method no-project \
  --component all \
  --train-env local_legacy \
  --train-data sec \
  --rl-reward final-answer-recall \
  --rl-data /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data \
  --sec-corpus-root /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus \
  --train-steps 200 \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 1 \
  --train-heartbeat-every 8 \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --out "${OUT}" \
  --resume
