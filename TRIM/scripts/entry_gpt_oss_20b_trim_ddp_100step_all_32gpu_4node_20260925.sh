#!/usr/bin/env bash
# Container entry for Harness-1 BC+ TRIM (OPD) GPT-OSS-20B (original mxfp4,
# non-dequant per user instruction) via torch_ddp_lora backend on FOUR 8-GPU
# gm402-80g nodes (4 nodes x 8 cards = 32 GPUs, 100 steps, --component all =
# H_full, all advanced components ON). --train_method trim (NOT rl).
#
# This is the TRIM/ALL 100-step MULTI-NODE variant. Derived from
# entry_gpt_oss_20b_rl_ddp_100step_zero_32gpu_4node_20260925.sh (the 100-step
# RL/ZERO 4-node job ghynp). Deltas vs that entry:
#   * --train_method rl -> trim  (OPD teacher projection, NOT on-policy RL/CISPO)
#   * --component zero -> all    (H_full: all advanced components ON; zero was H_zero)
#   * --rdzv-id gptoss20b-rl-32-100step -> gptoss20b-trim-32-100step (distinct
#     rendezvous, so a simultaneous RL/ZERO 4-node run and this TRIM/ALL run do
#     not collide on the same torchrun rendezvous id even if both live at once)
#   * preflight adds curate-clipping tolerant-skip patch assertions
#     (action_is_visible + skip call site + n_skip_clipped_doc). RL/ZERO did NOT
#     assert these (zero-mask rollouts never trigger the teacher-curate-references-
#     clipped-docs path); TRIM/ALL DOES and the single-node TRIM/ALL recov1 (ethqp)
#     proved this patch is REQUIRED or step0 aborts on doc 0000950103-25-007493_26.
#   * --out "${OUT}" unchanged in MECHANISM (still env-injected, shared across
#     pods); only the OUT *value* in the submit script points at the trim/all
#     output dir.
# Everything else is identical: mb=1, gpu-mem 0.90, 4 nodes, unset rank env,
# model cp /tmp per node, four code-tree patches asserted in preflight.
#
# === Multi-node launch path ===
# run_train_multinode.py is the real multi-node launcher (unlike run_train.py).
# Each node runs ONE Scheme A process that itself torchruns --nnodes 4
# --nproc_per_node 8 (world_size=32), shards on-policy query groups across 32
# ranks, gathers rollout trajectories onto rank 0 via shared disk
# (<out>/tmp/rollout_shards), rank 0 runs the DDP-LoRA CISPO optimizer step,
# then every rank barriers and reloads the fresh adapter for the next rollout.
#
# === CloudML custom_train framework=pytorch nodeNumber=4 (verified by 7nzzn) ===
# Creates a PyTorchJob CRD (master-0 + worker-0/1/2 pods + init-pytorch). The
# 2026-09-15 probe's three risks are all RESOLVED by the 7nzzn run:
#   * starfs-user-pv FailedMount: did NOT recur (4 pods deploying->running).
#   * cross-node object collectives: SAFE — PyTorch auto-uses Gloo for
#     rendezvous + broadcast_object_list/all_gather_object, nccl for DDP
#     gradient all-reduce. 7nzzn stdout "[Gloo] Rank 0-31 connected to 31 peers".
#   * env injection: custom_train injects MASTER_ADDR (master pod DNS) +
#     NODE_RANK, but NOT RANK/WORLD_SIZE/PET_* (7nzzn PREFLIGHT: rank_env=None,
#     world_size_env=None, pet_nproc_per_node=None). The unset-RANK block below
#     is defensive insurance (harmless).
#
# === All known gpt-oss mxfp4 + torch_ddp_lora pitfalls (carried from 200step) ===
#   * trainer_adapter.py:478 startup_timeout_s=3600.0 (FIX for 900s worker-start
#     timeout when DDP ranks each read 13GB mxfp4). Even with the per-node /tmp
#     cp below, 8 ranks/node load from local /tmp concurrently (~2min/engine);
#     3600s is headroom.
#   * trainer_adapter.py resolve_queries 4-tuple unpack (FIX for 724d9fa ValueError).
#   * trainer_adapter.py:501 client.start() EADDRINUSE retry x3 (FIX for the
#     per-rank vLLM TP=1 TCPStore port race at step>=1; 32 ranks并发更易撞端口).
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
# 74.7GB/80GB constant per rank on single-node 8-GPU runs and 7nzzn). With 4
# nodes x 8 ranks = 32 replicas, per-rank memory budget is UNCHANGED (each rank
# is still TP=1 single-GPU), so mb1 + gpu-mem 0.90 still fits 80GB. NOTE: the
# reference command in the /goal used mb=4; that value FAILED on t-k6v0p (step0
# student forward OOM, 8 ranks simultaneous). mb=1 is the verified-safe floor.
#
# torch_ddp_lora path: NO CUDA_LAUNCH_BLOCKING (DDP is real multi-proc torchrun,
# not the accelerate device_map=auto single-proc that needs the sync crutch).
# NO DISABLE_CUSTOM_ALL_REDUCE (per-rank vLLM TP=1, no TP all-reduce to fix).
# trainer_adapter.py:181 hardcodes init_dist_if_needed(backend="nccl"); NCCL
# backend supports broadcast_object_list via internal tensor serialization, so
# the object collectives work under NCCL (verified by 7nzzn world_size=32).
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# --- Multi-node distributed env guard ---
# CloudML custom_train framework=pytorch nodeNumber=4 gives each pod a distinct
# NODE_RANK and a shared MASTER_ADDR (node0's DNS name, e.g. tj-...-master-0).
# 7nzzn confirmed custom_train injects MASTER_ADDR + NODE_RANK but NOT
# RANK/WORLD_SIZE/PET_*. The unset block below is defensive insurance: if a
# future CRD change starts injecting RANK/WORLD_SIZE, under_torchrun() would
# short-circuit main() into _worker WITHOUT launching torchrun, so the 4 nodes
# would never rendezvous. Clearing these makes the launch deterministic.
#   1. Capture MASTER_ADDR + NODE_RANK BEFORE clearing injected rank env.
#   2. unset RANK / WORLD_SIZE / LOCAL_RANK / LOCAL_WORLD_SIZE / PET_* / the
#      torchelastic markers, so under_torchrun() returns False and main() takes
#      the needs_torchrun() -> _exec_torchrun branch (self-launched torchrun).
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set. Each pod needs its own rank id (0..3)." >&2
  exit 1
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "FATAL: MASTER_ADDR is not set. CloudML custom_train multi-node must inject" \
       "node0's IP/DNS as MASTER_ADDR. If this pod has no MASTER_ADDR, the job will" \
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
# step 0 backward crashes with FileNotFoundError (korur first-attempt bug, FIXED on
# 7nzzn recov1). OUT is injected via submit --env so all 4 pods share the exact same
# path. Never use $(date) here.
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
# curate-clipping tolerant-skip patch (REQUIRED for --component all / TRIM, NOT
# needed for rl/zero): the single-node TRIM/ALL first attempt (x2j3u) aborted at
# step0 datum because a teacher curate referenced docs dropped by context
# clipping (doc 0000950103-25-007493_26 / 0001213900-25-049013_54). The fix:
# action_is_visible() returns bool (instead of assert_abort), and
# tinker_opd_datum skips the unprojectable step + counts it in
# n_skip_clipped_doc. Without this patch, multi-node TRIM/ALL would hit the same
# deterministic step0 abort. Verified present in the JuiceFS code tree above.
ae = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/action_encoding.py").read_text()
assert "def action_is_visible(" in ae, "action_is_visible patch missing"
tod = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/training/tinker_opd_datum.py").read_text()
assert "action_is_visible(step.target_action, visible)" in tod, "skip-clipped-doc call site missing"
assert "n_skip_clipped_doc" in tod, "skip counter missing"
from trim.training.action_encoding import action_is_visible
assert action_is_visible({"name": "curate", "arguments": {"add_ids": ["X"]}}, ["Y"]) is False
assert action_is_visible({"name": "curate", "arguments": {"add_ids": ["X"]}}, ["X"]) is True
print("PATCH_CHECK_OK", {"startup_timeout_s": True, "resolve_queries_4tuple": True, "eaddrinuse_retry": True, "skip_clipped_doc": True}, flush=True)
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
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py \
  --nnodes 4 \
  --nproc-per-node 8 \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --dist-backend nccl \
  --rdzv-backend static \
  --rdzv-id gptoss20b-trim-32-100step \
  --training-backend torch_ddp_lora \
  --harness Harness-1 \
  --benchmark bcplus_full \
  --model_name /tmp/gpt-oss-20b \
  --train_method trim \
  --component all \
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
  --out "${OUT}"
