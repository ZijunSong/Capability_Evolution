#!/usr/bin/env bash
# CloudML imageCommand entry — H0 zero rollout (fixed-base measurement before any
# harness search or TRIM update) on bcplus_full (830 queries) for Qwen3-4B-Instruct-2507
# (dense bf16, NOT mxfp4, NO adapter — this is the bare-base H0 measurement).
#
# FOUR 8-GPU gm402-80g nodes (4 nodes x 8 cards = 32 GPUs). layout = 4x8: each node
# runs run_h0_zero_rollout.py --nnodes 4 --rollout-replicas 8 --node-rank $NODE_RANK,
# which (a) shards the 830-query pool round-robin across 32 replicas, (b) calls
# run_eval.py --harness Harness-1 --benchmark bcplus_full --component zero
# --evaluation-path legacy_local --tp 8 --tensor-parallel-size 1 --eval-query-ids <node shard>
# per node → 8 single-GPU vLLM engines (TP=1) per node = 32 total, each scoring its query
# shard with the BC+ local_bm25 retrieval, (c) writes NODE_DONE.json; node 0 polls the
# shared --out for all 4 NODE_DONE.json, merges PER_QUERY.jsonl traces, and writes
# ZERO_ROLLOUT.json with evidence_recall (the H0 fixed-base number).
#
# === run_h0_zero_rollout.py does NOT use torchrun/Gloo/rdzv ===
# It is a FILE-BASED multi-node coordinator: all 4 nodes share the same JuiceFS --out
# dir (injected via --env OUT=...); node 0 waits for out/nodes/node{0..3}/NODE_DONE.json.
# So this entry does NOT call run_train_multinode.py and does NOT pass --master-addr /
# --rdzv-id / --nproc-per-node. The only distributed env we need is NODE_RANK (to pick
# this node's shard) and a shared OUT. MASTER_ADDR is NOT required (kept as a soft check
# that warns instead of aborting, since legacy_local eval spawns its own per-replica
# vLLM subprocesses with no cross-node rendezvous).
#
# === Multi-node CloudML env injection (verified by iowty/9ixnl/7nzzn) ===
# custom_train framework=pytorch nodeNumber=4 creates a PyTorchJob CRD. CloudML injects
# NODE_RANK on every pod. We unset RANK/WORLD_SIZE/PET_* defensively so run_eval.py's
# legacy_local path (which spawns trim.eval.eval_shard_worker subprocesses) does not
# inherit stale torchrun-style rank env.
#
# === Qwen3-4B specifics (dense bf16) ===
#   * NO triton_kernels / Marlin / mxfp4 dance. vLLM 0.19.1 loads Qwen3 via standard bf16.
#   * SCAPE_FORCE_LOCAL_HARMONY=1 SET (harmless on Qwen3; is_hf_chat_family=True).
#   * NO adapter (H0 = bare base measurement). gpu-mem 0.90 (Qwen3-4B 7.6GB weights,
#     eval-only no backward, KV cache can use the full budget; matches the verified
#     Qwen3-4B eval entries).
#   * Model cp to /tmp: 4 nodes x 8 vLLM engines = 32 concurrent JuiceFS readers of the
#     7.6G model on first shard; per-node local cp (~1min) defensively avoids contention.
#
# === Deps: minimal Harness-G/legacy_local eval set (NOT the heavy training set) ===
# torch2.10.0 / vllm0.19.1 / transformers4.57.6 / pyserini2.0.0 + pyjnius + structlog +
# openai_harmony (Qwen3 uses HF chat template but the import chain still needs it) +
# peft (eval worker import) + tiktoken bundle. chromadb NOT installed; legacy_local eval
# with SCAPE_LOCAL_OPENAI_EMBEDDINGS=1 skips it (do NOT set HARNESS1_FORBID_CHROMA — that
# is harness-1 upstream_api only).
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# --- Multi-node distributed env guard ---
NODE_RANK="${NODE_RANK:-${PET_NODE_RANK:-${NODE_INDEX:-${NODE_NUMBER:-}}}}"
if [ -z "${NODE_RANK:-}" ]; then
  echo "FATAL: NODE_RANK is not set. Each pod needs its own rank id (0..3)." >&2
  exit 1
fi
# legacy_local eval has no cross-node rendezvous, so MASTER_ADDR is not strictly
# required; warn only (do not abort) if it is missing.
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "WARN: MASTER_ADDR not set (legacy_local H0 rollout does not rendezvous cross-node; continuing)." >&2
fi
export NODE_RANK

# Strip any CloudML-injected torchrun-style rank env so the legacy_local eval path
# (which spawns its own per-replica vLLM subprocesses) does not see stale rank env.
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "H0_ZERO_ENV node_rank=${NODE_RANK} host=$(hostname) master_addr=${MASTER_ADDR:-unset}"

# --- Shared output directory (MUST be identical on every node) ---
# run_h0_zero_rollout.py writes per-node shards to <out>/nodes/node{rank}/ and node 0
# polls <out>/nodes/node{0..3}/NODE_DONE.json. If each node uses a different <out>, node
# 0 never sees the other shards and merge times out. OUT is injected via submit --env.
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node H0 rollout requires a shared --out path injected via --env OUT=... (identical on every pod). Aborting." >&2
  exit 1
fi
export OUT
echo "H0_ZERO_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# Qwen3-4B: SCAPE_FORCE_LOCAL_HARMONY=1 (harmless on Qwen). Placeholder key so
# pyserini.encode._openai module-level openai.OpenAI() does not raise at import
# (legacy_local eval uses local BM25, never calls OpenAI API).
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1

# tiktoken bundle: harness import chain (openai_harmony) still needs o200k_base.
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

# --- Presence checks on the mounted JuiceFS tree ---
nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_h0_zero_rollout.py
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_eval.py
test -f /mnt/songzijun/models/Qwen3-4B-Instruct-2507/config.json
# BC+ corpus + bm25 index (bcplus_full retrieval backend)
test -f /mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl
test -d /mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25
# bcplus_full query pool (830) manifest
test -f /mnt/songzijun/Capability_Evolution/TRIM/manifests/browsecomp_plus_830/SPLIT.json

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

# --- Install four-pack + eval import chain deps (internal PyPI) ---
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog \
  tqdm regex sympy latex2sympy2 word2number sentencepiece "pyserini==2.0.0" pyjnius \
  openai tenacity anthropic
python3 -m pip install --no-cache-dir --no-deps --index-url "${PYPI_INDEX}" "openai_harmony" \
  || echo "[deps] WARN: openai_harmony --no-deps install issue"

# --- ANSERINI_CLASSPATH for PyJNI/Lucene ---
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Preflight: assert four-pack versions, 8 GPUs (per node), JVM, import chain,
#     Qwen3-4B dense-bf16 model, bcplus_full Lucene index, and the h0_zero_rollout entry ---
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
    "gpu_count": torch.cuda.device_count(),
    "JVM_PATH": os.environ["JVM_PATH"],
    "JAVA_HOME": os.environ["JAVA_HOME"],
    "TIKTOKEN_ENCODINGS_BASE": os.environ.get("TIKTOKEN_ENCODINGS_BASE"),
    "o200k_base_present": os.path.isfile(
        os.path.join(os.environ["TIKTOKEN_ENCODINGS_BASE"], "o200k_base.tiktoken")
    ),
    "java_ok": True,
    "host": socket.gethostname(),
    "node_rank": os.environ.get("NODE_RANK"),
    "master_addr": os.environ.get("MASTER_ADDR"),
    "rank_env": os.environ.get("RANK"),
    "world_size_env": os.environ.get("WORLD_SIZE"),
    "pet_nproc_per_node": os.environ.get("PET_NPROC_PER_NODE"),
    "pet_node_rank": os.environ.get("PET_NODE_RANK"),
    "out": os.environ.get("OUT"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["vllm"] == "0.19.1", f"vllm={versions['vllm']} != 0.19.1"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["pyserini"] == "2.0.0", f"pyserini={versions['pyserini']} != 2.0.0"
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
assert versions["o200k_base_present"]
assert versions["node_rank"] is not None, "NODE_RANK not set on this pod"
assert versions["out"], "OUT not set"

# Qwen3-4B model probe: dense bf16 (NOT mxfp4).
model_cfg_path = "/mnt/songzijun/models/Qwen3-4B-Instruct-2507/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "torch_dtype": model_cfg.get("torch_dtype"),
    "quant_method": qcfg.get("quant_method", "none"),
}, flush=True)
assert model_cfg.get("architectures") == ["Qwen3ForCausalLM"], \
    f"unexpected architectures={model_cfg.get('architectures')}"
assert not qcfg.get("quant_method"), \
    f"Qwen3-4B must be dense bf16, got quant_method={qcfg.get('quant_method')}"

# Lucene index probe (bcplus_full BM25 backend = BrowseComp-Plus/indexes/bm25).
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25")
n_docs = s.num_docs if not callable(s.num_docs) else s.num_docs()
print("LUCENE_OK", {"num_docs": n_docs}, flush=True)
assert n_docs > 0

# Import-chain smoke: the modules run_eval.py legacy_local + eval_shard_worker reach at startup.
from trim.eval.transfer_benchmarks import load_eval_benchmark  # noqa: F401
from trim.eval.eval_parallel import shard_rows_round_robin, assign_replica_gpus  # noqa: F401
from trim.eval.harness1_metrics import summarize_quality_and_timing  # noqa: F401
from trim.rsi.canonical import sha256_json  # noqa: F401
from trim.rsi.splits import load_id_file, filter_rows_by_ids  # noqa: F401 (the new 1eb69a0 filter)
import trim.eval.harness_g_contract  # noqa: F401 (is_formal_harness_g_eval, legacy_local eval marker)
print("IMPORT_OK", {"legacy_local_eval_chain": True}, flush=True)
PY

# --- Copy Qwen3-4B model weights to local /tmp before exec (defensive) ---
MODEL_SRC=/mnt/songzijun/models/Qwen3-4B-Instruct-2507
MODEL_LOCAL=/tmp/qwen3-4b
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"
test -f "$MODEL_LOCAL/model.safetensors.index.json" || true
echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_h0_zero_rollout.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_eval.py
cd /mnt/songzijun/Capability_Evolution

# --- H0 zero rollout exec ---
# --tensor-parallel-size 1 + --rollout-replicas 8 (per node) → 8 single-GPU vLLM engines
# per node = 32 total across 4 nodes. component=zero (0/10 mask, student-only, no harness
# search). legacy_local = graph-free eval runtime. bcplus_full = 830-query pool. No adapter
# (bare base = H0 fixed-base measurement). Out is the shared JuiceFS path.
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_h0_zero_rollout.py \
  --nnodes 4 \
  --rollout-replicas 8 \
  --node-rank "${NODE_RANK}" \
  --tensor-parallel-size 1 \
  --model_name /tmp/qwen3-4b \
  --out "${OUT}"
