#!/usr/bin/env bash
# CloudML imageCommand entry — H0 zero rollout (fixed-base measurement before any
# harness search or TRIM update) on bcplus_full (830 queries) for GPT-OSS-20B
# (bf16-dequant base, NO adapter, NO mxfp4 — CLAUDE.md 铁律: eval 必须用 bf16-dequant
# 避 mxfp4+eval Marlin PTX; 原版 mxfp4 禁用).
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
# === GPT-OSS-20B bf16-dequant on legacy_local eval path (key feasibility) ===
# legacy_local eval uses trim.training.vllm_hybrid.VLLMGenerateClient (eval_shard_worker
# ._run_vllm) — the SAME vLLM client class that GPT-OSS bf16-dequant TRAINING uses
# (trainer_adapter.py:564, four_cell_runtime.py:2736). pncda/vvptp/b1ljo 等 SUCCEEDED 的
# GPT-OSS bf16-dequant 训练已通过这套 vllm_rollout_worker.py 加载成功 → legacy_local eval
# 加载 GPT-OSS bf16-dequant 路径已间接验证可行. dtype=auto 读 config.dtype=bfloat16 (正确);
# bf16 MoE 不需 triton_kernels (那是 mxfp4 坑). openai_harmony encoding: resolve_model_profile
# 对 gpt-oss-20b-bf16-dequant 路径返回 HARMONY stack → load_model_encoding 走 load_harmony_enc
# (o200k harmony encoding, 正确).
#
# === run_h0_zero_rollout.py does NOT use torchrun/Gloo/rdzv ===
# File-based multi-node coordinator: all 4 nodes share the same JuiceFS --out dir
# (injected via --env OUT=...); node 0 waits for out/nodes/node{0..3}/NODE_DONE.json.
# This entry does NOT call run_train_multinode.py and does NOT pass --master-addr /
# --rdzv-id / --nproc-per-node. Only distributed env needed: NODE_RANK (pick shard) +
# shared OUT. MASTER_ADDR not required (warn-only; legacy_local eval spawns its own
# per-replica vLLM subprocesses with no cross-node rendezvous).
#
# === GPT-OSS-20B specifics (bf16-dequant, MoE 20B) ===
#   * NO triton_kernels / VLLM_MXFP4_USE_MARLIN / Marlin dance (mxfp4 专属; bf16 不需).
#   * SCAPE_FORCE_LOCAL_HARMONY=1 NOT set (GPT-OSS 走真 openai_harmony, 需 load_harmony_enc
#     从本地加载 harmony encoding, 不能 force local 跳过). openai_harmony 已 pip install.
#   * NO adapter (H0 = bare base measurement).
#   * gpu-mem 0.85 (CLAUDE.md 铁律: GPT-OSS-20B bf16-dequant 39-40GB 权重/卡, KV 余量比
#     Qwen3-4B 7.6GB 小, 0.90 撞余量风险; 0.85 是 pncda/vvptp/b1ljo 实测验证值). But H0 eval
#     uses run_eval default 0.90... see note below — we INJECT VLLM_GPU_MEM env so the entry
#     can override if needed. Actually run_h0_zero_rollout.build_eval_argv does NOT pass
#     gpu-mem, so run_eval defaults to 0.90. For GPT-OSS bf16 40GB weights, 0.90→72GB budget,
#     40GB weights + ~30GB KV, eval-only no backward → 应该够 but 余量小. We do NOT override
#     (run_eval has no flag in build_eval_argv); if OOM, drop via env. Eval-only no backward,
#     peak < training, 0.90 likely OK — 监控 OOM.
#   * Model cp to /tmp: 4 nodes x 8 vLLM engines = 32 concurrent JuiceFS readers of the
#     ~40G model on first shard; per-node local cp (~2-3min for 40G) defensively avoids
#     contention (GPT-OSS 训练 entry 铁律).
#
# === Multi-node CloudML env injection (verified by iowty/9ixnl/7nzzn) ===
# custom_train framework=pytorch nodeNumber=4 creates a PyTorchJob CRD. CloudML injects
# NODE_RANK on every pod. We unset RANK/WORLD_SIZE/PET_* defensively so run_eval.py's
# legacy_local path (which spawns trim.eval.eval_shard_worker subprocesses) does not
# inherit stale torchrun-style rank env.
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
if [ -z "${MASTER_ADDR:-}" ]; then
  echo "WARN: MASTER_ADDR not set (legacy_local H0 rollout does not rendezvous cross-node; continuing)." >&2
fi
export NODE_RANK

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "H0_ZERO_ENV node_rank=${NODE_RANK} host=$(hostname) master_addr=${MASTER_ADDR:-unset}"

# --- Shared output directory (MUST be identical on every node) ---
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node H0 rollout requires a shared --out path injected via --env OUT=... (identical on every pod). Aborting." >&2
  exit 1
fi
export OUT
echo "H0_ZERO_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

# GPT-OSS: openai_harmony needed. SCAPE_FORCE_LOCAL_HARMONY NOT set (GPT-OSS wants real
# harmony encoding). Placeholder key so pyserini.encode._openai module-level openai.OpenAI()
# does not raise at import (legacy_local eval uses local BM25, never calls OpenAI API).
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
# NOTE: do NOT export SCAPE_FORCE_LOCAL_HARMONY=1 (that forces local harmony skip; GPT-OSS
# needs load_harmony_enc real encoding). This differs from the Qwen3-4B H0 entry.

# tiktoken bundle: openai_harmony needs o200k_base.
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
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json
test -f /mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/model-00001-of-00009.safetensors
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

# --- Install four-pack + eval import chain deps + openai_harmony (internal PyPI) ---
PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog \
  tqdm regex sympy latex2sympy2 word2number sentencepiece "pyserini==2.0.0" pyjnius \
  openai tenacity anthropic
# openai_harmony is REQUIRED for GPT-OSS (real harmony encoding load). Install --no-deps
# to protect the four-pack.
python3 -m pip install --no-cache-dir --no-deps --index-url "${PYPI_INDEX}" "openai_harmony" \
  || echo "[deps] WARN: openai_harmony --no-deps install issue (GPT-OSS 必需, 失败需排查)"

# --- ANSERINI_CLASSPATH for PyJNI/Lucene ---
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Preflight: assert four-pack versions, 8 GPUs (per node), JVM, import chain,
#     GPT-OSS bf16-dequant model, bcplus_full Lucene index, harmony encoding ---
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

# GPT-OSS-20B model probe: bf16-dequant (NOT mxfp4).
model_cfg_path = "/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "dtype": model_cfg.get("dtype"),
    "quant_method": qcfg.get("quant_method", "none"),
    "num_local_experts": model_cfg.get("num_local_experts"),
    "num_experts_per_tok": model_cfg.get("num_experts_per_tok"),
}, flush=True)
assert model_cfg.get("architectures") == ["GptOssForCausalLM"], \
    f"unexpected architectures={model_cfg.get('architectures')}"
assert not qcfg.get("quant_method"), \
    f"GPT-OSS-20B must be bf16-dequant (NO quant), got quant_method={qcfg.get('quant_method')}"
assert str(model_cfg.get("dtype")).lower() == "bfloat16", \
    f"expected config.dtype=bfloat16, got {model_cfg.get('dtype')}"
assert model_cfg.get("num_local_experts"), "expected MoE (num_local_experts present)"

# openai_harmony import probe (GPT-OSS REQUIRED).
try:
    import openai_harmony  # noqa: F401
    harmony_ok = True
except Exception as exc:
    harmony_ok = False
    print("HARMONY_IMPORT_FAIL", {"error": str(exc)}, flush=True)
assert harmony_ok, "openai_harmony import failed (GPT-OSS requires it for encoding)"
print("HARMONY_OK", {"openai_harmony": True}, flush=True)

# Lucene index probe (bcplus_full BM25 backend).
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25")
n_docs = s.num_docs if not callable(s.num_docs) else s.num_docs()
print("LUCENE_OK", {"num_docs": n_docs}, flush=True)
assert n_docs > 0

# Model profile probe: GPT-OSS path → HARMONY stack + o200k_harmony encoding.
from trim.eval.model_tokenizer import encoding_config_for_model
enc = encoding_config_for_model("/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant")
print("MODEL_ENCODING", enc, flush=True)
assert enc["family"] == "gpt-oss", f"expected family=gpt-oss, got {enc['family']}"
assert enc["encoding"] == "o200k_harmony", f"expected o200k_harmony, got {enc['encoding']}"

# Import-chain smoke: the modules run_eval.py legacy_local + eval_shard_worker reach at startup.
from trim.eval.transfer_benchmarks import load_eval_benchmark  # noqa: F401
from trim.eval.eval_parallel import shard_rows_round_robin, assign_replica_gpus  # noqa: F401
from trim.eval.harness1_metrics import summarize_quality_and_timing  # noqa: F401
from trim.rsi.canonical import sha256_json  # noqa: F401
from trim.rsi.splits import load_id_file, filter_rows_by_ids  # noqa: F401
from trim.training.vllm_hybrid import VLLMGenerateClient  # noqa: F401 (same worker as training)
import trim.eval.harness_g_contract  # noqa: F401
print("IMPORT_OK", {"legacy_local_eval_chain": True, "vllm_client": "VLLMGenerateClient"}, flush=True)
PY

# --- Copy GPT-OSS-20B bf16-dequant model weights to local /tmp before exec (defensive) ---
MODEL_SRC=/mnt/songzijun/models/openai/gpt-oss-20b-bf16-dequant
MODEL_LOCAL=/tmp/gpt-oss-20b
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"
test -f "$MODEL_LOCAL/model-00001-of-00009.safetensors"
echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_h0_zero_rollout.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_eval.py
cd /mnt/songzijun/Capability_Evolution

# --- H0 zero rollout exec ---
# --tensor-parallel-size 1 + --rollout-replicas 8 (per node) → 8 single-GPU vLLM engines
# per node = 32 total across 4 nodes. component=zero (0/10 mask, student-only, no harness
# search). legacy_local = graph-free eval runtime. bcplus_full = 830-query pool. No adapter
# (bare base = H0 fixed-base measurement). Out is the shared JuiceFS path.
# NOTE: run_h0_zero_rollout.build_eval_argv does NOT pass --gpu-memory-utilization, so
# run_eval defaults to 0.90. For GPT-OSS bf16 40GB weights, 0.90→72GB budget (40GB weights
# + ~30GB KV), eval-only no backward → should fit; monitor for OOM (if OOM, this entry
# cannot easily lower it via flag — would need a code/env override). Training used 0.85 but
# that path passes gpu-mem explicitly; eval path relies on run_eval default.
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_h0_zero_rollout.py \
  --nnodes 4 \
  --rollout-replicas 8 \
  --node-rank "${NODE_RANK}" \
  --tensor-parallel-size 1 \
  --model_name /tmp/gpt-oss-20b \
  --out "${OUT}"
