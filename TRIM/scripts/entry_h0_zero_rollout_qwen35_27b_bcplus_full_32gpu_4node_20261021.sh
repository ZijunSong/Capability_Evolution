#!/usr/bin/env bash
# CloudML imageCommand entry — H0 zero rollout (fixed-base measurement before any
# harness search or TRIM update) on bcplus_full (830 queries) for Qwen3.5-27B
# (Qwen3_5ForConditionalGeneration, multimodal base, 27B dense bf16, ~52GiB/card,
# NO adapter, chat_template already set to no-thinking — see *.orig_thinking_on_* backups).
#
# Fork of entry_h0_zero_rollout_qwen35_9b_bcplus_full_32gpu_4node_20261015.sh — same
# architecture (Qwen3_5ForConditionalGeneration) + same legacy_local eval path, only
# the model path / weight size / model_local differ.
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
# === vLLM 0.19.1 SUPPORTS Qwen3_5ForConditionalGeneration (key feasibility) ===
# Qwen3.5-27B IS the verified reference: 9 月 fcanl(zero)/iytmg(all) 已验证支持,
# bg3xs smoke preflight (2026-10-15) 铁证 Qwen3_5ForConditionalGeneration SUPPORTED on
# same four-pack (vllm 0.19.1 + transformers 4.57.6). This entry's preflight probes
# ModelRegistry directly; exit 5 if NOT registered (won't waste 32 GPU).
#
# === legacy_local eval path ===
# run_h0_zero_rollout → run_eval.py --evaluation-path legacy_local → eval_shard_worker
# ._run_vllm → trim.training.vllm_hybrid.VLLMGenerateClient (vllm_rollout_worker.py LLM(
# dtype="auto")). dtype=auto 读 text_config.dtype=bfloat16 (正确). encoding 走 HF chat
# template (profile=QWEN35 family=hf_chat encoding=hf_chat_tools, 非 harmony). legacy_local
# worker 不传 reasoning-parser/tool-call-parser — H0 component=zero 是 student-only 生成不
# 调工具, 不需要 tool parser.
#
# === run_h0_zero_rollout.py does NOT use torchrun/Gloo/rdzv ===
# File-based multi-node coordinator: all 4 nodes share the same JuiceFS --out dir
# (injected via --env OUT=...); node 0 waits for out/nodes/node{0..3}/NODE_DONE.json.
# This entry does NOT call run_train_multinode.py and does NOT pass --master-addr /
# --rdzv-id / --nproc-per-node. Only distributed env needed: NODE_RANK (pick shard) +
# shared OUT. MASTER_ADDR not required (warn-only).
#
# === Qwen3.5-27B specifics (dense bf16 27B multimodal base, no-thinking) ===
#   * NO triton_kernels / Marlin / mxfp4 dance (Qwen3.5 is dense bf16, no quant).
#   * SCAPE_FORCE_LOCAL_HARMONY=1 SET (Qwen3.5 is_hf_chat_family=True, 走 HF chat template
#     非 harmony; 无害, 跳过 harmony 加载).
#   * NO adapter (H0 = bare base measurement).
#   * gpu-mem: run_eval default 0.90 (run_h0_zero_rollout.build_eval_argv 不传 gpu-mem).
#     27B = 52GiB weights/card → 0.90→72GB budget, ~20GB left for KV cache. eval-only no
#     backward + no-think (chat_template 已改关 thinking, 生成短) → 20GB KV should fit.
#     27B base eval (upstream_api path) 实测 0.90+max_num_seqs=6+max_model_len=65536 跑通;
#     legacy_local 不传 max_num_seqs (vLLM 默认 256) + max_model_len 默认 32768, KV 可能更费
#     but no-think 短生成抵消. 主要风险点: OOM — 若 OOM 需 code override (build_eval_argv
#     不传 gpu-mem flag), 报告后问用户. monitor "OutOfMemoryError"/"CUDA out of memory".
#   * chat_template: 27B 已改关 thinking (有 *.orig_thinking_on_* 备份), no-think 防止
#     reasoning token 爆 + tool call 吞. 与 9B entry (thinking 开) 不同, 27B 用 no-think.
#   * Model cp to /tmp: 4 nodes x 8 vLLM engines = 32 concurrent JuiceFS readers of the
#     ~52G model on first shard; per-node local cp (~3-5min for 52G) defensively avoids
#     contention.
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

# Qwen3.5-27B: SCAPE_FORCE_LOCAL_HARMONY=1 (走 HF chat template 非 harmony, 无害).
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1

# tiktoken bundle: openai_harmony import chain needs o200k_base.
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
test -f /mnt/songzijun/models/Qwen3.5-27B/config.json
test -f /mnt/songzijun/models/Qwen3.5-27B/model.safetensors.index.json
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

# --- Preflight: assert four-pack versions, 8 GPUs (per node), JVM, Qwen3_5 arch
#     registration, Qwen3.5-27B model, bcplus_full Lucene index ---
python3 - <<'PY'
import importlib.metadata
import json
import os
import socket
import sys
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

# Qwen3.5-27B model probe: Qwen3_5ForConditionalGeneration dense bf16 (no quant).
model_cfg_path = "/mnt/songzijun/models/Qwen3.5-27B/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
tc = model_cfg.get("text_config") or {}
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {
    "architectures": model_cfg.get("architectures"),
    "model_type": model_cfg.get("model_type"),
    "text_config.model_type": tc.get("model_type"),
    "text_config.dtype": tc.get("dtype"),
    "text_config.hidden_size": tc.get("hidden_size"),
    "text_config.num_hidden_layers": tc.get("num_hidden_layers"),
    "quant_method": qcfg.get("quant_method", "none"),
}, flush=True)
assert model_cfg.get("architectures") == ["Qwen3_5ForConditionalGeneration"], \
    f"unexpected architectures={model_cfg.get('architectures')}"
assert not qcfg.get("quant_method"), \
    f"Qwen3.5-27B must be dense bf16, got quant_method={qcfg.get('quant_method')}"
assert str(tc.get("dtype")).lower() == "bfloat16", \
    f"expected text_config.dtype=bfloat16, got {tc.get('dtype')}"

# CRITICAL: probe vLLM ModelRegistry for Qwen3_5ForConditionalGeneration.
# Qwen3.5-27B IS the verified reference (9 月 fcanl/iytmg). 不支持则 exit 5 (不浪费 32 GPU).
arch = "Qwen3_5ForConditionalGeneration"
try:
    from vllm.model_executor.models.registry import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
    if arch in supported:
        print("ARCH_OK", {"arch": arch, "registered": True}, flush=True)
    else:
        qwen_archs = sorted(a for a in supported if "qwen3" in a.lower())
        print("ARCH_NOT_SUPPORTED", {"arch": arch, "qwen3_family_registered": qwen_archs[:15]}, flush=True)
        sys.exit(5)
except SystemExit:
    raise
except Exception as e:
    print("ARCH_PROBE_FAIL", {"error": f"{type(e).__name__}: {e}"}, flush=True)
    raise

# Lucene index probe (bcplus_full BM25 backend).
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher("/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/bm25")
n_docs = s.num_docs if not callable(s.num_docs) else s.num_docs()
print("LUCENE_OK", {"num_docs": n_docs}, flush=True)
assert n_docs > 0

# Model profile + encoding probe: Qwen3.5-27B → QWEN35 profile (hf_chat, hf_chat_tools).
from trim.eval.model_profiles import resolve_model_profile
from trim.eval.model_tokenizer import encoding_config_for_model
prof = resolve_model_profile("/mnt/songzijun/models/Qwen3.5-27B")
enc = encoding_config_for_model("/mnt/songzijun/models/Qwen3.5-27B")
print("MODEL_PROFILE", {"family": prof.family, "stack": prof.stack, "label": prof.label,
                        "encoding": enc["encoding"]}, flush=True)
assert prof.family == "hf_chat", f"expected hf_chat, got {prof.family}"

# Import-chain smoke.
from trim.eval.transfer_benchmarks import load_eval_benchmark  # noqa: F401
from trim.eval.eval_parallel import shard_rows_round_robin, assign_replica_gpus  # noqa: F401
from trim.eval.harness1_metrics import summarize_quality_and_timing  # noqa: F401
from trim.rsi.canonical import sha256_json  # noqa: F401
from trim.rsi.splits import load_id_file, filter_rows_by_ids  # noqa: F401
from trim.training.vllm_hybrid import VLLMGenerateClient  # noqa: F401
import trim.eval.harness_g_contract  # noqa: F401
print("IMPORT_OK", {"legacy_local_eval_chain": True}, flush=True)
PY

# --- cp model to /tmp BEFORE vLLM init (JuiceFS concurrent-read deadlock fix) ---
# LESSON (hard-won): removing cp and pointing --model_name at JuiceFS directly FAILED —
# 8 vLLM engines per node all read the 52G model concurrently from JuiceFS, IO contention
# makes each shard load ~348s/shard × 11 shards, load never finishes → shard returncode=1
# → "eval shards failed". cp /tmp is REQUIRED: vLLM loads from local /tmp fast (prev run's
# workers had 8 READY each from /tmp). The prev cp-run's master(rank0) cp hang was cp
# itself slow on JuiceFS for 52G, NOT the approach being wrong. Keep cp; 52G cp ~3-8min.
# NOTE: prev noinlinecp run confirmed 27B has NO OOM (52GiB + ~20GB KV fits, 8 READY).
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-27B
MODEL_LOCAL=/tmp/qwen35-27b
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  echo "MODEL_CP_START $MODEL_SRC -> $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK} $(date -Iseconds 2>/dev/null)"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
  echo "MODEL_CP_DONE $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK} $(date -Iseconds 2>/dev/null)"
fi
test -f "$MODEL_LOCAL/config.json"
echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_h0_zero_rollout.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_eval.py
cd /mnt/songzijun/Capability_Evolution

# --- H0 zero rollout exec ---
# --tensor-parallel-size 1 + --rollout-replicas 8 (per node) → 8 single-GPU vLLM engines
# per node = 32 total. component=zero (0/10 mask, student-only, no harness search).
# legacy_local graph-free eval runtime. bcplus_full = 830-query pool. No adapter.
# model_name = /tmp local copy (cp'd above; vLLM loads fast vs 348s/shard from JuiceFS).
# NOTE: run_h0_zero_rollout.build_eval_argv 不传 gpu-mem; run_eval 默认 0.90. 27B 52GB
# weights → ~20GB KV budget. no-think + eval-only 应够 (前次 worker 实测无 OOM). monitor OOM.
exec python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_h0_zero_rollout.py \
  --nnodes 4 \
  --rollout-replicas 8 \
  --node-rank "${NODE_RANK}" \
  --tensor-parallel-size 1 \
  --model_name "$MODEL_LOCAL" \
  --out "${OUT}"
