#!/usr/bin/env bash
# CloudML imageCommand entry — Harness-G / Qwen3-4B-Instruct-2507 / trim / all
# 4 nodes × 8 gm402-80g = 32 GPUs. torch_ddp_lora. Original model (dense bf16).
#
# === EVALHOOK VARIANT === (see qwen3_4b_instruct_rl_zero entry for full gate doc)
#   - EVAL_TARGET_STEP (必填): 50/100/150/200 → --train-steps
#   - RESUME (默认 0): 1 时加 --resume
#
# === Harness-G / trim / all ===
#   - train_method trim = CISPO + projected actions + SEED-scale gap (scape+seed alias)
#   - opd_loss=OPD_LOSS_PROJECTED_GAP, opd_states_per_trajectory=-1 (all actions), lambda_opd=0.01 (SEED)
#   - component all = Harness-G 8 graph-menu components (answer_with/bridge_entities/entity_synonyms/
#     sentence_neighbors/hybrid_init_retrieve/invalid_target_filter/lookup_dedup/snc_frontier)
#     自包含，不需 verifier（Harness-1 all 才需要）
#   - train-data 默认 bcplus_train_664, retrieval_backend 默认 local_bm25, 需 corpus graph
#
# === Qwen3-4B-Instruct-2507 (dense bf16) ===
#   - mb4 (Qwen3-4B dense 能 grad-ckpt; CLAUDE.md mb 决策铁律)
#   - SCAPE_FORCE_LOCAL_HARMONY 不设（Qwen3 不走 Harmony）
#   - 无 mxfp4/triton 修复
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
  echo "FATAL: MASTER_ADDR is not set. CloudML custom_train multi-node must inject node0's IP as MASTER_ADDR." >&2
  exit 1
fi
: "${MASTER_PORT:=29500}"
export MASTER_ADDR NODE_RANK MASTER_PORT

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK \
      GROUP_WORLD_SIZE TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT \
      TORCHELASTIC_MAX_RESTARTS TORCHELASTIC_USE_AGENT_STORE \
      PET_MASTER_ADDR PET_MASTER_PORT PET_NNODES PET_NPROC_PER_NODE PET_NODE_RANK

echo "MULTINODE_ENV master_addr=${MASTER_ADDR} node_rank=${NODE_RANK} master_port=${MASTER_PORT} host=$(hostname)"

if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set. Multi-node requires a shared --out path injected via --env OUT=... Aborting." >&2
  exit 1
fi
export OUT
echo "MULTINODE_OUT out=${OUT} host=$(hostname) node_rank=${NODE_RANK}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
test -f /tmp/tiktoken-bundle/share/tiktoken/o200k_base.tiktoken
test -f /tmp/tiktoken-bundle/share/tiktoken/cl100k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

# --- Presence checks ---
nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py
test -f /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
test -f /mnt/songzijun/models/Qwen3-4B-Instruct-2507/config.json
test -f /mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/harness_g_corpus_graph.pkl
test -f /mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD/manifests/COMPONENT_SWEEP_TRAIN_POOL.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/manifests/browsecomp_plus_830/SPLIT.json

# --- JVM discovery ---
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

PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog \
  tqdm regex sympy latex2sympy2 word2number sentencepiece pyserini==2.0.0 pyjnius \
  openai tenacity anthropic
python3 -m pip install --no-cache-dir --no-deps --index-url "${PYPI_INDEX}" "openai_harmony" \
  || echo "[deps] WARN: openai_harmony --no-deps install issue (Qwen3 不触发 Harmony，无害)"

PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# --- Preflight ---
python3 - <<'PY'
import importlib.metadata, json, os, socket, torch, transformers, vllm
from jnius import autoclass
autoclass("java.lang.String")
versions = {
    "torch": torch.__version__, "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "pyserini": importlib.metadata.version("pyserini"),
    "peft": importlib.metadata.version("peft"),
    "structlog": importlib.metadata.version("structlog"),
    "gpu_count": torch.cuda.device_count(),
    "JVM_PATH": os.environ["JVM_PATH"], "JAVA_HOME": os.environ["JAVA_HOME"],
    "host": socket.gethostname(),
    "master_addr": os.environ.get("MASTER_ADDR"), "node_rank": os.environ.get("NODE_RANK"),
    "rank_env": os.environ.get("RANK"), "world_size_env": os.environ.get("WORLD_SIZE"),
}
print("PREFLIGHT", versions, flush=True)
assert versions["torch"].startswith("2.10.0"), f"torch={versions['torch']} != 2.10.0"
assert versions["vllm"] == "0.19.1", f"vllm={versions['vllm']} != 0.19.1"
assert versions["transformers"] == "4.57.6", f"transformers={versions['transformers']} != 4.57.6"
assert versions["pyserini"] == "2.0.0", f"pyserini={versions['pyserini']} != 2.0.0"
assert versions["gpu_count"] == 8, f"per-node gpu_count={versions['gpu_count']} != 8"
model_cfg_path = "/mnt/songzijun/models/Qwen3-4B-Instruct-2507/config.json"
with open(model_cfg_path, "r", encoding="utf-8") as fh:
    model_cfg = json.load(fh)
qcfg = model_cfg.get("quantization_config") or {}
print("MODEL_PROBE", {"architectures": model_cfg.get("architectures"), "model_type": model_cfg.get("model_type"), "quant_method": qcfg.get("quant_method")}, flush=True)
assert model_cfg.get("architectures") == ["Qwen3ForCausalLM"]
assert not qcfg.get("quant_method"), f"Qwen3-4B-Instruct must be dense bf16, got quant={qcfg.get('quant_method')}"
from trim.training.harness_g_train import default_graph_path
gp = default_graph_path()
assert gp, "harness_g_corpus_graph.pkl not found"
print("GRAPH_PATH_OK", {"graph_index_path": gp}, flush=True)
from trim.integrations.verl.trainer_adapter import _actor_wrap
assert _actor_wrap("torch_ddp_lora") == "ddp"
print("DDP_LORA_WRAP_OK", {"wrap": _actor_wrap("torch_ddp_lora")}, flush=True)
import pathlib, re
ta = pathlib.Path("/mnt/songzijun/Capability_Evolution/TRIM/trim/integrations/verl/trainer_adapter.py").read_text()
assert "startup_timeout_s=3600.0" in ta, "startup_timeout_s=3600 patch missing"
assert re.search(r"train_rows,\s*eval_rows,\s*pool_meta,\s*_?frozen_points\s*=\s*resolve_queries", ta), "4-tuple unpack patch missing"
assert "EADDRINUSE" in ta, "EADDRINUSE retry patch missing"
print("PATCH_CHECK_OK", {"startup_timeout_s": True, "resolve_queries_4tuple": True, "eaddrinuse_retry": True}, flush=True)
PY

# --- Copy model to local /tmp ---
MODEL_SRC=/mnt/songzijun/models/Qwen3-4B-Instruct-2507
MODEL_LOCAL=/tmp/qwen3-4b-instruct
if [ ! -f "$MODEL_LOCAL/config.json" ]; then
  rm -rf "$MODEL_LOCAL"
  mkdir -p "$MODEL_LOCAL"
  cp -r "$MODEL_SRC"/. "$MODEL_LOCAL"/
fi
test -f "$MODEL_LOCAL/config.json"
echo "MODEL_LOCAL_READY $MODEL_LOCAL host=$(hostname) node_rank=${NODE_RANK}"

python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train_multinode.py
python3 -m py_compile /mnt/songzijun/Capability_Evolution/TRIM/scripts/run_train.py
cd /mnt/songzijun/Capability_Evolution

# --- Evalhook gate ---
: "${EVAL_TARGET_STEP:?EVAL_TARGET_STEP is required (50/100/150/200)}"
: "${RESUME:=0}"
TRAIN_STEPS_ARG="--train-steps ${EVAL_TARGET_STEP}"
RESUME_ARG=""
RDZV_ID="qwen34b-instruct-trim-harnessg-32"
if [ "${RESUME}" = "1" ]; then
  RESUME_ARG="--resume"
  RDZV_ID="qwen34b-instruct-trim-harnessg-32-resume"
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
  --harness Harness-G \
  --benchmark bcplus_full \
  --model_name /tmp/qwen3-4b-instruct \
  --train_method trim \
  --component all \
  --train-env local_legacy \
  --max-turns 40 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  ${TRAIN_STEPS_ARG} \
  --train-groups-per-step 32 \
  --group-size 8 \
  --train-micro-batch-size 4 \
  --train-heartbeat-every 8 \
  ${RESUME_ARG} \
  --out "${OUT}"
