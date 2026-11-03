#!/usr/bin/env bash
# Probe C 可学习性实验：Qwen3.5-9B 四组件联盟（esp/ccso/trieve/msc），1node8gpu。
# 链路：audit → probe_c_collect(prepare/worker×8/merge) → runner collect → cache(C0 漏斗
# +冻结 Teacher 缓存) → screen(C1 4→8 微更新，seed 外层循环 + 每候选 reset 回 θ0)。
# 状态源=真实 H0 rollout（train pool SEC，query 分组切 C_train/C_select/C_audit，
# 与官方测试集零 overlap 校验）；Student=H0，Teacher=联盟精确 add pair。
# 未注册成员（subtractive_curation/importance_tagging/content_dedup/evidence_graph/
# chunk_neighbors）在漏斗记 UNREGISTERED，不伪造监督。
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

# SEC 训练池/语料显式路径（default_sec_rl_data 兜底 /data/ppnm 不存在；与训练 entry 同源）。
export SCAPE_SEC_RL_DATA=/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data
export SCAPE_SEC_CORPUS_ROOT=/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus

# 占位 key 防 pyserini openai.OpenAI() import 报错（仅 local BM25）。
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

# tiktoken bundle（harness import chain 需要）。
python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT is not set (submit --env OUT=...)." >&2
  exit 1
fi
export OUT
echo "PROBE_C_OUT out=${OUT} host=$(hostname)"

nvidia-smi -L
test -f /mnt/songzijun/Capability_Evolution/TRIM/trim/probes/probe_c_runner.py
test -f /mnt/songzijun/Capability_Evolution/TRIM/trim/probes/probe_c_collect.py
test -f /mnt/songzijun/Capability_Evolution/TRIM/trim/probes/probe_c_qwen35_hooks.py
test -f /mnt/songzijun/Capability_Evolution/TRIM/configs/probes/probe_c_qwen35_9b_coalitions.json
test -f /mnt/songzijun/models/Qwen3.5-9B/config.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data/manifest.json
test -f /mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus/manifest.json

# --- JVM discovery for Pyserini/Lucene（SEC BM25 检索）---
JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
if [ -z "${JVM_PATH}" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
fi
test -n "${JVM_PATH}"
test -f "${JVM_PATH}"
export JVM_PATH
export JAVA_HOME=$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")
export LD_LIBRARY_PATH=$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}

# --- 运行时依赖（与 9B teacher 训练 entry 同配方：镜像 python3 裸，需内网 PyPI 补齐）---
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==5.17.0" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece "pyserini==2.0.0" openai tenacity anthropic \
  tensorboard

# --- ANSERINI_CLASSPATH for PyJNI/Lucene ---
PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"
export CLASSPATH="$ANSERINI_CLASSPATH"

# 模型 cp /tmp（JuiceFS 并发读铁律；8 vLLM worker 从 /tmp 本地读）。
MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/qwen35-9b
echo "MODEL_CP_START $(date -Is)"
cp -r "${MODEL_SRC}"/. "${MODEL_LOCAL}"/
echo "MODEL_CP_DONE $(date -Is)"
test -f "${MODEL_LOCAL}/config.json"

# Preflight：模块 import、transformers>=5.6（qwen3_5 注册）、架构探测、teacher 注册表。
python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/probe_c_preflight.py

# 冻结契约（model_revision/git_sha/state_source_dir 注入；占位符会被 audit 拒绝）。
GIT_SHA=$(git -C /mnt/songzijun/Capability_Evolution rev-parse HEAD || echo "unknown")
python3 -m trim.probes.probe_c_qwen35_hooks resolve-config \
  --base /mnt/songzijun/Capability_Evolution/TRIM/configs/probes/probe_c_qwen35_9b_coalitions.json \
  --out "${OUT}/config_resolved.json" \
  --model-path "${MODEL_LOCAL}" \
  --state-source-dir "${OUT}" \
  --output-root "${OUT}/runner" \
  --git-sha "${GIT_SHA}"

HOOKS=trim.probes.probe_c_qwen35_hooks
CFG="${OUT}/config_resolved.json"

# audit：未解析 revision / 未冻结 split / 错误 mask pair / loader 不支持 → 直接拒绝。
python3 -m trim.probes.probe_c_runner audit --config "${CFG}" --out "${OUT}/runner" --hooks-module ${HOOKS}

# collect：真实 H0 状态（θ0/zero mask，SEC train pool）。
# 幂等续跑：COLLECTION_MANIFEST.json 已存在 = 采集+merge 已完成，直接跳过（防重跑混 provenance）。
if [ ! -f "${OUT}/COLLECTION_MANIFEST.json" ]; then
python3 -m trim.probes.probe_c_collect prepare --model-path "${MODEL_LOCAL}" --out "${OUT}" \
  --n-train 128 --n-select 64 --n-audit 64 --train-data sec
PIDS=()
for R in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=${R} python3 -m trim.probes.probe_c_collect worker \
    --model-path "${MODEL_LOCAL}" --out "${OUT}" --worker-rank ${R} --workers 8 \
    --max-turns 40 --max-new 2048 --max-model-len 8192 --group-size 1 \
    --sample --train-env local_legacy --searcher sec \
    --gpu-memory-utilization 0.90 &
  PIDS+=($!)
done
FAIL=0
for P in "${PIDS[@]}"; do wait "${P}" || FAIL=1; done
test "${FAIL}" = "0"
python3 -m trim.probes.probe_c_collect merge --out "${OUT}"
else
  echo "COLLECTION_DONE resume: $(ls "${OUT}/states_raw" | wc -l) shards reused"
fi

# runner：严格校验入库 → C0/cache → C1/screen（四联盟共享同一份状态）。
python3 -m trim.probes.probe_c_runner collect --config "${CFG}" --out "${OUT}/runner" --hooks-module ${HOOKS}
python3 -m trim.probes.probe_c_runner cache   --config "${CFG}" --out "${OUT}/runner" --hooks-module ${HOOKS}
python3 -m trim.probes.probe_c_runner screen  --config "${CFG}" --out "${OUT}/runner" --hooks-module ${HOOKS}

echo "PROBE_C_DONE $(date -Is)"
ls -la "${OUT}/runner" || true
