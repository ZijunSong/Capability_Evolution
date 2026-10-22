#!/usr/bin/env bash
# CloudML imageCommand entry — 在 1-node 8-GPU pod 上构建 harness_g_corpus_graph.pkl
# 原因：dev box memcg 上限低（~24GB），HarnessGGraphIndex 全内存（100k docs → 50GB+）
# 在 dev box 构建到 44k docs 时 OOM killed。gm402-80g pod 有 224 GiB 系统内存，足够。
#
# 输入：/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_corpus_full.jsonl (3.3GB, 100195 docs)
# 输出：/mnt/songzijun/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/harness_g_corpus_graph.pkl
#       + .METADATA.json + .quality.json
#
# 构建后 4 个训练 job 的 entry default_graph_path() 自动发现该 .pkl。
# 本 job 跑完即 Succeeded，不占长卡（~10-20min build）。
set -euxo pipefail

JFS_ROOT="/mnt/songzijun"
TRIM_ROOT="${JFS_ROOT}/Capability_Evolution/TRIM"
CORPUS_JSONL="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus/data/browsecomp_plus_corpus_full.jsonl"
OUT_PKL="${JFS_ROOT}/Capability_Evolution/SCOPE/external/BrowseComp-Plus/indexes/harness_g_corpus_graph.pkl"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${TRIM_ROOT}:${JFS_ROOT}/Capability_Evolution/SCAPE-EasyOPD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0

# 清掉 CloudML 注入的 torchrun-style env（本 job 不走 torchrun）
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK NODE_RANK MASTER_ADDR MASTER_PORT PET_NPROC_PER_NODE 2>/dev/null || true

# Placeholder OpenAI key
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1

echo "[build] corpus_jsonl=${CORPUS_JSONL} out_pkl=${OUT_PKL}"
nvidia-smi -L 2>&1 | head
nvidia-smi --query-gpu=memory.total --format=csv 2>&1 | head -2
free -g 2>&1 | head -3

# presence
test -f "${CORPUS_JSONL}"
test -f "${TRIM_ROOT}/scripts/build_harness_g_corpus_graph.py"
mkdir -p "$(dirname "${OUT_PKL}")"

# JVM（build script 用 pyserini validate_loaded_graph；pyserini import 需 JVM）
JVM_PATH="$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)"
if [ -z "${JVM_PATH}" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH="$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)"
fi
test -n "${JVM_PATH}" && test -f "${JVM_PATH}"
ldd "${JVM_PATH}" >/dev/null
export JVM_PATH JAVA_HOME="$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")"
export LD_LIBRARY_PATH="$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}"

PYPI_INDEX="https://pkgs.d.xiaomi.net/artifactory/api/pypi/pypi-virtual/simple"
python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  --index-url "${PYPI_INDEX}" \
  "torch==2.10.0" "transformers==4.57.6" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" peft structlog \
  tqdm regex sympy latex2sympy2 word2number sentencepiece pyserini==2.0.0 pyjnius \
  openai tenacity anthropic

PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH="$(find "${PYSERINI_ROOT}/resources/jars" -type f -name '*.jar' -print | paste -sd: -)"
export CLASSPATH="${ANSERINI_CLASSPATH}"

python3 -c 'from jnius import autoclass; autoclass("java.lang.String"); print("[preflight][jvm] java_ok")'

cd "${TRIM_ROOT}"

echo "[build] exec build_harness_g_corpus_graph.py"
exec python3 scripts/build_harness_g_corpus_graph.py \
  --corpus-jsonl "${CORPUS_JSONL}" \
  --out "${OUT_PKL}" \
  --scope corpus \
  --source-corpus browsecomp_plus \
  --batch-size 500 \
  --progress-every 2000
