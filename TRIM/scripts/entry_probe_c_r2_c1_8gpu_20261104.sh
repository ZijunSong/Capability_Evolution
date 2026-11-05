#!/usr/bin/env bash
# Probe C R2：修复后 C1 双 seed 4→8（guide §16 R2）。
# 复用 R1 cache（auto_populate_first_search READY 通道）+ 旧 r3 状态；
# 真实 checkpoint pre/4/8（weights+optimizer+RNG 原子落盘）；
# 持久 sampler session（4+4==8 契约）；面板 pre/4/8 原始量+生成记录全留档。
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRIM_GPU_KEEPALIVE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/songzijun/Capability_Evolution/TRIM:/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD

export SCAPE_SEC_RL_DATA=/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-rl-data
export SCAPE_SEC_CORPUS_ROOT=/mnt/songzijun/Capability_Evolution/TRIM/data/harness-1-sec-corpus
export OPENAI_API_KEY="${OPENAI_API_KEY:-**-****-***********-***-*-****-***}"
export SCAPE_LOCAL_OPENAI_EMBEDDINGS=1
export SCAPE_FORCE_LOCAL_HARMONY=1
export SCAPE_CHROMA_PATH=/tmp/empty_chroma
mkdir -p "$SCAPE_CHROMA_PATH"

python3 -c "import zipfile; zipfile.ZipFile('/mnt/songzijun/tiktoken-bundle.zip').extractall('/tmp')"
export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken-bundle/share/tiktoken
export TIKTOKEN_RS_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_rs_cache
export TIKTOKEN_CACHE_DIR=/tmp/tiktoken-bundle/tiktoken_cache

if [ -z "${OUT:-}" ]; then echo "FATAL: OUT not set" >&2; exit 1; fi
if [ -z "${OLD_RUN:-}" ]; then echo "FATAL: OLD_RUN not set" >&2; exit 1; fi
if [ -z "${R1_DIR:-}" ]; then echo "FATAL: R1_DIR not set" >&2; exit 1; fi
export OUT OLD_RUN R1_DIR
echo "R2_C1 out=${OUT} old_run=${OLD_RUN} r1=${R1_DIR} host=$(hostname)"

test -f "${OLD_RUN}/states/all_states.jsonl"
test -f "${R1_DIR}/cache/auto_populate_first_search/CACHED_TARGETS.jsonl"
test -f /mnt/songzijun/Capability_Evolution/TRIM/trim/probes/probe_c_r2_c1.py
test -f /mnt/songzijun/models/Qwen3.5-9B/config.json

JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
if [ -z "${JVM_PATH}" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
fi
test -n "${JVM_PATH}"
export JVM_PATH
export JAVA_HOME=$(dirname "$(dirname "$(dirname "${JVM_PATH}")")")
export LD_LIBRARY_PATH=$(dirname "${JVM_PATH}"):${LD_LIBRARY_PATH:-}

python3 -m pip install --no-cache-dir --ignore-installed --upgrade-strategy only-if-needed \
  "torch==2.10.0" "transformers==5.17.0" "accelerate==1.13.0" "datasets==4.8.4" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "vllm==0.19.1" peft structlog chromadb \
  tqdm regex sympy latex2sympy2 word2number sentencepiece "pyserini==2.0.0" openai tenacity anthropic \
  tensorboard

PYSERINI_ROOT=$(python3 -c 'import pathlib, pyserini; print(pathlib.Path(pyserini.__file__).resolve().parent)')
export ANSERINI_CLASSPATH=$(find "$PYSERINI_ROOT/resources/jars" -type f -name '*.jar' -print | paste -sd: -)
test -n "$ANSERINI_CLASSPATH"

MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/qwen35-9b
echo "MODEL_CP_START $(date -Is)"
cp -r "${MODEL_SRC}"/. "${MODEL_LOCAL}"/
echo "MODEL_CP_DONE $(date -Is)"
test -f "${MODEL_LOCAL}/config.json"

python3 /mnt/songzijun/Capability_Evolution/TRIM/scripts/probe_c_preflight.py

GIT_SHA=$(git -C /mnt/songzijun/Capability_Evolution rev-parse HEAD || echo "unknown")
python3 -m trim.probes.probe_c_qwen35_hooks resolve-config \
  --base /mnt/songzijun/Capability_Evolution/TRIM/configs/probes/probe_c_qwen35_9b_coalitions.json \
  --out "${OUT}/config_resolved.json" \
  --model-path "${MODEL_LOCAL}" \
  --state-source-dir "${OLD_RUN}" \
  --output-root "${OUT}" \
  --git-sha "${GIT_SHA}" \
  --generation-max-new 2048 --generation-max-model-len 8192 --generation-max-turns 40

CFG="${OUT}/config_resolved.json"
# R2：双 seed 4→8，主 budget=8 汇总；BORDERLINE@8 且可测才扩 16（--extend-16 由用户决定，默认关）。
python3 -m trim.probes.probe_c_r2_c1 \
  --config "${CFG}" \
  --out "${OUT}" \
  --r1-dir "${R1_DIR}" \
  --states-dir "${OLD_RUN}" \
  --members auto_populate_first_search \
  --seeds 11,23 \
  --min-select-supported-states 8 \
  --regen-limit 16

echo "R2_C1_DONE $(date -Is)"
python3 -c "import json;p='${OUT}/Qwen_Qwen3.5-9B/R2_C1_SUMMARY.json';print(open(p).read())"
