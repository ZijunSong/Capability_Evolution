#!/usr/bin/env bash
# Probe C R1 测量 smoke（guide §16 R1）：16-32 题双单成员，复用旧 r3 真实 H0 状态。
# 验收（未达标不训练）：same-state 目标进 K / 重复打分一致 / Teacher view 记录 /
# 128 vs 2048 预算对照（截断假失败定量）/ 零跨 split 泄漏。
# 状态源=OLD_RUN（r3），不重采集；产出 R1_SMOKE_REPORT.json + PREFLIGHT_OK.json。
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
if [ -z "${OLD_RUN:-}" ]; then echo "FATAL: OLD_RUN not set (r3 states dir)" >&2; exit 1; fi
export OUT OLD_RUN
echo "R1_SMOKE out=${OUT} old_run=${OLD_RUN} host=$(hostname)"

test -f "${OLD_RUN}/states/all_states.jsonl" || { echo "FATAL: ${OLD_RUN}/states missing" >&2; exit 1; }
test -f /mnt/songzijun/Capability_Evolution/TRIM/trim/probes/probe_c_r1_smoke.py
test -f /mnt/songzijun/Capability_Evolution/TRIM/trim/training/action_validation.py
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
# 生成契约显式与旧 r3 采集同源（--max-new 2048）；绝不静默默认。
python3 -m trim.probes.probe_c_qwen35_hooks resolve-config \
  --base /mnt/songzijun/Capability_Evolution/TRIM/configs/probes/probe_c_qwen35_9b_coalitions.json \
  --out "${OUT}/config_resolved.json" \
  --model-path "${MODEL_LOCAL}" \
  --state-source-dir "${OLD_RUN}" \
  --output-root "${OUT}/runner" \
  --git-sha "${GIT_SHA}" \
  --generation-max-new 2048 --generation-max-model-len 8192 --generation-max-turns 40

HOOKS=trim.probes.probe_c_qwen35_hooks
CFG="${OUT}/config_resolved.json"
python3 -c "import json;c=json.load(open('${CFG}'));g=c['generation'];assert g['max_new_tokens']==2048,g;print('GEN_CONTRACT',g)"

# R1 smoke：双单成员 auto_populate_first_search + sentence_compress（GPU 打分+regen）。
python3 -m trim.probes.probe_c_r1_smoke \
  --config "${CFG}" \
  --out "${OUT}" \
  --states-dir "${OLD_RUN}" \
  --members auto_populate_first_search,sentence_compress \
  --n-train-queries 16 --n-select-queries 16

echo "R1_SMOKE_DONE $(date -Is)"
cat "${OUT}/runner/Qwen_Qwen3.5-9B/acceptance_gate.json" 2>/dev/null || \
  python3 -c "import json;j=json.load(open('${OUT}/runner/Qwen_Qwen3.5-9B/R1_SMOKE_REPORT.json'));print(json.dumps(j['acceptance_gate'],indent=1))"
