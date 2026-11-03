#!/usr/bin/env bash
# Probe C2 闭环评测：Qwen3.5-9B 1node8gpu。C_select 完整 episode（final-answer-recall J）。
# 5 条件：no_update(θ0,H0) / runtime_m(θ0,H_m) / harnessopd_m(θ^8,H0) / rl_only(θ^8,H0)
# / same_view(θ^8,H0)；compare 出 G_m^8 / E_m^8（paired bootstrap）。
# 先跑两个无需 adapter 的条件，再等 3 个 8-step adapter（上限 12h），最后 compare。
set -euxo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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

: "${OUT:?FATAL: OUT must be set}"
: "${STATES_OUT:?FATAL: STATES_OUT must be set}"
: "${OUT_HM:?FATAL: OUT_HM (harnessopd_m training OUT) must be set}"
: "${OUT_RL:?FATAL: OUT_RL (rl_only training OUT) must be set}"
: "${OUT_SV:?FATAL: OUT_SV (same_view training OUT) must be set}"

JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
if [ -z "${JVM_PATH}" ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
  JVM_PATH=$(find /usr /opt -type f -name libjvm.so 2>/dev/null | head -1 || true)
fi
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
export CLASSPATH="$ANSERINI_CLASSPATH"

nvidia-smi -L
test -f /mnt/songzijun/models/Qwen3.5-9B/config.json
test -f "${STATES_OUT}/SPLIT_ASSIGNMENT.json"
mkdir -p "${OUT}/c2"

MODEL_SRC=/mnt/songzijun/models/Qwen3.5-9B
MODEL_LOCAL=/tmp/qwen35-9b
echo "MODEL_CP_START $(date -Is)"
cp -r "${MODEL_SRC}"/. "${MODEL_LOCAL}"/
echo "MODEL_CP_DONE $(date -Is)"

C2E="python3 -m trim.probes.probe_c_c2_eval"

# 条件 1/2：无需 adapter，先跑。
${C2E} run --model-path "${MODEL_LOCAL}" --states-out "${STATES_OUT}" \
  --condition no_update --mask h0 --out "${OUT}/c2"
${C2E} run --model-path "${MODEL_LOCAL}" --states-out "${STATES_OUT}" \
  --condition runtime_m --mask hm --out "${OUT}/c2"

# 等 3 个 8-step adapter（正式训练另有 job；上限 12h）。
find_adapter() {
  # pipefail: a missing checkpoints dir makes find exit 1 — never let that
  # kill the wait loop.
  { find "$1"/checkpoints -maxdepth 3 -type d -name adapter -path "*step_00008*" 2>/dev/null || true; } | head -1 || true
}
WAIT_MAX_S=43200
WAITED=0
while true; do
  AD_HM=$(find_adapter "${OUT_HM}")
  AD_RL=$(find_adapter "${OUT_RL}")
  AD_SV=$(find_adapter "${OUT_SV}")
  if [ -n "${AD_HM}" ] && [ -n "${AD_RL}" ] && [ -n "${AD_SV}" ]; then
    break
  fi
  if [ "${WAITED}" -ge "${WAIT_MAX_S}" ]; then
    echo "FATAL: adapters not ready after ${WAIT_MAX_S}s (hm=${AD_HM:-none} rl=${AD_RL:-none} sv=${AD_SV:-none})" >&2
    exit 1
  fi
  echo "C2_EVAL waiting adapters... waited=${WAITED}s"
  sleep 300
  WAITED=$((WAITED + 300))
done
echo "C2_EVAL adapters: hm=${AD_HM} rl=${AD_RL} sv=${AD_SV}"

${C2E} run --model-path "${MODEL_LOCAL}" --states-out "${STATES_OUT}" \
  --condition harnessopd_m --adapter "${AD_HM}" --mask h0 --out "${OUT}/c2"
${C2E} run --model-path "${MODEL_LOCAL}" --states-out "${STATES_OUT}" \
  --condition rl_only --adapter "${AD_RL}" --mask h0 --out "${OUT}/c2"
${C2E} run --model-path "${MODEL_LOCAL}" --states-out "${STATES_OUT}" \
  --condition same_view --adapter "${AD_SV}" --mask h0 --out "${OUT}/c2"

${C2E} compare --out "${OUT}/c2"
echo "PROBE_C_C2_DONE $(date -Is)"
