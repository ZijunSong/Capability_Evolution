#!/usr/bin/env bash
# SCOPE E0 Capability Distillability Map — 100q pilot
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
ROOT_OUT="${ROOT_OUT:-$REPO_ROOT/outputs/scope_e0_distillability}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-100}"
SEED="${SEED:-42}"
VLLM_PORT="${VLLM_PORT:-8776}"
PARALLEL="${PARALLEL:-2}"
CAPABILITIES="${CAPABILITIES:-duplicate_evidence,stop_decision,evidence_curation,verification_decision,external_verification,deterministic_truncation}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export JAVA_HOME="${JAVA_HOME:-$CONDA_PREFIX/lib/jvm}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export JVM_PATH="${JVM_PATH:-$JAVA_HOME/lib/server/libjvm.so}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$REPO_ROOT/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}"
export BROWSECOMPPLUS_QUERIES_PATH="${BROWSECOMPPLUS_QUERIES_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/queries.tsv}"
export BROWSECOMP_BM25_INDEX_PATH="${BROWSECOMP_BM25_INDEX_PATH:-$REPO_ROOT/external/BrowseComp-Plus/indexes/bm25}"
export CHAT_MIN_TURNS_BEFORE_END="${CHAT_MIN_TURNS_BEFORE_END:-8}"
export CHAT_MIN_CURATED_BEFORE_END="${CHAT_MIN_CURATED_BEFORE_END:-1}"
export CHAT_MAX_WM_CHARS="${CHAT_MAX_WM_CHARS:-18000}"

cd "${REPO_ROOT}"
mkdir -p "${ROOT_OUT}"

# Start vLLM if not already running
bash "${REPO_ROOT}/scripts/start_e0_vllm.sh"
export base_url="http://127.0.0.1:${VLLM_PORT}/v1"
export api_key="EMPTY"
export model_name="${SERVED_MODEL_NAME:-e0-harness-policy}"

echo "=== SCOPE E0 Distillability 100q Pilot ==="
echo "Output: ${ROOT_OUT}"
echo "Capabilities: ${CAPABILITIES}"
echo

IFS=',' read -ra CAPS <<< "${CAPABILITIES}"

for CAP in "${CAPS[@]}"; do
  CAP="$(echo "${CAP}" | xargs)"
  echo "--- ${CAP} ---"

  echo "[${CAP}] FULL (reuse if matched) ..."
  python training/scope/distillability/runner.py \
    --capability "${CAP}" \
    --mode full \
    --output-dir "${ROOT_OUT}" \
    --limit "${LIMIT}" \
    --seed "${SEED}" \
    --model-path "${MODEL_PATH}" \
    --parallel "${PARALLEL}" \
    --vllm-port "${VLLM_PORT}" \
    --no-manage-vllm \
    --vllm-url "http://127.0.0.1:${VLLM_PORT}/v1"

  echo "[${CAP}] OFF ..."
  python training/scope/distillability/runner.py \
    --capability "${CAP}" \
    --mode off \
    --output-dir "${ROOT_OUT}" \
    --limit "${LIMIT}" \
    --seed "${SEED}" \
    --model-path "${MODEL_PATH}" \
    --parallel "${PARALLEL}" \
    --vllm-port "${VLLM_PORT}" \
    --no-manage-vllm \
    --vllm-url "http://127.0.0.1:${VLLM_PORT}/v1" \
    --resume

  if [[ "${CAP}" != "deterministic_truncation" ]]; then
    echo "[${CAP}] PROC ..."
    python training/scope/distillability/runner.py \
      --capability "${CAP}" \
      --mode proc \
      --output-dir "${ROOT_OUT}" \
      --limit "${LIMIT}" \
      --seed "${SEED}" \
      --model-path "${MODEL_PATH}" \
      --parallel "${PARALLEL}" \
      --vllm-port "${VLLM_PORT}" \
      --no-manage-vllm \
      --vllm-url "http://127.0.0.1:${VLLM_PORT}/v1" \
      --resume
  fi
  echo
done

echo "=== Building distillability map ==="
python training/scope/distillability/build_map.py \
  --root "${ROOT_OUT}" \
  --output-map "${REPO_ROOT}/artifacts/capability/distillability_map.json" \
  --output-report "${ROOT_OUT}/E0_REPORT.md"

echo "Done. See ${ROOT_OUT}/E0_REPORT.md"

if [[ "${E0_CLEANUP_VLLM:-1}" == "1" ]]; then
  echo "=== Stopping E0 vLLM ==="
  bash "${REPO_ROOT}/scripts/stop_e0_vllm.sh" || true
fi
