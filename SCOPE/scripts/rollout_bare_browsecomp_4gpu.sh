#!/usr/bin/env bash
# Full BrowseComp+ bare rollout on GPUs 4-7: tau ~ pi_theta(x), no Harness.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/bare_rollout_browsecomp_full}"
SPLIT="${SPLIT:-all}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
VLLM_PORT="${VLLM_PORT:-8770}"
RESUME="${RESUME:-1}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="${REPO_ROOT}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export BROWSECOMPPLUS_ANSWERS_PATH="${BROWSECOMPPLUS_ANSWERS_PATH:-$REPO_ROOT/external/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl}"
export BROWSECOMPPLUS_QUERIES_PATH="${BROWSECOMPPLUS_QUERIES_PATH:-$REPO_ROOT/external/BrowseComp-Plus/topics-qrels/queries.tsv}"

cd "${REPO_ROOT}"

if [[ ! -s "${BROWSECOMPPLUS_ANSWERS_PATH}" && ! -s "${BROWSECOMPPLUS_QUERIES_PATH}" ]]; then
  echo "BrowseComp+ data missing; running setup_browsecomp_data.sh ..."
  bash "${REPO_ROOT}/scripts/setup_browsecomp_data.sh"
fi

echo "=== Bare BrowseComp+ FULL rollout (no Harness) ==="
echo "Model:          ${MODEL_PATH}"
echo "GPUs:           CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Split:          ${SPLIT} (limit=${LIMIT:-all})"
echo "max_new_tokens: ${MAX_NEW_TOKENS}"
echo "temperature:    ${TEMPERATURE}"
echo "max_model_len:  ${MAX_MODEL_LEN}"
echo "Output:         ${OUTPUT_DIR}"
echo

ARGS=(
  --model-path "${MODEL_PATH}"
  --split "${SPLIT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --vllm-port "${VLLM_PORT}"
  --tensor-parallel-size 4
  --output-dir "${OUTPUT_DIR}"
)
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi
if [[ "${LIMIT}" != "0" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

python training/rollout_bare_browsecomp.py "${ARGS[@]}"

echo
echo "Done. See ${OUTPUT_DIR}/bare_rollout_manifest.json"
