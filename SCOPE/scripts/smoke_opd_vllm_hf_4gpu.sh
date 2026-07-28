#!/usr/bin/env bash
# Smoke: vLLM rollout (GPUs 4-7) -> stop vLLM -> HF OPD train on same GPUs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/smoke_opd_vllm_hf_4gpu}"
VLLM_PORT="${VLLM_PORT:-8765}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

cd "${REPO_ROOT}"

echo "=== BiSHOP OPD smoke (vLLM rollout + HF train) ==="
echo "Model:  ${MODEL_PATH}"
echo "GPUs:   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Output: ${OUTPUT_DIR}"
echo

python training/smoke_opd_vllm_hf.py \
  --model-path "${MODEL_PATH}" \
  --queries-json tests/fixtures/browsecomp_sample_queries.json \
  --limit 2 \
  --max-new-tokens 24 \
  --vllm-port "${VLLM_PORT}" \
  --tensor-parallel-size 4 \
  --output-dir "${OUTPUT_DIR}"

echo
echo "Smoke test finished. See ${OUTPUT_DIR}/smoke_manifest.json"
