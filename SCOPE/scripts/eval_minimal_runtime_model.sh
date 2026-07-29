#!/usr/bin/env bash
# Compare base vs Dup-SDI merged model on Minimal Runtime BrowseComp+.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

BASE_MODEL="${BASE_MODEL:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:-$REPO_ROOT/outputs/dup_sdi_round1}"
MERGED_DIR="${MERGED_DIR:-$REPO_ROOT/outputs/dup_sdi_round1/merged_hf}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/dup_sdi_round1/minimal_runtime_eval}"
LIMIT="${LIMIT:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
VLLM_PORT="${VLLM_PORT:-8775}"

export PYTHONPATH="${REPO_ROOT}"

if [[ ! -d "${MERGED_DIR}" || "${FORCE_MERGE:-0}" == "1" ]]; then
  echo "[eval] Merging LoRA adapter -> ${MERGED_DIR}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" python training/merge_lora_hf.py \
    --base-model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --output "${MERGED_DIR}"
fi

echo "[eval] Minimal Runtime rollout (trained model)"
export CUDA_VISIBLE_DEVICES
export MODEL_PATH="${MERGED_DIR}"
export OUTPUT_DIR
export LIMIT
export VLLM_PORT
export RESUME=0
bash "${REPO_ROOT}/scripts/rollout_minimal_runtime_browsecomp.sh"

python "${REPO_ROOT}/scripts/build_phase0_compare.py" || true

echo "[eval] Compare against Phase0 baseline:"
echo "  baseline recall=2.45% reward=0.1208 (artifacts/baselines/minimal_runtime_metrics.json)"
echo "  trained output: ${OUTPUT_DIR}"
