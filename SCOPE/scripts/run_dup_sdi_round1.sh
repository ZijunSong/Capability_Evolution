#!/usr/bin/env bash
# Dup-only SDI Round-1: build dataset + LoRA train on GPU 4.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONPATH="${REPO_ROOT}"

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/artifacts/datasets/dup_sdi_round1}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/dup_sdi_round1}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/scope/sdi_dup_only.yaml}"
SAMPLES="${SAMPLES:-$REPO_ROOT/outputs/scope_v3_audit_100q/natural_100q/samples.jsonl}"

cd "${REPO_ROOT}"
mkdir -p "${DATASET_DIR}" "${OUTPUT_DIR}"

echo "[dup-sdi] Building dataset from ${SAMPLES}"
python training/build_dup_sdi_dataset.py \
  --samples "${SAMPLES}" \
  --out-dir "${DATASET_DIR}"

echo "[dup-sdi] Training (GPU=${CUDA_VISIBLE_DEVICES})"
python training/train_sdi_dup.py \
  --config "${CONFIG}" \
  --train "${DATASET_DIR}/train.jsonl" \
  --valid "${DATASET_DIR}/valid.jsonl" \
  --output-dir "${OUTPUT_DIR}"

echo "[dup-sdi] Valid capability eval"
python training/scope/eval_dup_capability.py \
  --valid "${DATASET_DIR}/valid.jsonl" \
  --model-path /data/ppnm/models/Qwen2.5-7B-Instruct \
  --adapter-path "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/capability_eval.json"

echo "[dup-sdi] Done. Artifacts: ${OUTPUT_DIR}"
