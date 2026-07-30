#!/usr/bin/env bash
# Barrier 4: build overfit128 dataset + operation_ce overfit training
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/outputs/scope_round4/logs"
GPU="${BARRIER4_GPU:-4}"
mkdir -p "${LOG_DIR}"

echo "[barrier4] $(date -Is) build overfit128 dataset"
python training/scope_round4/build_overfit128.py

echo "[barrier4] $(date -Is) train operation_ce overfit128 on GPU${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round4/run_overfit128.py \
  --epochs 10 \
  --class-balancing \
  2>&1 | tee "${LOG_DIR}/barrier4_overfit128.log"

echo "[barrier4] complete $(date -Is)"
