#!/usr/bin/env bash
# Phase 1 only: 8-GPU offline eval (fixed GPU assignment)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/outputs/scope_round4/logs"
mkdir -p "${LOG_DIR}/offline_retry"

declare -a OFFLINE=(
  "0:Base"
  "1:round3_compact_json"
  "2:round3_op_seed42"
  "3:round3_op_seed43"
  "4:round3_op_seed44"
  "5:round3_op_no_balance"
  "6:round3_correct_only"
  "7:round3_endorse_only"
)

pids=()
for entry in "${OFFLINE[@]}"; do
  IFS=':' read -r gpu variant <<< "${entry}"
  out="${REPO_ROOT}/outputs/scope_round4/postfix_replay/offline/${variant}.json"
  if [[ -f "${out}" ]]; then
    echo "[skip] ${variant} already done"
    continue
  fi
  log="${LOG_DIR}/offline_retry_${variant}.log"
  echo "[launch] GPU${gpu} ${variant}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round4/run_postfix_offline_eval.py \
    --variant "${variant}" > "${log}" 2>&1 &
  pids+=($!)
  sleep 3
done

for pid in "${pids[@]}"; do wait "${pid}" || true; done
echo "[done] offline eval $(date -Is)"
