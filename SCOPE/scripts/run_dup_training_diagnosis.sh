#!/usr/bin/env bash
# Dup-SDI training-loop diagnosis — nohup-safe with --resume.
# Usage:
#   bash scripts/run_dup_training_diagnosis.sh          # foreground
#   bash scripts/run_dup_training_diagnosis.sh --nohup    # background
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Force a single free GPU for diagnosis training (override inherited env).
export CUDA_VISIBLE_DEVICES="${DIAG_CUDA_DEVICE:-5}"
export PYTHONPATH="${REPO_ROOT}"

OUTPUT_DIR="${REPO_ROOT}/outputs/dup_sdi_round1/diagnosis"
ADAPTER="${ADAPTER:-$REPO_ROOT/outputs/dup_sdi_round1}"
OVERFIT_ADAPTER="${OVERFIT_ADAPTER:-$REPO_ROOT/outputs/dup_sdi_round1/minimal_runtime_smoke20/trained/overfit64}"
LOG_DIR="${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

NOHUP_MODE=0
EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--nohup" ]]; then
    NOHUP_MODE=1
  else
    EXTRA_ARGS+=("$arg")
  fi
done

RUN_ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --adapter "${ADAPTER}"
  --resume
)
if [[ -d "${OVERFIT_ADAPTER}" ]]; then
  RUN_ARGS+=(--overfit-adapter "${OVERFIT_ADAPTER}")
fi
RUN_ARGS+=("${EXTRA_ARGS[@]}")

cd "${REPO_ROOT}"

if [[ "${NOHUP_MODE}" -eq 1 ]]; then
  LOG_FILE="${LOG_DIR}/nohup_diagnosis_$(date +%Y%m%d_%H%M%S).log"
  echo "[diag] Starting nohup on GPU ${CUDA_VISIBLE_DEVICES} -> ${LOG_FILE}"
  nohup python training/run_dup_training_diagnosis.py "${RUN_ARGS[@]}" \
    > "${LOG_FILE}" 2>&1 &
  echo "[diag] PID=$!  tail -f ${LOG_FILE}"
else
  python training/run_dup_training_diagnosis.py "${RUN_ARGS[@]}"
  echo "[diag] Done: ${OUTPUT_DIR}/diagnosis_report.json"
fi
