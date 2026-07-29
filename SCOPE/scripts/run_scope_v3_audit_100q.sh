#!/usr/bin/env bash
# SCOPE v3 formal 100q audit: natural online + targeted valid-stop (separate dirs).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
ROOT_OUT="${ROOT_OUT:-$REPO_ROOT/outputs/scope_v3_audit_100q}"
NATURAL_OUT="${NATURAL_OUT:-$ROOT_OUT/natural_100q}"
TARGETED_OUT="${TARGETED_OUT:-$ROOT_OUT/targeted_valid_stop}"
SCOPE_CONFIG="${SCOPE_CONFIG:-$REPO_ROOT/configs/scope/sdi_dup_premature.yaml}"
HARNESS_CONFIG="${HARNESS_CONFIG:-$REPO_ROOT/harness/configs/modules_full_v2.yaml}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-100}"
SEED="${SEED:-42}"
VLLM_PORT="${VLLM_PORT:-8775}"
PARALLEL="${PARALLEL:-2}"
TARGETED_N="${TARGETED_N:-24}"
SKIP_TARGETED="${SKIP_TARGETED:-0}"
SKIP_NATURAL="${SKIP_NATURAL:-0}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"
mkdir -p "${ROOT_OUT}"

echo "=== SCOPE v3 formal audit ==="
echo "Harness (Phase0 Full v2): ${HARNESS_CONFIG}"
echo "Natural 100q:             ${NATURAL_OUT}"
echo "Targeted valid-stop:      ${TARGETED_OUT}"
echo

if [[ "${SKIP_TARGETED}" != "1" ]]; then
  echo "[1/2] Targeted valid-stop probe (synthetic, NOT training data) ..."
  python training/audit_scope_v3_targeted_valid_stop.py \
    --output-dir "${TARGETED_OUT}" \
    --n-probes "${TARGETED_N}"
  echo
fi

if [[ "${SKIP_NATURAL}" != "1" ]]; then
  echo "[2/2] Natural 100q online audit ..."
  export OUTPUT_DIR="${NATURAL_OUT}"
  export HARNESS_CONFIG
  export SCOPE_CONFIG
  export LIMIT
  export SEED
  export VLLM_PORT
  export PARALLEL
  export RESUME="${RESUME:-1}"
  export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-scope-v3-audit100}"
  bash "${REPO_ROOT}/scripts/run_scope_v3_protocol_smoke20.sh"
fi

echo
echo "=== Done ==="
echo "Targeted: ${TARGETED_OUT}/summary.json"
echo "Natural:  ${NATURAL_OUT}/summary.json"
