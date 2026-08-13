#!/usr/bin/env bash
# Stage S four-grid closed-loop eval (harness-1, query-disjoint test split).
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/true_scape_evidence_graph/stage_s}"
STAGE_L="${SCAPE_ROOT}/outputs/true_scape_evidence_graph/stage_l_retry"
MODEL_PATH="${MODEL_PATH:-}"
LIMIT="${LIMIT:-200}"
SPLIT="${SPLIT:-test}"

if [[ -z "${MODEL_PATH}" ]]; then
  CKPT=$(find "${STAGE_L}" -path '*/weighted_L8K_s42/hf_merged' -type d 2>/dev/null | head -1)
  [[ -z "$CKPT" ]] && CKPT=$(find "${SCAPE_ROOT}/outputs/true_scape_evidence_graph/stage_l" -path '*/main_L8K_s42/hf_merged' -type d 2>/dev/null | head -1)
  MODEL_PATH="${CKPT:-/data/ppnm/models/harness-1}"
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.65}"
export MAX_MODEL_LEN_VLLM="${MAX_MODEL_LEN_VLLM:-16384}"

echo "[stage_s] model=${MODEL_PATH} limit=${LIMIT}"

# S2: trained + H_-graph; S3: trained + H_full (S0/S1 from pre-stage LOO)
for spec in "2:S2_trained_minus_graph:evidence_graph" "3:S3_trained_full:"; do
  IFS=: read -r gpu name comp <<<"$spec"
  out="${OUT_ROOT}/${name}"
  [[ -f "${out}/DONE" ]] && { echo "[skip] ${name}"; continue; }
  rm -f "${out}/worker.pid" "${out}/vllm.pid" 2>/dev/null || true
  nohup env GPU="$gpu" JOB_NAME="$name" COMPONENT="$comp" \
    OUT_ROOT="${OUT_ROOT}" LIMIT="${LIMIT}" SPLIT="${SPLIT}" \
    MODEL_PATH="${MODEL_PATH}" \
    bash "${SCAPE_ROOT}/scripts/run_loo_worker.sh" \
    >"${OUT_ROOT}/logs/${name}.log" 2>&1 &
  echo $! >"${OUT_ROOT}/pids/${name}.pid"
  echo "[launch] ${name} gpu=${gpu}"
done
