#!/usr/bin/env bash
# Expand MICRO_PASS winner to L8K + baseline ablations (SCAPE-0813-H20 §6).
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/true_scape_candidate_b_tournament}"
DATA_DIR="${OUT_ROOT}/data"
STAGE_L="${OUT_ROOT}/stage_l_8k"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/harness-1}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${OUT_ROOT}/logs_8k"
PID_DIR="${OUT_ROOT}/pids_8k"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

WINNER="${WINNER:-}"
if [[ -z "${WINNER}" ]]; then
  if [[ -f "${OUT_ROOT}/CANDIDATE_B_FINAL.json" ]]; then
    WINNER="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${OUT_ROOT}/CANDIDATE_B_FINAL.json'))['winner_component_id'])")"
  else
    echo "WINNER not set and CANDIDATE_B_FINAL.json missing" >&2
    exit 1
  fi
fi

mkdir -p "${STAGE_L}" "${LOG_DIR}" "${PID_DIR}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

run_cell() {
  local gpu="$1" tag="$2" n="$3" seed="$4" loss="$5"
  local out="${STAGE_L}/gpu${gpu}/${tag}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] gpu${gpu} ${tag}"
    return 0
  fi
  local train="${DATA_DIR}/${WINNER}_TRAIN_8K.jsonl"
  local valid="${DATA_DIR}/${WINNER}_VALID_512.jsonl"
  local test="${DATA_DIR}/${WINNER}_TEST_512.jsonl"
  mkdir -p "${out}"
  echo "[launch] gpu${gpu} ${tag} winner=${WINNER} n=${n} seed=${seed} loss=${loss}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_true_scape_stage_l_cell.py" \
    --out "${out}" \
    --model-path "${MODEL_PATH}" \
    --train-jsonl "${train}" \
    --valid-jsonl "${valid}" \
    --test-jsonl "${test}" \
    --component-id "${WINNER}" \
    --n-samples "${n}" \
    --seed "${seed}" \
    --loss-path "${loss}" \
    --gpu 0 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    >"${LOG_DIR}/gpu${gpu}_${tag}.log" 2>&1
  touch "${out}/DONE"
}

run_gpu_queue() {
  local gpu="$1"
  shift
  local log="${LOG_DIR}/gpu${gpu}_queue.log"
  {
    echo "[$(date -Iseconds)] gpu${gpu} 8k queue start winner=${WINNER}"
    while [[ $# -gt 0 ]]; do
      run_cell "${gpu}" "$@"
      shift 4
    done
    echo "[$(date -Iseconds)] gpu${gpu} 8k queue ALL_DONE"
    touch "${STAGE_L}/gpu${gpu}/ALL_DONE"
  } >>"${log}" 2>&1
}

launch_bg() {
  local gpu="$1"; shift
  run_gpu_queue "${gpu}" "$@" &
  echo $! >"${PID_DIR}/gpu${gpu}.pid"
  echo "[bg] gpu${gpu} pid=$(cat "${PID_DIR}/gpu${gpu}.pid")"
}

if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "0" ]]; then
  launch_bg 0 "W_L8K_s42" 8000 42 tool_token_kl
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "1" ]]; then
  launch_bg 1 "W_L8K_s43" 8000 43 tool_token_kl
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "2" ]]; then
  launch_bg 2 "W_L8K_s44" 8000 44 tool_token_kl
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "3" ]]; then
  launch_bg 3 "W_name_only_L8K" 8000 42 tool_name_only_kl
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "4" ]]; then
  launch_bg 4 "W_args_only_L8K" 8000 42 args_only_kl
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "5" ]]; then
  launch_bg 5 "W_action_ce_L8K" 8000 42 action_ce
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "6" ]]; then
  launch_bg 6 "W_full_response_L8K" 8000 42 full_response_kl
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "7" ]]; then
  launch_bg 7 "W_offpolicy_L8K" 8000 42 offpolicy_matched
fi

echo "[launch] Winner 8K expansion started winner=${WINNER}"
