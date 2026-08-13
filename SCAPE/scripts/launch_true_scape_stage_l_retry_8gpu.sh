#!/usr/bin/env bash
# Stage L retry — weighted name/args tool-token KL (Part H, one allowed retry).
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/true_scape_evidence_graph}"
STAGE_L="${OUT_ROOT}/stage_l_retry"
DATA_DIR="${OUT_ROOT}/data"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/harness-1}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${OUT_ROOT}/logs_retry"
PID_DIR="${OUT_ROOT}/pids_retry"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${OUT_ROOT}" "${STAGE_L}" "${LOG_DIR}" "${PID_DIR}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

TRAIN="${DATA_DIR}/EG_TRAIN_8K.jsonl"
VALID="${DATA_DIR}/EG_VALID_1K.jsonl"
TEST="${DATA_DIR}/EG_TEST_1K.jsonl"

run_cell() {
  local gpu="$1" tag="$2" n="$3" seed="$4"
  local out="${STAGE_L}/gpu${gpu}/${tag}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] gpu${gpu} ${tag}"
    return 0
  fi
  mkdir -p "${out}"
  echo "[launch] gpu${gpu} ${tag} n=${n} seed=${seed} loss=weighted_tool_token_kl"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_true_scape_stage_l_cell.py" \
    --out "${out}" \
    --model-path "${MODEL_PATH}" \
    --train-jsonl "${TRAIN}" \
    --valid-jsonl "${VALID}" \
    --test-jsonl "${TEST}" \
    --component-id evidence_graph \
    --n-samples "${n}" \
    --seed "${seed}" \
    --loss-path weighted_tool_token_kl \
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
    echo "[$(date -Iseconds)] gpu${gpu} retry queue start"
    while [[ $# -gt 0 ]]; do
      run_cell "${gpu}" "$@"
      shift 3
    done
    echo "[$(date -Iseconds)] gpu${gpu} retry queue ALL_DONE"
    touch "${STAGE_L}/gpu${gpu}/ALL_DONE"
  } >>"${log}" 2>&1
}

launch_bg() {
  local gpu="$1"; shift
  run_gpu_queue "${gpu}" "$@" &
  echo $! >"${PID_DIR}/gpu${gpu}.pid"
  echo "[bg] gpu${gpu} pid=$(cat "${PID_DIR}/gpu${gpu}.pid")"
}

# 8-GPU parallel: main seeds 42/43/44 at 512/2k/8k
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "0" ]]; then
  launch_bg 0 "weighted_L512_s42" 512 42 "weighted_L2K_s42" 2000 42 "weighted_L8K_s42" 8000 42
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "1" ]]; then
  launch_bg 1 "weighted_L512_s43" 512 43 "weighted_L2K_s43" 2000 43 "weighted_L8K_s43" 8000 43
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "2" ]]; then
  launch_bg 2 "weighted_L2K_s44" 2000 44 "weighted_L8K_s44" 8000 44
fi
# GPU3-7: extra seeds for robustness (same weighted loss)
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "3" ]]; then
  launch_bg 3 "weighted_L8K_s45" 8000 45
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "4" ]]; then
  launch_bg 4 "weighted_L8K_s46" 8000 46
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "5" ]]; then
  launch_bg 5 "weighted_L8K_s47" 8000 47
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "6" ]]; then
  launch_bg 6 "weighted_L8K_s48" 8000 48
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "7" ]]; then
  launch_bg 7 "weighted_L8K_s49" 8000 49
fi

echo "[launch] weighted Stage L retry started under ${STAGE_L}"
