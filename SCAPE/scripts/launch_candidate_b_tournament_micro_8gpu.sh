#!/usr/bin/env bash
# Candidate-B micro-learnability tournament — Stage L micro (L512/L2K only).
# GPU map per SCAPE-0813-H20 §3.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/true_scape_candidate_b_tournament}"
DATA_DIR="${OUT_ROOT}/data"
STAGE_L="${OUT_ROOT}/stage_l_micro"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/harness-1}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${OUT_ROOT}/logs_micro"
PID_DIR="${OUT_ROOT}/pids_micro"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${OUT_ROOT}" "${DATA_DIR}" "${STAGE_L}" "${LOG_DIR}" "${PID_DIR}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${GPU_ONLY:-}" ]]; then
  echo "[preflight]"
  "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/preflight_scape.py" \
    --model-path "${MODEL_PATH}" \
    --json-out "${OUT_ROOT}/PREFLIGHT.json"

  echo "[data] build SC/IT/VT splits"
  "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/build_candidate_b_tournament_splits.py" \
    --out-dir "${DATA_DIR}"
fi

run_cell() {
  local gpu="$1" tag="$2" component="$3" n="$4" seed="$5" loss="$6"
  local out="${STAGE_L}/gpu${gpu}/${tag}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] gpu${gpu} ${tag}"
    return 0
  fi
  local train="${DATA_DIR}/${component}_TRAIN_8K.jsonl"
  local valid="${DATA_DIR}/${component}_VALID_512.jsonl"
  local test="${DATA_DIR}/${component}_TEST_512.jsonl"
  mkdir -p "${out}"
  echo "[launch] gpu${gpu} ${tag} component=${component} n=${n} seed=${seed} loss=${loss}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_true_scape_stage_l_cell.py" \
    --out "${out}" \
    --model-path "${MODEL_PATH}" \
    --train-jsonl "${train}" \
    --valid-jsonl "${valid}" \
    --test-jsonl "${test}" \
    --component-id "${component}" \
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
    echo "[$(date -Iseconds)] gpu${gpu} micro queue start"
    while [[ $# -gt 0 ]]; do
      run_cell "${gpu}" "$@"
      shift 5
    done
    echo "[$(date -Iseconds)] gpu${gpu} micro queue ALL_DONE"
    touch "${STAGE_L}/gpu${gpu}/ALL_DONE"
  } >>"${log}" 2>&1
}

launch_bg() {
  local gpu="$1"; shift
  run_gpu_queue "${gpu}" "$@" &
  echo $! >"${PID_DIR}/gpu${gpu}.pid"
  echo "[bg] gpu${gpu} pid=$(cat "${PID_DIR}/gpu${gpu}.pid")"
}

if [[ -z "${GPU_ONLY:-}" ]]; then
  for g in 0 1 2 3 4 5 6 7; do
    pf="${PID_DIR}/gpu${g}.pid"
    if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
      echo "[cleanup] killing stale gpu${g} pid=$(cat "$pf")"
      kill "$(cat "$pf")" 2>/dev/null || true
    fi
  done
  sleep 2
fi

# GPU0: SC seed42
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "0" ]]; then
  launch_bg 0 \
    "SC_L512_s42" subtractive_curation 512 42 tool_token_kl \
    "SC_L2K_s42" subtractive_curation 2000 42 tool_token_kl
fi
# GPU1: SC seed43
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "1" ]]; then
  launch_bg 1 \
    "SC_L512_s43" subtractive_curation 512 43 tool_token_kl \
    "SC_L2K_s43" subtractive_curation 2000 43 tool_token_kl
fi
# GPU2: IT seed42
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "2" ]]; then
  launch_bg 2 \
    "IT_L512_s42" importance_tagging 512 42 tool_token_kl \
    "IT_L2K_s42" importance_tagging 2000 42 tool_token_kl
fi
# GPU3: IT seed43
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "3" ]]; then
  launch_bg 3 \
    "IT_L512_s43" importance_tagging 512 43 tool_token_kl \
    "IT_L2K_s43" importance_tagging 2000 43 tool_token_kl
fi
# GPU4: VT seed42 (natural only — synthetic collection uses natural states)
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "4" ]]; then
  launch_bg 4 \
    "VT_L512_s42" verify_tool 512 42 tool_token_kl \
    "VT_L2K_s42" verify_tool 2000 42 tool_token_kl
fi
# GPU5: VT seed43
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "5" ]]; then
  launch_bg 5 \
    "VT_L512_s43" verify_tool 512 43 tool_token_kl \
    "VT_L2K_s43" verify_tool 2000 43 tool_token_kl
fi
# GPU6: SC action-CE @2K
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "6" ]]; then
  launch_bg 6 \
    "SC_action_ce_L2K_s42" subtractive_curation 2000 42 action_ce
fi
# GPU7: SC tool-name-only @2K
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "7" ]]; then
  launch_bg 7 \
    "SC_name_only_L2K_s42" subtractive_curation 2000 42 tool_name_only_kl
fi

echo "[launch] Candidate-B micro tournament started under ${OUT_ROOT}"
