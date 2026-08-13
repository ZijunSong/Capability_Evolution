#!/usr/bin/env bash
# True SCAPE evidence_graph Stage L — 8×H20 parallel queue per SCAPE-0813-H20 Part F.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/true_scape_evidence_graph}"
DATA_DIR="${OUT_ROOT}/data"
STAGE_L="${OUT_ROOT}/stage_l"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/harness-1}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${OUT_ROOT}/logs"
PID_DIR="${OUT_ROOT}/pids"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${OUT_ROOT}" "${DATA_DIR}" "${STAGE_L}" "${LOG_DIR}" "${PID_DIR}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

echo "[preflight] scape + harness-1"
if [[ -z "${GPU_ONLY:-}" ]]; then
  "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/preflight_scape.py" \
    --model-path "${MODEL_PATH}" \
    --json-out "${OUT_ROOT}/PREFLIGHT.json"

  echo "[data] build EG_TRAIN_8K / EG_VALID_1K / EG_TEST_1K"
  "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/build_eg_same_state_splits.py" \
    --out-dir "${DATA_DIR}"
fi

TRAIN="${DATA_DIR}/EG_TRAIN_8K.jsonl"
VALID="${DATA_DIR}/EG_VALID_1K.jsonl"
TEST="${DATA_DIR}/EG_TEST_1K.jsonl"

run_cell() {
  local gpu="$1" tag="$2" n="$3" seed="$4" loss="$5"
  local out="${STAGE_L}/gpu${gpu}/${tag}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] gpu${gpu} ${tag}"
    return 0
  fi
  mkdir -p "${out}"
  echo "[launch] gpu${gpu} ${tag} n=${n} seed=${seed} loss=${loss}"
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
    echo "[$(date -Iseconds)] gpu${gpu} queue start"
    while [[ $# -gt 0 ]]; do
      run_cell "${gpu}" "$@"
      shift 4
    done
    echo "[$(date -Iseconds)] gpu${gpu} queue ALL_DONE"
    touch "${STAGE_L}/gpu${gpu}/ALL_DONE"
  } >>"${log}" 2>&1
}

# Stop stale per-gpu workers on this OUT_ROOT only (full launch, not relaunch)
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

# Part F queues — each cell starts fresh from base (independent in run_cell)
launch_bg() {
  local gpu="$1"; shift
  run_gpu_queue "${gpu}" "$@" &
  echo $! >"${PID_DIR}/gpu${gpu}.pid"
  echo "[bg] gpu${gpu} pid=$(cat "${PID_DIR}/gpu${gpu}.pid")"
}

# GPU0: main seed42
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "0" ]]; then
  launch_bg 0 \
    "main_L512_s42" 512 42 tool_token_kl \
    "main_L2K_s42" 2000 42 tool_token_kl \
    "main_L8K_s42" 8000 42 tool_token_kl
fi

# GPU1: main seed43
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "1" ]]; then
  launch_bg 1 \
    "main_L512_s43" 512 43 tool_token_kl \
    "main_L2K_s43" 2000 43 tool_token_kl \
    "main_L8K_s43" 8000 43 tool_token_kl
fi

# GPU2: seed44
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "2" ]]; then
  launch_bg 2 \
    "main_L2K_s44" 2000 44 tool_token_kl \
    "main_L8K_s44" 8000 44 tool_token_kl
fi

# GPU3-7 baselines
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "3" ]]; then
  launch_bg 3 "baseline_action_ce_L2K" 2000 42 action_ce "baseline_action_ce_L8K" 8000 42 action_ce
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "4" ]]; then
  launch_bg 4 "baseline_full_response_L2K" 2000 42 full_response_kl "baseline_full_response_L8K" 8000 42 full_response_kl
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "5" ]]; then
  launch_bg 5 "baseline_offpolicy_L2K" 2000 42 offpolicy_matched "baseline_offpolicy_L8K" 8000 42 offpolicy_matched
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "6" ]]; then
  launch_bg 6 "baseline_name_only_L2K" 2000 42 tool_name_only_kl "baseline_name_only_L8K" 8000 42 tool_name_only_kl
fi
if [[ -z "${GPU_ONLY:-}" || "${GPU_ONLY}" == "7" ]]; then
  launch_bg 7 "baseline_args_only_L2K" 2000 42 args_only_kl "baseline_args_only_L8K" 8000 42 args_only_kl
fi

if [[ -z "${GPU_ONLY:-}" ]]; then
  echo "[launch] 8 GPU queues started under ${OUT_ROOT}"
else
  echo "[launch] GPU ${GPU_ONLY} queue (relaunch) under ${OUT_ROOT}"
fi

cat > "${OUT_ROOT}/GPU_QUEUE.json" <<'EOF'
{
  "0": {"job": "EG L512/2K/8K + heldout", "seed": 42, "loss": "tool_token_kl"},
  "1": {"job": "EG L512/2K/8K + heldout", "seed": 43, "loss": "tool_token_kl"},
  "2": {"job": "EG L2K/8K + heldout", "seed": 44, "loss": "tool_token_kl"},
  "3": {"job": "same_state_action_ce", "loss": "action_ce"},
  "4": {"job": "full_response_kl", "loss": "full_response_kl"},
  "5": {"job": "offpolicy_harness_trace", "loss": "offpolicy_matched"},
  "6": {"job": "tool_name_only_kl", "loss": "tool_name_only_kl"},
  "7": {"job": "args_only_kl", "loss": "args_only_kl"}
}
EOF
