#!/usr/bin/env bash
# SCAPE-0814-H20 Phase C2 — Clean Graph-Hybrid micro 512/2K on 8×H20.
set -euo pipefail
trap '' HUP

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/0814_clean_mechanism}"
GH_DATA="${GH_DATA:-${SCAPE_ROOT}/outputs/0813_next_h20/graph_hybrid/data}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${OUT_ROOT}/logs"
PID_DIR="${OUT_ROOT}/pids"
MICRO="${OUT_ROOT}/micro"
BASE_OSS="${BASE_OSS:-/data/ppnm/models/gpt-oss-20b}"
HARNESS1="${HARNESS1:-/data/ppnm/models/harness-1}"

mkdir -p "${MICRO}" "${LOG_DIR}" "${PID_DIR}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# Resolve preferred clean base (TOOL s42 preferred).
TOOL_FULL="${OUT_ROOT}/sft/gpu2/tool_s42_full/lora_checkpoint"
TOOL_SMOKE="${OUT_ROOT}/sft/gpu2/tool_s42_smoke/lora_checkpoint"
FULL_FULL="${OUT_ROOT}/sft/gpu0/full_s42_full/lora_checkpoint"
FULL_SMOKE="${OUT_ROOT}/sft/gpu0/full_s42_smoke/lora_checkpoint"
if [[ -d "${TOOL_FULL}" ]]; then TOOL_BASE="${TOOL_FULL}"
elif [[ -d "${TOOL_SMOKE}" ]]; then TOOL_BASE="${TOOL_SMOKE}"
else TOOL_BASE="${BASE_OSS}"; fi
if [[ -d "${FULL_FULL}" ]]; then FULL_BASE="${FULL_FULL}"
elif [[ -d "${FULL_SMOKE}" ]]; then FULL_BASE="${FULL_SMOKE}"
else FULL_BASE="${BASE_OSS}"; fi

LOSS_MAIN="${LOSS_MAIN:-route_kl}"
if [[ -f "${SCAPE_ROOT}/imports/h100_4/H1004_OBJECTIVE_HANDOFF.json" ]]; then
  rec=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${SCAPE_ROOT}/imports/h100_4/H1004_OBJECTIVE_HANDOFF.json')); print(d.get('recommendation') or d.get('NEXT') or '')")
  case "${rec}" in
    TOKEN_OBJECTIVE_OK) LOSS_MAIN=tool_name_only_kl ;;
    NO_OBJECTIVE_RESCUE) echo "[c2] H100-4 NO_OBJECTIVE_RESCUE — still running micro for the record" ;;
  esac
fi

TRAIN="${GH_DATA}/GH_TRAIN_8K.jsonl"
VALID="${GH_DATA}/GH_VALID_1K.jsonl"
TEST="${GH_DATA}/GH_TEST_1K.jsonl"

run_cell() {
  local gpu="$1" tag="$2" n="$3" seed="$4" loss="$5" model="$6"
  local out="${MICRO}/gpu${gpu}/${tag}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] gpu${gpu} ${tag}"
    return 0
  fi
  mkdir -p "${out}"
  echo "[launch] gpu${gpu} ${tag} n=${n} seed=${seed} loss=${loss} model=${model}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_true_scape_stage_l_cell.py" \
      --out "${out}" \
      --model-path "${model}" \
      --train-jsonl "${TRAIN}" \
      --valid-jsonl "${VALID}" \
      --test-jsonl "${TEST}" \
      --component-id evidence_graph_hybrid \
      --n-samples "${n}" \
      --seed "${seed}" \
      --loss-path "${loss}" \
      --gpu 0 \
      --epochs 1 \
      --batch-size 1 \
      >"${LOG_DIR}/c2_gpu${gpu}_${tag}.log" 2>&1
  touch "${out}/DONE"
}

run_gpu_queue() {
  local gpu="$1"
  local log="${LOG_DIR}/c2_gpu${gpu}_queue.log"
  {
    echo "[$(date -Iseconds)] gpu${gpu} C2 start main_loss=${LOSS_MAIN}"
    case "${gpu}" in
      0) run_cell 0 tool_route_s42_L512 512 42 "${LOSS_MAIN}" "${TOOL_BASE}"; run_cell 0 tool_route_s42_L2K 2000 42 "${LOSS_MAIN}" "${TOOL_BASE}" ;;
      1) run_cell 1 tool_route_s43_L512 512 43 "${LOSS_MAIN}" "${TOOL_BASE}"; run_cell 1 tool_route_s43_L2K 2000 43 "${LOSS_MAIN}" "${TOOL_BASE}" ;;
      2) run_cell 2 full_route_s42_L512 512 42 "${LOSS_MAIN}" "${FULL_BASE}"; run_cell 2 full_route_s42_L2K 2000 42 "${LOSS_MAIN}" "${FULL_BASE}" ;;
      3) run_cell 3 full_route_s43_L512 512 43 "${LOSS_MAIN}" "${FULL_BASE}"; run_cell 3 full_route_s43_L2K 2000 43 "${LOSS_MAIN}" "${FULL_BASE}" ;;
      4) run_cell 4 tool_name_only_s42_L2K 2000 42 tool_name_only_kl "${TOOL_BASE}" ;;
      5) run_cell 5 tool_uniform_s42_L2K 2000 42 tool_token_kl "${TOOL_BASE}" ;;
      6) run_cell 6 tool_action_ce_s42_L2K 2000 42 action_ce "${TOOL_BASE}" ;;
      7)
        CUDA_VISIBLE_DEVICES=7 "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_clean_v2v3_scorer.py" \
          --out "${OUT_ROOT}/evals/gpu7/c2_clean_notrain_v2v3" \
          --model-path "${TOOL_BASE}" --gpu 0 --n 128 --seed 8141 --tag c2_clean_notrain \
          >"${LOG_DIR}/c2_gpu7_clean_notrain.log" 2>&1 || true
        CUDA_VISIBLE_DEVICES=7 "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_clean_v2v3_scorer.py" \
          --out "${OUT_ROOT}/evals/gpu7/c2_harness1_v2v3" \
          --model-path "${HARNESS1}" --gpu 0 --n 128 --seed 8141 --tag c2_released_harness1 \
          >"${LOG_DIR}/c2_gpu7_harness1.log" 2>&1 || true
        CUDA_VISIBLE_DEVICES=7 "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/score_clean_micro_v2.py" \
          --out-dir "${OUT_ROOT}" \
          >"${LOG_DIR}/c2_gpu7_score.log" 2>&1 || true
        ;;
    esac
    echo "[$(date -Iseconds)] gpu${gpu} C2 ALL_DONE"
    mkdir -p "${MICRO}/gpu${gpu}"
    touch "${MICRO}/gpu${gpu}/ALL_DONE"
  } >>"${log}" 2>&1
}

if [[ -n "${GPU_ONLY:-}" ]]; then
  run_gpu_queue "${GPU_ONLY}"
else
  for g in 0 1 2 3 4 5 6 7; do
    run_gpu_queue "${g}" &
    echo $! >"${PID_DIR}/c2_gpu${g}.pid"
  done
  echo "[launch] C2 micro 8-GPU under ${MICRO} loss=${LOSS_MAIN}"
fi
