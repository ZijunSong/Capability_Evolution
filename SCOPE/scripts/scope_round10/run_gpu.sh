#!/usr/bin/env bash
# Train + eval + HF/vLLM parity for one Round 10 variant on one GPU.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope10_setup

GPU="${1:?gpu}"
VARIANT="${TRAINING_VARIANTS[$GPU]}"
PORT="$(scope10_port_for_gpu "${GPU}")"
VDIR="${OUT}/training/${VARIANT}"
MARKER="${VDIR}/DONE"

if [[ -f "${MARKER}" ]]; then
  scope10_log "Skip ${VARIANT}"
  exit 0
fi

mkdir -p "${VDIR}/eval_offline_valid" "${VDIR}/eval_live_valid" "${VDIR}/eval_live_test"

if [[ "${VARIANT}" == "rollback_calibration_only" ]]; then
  scope10_log "${VARIANT}: calibration-only baseline (no train)"
  CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round10/calibrate_binary.py --gpu cuda:0 \
    >> "${LOG_DIR}/${VARIANT}.log" 2>&1
  touch "${MARKER}"
  scope10_log "${VARIANT} DONE"
  exit 0
fi

# Dataset SHA gate
DATASET_NAME=$(python -c "
import json
from pathlib import Path
v = '${VARIANT}'
m = {
  'rollback_live_aligned_seed42': 'D2_mixed_aligned',
  'rollback_live_aligned_seed43': 'D2_mixed_aligned',
  'rollback_live_aligned_seed44': 'D2_mixed_aligned',
  'rollback_live_only_seed42': 'D1_live_only',
  'rollback_offline_only_binary_seed42': 'D0_offline_only',
  'rollback_hard_continue_seed42': 'D3_mixed_hard_continue',
  'rollback_source_token_seed42': 'D4_source_token',
}
print(m[v])
")
GATE="${DATA_DIR}/binary_datasets/${DATASET_NAME}/DATASET_GATE.json"
python -c "import json,sys; g=json.load(open('${GATE}')); sys.exit(0 if g.get('gate_pass') else 2)"

scope10_log "${VARIANT} train"
CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round10/run_training.py \
  --variant "${VARIANT}" --gpu cuda:0 \
  >> "${LOG_DIR}/${VARIANT}_train.log" 2>&1

MODEL="$(scope10_merged_model "${VARIANT}")"
if [[ ! -f "${MODEL}/config.json" ]]; then
  scope10_log "ERROR: missing merged model ${MODEL}"
  exit 1
fi

run_eval_split() {
  local split="$1" inp="$2"
  local hf_out="${VDIR}/eval_${split}/hf_replay.jsonl"
  local vllm_out="${VDIR}/eval_${split}/vllm_replay.jsonl"
  local n_expected
  n_expected=$(wc -l < "${inp}" | tr -d ' ')
  if [[ ! -f "${hf_out}" ]] || [[ "$(wc -l < "${hf_out}" | tr -d ' ')" -lt "${n_expected}" ]]; then
    rm -f "${hf_out}"
    scope10_log "${VARIANT} HF eval ${split}"
    CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/replay_frozen_hf.py \
      --model-path "${MODEL}" --input "${inp}" --output "${hf_out}" --device cuda:0 \
      >> "${LOG_DIR}/${VARIANT}_${split}_hf.log" 2>&1
  fi
  if [[ ! -f "${vllm_out}" ]] || [[ "$(wc -l < "${vllm_out}" | tr -d ' ')" -lt "${n_expected}" ]]; then
    rm -f "${vllm_out}"
    scope10_log "${VARIANT} vLLM eval ${split}"
    CUDA_VISIBLE_DEVICES="${GPU}" python training/scope_round9/run_vllm_replay_split.py \
      --model-path "${MODEL}" --input "${inp}" --output "${vllm_out}" --port "${PORT}" --gpu "${GPU}" \
      >> "${LOG_DIR}/${VARIANT}_${split}_vllm.log" 2>&1
  fi
}

run_eval_split offline_valid "${OFFLINE_VALID}"
run_eval_split live_valid "${LIVE_VALID}"
run_eval_split live_test "${LIVE_TEST}"

python - <<PY
import json
from pathlib import Path
from training.scope_round9.aggregate_wave_b_report import split_report

vdir = Path("${VDIR}")
variant = "${VARIANT}"
report = {
    "variant": variant,
    "offline_valid": split_report(vdir, "offline_valid"),
    "live_valid": split_report(vdir, "live_valid"),
    "live_test": split_report(vdir, "live_test"),
}
out = vdir / "TRAIN_AND_EVAL_REPORT.json"
out.write_text(json.dumps(report, indent=2) + "\n")
PY

scope10_stop_recorded "vllm_port_${PORT}" || true
touch "${MARKER}"
scope10_log "${VARIANT} DONE"
