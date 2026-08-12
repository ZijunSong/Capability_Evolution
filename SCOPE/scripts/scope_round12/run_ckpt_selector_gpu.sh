#!/usr/bin/env bash
# GPU job: oracle-operation Stage2 ranking for C11L / C11P on canonical base_live.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r12_setup

GPU="${1:?gpu}"
SEL="${2:?C11L|C11P}"
PORT="$(r12_port_for_gpu "${GPU}")"
MODEL="$(r12_model_for_job "${SEL}")"
VDIR="${OUT}/phase_a_ckpt_provenance/per_selector_scores"
OUTP="${VDIR}/${SEL}_oracle_replay.jsonl"
MARKER="${VDIR}/${SEL}_DONE"

if [[ -f "${MARKER}" ]]; then
  r12_log "Skip ${SEL} oracle replay (DONE)"
  exit 0
fi
mkdir -p "${VDIR}"
# Use A0 factorized eval (has stage2_text + candidates); filter happens in scoring
INP="${R11_DATA}/factorized_eval/A0/base_live.jsonl"
if [[ ! -f "${INP}" ]]; then
  python training/scope_round11/build_factorized_eval.py --view A0 \
    >> "${LOG_DIR}/build_factorized_eval_A0.log" 2>&1
fi

heartbeat() { date -Is > "${VDIR}/${SEL}_HEARTBEAT"; }
heartbeat
HB_PID=""
start_hb() { ( while true; do heartbeat; sleep 60; done ) & HB_PID=$!; }
stop_hb() {
  if [[ -n "${HB_PID}" ]] && kill -0 "${HB_PID}" 2>/dev/null; then
    kill "${HB_PID}" 2>/dev/null || true
  fi
}
trap 'stop_hb; r12_stop_recorded "vllm_port_${PORT}" || true' EXIT

n_expected=$(wc -l < "${INP}" | tr -d ' ')
if [[ -f "${OUTP}" ]] && [[ "$(wc -l < "${OUTP}" | tr -d ' ')" -ge "${n_expected}" ]]; then
  r12_log "${SEL} oracle replay already complete"
  touch "${MARKER}"
  exit 0
fi

rm -f "${OUTP}"
r12_log "${SEL} oracle_op + stage2 ranker on GPU${GPU}"
start_hb
SCOPE_VLLM_OUT_ROOT="${OUT}" \
  python training/scope_round11/run_vllm_factorized_split.py \
  --model-path "${MODEL}" --input "${INP}" --output "${OUTP}" \
  --port "${PORT}" --gpu "${GPU}" \
  --use-stage2-ranker --operation-from-oracle \
  >> "${LOG_DIR}/ckpt_${SEL}_oracle.log" 2>&1
stop_hb
touch "${MARKER}"
r12_log "${SEL} DONE"
