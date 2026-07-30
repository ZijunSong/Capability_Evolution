#!/usr/bin/env bash
# Phase B: 8-GPU forensic audit (sequential per GPU, parallel across GPUs)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round6/_common.sh"
scope6_setup

PB="${OUT}/phase_b"
LOG="${LOG_DIR}/phase_b"

run_gpu0() {
  scope6_log "GPU0: seed42 parity + replay"
  scope6_run_gpu 0 "${LOG}/gpu0.log" python training/scope_round6/run_parity_audit.py \
    --mode adapter_merged --scorer o7_42 --gpu cuda:0 --n-states 522
  scope6_run_gpu 0 "${LOG}/gpu0.log" python training/scope_round6/replay_closed_loop_states.py \
    --checkpoint o7_42 --state-source o7_42 --gpu cuda:0
  touch "${PB}/GPU0_DONE"
}

run_gpu1() {
  scope6_log "GPU1: seed43"
  scope6_run_gpu 1 "${LOG}/gpu1.log" python training/scope_round6/run_parity_audit.py \
    --mode adapter_merged --scorer o7_43 --gpu cuda:0 --n-states 522
  scope6_run_gpu 1 "${LOG}/gpu1.log" python training/scope_round6/replay_closed_loop_states.py \
    --checkpoint o7_43 --state-source o7_43 --gpu cuda:0
  touch "${PB}/GPU1_DONE"
}

run_gpu2() {
  scope6_log "GPU2: seed44"
  scope6_run_gpu 2 "${LOG}/gpu2.log" python training/scope_round6/run_parity_audit.py \
    --mode adapter_merged --scorer o7_44 --gpu cuda:0 --n-states 522
  scope6_run_gpu 2 "${LOG}/gpu2.log" python training/scope_round6/replay_closed_loop_states.py \
    --checkpoint o7_44 --state-source o7_44 --gpu cuda:0
  touch "${PB}/GPU2_DONE"
}

run_gpu3() {
  scope6_log "GPU3: seed42 cross-state"
  for src in base o7_43 o7_44; do
    scope6_run_gpu 3 "${LOG}/gpu3.log" python training/scope_round6/replay_closed_loop_states.py \
      --checkpoint o7_42 --state-source "${src}" --gpu cuda:0
  done
  touch "${PB}/GPU3_DONE"
}

run_gpu4() {
  scope6_log "GPU4: seed43 cross-state"
  for src in base o7_42 o7_44; do
    scope6_run_gpu 4 "${LOG}/gpu4.log" python training/scope_round6/replay_closed_loop_states.py \
      --checkpoint o7_43 --state-source "${src}" --gpu cuda:0
  done
  touch "${PB}/GPU4_DONE"
}

run_gpu5() {
  scope6_log "GPU5: seed44 cross-state"
  for src in base o7_42 o7_43; do
    scope6_run_gpu 5 "${LOG}/gpu5.log" python training/scope_round6/replay_closed_loop_states.py \
      --checkpoint o7_44 --state-source "${src}" --gpu cuda:0
  done
  touch "${PB}/GPU5_DONE"
}

run_gpu6() {
  scope6_log "GPU6: compact_json + semantics audit"
  scope6_run_gpu 6 "${LOG}/gpu6.log" python training/scope_round6/cross_score_matrix.py \
    --scorer o7_42 --state-source valid522 --gpu cuda:0
  scope6_run_gpu 6 "${LOG}/gpu6.log" python training/scope_round6/metric_semantics_audit.py
  touch "${PB}/GPU6_DONE"
}

run_gpu7() {
  scope6_log "GPU7: state shift + input audit"
  for seed in 42 43 44; do
    scope6_run_gpu 7 "${LOG}/gpu7.log" python training/scope_round6/analyze_state_shift.py \
      --scorer "o7_${seed}" --state-source "o7_${seed}" --gpu cuda:0
  done
  touch "${PB}/GPU7_DONE"
}

finalize_phase_b() {
  scope6_log "Building full cross-score matrix"
  python training/scope_round6/cross_score_matrix.py --gpu cuda:0
  python training/scope_round6/root_cause_gate.py
  python training/scope_round6/build_round6_report.py
  touch "${OUT}/PHASE_B_COMPLETE"
  scope6_set_stage "phase_b_done"
}

GPU_ID="${1:-all}"
case "${GPU_ID}" in
  0) run_gpu0 ;;
  1) run_gpu1 ;;
  2) run_gpu2 ;;
  3) run_gpu3 ;;
  4) run_gpu4 ;;
  5) run_gpu5 ;;
  6) run_gpu6 ;;
  7) run_gpu7 ;;
  all)
    pids=()
    for g in 0 1 2 3 4 5 6 7; do
      bash "$0" "${g}" &
      pids+=($!)
      sleep 5
    done
    for pid in "${pids[@]}"; do wait "${pid}" || true; done
    finalize_phase_b
    ;;
  *) echo "Unknown GPU ${GPU_ID}"; exit 1 ;;
esac
