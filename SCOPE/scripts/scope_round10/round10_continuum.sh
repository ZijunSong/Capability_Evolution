#!/usr/bin/env bash
# Round 10 full pipeline: Barrier 0 → 5 → optional Wave C
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope10_setup

run_barrier0() {
  scope10_log "Barrier 0: Round 9 closure"
  python training/scope_round10/round9_closure.py \
    2>&1 | tee "${LOG_DIR}/barrier0.log"
  touch "${MARKER_DIR}/barrier0.DONE"
}

run_barrier1() {
  scope10_log "Barrier 1: schema audit"
  python training/scope_round10/schema_audit.py \
    2>&1 | tee "${LOG_DIR}/barrier1.log"
  touch "${MARKER_DIR}/barrier1.DONE"
}

run_barrier2() {
  scope10_log "Barrier 2: live split"
  python training/scope_round10/live_split.py \
    2>&1 | tee "${LOG_DIR}/barrier2_split.log"
  scope10_log "Barrier 2: prior shift (8 GPU parallel)"
  bash "$(dirname "$0")/launch_prior_shift_8gpu.sh" \
    2>&1 | tee "${LOG_DIR}/barrier2_prior.log"
  touch "${MARKER_DIR}/barrier2.DONE"
}

run_barrier3() {
  scope10_log "Barrier 3: binary calibration (3 GPU parallel)"
  bash "$(dirname "$0")/launch_barrier3_3gpu.sh" \
    2>&1 | tee "${LOG_DIR}/barrier3.log"
  touch "${MARKER_DIR}/barrier3.DONE"
}

run_barrier4() {
  scope10_log "Barrier 4: build datasets"
  python training/scope_round10/build_datasets.py \
    2>&1 | tee "${LOG_DIR}/barrier4.log"
  touch "${MARKER_DIR}/barrier4.DONE"
}

run_barrier5_micro() {
  scope10_log "Barrier 5.1: micro-overfit (8 GPU parallel)"
  bash "$(dirname "$0")/launch_micro_overfit_8gpu.sh" \
    2>&1 | tee "${LOG_DIR}/barrier5_micro.log"
  touch "${MARKER_DIR}/barrier5_micro.DONE"
}

run_barrier5_train() {
  scope10_log "Barrier 5.2: 8-GPU training"
  bash "$(dirname "$0")/launch_training_8gpu.sh" \
    2>&1 | tee "${LOG_DIR}/barrier5_train.log"
  touch "${MARKER_DIR}/barrier5_train.DONE"
}

run_wave_c_if_gate() {
  local gate="${OUT}/ROUND10_OFFLINE_GATE.json"
  if [[ ! -f "${gate}" ]]; then
    scope10_log "No offline gate file"
    return 2
  fi
  local pass
  pass=$(python -c "import json; print(json.load(open('${gate}')).get('offline_gate_pass', False))")
  if [[ "${pass}" != "True" ]]; then
    scope10_log "Offline Gate FAIL — skip closed-loop"
    return 2
  fi
  scope10_log "Offline Gate PASS — launching smoke20"
  bash "${REPO_ROOT}/scripts/scope_round9/launch_wave_c_smoke20_8gpu.sh" \
    2>&1 | tee "${LOG_DIR}/wave_c_smoke.log" || true
}

write_final() {
  python training/scope_round10/write_final_report.py
  scope10_log "Final report written"
}

main() {
  [[ -f "${MARKER_DIR}/barrier0.DONE" ]] || run_barrier0
  [[ -f "${MARKER_DIR}/barrier1.DONE" ]] || run_barrier1
  [[ -f "${MARKER_DIR}/barrier2.DONE" ]] || run_barrier2
  [[ -f "${MARKER_DIR}/barrier3.DONE" ]] || run_barrier3
  [[ -f "${MARKER_DIR}/barrier4.DONE" ]] || run_barrier4
  [[ -f "${MARKER_DIR}/barrier5_micro.DONE" ]] || run_barrier5_micro
  [[ -f "${MARKER_DIR}/barrier5_train.DONE" ]] || run_barrier5_train
  run_wave_c_if_gate || true
  write_final
  scope10_log "Round 10 continuum complete"
}

main "$@"
