#!/usr/bin/env bash
# Wave0: parallel Dup retirement calibration across GPUs 0-4
set -euo pipefail
source "$(dirname "$0")/_common.sh"
r14_setup

OUT_GPU="${OUT}/gpu0_dup_anchor"
PARALLEL="${R14_PARALLEL:-16}"
MANIFEST="${R14_FRESH100}"
LOG="${LOG_DIR}/wave0_dup_calibrate.log"

r14_log "wave0 dup calibrate start parallel=${PARALLEL}"

run_cond() {
  local gpu="$1" cond="$2" seed="${3:-42}"
  local out="${OUT_GPU}/${cond}"
  if [[ "${cond}" == "T_OFF" ]]; then
    out="${OUT_GPU}/T_OFF_seed${seed}"
  fi
  mkdir -p "${out}"
  r14_log "wave0 GPU${gpu} ${cond} seed=${seed} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round14/run_module_retirement_eval.py \
    --capability duplicate_evidence \
    --manifest "${MANIFEST}" \
    --output-dir "${out}" \
    --gpu "${gpu}" \
    --seed "${seed}" \
    --conditions "${cond%%_seed*}" \
    --temperature 0.0 \
    --parallel "${PARALLEL}" \
    --flat-output \
    --resume \
    --run-closed-loop \
    >> "${LOG}" 2>&1 &
  echo $! > "${PID_DIR}/wave0_gpu${gpu}.pid"
}

run_cond 0 B_OFF 42
run_cond 1 B_ON 42
run_cond 2 T_OFF 42
run_cond 3 T_OFF 43
run_cond 4 T_OFF 44

# Heartbeat while wave0 jobs run (guardian-safe)
(
  while true; do
    r14_touch_hb "${OUT_GPU}/HEARTBEAT"
    sleep 60
    # exit when all child pids gone
    alive=0
    for g in 0 1 2 3 4; do
      if [[ -f "${PID_DIR}/wave0_gpu${g}.pid" ]] && kill -0 "$(cat "${PID_DIR}/wave0_gpu${g}.pid")" 2>/dev/null; then
        alive=1
      fi
    done
    [[ "${alive}" -eq 0 ]] && break
  done
) &
HB_PID=$!

r14_log "wave0 waiting for 5 jobs..."
wait || true
kill "${HB_PID}" 2>/dev/null || true

python training/scope_round14/aggregate_dup_anchor.py \
  --anchor-dir "${OUT_GPU}" \
  --resume \
  >> "${LOG}" 2>&1

r14_touch_hb "${OUT_GPU}/HEARTBEAT"
r14_log "wave0 dup calibrate complete"
