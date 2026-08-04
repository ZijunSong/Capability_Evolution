#!/usr/bin/env bash
# Resume remaining Phase 1: O7 dup retention + rollback collection (nohup)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

scope8_log "Phase 1 RESUME launch (O7 seeds + rollback collection)"

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "${NGPU}" -lt 8 ]]; then
  scope8_log "ERROR: need 8 GPUs, found ${NGPU}"
  exit 1
fi

for P in 9300 9301 9302 9303 9304 9305 9306 9307; do
  if ss -lntp 2>/dev/null | grep -q ":${P} "; then
    scope8_log "WARN: port ${P} in use — may conflict"
  fi
done

for G in 0 1 2 3 4 5 6 7; do
  DELAY=$((G * 60))
  (
    if [[ "${DELAY}" -gt 0 ]]; then sleep "${DELAY}"; fi
    CUDA_VISIBLE_DEVICES="${G}" bash "${REPO_ROOT}/scripts/scope_round8/run_gpu${G}_queue.sh"
  ) >> "${LOG_DIR}/gpu${G}_resume.log" 2>&1 &
  echo $! > "${PID_DIR}/gpu${G}.pid"
  scope8_log "Resumed GPU${G} pid=$(cat "${PID_DIR}/gpu${G}.pid") delay=${DELAY}s"
done

scope8_log "Phase 1 resume launched. Monitor: bash scripts/scope_round8/status.sh"
