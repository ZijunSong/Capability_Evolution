#!/usr/bin/env bash
# Launch Round 8 Phase 1 on 8 GPUs (nohup)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

scope8_log "Phase 1 launch"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "${NGPU}" -lt 8 ]]; then
  scope8_log "ERROR: need 8 GPUs, found ${NGPU}"
  exit 1
fi

bash "${REPO_ROOT}/scripts/scope_round8/preflight.sh"

for P in 9300 9301 9302 9303 9304 9305 9306 9307; do
  if ss -lntp 2>/dev/null | grep -q ":${P} "; then
    scope8_log "WARN: port ${P} in use"
  fi
done

for G in 0 1 2 3 4 5 6 7; do
  DELAY=$((G * 60))
  (
    if [[ "${DELAY}" -gt 0 ]]; then sleep "${DELAY}"; fi
    CUDA_VISIBLE_DEVICES="${G}" bash "${REPO_ROOT}/scripts/scope_round8/run_gpu${G}_queue.sh"
  ) >> "${LOG_DIR}/gpu${G}.log" 2>&1 &
  echo $! > "${PID_DIR}/gpu${G}.pid"
  scope8_log "Started GPU${G} pid=$(cat "${PID_DIR}/gpu${G}.pid") delay=${DELAY}s"
done

scope8_log "Phase 1 launched. Monitor: bash scripts/scope_round8/status.sh"
