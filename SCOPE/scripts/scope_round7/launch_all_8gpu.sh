#!/usr/bin/env bash
# Launch all 8 GPU Round 7 queues with staggered harness startup
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

scope7_log "Round 7 launch preflight"

# Check 8 GPUs visible
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "${NGPU}" -lt 8 ]]; then
  scope7_log "ERROR: need 8 GPUs, found ${NGPU}"
  exit 1
fi

# Check ports 9200-9207
for P in 9200 9201 9202 9203 9204 9205 9206 9207; do
  if ss -lntp 2>/dev/null | grep -q ":${P} "; then
    scope7_log "WARN: port ${P} in use"
  fi
done

# Check frozen assets
for P in \
  "${BASE_MODEL}" \
  "${MANIFEST}" \
  "${VALID522}" \
  "${R5}/merged/o7_r64_seed42" \
  "${R5}/merged/o7_r64_seed43" \
  "${R5}/merged/o7_r64_seed44" \
  "${REPO_ROOT}/harness/configs/modules_minimal_v2.yaml"; do
  if [[ ! -e "${P}" ]]; then
    scope7_log "ERROR: missing asset ${P}"
    exit 1
  fi
done

# Environment snapshot
python training/scope_round7/preflight_snapshot.py >> "${LOG_DIR}/preflight.log" 2>&1

scope7_log "Launching GPU queues (PARALLEL=${PARALLEL})"

# GPU4-7 start immediately
for G in 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${G}" nohup bash "${REPO_ROOT}/scripts/scope_round7/run_gpu${G}_queue.sh" \
    > "${LOG_DIR}/gpu${G}.log" 2>&1 &
  echo $! > "${PID_DIR}/gpu${G}.pid"
  scope7_log "Started GPU${G} pid=$(cat "${PID_DIR}/gpu${G}.pid")"
done

# GPU0-3 staggered harness (75s apart)
for G in 0 1 2 3; do
  DELAY=$((G * 75))
  (
    sleep "${DELAY}"
    CUDA_VISIBLE_DEVICES="${G}" bash "${REPO_ROOT}/scripts/scope_round7/run_gpu${G}_queue.sh" \
      >> "${LOG_DIR}/gpu${G}.log" 2>&1
  ) &
  echo $! > "${PID_DIR}/gpu${G}.pid"
  scope7_log "Scheduled GPU${G} pid=$(cat "${PID_DIR}/gpu${G}.pid") delay=${DELAY}s"
done

scope7_log "All queues launched. Monitor: bash scripts/scope_round7/status.sh"
