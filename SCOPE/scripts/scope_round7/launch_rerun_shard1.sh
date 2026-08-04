#!/usr/bin/env bash
# Launch shard1 rerun (parity fix) on GPU0-3 with harness stagger
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

export SCOPE7_NO_RESUME=1
export PARALLEL="${PARALLEL:-64}"

scope7_log "=== Round7 shard1 RERUN launch (parity fix) PARALLEL=${PARALLEL} ==="

for P in 9200 9201 9202 9203; do
  if ss -lntp 2>/dev/null | grep -q ":${P} "; then
    scope7_log "WARN: port ${P} in use"
  fi
done

mkdir -p "${OUT}/contract_trace/live_rerun" "${OUT}/holdout_tau0_rerun"

# GPU0-3 staggered harness start (75s)
VARIANTS=(base seed42 seed43 seed44)
for i in 0 1 2 3; do
  DELAY=$((i * 75))
  G="${i}"
  V="${VARIANTS[$i]}"
  (
    sleep "${DELAY}"
    bash "${REPO_ROOT}/scripts/scope_round7/run_rerun_shard1_worker.sh" "${G}" "${V}" \
      >> "${LOG_DIR}/rerun_gpu${G}.log" 2>&1
  ) &
  echo $! > "${PID_DIR}/rerun_gpu${G}.pid"
  scope7_log "Scheduled rerun GPU${G} (${V}) pid=$(cat "${PID_DIR}/rerun_gpu${G}.pid") delay=${DELAY}s"
done

scope7_log "Rerun launched. Monitor: tail -f ${LOG_DIR}/rerun_gpu*.log"
