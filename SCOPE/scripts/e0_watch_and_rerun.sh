#!/usr/bin/env bash
# Wait for current E0 orchestrator to finish, then relaunch nohup (resume incomplete jobs).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_OUT="${ROOT_OUT:-$REPO_ROOT/outputs/scope_e0_distillability}"
PID_FILE="${ROOT_OUT}/nohup_master.pid"
LOG="${ROOT_OUT}/nohup_master.log"

log() { echo "[$(date '+%F %T')] [e0-watch] $*"; }

wait_for_orchestrator() {
  while true; do
    if [[ -f "${PID_FILE}" ]]; then
      pid="$(cat "${PID_FILE}")"
      if ps -p "${pid}" >/dev/null 2>&1; then
        sleep 60
        continue
      fi
    fi
    # Also check flock lock holder
    if lsof "${ROOT_OUT}/.e0_orchestrator.lock" >/dev/null 2>&1; then
      sleep 60
      continue
    fi
    break
  done
}

has_incomplete() {
  bash "${REPO_ROOT}/scripts/e0_status.sh" | grep -E '✗|~' | grep -q .
}

log "Watching orchestrator (pid_file=${PID_FILE})"
while has_incomplete; do
  wait_for_orchestrator
  log "Orchestrator idle"
  if ! has_incomplete; then
    break
  fi
  log "Incomplete jobs remain — relaunching nohup"
  cd "${REPO_ROOT}"
  nohup bash scripts/run_e0_distillability_nohup.sh >> "${LOG}" 2>&1 &
  echo $! > "${PID_FILE}"
  log "Relaunched pid=$(cat "${PID_FILE}")"
  sleep 30
done
log "All jobs complete"
if [[ "${E0_CLEANUP_VLLM:-1}" == "1" ]]; then
  log "Stopping E0 vLLM"
  bash "${REPO_ROOT}/scripts/stop_e0_vllm.sh" || log "WARN stop_e0_vllm failed"
fi
