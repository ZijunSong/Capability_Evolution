#!/usr/bin/env bash
# Stop vLLM started for E0 distillability experiments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_PORT="${VLLM_PORT:-8776}"
PID_FILE="${PID_FILE:-$REPO_ROOT/outputs/scope_e0_distillability/vllm_server.pid}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-e0-harness-policy}"

_stop_pid() {
  local pid="$1"
  [[ -z "${pid}" ]] && return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  echo "[stop_e0_vllm] Stopping pid=${pid} ..."
  kill "${pid}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 2
  done
  echo "[stop_e0_vllm] Force kill pid=${pid}"
  kill -9 "${pid}" 2>/dev/null || true
}

_stopped=0

if [[ -f "${PID_FILE}" ]]; then
  vpid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${vpid}" ]]; then
    _stop_pid "${vpid}"
    _stopped=1
  fi
  rm -f "${PID_FILE}"
fi

# Fallback: parent vllm serve on this port / served name (e.g. stale pid file).
while IFS= read -r pid; do
  [[ -z "${pid}" ]] && continue
  _stop_pid "${pid}"
  _stopped=1
done < <(pgrep -f "vllm serve.*--port ${VLLM_PORT}" 2>/dev/null || true)

while IFS= read -r pid; do
  [[ -z "${pid}" ]] && continue
  _stop_pid "${pid}"
  _stopped=1
done < <(pgrep -f "vllm serve.*${SERVED_MODEL_NAME}" 2>/dev/null || true)

if [[ "${_stopped}" == "1" ]]; then
  echo "[stop_e0_vllm] vLLM stopped (port=${VLLM_PORT}, model=${SERVED_MODEL_NAME})"
else
  echo "[stop_e0_vllm] No E0 vLLM process found (port=${VLLM_PORT})"
fi
