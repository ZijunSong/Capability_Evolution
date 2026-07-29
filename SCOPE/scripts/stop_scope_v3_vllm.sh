#!/usr/bin/env bash
# Stop vLLM started for SCOPE v3 protocol smoke / audit (port 8774, scope-v3-smoke).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_PORT="${VLLM_PORT:-8774}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-scope-v3-smoke}"
PID_CANDIDATES=(
  "${PID_FILE:-}"
  "$REPO_ROOT/outputs/scope_v3_protocol_smoke20/vllm_server.pid"
  "$REPO_ROOT/outputs/scope_v3_audit_100q/natural_100q/vllm_server.pid"
)

_stop_pid() {
  local pid="$1"
  [[ -z "${pid}" ]] && return 0
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  echo "[stop_scope_v3_vllm] Stopping pid=${pid} ..."
  kill "${pid}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "${pid}" 2>/dev/null || return 0
    sleep 2
  done
  kill -9 "${pid}" 2>/dev/null || true
}

_stopped=0

for pf in "${PID_CANDIDATES[@]}"; do
  [[ -z "${pf}" || ! -f "${pf}" ]] && continue
  vpid="$(cat "${pf}" || true)"
  if [[ -n "${vpid}" ]]; then
    _stop_pid "${vpid}"
    _stopped=1
  fi
  rm -f "${pf}"
done

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
  echo "[stop_scope_v3_vllm] vLLM stopped (port=${VLLM_PORT}, model=${SERVED_MODEL_NAME})"
else
  echo "[stop_scope_v3_vllm] No scope-v3 vLLM process found (port=${VLLM_PORT})"
fi
