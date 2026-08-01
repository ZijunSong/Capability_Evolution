#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round6/_common.sh"
scope6_setup
LOG="${LOG_DIR}/seed43_shard2_resume.log"
PID_FILE="${LOG_DIR}/seed43_shard2_resume.pid"
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Already running pid=$(cat "${PID_FILE}")"
  exit 0
fi
nohup bash "${REPO_ROOT}/scripts/scope_round6/resume_seed43_shard2.sh" >> "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "Started seed43/shard2 resume pid=$(cat "${PID_FILE}") log=${LOG}"
