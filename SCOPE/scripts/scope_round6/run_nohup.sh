#!/usr/bin/env bash
# Launch Round 6 master pipeline under nohup
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round6/_common.sh"
scope6_setup

MASTER_LOG="${LOG_DIR}/round6_master.log"
PID_FILE="${LOG_DIR}/round6_master.pid"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Round 6 already running pid=$(cat "${PID_FILE}")"
  exit 0
fi

nohup bash "${REPO_ROOT}/scripts/scope_round6/run_round6_pipeline.sh" \
  >> "${MASTER_LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "Started Round 6 pid=$(cat "${PID_FILE}") log=${MASTER_LOG}"
