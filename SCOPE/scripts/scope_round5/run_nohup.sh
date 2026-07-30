#!/usr/bin/env bash
# Round 5 主入口 — 启动 pipeline supervisor（nohup，各阶段自动衔接）
#
# 用法:
#   bash scripts/scope_round5/run_nohup.sh
#   bash scripts/scope_round5/run_nohup.sh --fresh   # 若旧 supervisor 已退出则强制重启
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round5/_common.sh"
scope5_setup

if [[ -f "${SUPERVISOR_PID}" ]]; then
  oldpid="$(cat "${SUPERVISOR_PID}")"
  if kill -0 "${oldpid}" 2>/dev/null && [[ "${1:-}" != "--fresh" ]]; then
    echo "Pipeline supervisor already running pid=${oldpid}"
    echo "  stage: $(scope5_get_stage)"
    echo "  log:   ${SUPERVISOR_LOG}"
    echo "  status: bash scripts/scope_round5/status.sh"
    exit 0
  fi
fi

# 停止旧的 wait-eval / master（若存在）
for pf in "${LOG_DIR}/b4_wait_eval.pid" "${LOG_DIR}/round5_master.pid"; do
  if [[ -f "${pf}" ]]; then
    opid="$(cat "${pf}")"
    kill "${opid}" 2>/dev/null || true
    rm -f "${pf}"
  fi
done

nohup bash "${REPO_ROOT}/scripts/scope_round5/pipeline_supervisor.sh" >> "${SUPERVISOR_LOG}" 2>&1 &
echo $! > "${SUPERVISOR_PID}"

echo "Pipeline supervisor launched (auto-chaining stages)"
echo "  pid:   $(cat "${SUPERVISOR_PID}")"
echo "  stage: $(scope5_get_stage)"
echo "  log:   ${SUPERVISOR_LOG}"
echo "  tail:  tail -f ${SUPERVISOR_LOG}"
