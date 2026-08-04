#!/usr/bin/env bash
# Resume failed Round 7 queue stages
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

GPU="${1:-}"
if [[ -z "${GPU}" ]]; then
  echo "Usage: $0 <gpu_id>"
  exit 1
fi

scope7_log "Resuming GPU${GPU} queue"
CUDA_VISIBLE_DEVICES="${GPU}" bash "${REPO_ROOT}/scripts/scope_round7/run_gpu${GPU}_queue.sh" \
  >> "${LOG_DIR}/gpu${GPU}_resume.log" 2>&1
