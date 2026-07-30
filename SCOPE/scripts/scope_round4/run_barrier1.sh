#!/usr/bin/env bash
# Round 4 Barrier 1: metric audit (CPU) — nohup safe
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/outputs/scope_round4/logs"
mkdir -p "${LOG_DIR}"

echo "[barrier1] $(date -Is) starting unit tests + forced episode + offline eval"

# 1.3 Forced episode (CPU, fast)
python training/scope_round4/run_forced_episode.py \
  > "${LOG_DIR}/barrier1_forced_episode.log" 2>&1
echo "[barrier1] forced episode done"

# 1.1 Offline metric audit (GPU sequential — one model at a time)
nohup python training/scope_round4/run_offline_metric_audit.py \
  > "${LOG_DIR}/barrier1_offline_eval.log" 2>&1 &
echo $! > "${LOG_DIR}/barrier1_offline_eval.pid"
echo "[barrier1] offline eval launched PID=$(cat "${LOG_DIR}/barrier1_offline_eval.pid")"
echo "[barrier1] tail -f ${LOG_DIR}/barrier1_offline_eval.log"
