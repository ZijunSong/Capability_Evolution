#!/usr/bin/env bash
# Round 5 B3 — 8 GPU micro-overfit tournament (nohup per GPU)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round5/_common.sh"
scope5_setup

declare -a OBJECTIVES=(O0 O1 O2 O3 O4 O5 O6 O7)
declare -a JOBS=()

for i in "${!OBJECTIVES[@]}"; do
  obj="${OBJECTIVES[$i]}"
  job="gpu${i}_${obj}"
  out_dir="${OUT}/micro_overfit/${obj}"
  logfile="${LOG_DIR}/b3_${job}.log"

  if [[ -f "${out_dir}/summary.json" ]]; then
    scope5_mark_done "b3" "${job}"
    echo "[skip] ${obj} summary exists"
    continue
  fi

  scope5_launch_nohup b3 "${job}" "${logfile}" \
    "cd '${REPO_ROOT}' && source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && export PYTHONPATH='${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${i} python training/scope_round5/run_micro_overfit.py --objective ${obj} --gpu cuda:0"
  JOBS+=("${job}")
  sleep 2
done

scope5_wait_pids b3 "${JOBS[@]}"
python training/scope_round5/build_micro_matrix.py
echo "B3 complete $(date -Is)"
