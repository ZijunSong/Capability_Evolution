#!/usr/bin/env bash
# Round 5 B4 — full 1807/522 objective screen (nohup-safe)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round5/_common.sh"
scope5_setup

OUT="${REPO_ROOT}/outputs/scope_round5/b4_full"
B4_LOG="${LOG_DIR}/b4"
mkdir -p "${OUT}" "${B4_LOG}" "${OUT}/offline"

declare -a TRAIN_JOBS=()

launch_train() {
  local gpu="$1" variant="$2" seed="$3"
  local tag="${variant}_seed${seed}"
  local job="train_${tag}"
  local logfile="${B4_LOG}/train_${tag}.log"

  if [[ -f "${OUT}/${tag}/DONE" ]]; then
    scope5_mark_done "b4" "${job}"
    echo "[skip] ${tag} DONE file exists"
    return
  fi

  # Detect training already running (from any session)
  if pgrep -f "run_b4_train.py --variant ${variant} --seed ${seed}" >/dev/null 2>&1; then
    echo "[skip] ${tag} already running (pgrep)"
    TRAIN_JOBS+=("${job}")
    return
  fi

  local d pidfile
  d="$(scope5_job_dir b4 "${job}")"
  pidfile="${d}/pid"
  if [[ -f "${pidfile}" ]] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    echo "[skip] ${tag} already running pid=$(cat "${pidfile}")"
    TRAIN_JOBS+=("${job}")
    return
  fi

  scope5_launch_nohup b4 "${job}" "${logfile}" \
    "cd '${REPO_ROOT}' && source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && export PYTHONPATH='${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=${gpu} python training/scope_round5/run_b4_train.py --variant ${variant} --seed ${seed} --gpu cuda:0"
  TRAIN_JOBS+=("${job}")
}

echo "=== B4 train launch $(date -Is) ==="
launch_train 0 o7_r64 42
launch_train 1 o7_r64 43
launch_train 2 o7_r64 44
launch_train 3 compact_json 42
launch_train 4 compact_json 43
launch_train 5 compact_json 44

echo "=== B4 waiting for training jobs ==="
scope5_wait_pids b4 "${TRAIN_JOBS[@]}"

# Also wait for DONE files (in case jobs were started outside this script)
for seed in 42 43 44; do
  for variant in o7_r64 compact_json; do
    tag="${variant}_seed${seed}"
    if [[ ! -f "${OUT}/${tag}/DONE" ]]; then
      echo "[warn] ${tag} missing DONE after wait"
    fi
  done
done

echo "=== B4 offline eval $(date -Is) ==="
scope5_launch_nohup b4 eval_baseline "${B4_LOG}/eval_baseline.log" \
  "cd '${REPO_ROOT}' && source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && export PYTHONPATH='${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=6 python training/scope_round5/run_b4_offline_eval.py --adapter '${REPO_ROOT}/outputs/scope_round3/training/round3_op_main_seed42/adapter' --variant round3_operation_ce --loss-mode operation_ce --output '${OUT}/offline/round3_op_seed42.json' --gpu cuda:0"

for seed in 42 43 44; do
  scope5_launch_nohup b4 "eval_o7_${seed}" "${B4_LOG}/eval_o7_${seed}.log" \
    "cd '${REPO_ROOT}' && source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && export PYTHONPATH='${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=6 python training/scope_round5/run_b4_offline_eval.py --adapter '${OUT}/o7_r64_seed${seed}/adapter' --variant o7_r64_seed${seed} --loss-mode discriminative_ce --output '${OUT}/offline/o7_r64_seed${seed}.json' --gpu cuda:0"
  scope5_launch_nohup b4 "eval_compact_${seed}" "${B4_LOG}/eval_compact_${seed}.log" \
    "cd '${REPO_ROOT}' && source '${CONDA_BASE}/etc/profile.d/conda.sh' && conda activate '${ENV_NAME}' && export PYTHONPATH='${REPO_ROOT}' && CUDA_VISIBLE_DEVICES=7 python training/scope_round5/run_b4_offline_eval.py --adapter '${OUT}/compact_json_seed${seed}/adapter' --variant compact_json_seed${seed} --loss-mode sample_normalized_action_ce --compact-target --output '${OUT}/offline/compact_json_seed${seed}.json' --gpu cuda:0"
done

scope5_wait_pids b4 eval_baseline eval_o7_42 eval_o7_43 eval_o7_44 eval_compact_42 eval_compact_43 eval_compact_44

python training/scope_round5/run_b4_gate.py | tee "${B4_LOG}/gate.log"
echo "B4 complete $(date -Is)" | tee "${OUT}/B4_COMPLETE"
