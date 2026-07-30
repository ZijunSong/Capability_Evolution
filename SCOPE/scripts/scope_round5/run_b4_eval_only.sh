#!/usr/bin/env bash
# B4 offline eval + gate only (assumes training DONE files exist)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round5/_common.sh"
scope5_setup

OUT="${REPO_ROOT}/outputs/scope_round5/b4_full"
B4_LOG="${LOG_DIR}/b4"
mkdir -p "${OUT}/offline" "${B4_LOG}"

CUDA_VISIBLE_DEVICES=6 python training/scope_round5/run_b4_offline_eval.py \
  --adapter "${REPO_ROOT}/outputs/scope_round3/training/round3_op_main_seed42/adapter" \
  --variant round3_operation_ce --loss-mode operation_ce \
  --output "${OUT}/offline/round3_op_seed42.json" --gpu cuda:0 || true

for seed in 42 43 44; do
  CUDA_VISIBLE_DEVICES=6 python training/scope_round5/run_b4_offline_eval.py \
    --adapter "${OUT}/o7_r64_seed${seed}/adapter" \
    --variant "o7_r64_seed${seed}" --loss-mode discriminative_ce \
    --output "${OUT}/offline/o7_r64_seed${seed}.json" --gpu cuda:0
  CUDA_VISIBLE_DEVICES=7 python training/scope_round5/run_b4_offline_eval.py \
    --adapter "${OUT}/compact_json_seed${seed}/adapter" \
    --variant "compact_json_seed${seed}" --loss-mode sample_normalized_action_ce \
    --compact-target \
    --output "${OUT}/offline/compact_json_seed${seed}.json" --gpu cuda:0
done

python training/scope_round5/run_b4_gate.py | tee "${B4_LOG}/gate.log"
echo "B4 eval+gate complete $(date -Is)" > "${OUT}/B4_COMPLETE"
