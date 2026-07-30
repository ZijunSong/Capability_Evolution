#!/usr/bin/env bash
# Round 4 Barrier 3: postfix offline eval (8 GPU) + 50q closed-loop (max 4 vLLM)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/ppnm/miniconda3}"
ENV_NAME="${BISHOP_CONDA_ENV:-bishop}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

BASE_MODEL="${BASE_MODEL:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
MERGED_ROOT="${REPO_ROOT}/outputs/scope_round3/merged"
OUT_ROOT="${REPO_ROOT}/outputs/scope_round4/postfix_replay"
LOG_DIR="${REPO_ROOT}/outputs/scope_round4/logs"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
mkdir -p "${OUT_ROOT}/offline" "${OUT_ROOT}/closed_loop" "${LOG_DIR}"

echo "[barrier3] $(date -Is) offline eval 8-way parallel"

declare -a OFFLINE_JOBS=(
  "0:Base:${BASE_MODEL}"
  "1:round3_compact_json:${MERGED_ROOT}/round3_compact_json_sample_norm"
  "2:round3_op_seed42:${MERGED_ROOT}/round3_op_main_seed42"
  "3:round3_op_seed43:${MERGED_ROOT}/round3_op_main_seed43"
  "4:round3_op_seed44:${MERGED_ROOT}/round3_op_main_seed44"
  "5:round3_op_no_balance:${MERGED_ROOT}/round3_op_no_balance"
  "6:round3_correct_only:${MERGED_ROOT}/round3_correct_only_op"
  "7:round3_endorse_only:${MERGED_ROOT}/round3_endorse_only_op"
)

OFFLINE_PIDS=()
for entry in "${OFFLINE_JOBS[@]}"; do
  IFS=':' read -r gpu variant model_path <<< "${entry}"
  log="${LOG_DIR}/barrier3_offline_${variant}.log"
  echo "[barrier3-offline] GPU${gpu} ${variant}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python -c "
import json, sys
from pathlib import Path
sys.path.insert(0, '${REPO_ROOT}')
from training.scope.dup_diagnostics import load_jsonl
from training.scope.eval_dup_capability import evaluate_capability
from training.scope.sdi_trainer import DupSDITrainer, SDITrainConfig

valid = load_jsonl('${REPO_ROOT}/artifacts/datasets/dup_sdi_round3/valid.jsonl')
cfg = SDITrainConfig(
    model_path='${model_path}',
    output_dir=Path('/tmp/r4_postfix_${variant}'),
    loss_mode='operation_ce' if 'op_seed' in '${variant}' or '${variant}' == 'Base' else 'sample_normalized_action_ce',
    compact_target=True,
    eval_only=True,
)
trainer = DupSDITrainer(cfg)
report = evaluate_capability(trainer, valid)
report['variant'] = '${variant}'
out = Path('${OUT_ROOT}/offline/${variant}.json')
out.write_text(json.dumps(report, indent=2) + '\n')
print('done', '${variant}', report.get('macro_f1'))
" > "${log}" 2>&1 &
  OFFLINE_PIDS+=($!)
  sleep 2
done
wait "${OFFLINE_PIDS[@]}" || true
echo "[barrier3] offline eval done"

run_cl_shard() {
  local gpu=$1 variant=$2 model_path=$3 port=$4 shard=$5
  local out="${OUT_ROOT}/closed_loop/${variant}/${shard}"
  [[ -f "${out}/summary.json" ]] && echo "[skip] ${variant}/${shard}" && return 0
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" \
    --n-shards 8 \
    --model-path "${model_path}" \
    --vllm-port "${port}" \
    --dup-operation \
    --parallel 1 \
    > "${LOG_DIR}/barrier3_cl_${variant}_${shard}.log" 2>&1
}

run_cl_variant() {
  local gpu=$1 variant=$2 model_path=$3 port=$4
  echo "[barrier3-cl] ${variant} GPU${gpu} port${port}"
  run_cl_shard "${gpu}" "${variant}" "${model_path}" "${port}" shard0 &
  local p0=$!
  run_cl_shard "${gpu}" "${variant}" "${model_path}" "$((port+1))" shard1 &
  local p1=$!
  wait "${p0}" "${p1}"
}

echo "[barrier3] Wave A closed-loop (4 models, staggered)"
declare -a WAVE_A=(
  "0:base:${BASE_MODEL}:9000"
  "1:compact_json:${MERGED_ROOT}/round3_compact_json_sample_norm:9010"
  "2:op_seed42:${MERGED_ROOT}/round3_op_main_seed42:9020"
  "3:op_seed43:${MERGED_ROOT}/round3_op_main_seed43:9030"
)

WAVE_A_PIDS=()
for entry in "${WAVE_A[@]}"; do
  IFS=':' read -r gpu variant model_path port <<< "${entry}"
  (
    sleep $((RANDOM % 30 + 60))
    run_cl_variant "${gpu}" "${variant}" "${model_path}" "${port}"
  ) > "${LOG_DIR}/barrier3_waveA_${variant}.log" 2>&1 &
  WAVE_A_PIDS+=($!)
done
wait "${WAVE_A_PIDS[@]}" || true

echo "[barrier3] Wave B: op_seed44"
(
  sleep $((RANDOM % 30 + 60))
  run_cl_variant 0 op_seed44 "${MERGED_ROOT}/round3_op_main_seed44" 9040
) > "${LOG_DIR}/barrier3_waveB_op_seed44.log" 2>&1

echo "[barrier3] complete $(date -Is)"
