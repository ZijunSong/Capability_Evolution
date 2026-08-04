#!/usr/bin/env bash
# Phase 2 — 8 GPU parallel rollback training
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

TRAIN_DIR="${REPO_ROOT}/artifacts/datasets/scope_round8/rollback_sdi"
PHASE2_OUT="${OUT}/phase2_training"
LOG="${LOG_DIR}/phase2"
mkdir -p "${PHASE2_OUT}" "${LOG}"

PHASE2_PIDS=()

launch() {
  local gpu="$1" variant="$2"
  local logfile="${LOG}/train_${variant}.log"
  local out="${PHASE2_OUT}/${variant}"
  if [[ -f "${out}/DONE" ]]; then
    scope8_log "Skip Phase2 ${variant} (DONE)"
    return 0
  fi
  scope8_log "Phase2 train GPU${gpu} ${variant}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python training/scope_round8/run_phase2_train.py \
    --variant "${variant}" \
    --gpu cuda:0 \
    --train "${TRAIN_DIR}/train.jsonl" \
    --valid "${TRAIN_DIR}/valid.jsonl" \
    --output-dir "${PHASE2_OUT}" \
    >> "${logfile}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${PID_DIR}/phase2_${variant}.pid"
  PHASE2_PIDS+=("${pid}")
}

# GPU mapping per 0802-todo1.md §7.1
launch 0 rollback_o7_seed42
launch 1 rollback_o7_seed43
launch 2 rollback_o7_seed44
launch 3 rollback_endorse_only
launch 4 rollback_prompt_hint_distill
launch 5 rollback_trajectory_imitation
launch 6 rollback_correct_only
launch 7 rollback_soft_replan_only

if ((${#PHASE2_PIDS[@]} > 0)); then
  scope8_log "Waiting for ${#PHASE2_PIDS[@]} Phase2 training jobs"
  wait "${PHASE2_PIDS[@]}"
fi

scope8_log "Phase 2 training jobs finished"
for v in rollback_o7_seed42 rollback_o7_seed43 rollback_o7_seed44 rollback_endorse_only \
  rollback_prompt_hint_distill rollback_trajectory_imitation rollback_correct_only rollback_soft_replan_only; do
  if [[ -f "${PHASE2_OUT}/${v}/DONE" ]]; then
    scope8_log "Merge LoRA ${v}"
    python training/merge_lora_hf.py \
      --base-model /data/ppnm/models/Qwen2.5-7B-Instruct \
      --adapter "${PHASE2_OUT}/${v}" \
      --output "${OUT}/merged/${v}" \
      >> "${LOG}/merge_${v}.log" 2>&1 || true
  fi
done
scope8_log "Phase 2 complete"
