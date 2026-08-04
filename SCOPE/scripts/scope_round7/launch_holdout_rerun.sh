#!/usr/bin/env bash
# Launch holdout shard2/3 for gate-passed rerun variants
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup
export PARALLEL="${PARALLEL:-64}"

run_holdout() {
  local gpu="$1" variant="$2" shard="$3" port="$4"
  local LIVE_ROOT="${OUT}/contract_trace/live_rerun"
  local HO="${OUT}/holdout_tau0_rerun"
  case "${variant}" in
    base) SEED=0; MODEL="${BASE_MODEL}"; LABEL="base" ;;
    seed42) SEED=42; MODEL="${R5}/merged/o7_r64_seed42"; LABEL="o7_r64_seed42" ;;
    seed43) SEED=43; MODEL="${R5}/merged/o7_r64_seed43"; LABEL="o7_r64_seed43" ;;
    seed44) SEED=44; MODEL="${R5}/merged/o7_r64_seed44"; LABEL="o7_r64_seed44" ;;
  esac
  local live="${LIVE_ROOT}/$( [[ $variant == base ]] && echo base_shard1_tau0 || echo o7_${variant}_shard1_tau0 )"
  if ! scope7_gate_passed "${live}"; then
    scope7_log "Skip holdout ${variant} ${shard}: gate not passed"
    return 0
  fi
  local out="${HO}/${variant}_${shard}"
  if [[ -f "${out}/episodes.jsonl" ]] && [[ $(wc -l < "${out}/episodes.jsonl") -ge 25 ]]; then
    scope7_log "Skip holdout ${variant} ${shard}: already complete"
    return 0
  fi
  scope7_log "Holdout GPU${gpu} ${variant} ${shard}"
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/hmin_v2_dup_rollout.py \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" --n-shards 4 \
    --model-path "${MODEL}" \
    --vllm-port "${port}" \
    --dup-operation --decision-threshold 0 \
    --dup-seed "${SEED}" --checkpoint-label "${LABEL}" \
    --round7-trace --parallel "${PARALLEL}" --resume \
    >> "${LOG_DIR}/holdout_${variant}_${shard}.log" 2>&1
}

# GPU0: base shard2, GPU1: base shard3, GPU4: seed42 shard2, GPU5: seed42 shard3
run_holdout 0 base shard2 9230 &
sleep 30
run_holdout 1 base shard3 9231 &
sleep 30
run_holdout 4 seed42 shard2 9234 &
sleep 30
run_holdout 5 seed42 shard3 9235 &
wait
scope7_log "Holdout batch complete"
