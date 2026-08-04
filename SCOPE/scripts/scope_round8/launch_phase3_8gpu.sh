#!/usr/bin/env bash
# Phase 3 — 8 GPU parallel closed-loop (after Offline Gate)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

OFFLINE_GATE="${OUT}/OFFLINE_GATE.json"
PHASE3_OUT="${OUT}/phase3_closed_loop"
LOG="${LOG_DIR}/phase3"
MANIFEST="${MANIFEST_100}"
HARNESS_CFG="${REPO_ROOT}/harness/configs/agent_core_recovery.yaml"
PARALLEL_PHASE3="${PARALLEL_PHASE3:-16}"

if [[ ! -f "${OFFLINE_GATE}" ]]; then
  scope8_log "ERROR: missing Offline Gate — run check_offline_gate.py first"
  exit 1
fi
if ! python3 -c "import json; g=json.load(open('${OFFLINE_GATE}')); exit(0 if g.get('phase3_eligible') else 1)"; then
  scope8_log "ERROR: Phase 3 not eligible — see ${OFFLINE_GATE}"
  exit 1
fi

mkdir -p "${PHASE3_OUT}" "${LOG}"

PHASE3_PIDS=()

launch_shard() {
  local gpu="$1" variant="$2" shard="$3" port="$4"
  local merged="$5"
  local extra_flags="$6"
  local out="${PHASE3_OUT}/${variant}/${shard}"
  local logfile="${LOG}/${variant}_${shard}.log"
  local n
  n=$(scope8_count_episodes "${out}/episodes.jsonl")
  if [[ "${n}" -ge 25 ]] && [[ -f "${out}/summary.json" ]]; then
    scope8_log "Skip Phase3 ${variant} ${shard} (${n}/25)"
    return 0
  fi
  mkdir -p "${out}"
  scope8_log "Phase3 GPU${gpu} ${variant} ${shard}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python training/scope_round8/rollback_closed_loop_rollout.py \
    --output-dir "${out}" \
    --variant "${variant}" \
    --manifest "${MANIFEST}" \
    --shard "${shard}" --n-shards 4 \
    --model-path "${BASE_MODEL}" \
    --merged-path "${merged}" \
    --harness-config "${HARNESS_CFG}" \
    --vllm-port "${port}" \
    --parallel "${PARALLEL_PHASE3}" \
    --rollback-operation \
    ${extra_flags} \
    --resume \
    >> "${logfile}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${PID_DIR}/phase3_${variant}_${shard}.pid"
  PHASE3_PIDS+=("${pid}")
}

launch_variant() {
  local gpu="$1" variant="$2" port="$3" merged="$4" extra_flags="$5"
  launch_shard "${gpu}" "${variant}" shard0 "${port}" "${merged}" "${extra_flags}"
  launch_shard "${gpu}" "${variant}" shard1 "$((port+1))" "${merged}" "${extra_flags}"
  launch_shard "${gpu}" "${variant}" shard2 "$((port+2))" "${merged}" "${extra_flags}"
  launch_shard "${gpu}" "${variant}" shard3 "$((port+3))" "${merged}" "${extra_flags}"
}

# Note: one variant per GPU — shards run sequentially within each GPU queue
run_gpu_queue() {
  local gpu="$1" variant="$2" port="$3" merged="$4" extra_flags="$5"
  for shard in shard0 shard1 shard2 shard3; do
    launch_shard "${gpu}" "${variant}" "${shard}" "${port}" "${merged}" "${extra_flags}"
    if ((${#PHASE3_PIDS[@]} > 0)); then
      wait "${PHASE3_PIDS[-1]}"
      PHASE3_PIDS=()
    fi
    port=$((port + 1))
  done
}

MERGED_ROOT="${OUT}/merged"

scope8_log "Phase 3 closed-loop start"
run_gpu_queue 0 base_agent_core 9400 "${BASE_MODEL}" "" &
run_gpu_queue 1 rollback_o7_seed42 9410 "${MERGED_ROOT}/rollback_o7_seed42" "" &
run_gpu_queue 2 rollback_o7_seed43 9420 "${MERGED_ROOT}/rollback_o7_seed43" "" &
run_gpu_queue 3 rollback_o7_seed44 9430 "${MERGED_ROOT}/rollback_o7_seed44" "" &
run_gpu_queue 4 rollback_prompt_hint_distill 9440 "${MERGED_ROOT}/rollback_prompt_hint_distill" "--hint-distill" &
run_gpu_queue 5 rollback_trajectory_imitation 9450 "${MERGED_ROOT}/rollback_trajectory_imitation" "" &
run_gpu_queue 6 rollback_correct_only 9460 "${MERGED_ROOT}/rollback_correct_only" "" &
run_gpu_queue 7 rollback_soft_replan_only 9470 "${MERGED_ROOT}/rollback_soft_replan_only" "--soft-replan-only" &
wait

scope8_log "Phase 3 runs complete — aggregating gate"
python training/scope_round8/aggregate_phase3_gate.py
scope8_log "Phase 3 complete"
