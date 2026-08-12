#!/usr/bin/env bash
# Phase D: 100q final gate — 0808-todo1.md §6
# Same 8-GPU matched sharding as Phase C (Base + 3 seeds × 2 shards).
set -euo pipefail
source "$(dirname "$0")/_common_followup.sh"
followup_setup

GATE="${OUT}/SMOKE20_GATE.json"
if [[ ! -f "${GATE}" ]]; then
  followup_log "ERROR: missing SMOKE20_GATE.json"
  exit 2
fi
PASS=$(python -c "import json; print(json.load(open('${GATE}')).get('pass', False))")
if [[ "${PASS}" != "True" ]]; then
  followup_log "SMOKE20_GATE.pass=false — STOP before 100q"
  exit 3
fi

python "$(dirname "$0")/create_followup_manifests.py" --out-dir "${OUT}/manifests"
MANIFEST="${OUT}/manifests/final100.json"
HARNESS="${REPO_ROOT}/harness/configs/agent_core_recovery.yaml"
PARALLEL="${FOLLOWUP_PARALLEL:-16}"
ROOT="${OUT}/phase_d_final100"
mkdir -p "${ROOT}"

declare -a SLOTS=(
  "0|base|shard0"
  "1|base|shard1"
  "2|r10_main_noweight_seed42|shard0"
  "3|r10_main_noweight_seed42|shard1"
  "4|r10_main_noweight_seed43|shard0"
  "5|r10_main_noweight_seed43|shard1"
  "6|r10_main_noweight_seed44|shard0"
  "7|r10_main_noweight_seed44|shard1"
)

model_for() {
  local variant="$1"
  if [[ "${variant}" == "base" ]]; then
    echo "${BASE_MODEL}"
  else
    echo "${OUT}/phase_b/${variant}/merged"
  fi
}

PIDS=()
for slot in "${SLOTS[@]}"; do
  IFS='|' read -r gpu variant shard <<< "${slot}"
  model="$(model_for "${variant}")"
  out="${ROOT}/${variant}/${shard}"
  mkdir -p "${out}"
  expected=50
  n=0
  if [[ -f "${out}/episodes.jsonl" ]]; then
    n=$(wc -l < "${out}/episodes.jsonl" | tr -d ' ')
  fi
  if [[ "${n}" -ge "${expected}" && -f "${out}/summary.json" ]]; then
    followup_log "Skip final100 ${variant} ${shard} (${n}/${expected})"
    continue
  fi
  port="$(followup_port_for_gpu "${gpu}")"
  log="${LOG_DIR}/phase_d_${variant}_${shard}.log"
  followup_log "Phase D GPU${gpu} ${variant} ${shard}"
  (
    export PYTHONUNBUFFERED=1
    while true; do date -Is > "${out}/HEARTBEAT"; sleep 60; done
  ) &
  echo $! > "${PID_DIR}/phase_d_hb_${gpu}.pid"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python training/scope_round8/rollback_closed_loop_rollout.py \
      --output-dir "${out}" \
      --variant "${variant}" \
      --manifest "${MANIFEST}" \
      --shard "${shard}" --n-shards 2 \
      --model-path "${BASE_MODEL}" \
      --merged-path "${model}" \
      --harness-config "${HARNESS}" \
      --vllm-port "${port}" \
      --parallel "${PARALLEL}" \
      --rollback-operation \
      --resume \
      >> "${log}" 2>&1 &
  echo $! > "${PID_DIR}/phase_d_gpu${gpu}.pid"
  PIDS+=($!)
  sleep 2
done

fail=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    followup_log "ERROR: Phase D slot ${i} failed"
    fail=1
  fi
done

for f in "${PID_DIR}"/phase_d_hb_*.pid; do
  [[ -f "${f}" ]] || continue
  kill "$(cat "${f}")" 2>/dev/null || true
  rm -f "${f}"
done

python training/scope_round10/aggregate_followup_closed_loop_gate.py --mode final100 \
  >> "${LOG_DIR}/phase_d_aggregate.log" 2>&1
followup_log "Phase D final100 aggregate done"
exit "${fail}"
