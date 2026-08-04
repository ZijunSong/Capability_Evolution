#!/usr/bin/env bash
# Monitor Round 7 rerun + holdout; auto-complete replay/compare/gate/holdout/report.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope7_setup

LIVE_RERUN="${OUT}/contract_trace/live_rerun"
HO="${OUT}/holdout_tau0_rerun"
LOG="${LOG_DIR}/monitor_rerun.log"
TAG43="o7_seed43_shard1_rerun"
MODEL43="${R5}/merged/o7_r64_seed43"
LIVE43="${LIVE_RERUN}/o7_seed43_shard1_tau0"

log() { scope7_log "[monitor] $*"; }

gate_status() {
  local d="$1"
  python3 -c "import json; print(json.load(open('${d}/contract_gate.json'))['contract_gate_pass'])" 2>/dev/null || echo "-"
}

ep_count() {
  local f="${1}/episodes.jsonl"
  [[ -f "$f" ]] || { echo 0; return; }
  wc -l < "$f" 2>/dev/null || echo 0
}

seed43_busy() {
  pgrep -f "finish_seed43|seed43_pipeline|replay_live_trace_.*o7_seed43_shard1_rerun|python.*hmin_v2_dup_rollout.py.*seed43" >/dev/null 2>&1
}

maybe_finish_seed43() {
  if [[ "$(ep_count "${LIVE43}")" -lt 25 ]]; then return 0; fi
  if [[ "$(gate_status "${LIVE43}")" == "True" ]]; then return 0; fi
  if seed43_busy; then return 0; fi
  log "seed43 live complete; starting contract pipeline on GPU6"
  scope7_wait_gpu_free 6 3600 || true
  scope7_contract_pipeline 6 "${LIVE43}" "${MODEL43}" 9226 "${TAG43}"
}

maybe_launch_seed43_holdout() {
  if [[ "$(gate_status "${LIVE43}")" != "True" ]]; then return 0; fi
  local s2="${HO}/seed43_shard2" s3="${HO}/seed43_shard3"
  if [[ "$(ep_count "${s2}")" -ge 25 && "$(ep_count "${s3}")" -ge 25 ]]; then return 0; fi
  if pgrep -f "python.*hmin_v2_dup_rollout.py.*seed43_shard" >/dev/null 2>&1; then return 0; fi
  log "seed43 gate passed; launching holdout shard2/3"
  if [[ "$(ep_count "${s2}")" -lt 25 ]]; then
    scope7_wait_gpu_free 4 3600 || true
    PARALLEL=64 scope7_run_live 4 "${s2}" "${MODEL43}" 9236 shard2 43 o7_r64_seed43 "${TAG43}_shard2"
  fi
  if [[ "$(ep_count "${s3}")" -lt 25 ]]; then
    scope7_wait_gpu_free 5 3600 || true
    PARALLEL=64 scope7_run_live 5 "${s3}" "${MODEL43}" 9237 shard3 43 o7_r64_seed43 "${TAG43}_shard3"
  fi
}

print_status() {
  log "=== status round ==="
  for d in "${LIVE_RERUN}"/*/; do
    [[ -d "$d" ]] || continue
    log "  $(basename "$d"): ep=$(ep_count "$d") gate=$(gate_status "$d")"
  done
  for d in "${HO}"/*/; do
    [[ -d "$d" ]] || continue
    log "  holdout $(basename "$d"): ep=$(ep_count "$d")/25"
  done
}

all_gates_passed() {
  for v in base_shard1_tau0 o7_seed42_shard1_tau0 o7_seed43_shard1_tau0 o7_seed44_shard1_tau0; do
    [[ "$(gate_status "${LIVE_RERUN}/${v}")" == "True" ]] || return 1
  done
}

all_holdout_done() {
  for h in base_shard2 base_shard3 seed42_shard2 seed42_shard3 seed43_shard2 seed43_shard3 seed44_shard2 seed44_shard3; do
    [[ "$(ep_count "${HO}/${h}")" -ge 25 ]] || return 1
  done
}

ROUND=0
while (( ROUND < 480 )); do
  ROUND=$((ROUND + 1))
  print_status
  maybe_finish_seed43
  maybe_launch_seed43_holdout

  if all_gates_passed && all_holdout_done; then
    log "All gates + holdouts complete; building report"
    python training/scope_round7/build_round7_report.py >> "${LOG}" 2>&1 || true
    log "DONE"
    exit 0
  fi

  sleep 120
done

log "Monitor timeout after 16h"
exit 1
