#!/usr/bin/env bash
# Supplement rollback collection on missing shards (boost Gate 1C event count)
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope8_setup

collect_if_missing() {
  local gpu="$1" shard="$2" mode="$3" port="$4"
  local out="${OUT}/rollback_collection/${mode}/${shard}"
  local n=0
  if [[ -f "${out}/rollback_events.jsonl" ]]; then
    n=$(wc -l < "${out}/rollback_events.jsonl" | tr -d ' ')
  fi
  if [[ "${n}" -ge 400 ]]; then
    scope8_log "Skip supplement ${mode}/${shard} (${n} events)"
    return 0
  fi
  scope8_collect_rollback "${gpu}" "${shard}" "${BASE_MODEL}" "${port}" "${mode}" "supp_${mode}_${shard}"
}

scope8_log "Supplementary rollback collection"
collect_if_missing 4 shard2 natural 9314 &
collect_if_missing 5 shard3 natural 9315 &
collect_if_missing 6 shard0 injected 9316 &
collect_if_missing 7 shard1 injected 9317 &
wait
scope8_log "Supplementary rollback complete"
