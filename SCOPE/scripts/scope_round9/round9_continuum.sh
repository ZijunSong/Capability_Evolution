#!/usr/bin/env bash
# Round 9 continuum: wait Wave A → oracle → Wave B → (gates) → Wave C
set -euo pipefail
source "$(dirname "$0")/_common.sh"
scope9_setup

wait_wave_a() {
  scope9_log "CONTINUUM: waiting for Wave A"
  local idle_rounds=0
  while true; do
    local done_n
    done_n=$(find "${MARKER_DIR}" -maxdepth 1 -name 'wave_a_*.DONE' 2>/dev/null | wc -l | tr -d ' ' || echo 0)
    if [[ "${done_n}" -ge 8 ]]; then
      local fail=0
      for v in "${WAVE_A_VARIANTS[@]}"; do
        local report="${OUT}/wave_a/${v}/WAVE_A_REPORT.json"
        if [[ ! -f "${report}" ]]; then
          scope9_log "CONTINUUM: missing ${report}"
          fail=1
          continue
        fi
        local pass
        pass=$(python -c "import json; print(json.load(open('${report}')).get('barrier_a_pass', False))")
        if [[ "${pass}" != "True" ]]; then
          scope9_log "CONTINUUM: Barrier A FAIL ${v}"
          python -c "import json; d=json.load(open('${report}')); print(d.get('split_failures'))"
          fail=1
        fi
      done
      if [[ "${fail}" -eq 0 ]]; then
        scope9_log "CONTINUUM: Barrier A PASS"
        return 0
      fi
      scope9_log "CONTINUUM: Barrier A failed; not advancing to training"
      return 2
    fi

    # Detect terminal failure: no workers, reports exist, markers incomplete.
    local workers
    workers=$(pgrep -f 'run_wave_a_gpu.sh' | wc -l | tr -d ' ')
    local reports
    reports=$(find "${OUT}/wave_a" -mindepth 2 -maxdepth 2 -name 'WAVE_A_REPORT.json' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${workers}" -eq 0 && "${reports}" -ge 8 && "${done_n}" -lt 8 ]]; then
      scope9_log "CONTINUUM: Wave A workers finished with ${done_n}/8 DONE; Barrier A failed"
      return 2
    fi
    if [[ "${workers}" -eq 0 && "${done_n}" -lt 8 ]]; then
      idle_rounds=$((idle_rounds + 1))
      if [[ "${idle_rounds}" -ge 5 ]]; then
        scope9_log "CONTINUUM: Wave A idle with ${done_n}/8 DONE after retries; abort"
        return 2
      fi
    else
      idle_rounds=0
    fi
    sleep 60
  done
}

run_oracle() {
  scope9_log "CONTINUUM: oracle factorization"
  local replay="${OUT}/wave_a/rollback_o7_seed42/base_live/hf_replay.jsonl"
  if [[ ! -f "${replay}" ]]; then
    scope9_log "ERROR: missing ${replay}"
    return 1
  fi
  python training/scope_round9/aggregate_oracle_factorization.py \
    --replay "${replay}" \
    --output "${OUT}/diagnosis/ROOT_CAUSE_DECISION.json" \
    2>&1 | tee "${LOG_DIR}/oracle_factorization.log"
  cp -f "${OUT}/diagnosis/ROOT_CAUSE_DECISION.json" "${OUT}/ROOT_CAUSE_DECISION.json"
  python - <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path("outputs/scope_round9/ROOT_CAUSE_DECISION.json").read_text())
d = r.get("diagnosis", {})
print(json.dumps(d, indent=2))
# Stop training if oracle+oracle cannot reach sanity.
if "labels_candidates_or_aggregator" in d.get("primary_bottlenecks", []):
    print("ORACLE SANITY FAIL", file=sys.stderr)
    sys.exit(3)
PY
}

run_wave_b() {
  scope9_log "CONTINUUM: launching Wave B"
  bash scripts/scope_round9/launch_wave_b_train_8gpu.sh \
    2>&1 | tee -a "${LOG_DIR}/wave_b_launch.log"
}

run_wave_c_if_gate() {
  local gate="${OUT}/OFFLINE_GATE_ROUND9.json"
  if [[ ! -f "${gate}" ]]; then
    scope9_log "ERROR: missing offline gate"
    return 1
  fi
  local pass
  pass=$(python -c "import json; print(json.load(open('${gate}')).get('offline_gate_pass', False))")
  if [[ "${pass}" != "True" ]]; then
    scope9_log "CONTINUUM: Offline Gate FAIL — stop before closed-loop"
    return 2
  fi
  scope9_log "CONTINUUM: Offline Gate PASS — smoke20"
  bash scripts/scope_round9/launch_wave_c_smoke20_8gpu.sh \
    2>&1 | tee -a "${LOG_DIR}/wave_c_smoke_launch.log"
  local smoke="${OUT}/H20_SMOKE_GATE.json"
  local smoke_pass
  smoke_pass=$(python -c "import json; print(json.load(open('${smoke}')).get('smoke_pass', False))")
  if [[ "${smoke_pass}" != "True" ]]; then
    scope9_log "CONTINUUM: smoke20 FAIL — stop before 100q"
    return 2
  fi
  scope9_log "CONTINUUM: smoke20 PASS — final100"
  bash scripts/scope_round9/launch_wave_c_final100_8gpu.sh \
    2>&1 | tee -a "${LOG_DIR}/wave_c_final_launch.log"
}

write_diagnosis_report() {
  python training/scope_round9/write_round9_diagnosis_report.py \
    --output "${OUT}/ROUND9_DIAGNOSIS_REPORT.md" || true
}

main() {
  wait_wave_a
  run_oracle
  run_wave_b
  run_wave_c_if_gate || true
  write_diagnosis_report
  scope9_log "CONTINUUM: finished"
}

main "$@"
