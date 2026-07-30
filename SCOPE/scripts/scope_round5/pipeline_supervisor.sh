#!/usr/bin/env bash
# Round 5 统一 pipeline supervisor — 各阶段完成后自动进入下一阶段
# 设计为长期运行的 nohup 进程，可安全关闭终端
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/scope_round5/_common.sh"

POLL_SEC="${POLL_SEC:-60}"

run_stage_b0() {
  scope5_log "[B0] environment snapshot + unit tests"
  bash scripts/scope_round5/run_b0.sh >> "${LOG_DIR}/b0.log" 2>&1
  scope5_set_stage "b1"
}

run_stage_b1() {
  if ! scope5_gate_true "${OUT}/B1_PASS"; then
    scope5_log "[B1] observability audit"
    python training/scope_round5/run_observability_audit.py >> "${LOG_DIR}/b1_observability.log" 2>&1
  fi
  if ! scope5_gate_true "${OUT}/B1_PASS"; then
    scope5_log "B1 FAIL — pipeline stopped"
    scope5_set_stage "failed_b1"
    return 1
  fi
  scope5_log "B1 PASS"
  scope5_set_stage "b2"
}

run_stage_b2() {
  if ! scope5_gate_true "${OUT}/B2_PASS"; then
    scope5_log "[B2] objective audit"
    pytest tests/scope/test_operation_objectives.py -q >> "${LOG_DIR}/b2_unit_tests.log" 2>&1
    python training/scope_round5/run_b2_audit.py >> "${LOG_DIR}/b2_one_step.log" 2>&1
  fi
  if ! scope5_gate_true "${OUT}/B2_PASS"; then
    scope5_log "B2 FAIL — pipeline stopped"
    scope5_set_stage "failed_b2"
    return 1
  fi
  scope5_log "B2 PASS"
  scope5_set_stage "b3"
}

run_stage_b3() {
  if [[ ! -f "${OUT}/micro_overfit/MICRO_OVERFIT_MATRIX.md" ]]; then
    scope5_log "[B3] build nested datasets + micro-overfit tournament"
    python training/scope_round5/build_nested_datasets.py >> "${LOG_DIR}/b3_build_nested.log" 2>&1
    bash scripts/scope_round5/run_b3_8gpu.sh >> "${LOG_DIR}/b3_8gpu.log" 2>&1
    python training/scope_round5/build_micro_matrix.py >> "${LOG_DIR}/b3_matrix.log" 2>&1
  fi
  python - <<'PY'
from pathlib import Path
import json
out = Path("outputs/scope_round5/micro_overfit")
passed = []
for obj_dir in sorted(out.iterdir()):
    if not obj_dir.is_dir():
        continue
    summ = obj_dir / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        if s.get("all_pass"):
            passed.append(obj_dir.name)
Path("outputs/scope_round5/B3_PASSED_OBJECTIVES").write_text("\n".join(passed) + "\n")
if not passed:
    raise SystemExit("B3 FAIL")
PY
  scope5_log "B3 PASS — $(cat "${OUT}/B3_PASSED_OBJECTIVES")"
  scope5_set_stage "b4_train"
}

run_stage_b4_train() {
  local done n running
  done="$(scope5_b4_done_count)"
  running="$(scope5_b4_train_count)"
  scope5_log "[B4 train] done=${done}/6 running=${running}"

  if [[ "${done}" -eq 6 ]]; then
    scope5_set_stage "b4_eval"
    return 0
  fi

  if [[ "${running}" -eq 0 ]]; then
    scope5_log "[B4 train] launching 6 training jobs"
    bash scripts/scope_round5/run_b4_8gpu.sh >> "${LOG_DIR}/b4_8gpu.log" 2>&1 || true
  fi
  return 2  # still waiting
}

run_stage_b4_eval() {
  if [[ -f "${OUT}/b4_full/B4_COMPLETE" ]]; then
    scope5_set_stage "b4_gate"
    return 0
  fi
  local done running
  done="$(scope5_b4_done_count)"
  running="$(scope5_b4_train_count)"
  if [[ "${done}" -lt 6 ]] || [[ "${running}" -gt 0 ]]; then
    scope5_log "[B4 eval] waiting train done=${done}/6 running=${running}"
    return 2
  fi
  scope5_log "[B4 eval] offline evaluation"
  bash scripts/scope_round5/run_b4_eval_only.sh >> "${LOG_DIR}/b4/eval_only.log" 2>&1
  scope5_set_stage "b4_gate"
}

run_stage_b4_gate() {
  if [[ -f "${OUT}/B4_PASS" ]]; then
    scope5_log "B4 gate already computed: $(cat "${OUT}/B4_PASS")"
  else
    scope5_log "[B4 gate] evaluating Top-2 eligibility"
    python training/scope_round5/run_b4_gate.py >> "${LOG_DIR}/b4/gate.log" 2>&1
  fi
  if scope5_gate_true "${OUT}/B4_PASS"; then
    scope5_log "B4 PASS — proceed to B5"
    scope5_set_stage "b5"
  else
    scope5_log "B4 FAIL — pipeline stopped (no B5/B6)"
    scope5_set_stage "failed_b4"
    return 1
  fi
}

run_stage_b5() {
  if [[ -f "${OUT}/B5_COMPLETE" ]]; then
    scope5_set_stage "b6"
    return 0
  fi
  scope5_log "[B5] closed-loop 50q"
  bash scripts/scope_round5/run_b5_closed_loop.sh >> "${LOG_DIR}/b5_closed_loop.log" 2>&1
  date -Is > "${OUT}/B5_COMPLETE"
  scope5_log "B5 COMPLETE"
  scope5_set_stage "b6"
}

run_stage_b6() {
  if [[ -f "${OUT}/B6_COMPLETE" ]]; then
    scope5_set_stage "done"
    return 0
  fi
  scope5_log "[B6] closed-loop 100q (best objective × 3 seeds)"
  bash scripts/scope_round5/run_b6_closed_loop.sh >> "${LOG_DIR}/b6_closed_loop.log" 2>&1
  date -Is > "${OUT}/B6_COMPLETE"
  python training/scope_round5/build_round5_report.py >> "${LOG_DIR}/round5_report.log" 2>&1 || true
  date -Is > "${OUT}/ROUND5_COMPLETE"
  scope5_log "B6 COMPLETE — Round 5 pipeline finished"
  scope5_set_stage "done"
}

scope5_setup
touch "${SUPERVISOR_LOG}"

# Initialize stage from disk state if missing
if [[ ! -f "${STAGE_FILE}" ]]; then
  scope5_set_stage "$(scope5_detect_stage)"
fi

scope5_log "=== pipeline supervisor start (stage=$(scope5_get_stage)) ==="

while true; do
  stage="$(scope5_get_stage)"
  rc=0
  case "${stage}" in
    b0) run_stage_b0 || rc=$? ;;
    b1) run_stage_b1 || rc=$? ;;
    b2) run_stage_b2 || rc=$? ;;
    b3) run_stage_b3 || rc=$? ;;
    b4_train) run_stage_b4_train || rc=$? ;;
    b4_eval) run_stage_b4_eval || rc=$? ;;
    b4_gate) run_stage_b4_gate || rc=$? ;;
    b5) run_stage_b5 || rc=$? ;;
    b6) run_stage_b6 || rc=$? ;;
    done)
      scope5_log "Pipeline complete."
      exit 0
      ;;
    failed_*)
      scope5_log "Pipeline halted at stage=${stage}"
      exit 1
      ;;
    *)
      scope5_log "Unknown stage=${stage}, re-detecting"
      scope5_set_stage "$(scope5_detect_stage)"
      rc=2
      ;;
  esac

  if [[ "${rc}" -eq 2 ]]; then
    scope5_log "waiting (${POLL_SEC}s) stage=${stage}"
    sleep "${POLL_SEC}"
    continue
  fi
  if [[ "${rc}" -ne 0 ]]; then
    exit "${rc}"
  fi
  # stage advanced synchronously — loop immediately
done
