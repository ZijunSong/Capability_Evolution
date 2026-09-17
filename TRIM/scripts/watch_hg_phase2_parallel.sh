#!/usr/bin/env bash
# Watchdog: after GPT-zero finishes, ensure GPT-all and Qwen-all start in parallel.
set -euo pipefail

RUN_ID="${1:-20260914_104500_hg_test50_gpu34567_par}"
TRIM_ROOT="/data/ppnm/Capability_Evolution/TRIM"
PKG="/data/ppnm/trim_bcplus_test50_eval_harness_g"
LOG="${PKG}/logs/${RUN_ID}/nohup_launch.log"
GPT_ZERO_OUT="${TRIM_ROOT}/outputs/eval_hg_bcplus_test50_gpt-oss-20b_zero_gpu3457_${RUN_ID}"
PY="/data/ppnm/miniconda3/envs/bishop/bin/python"

log() { echo "[$(date -Is)] watchdog $*" | tee -a "${LOG}" >&2; }

already_phase2() {
  rg -q "START tag=gpt_all" "${LOG}" 2>/dev/null || \
    pgrep -f "eval_hg_bcplus_test50_gpt-oss-20b_all_gpu345_${RUN_ID}" >/dev/null 2>&1 || \
    pgrep -f "eval_hg_bcplus_test50_Qwen3-4B-Instruct-2507_all_gpu067_${RUN_ID}" >/dev/null 2>&1
}

log "watching GPT-zero for RUN_ID=${RUN_ID}"
while pgrep -f "eval_hg_bcplus_test50_gpt-oss-20b_zero_gpu3457_${RUN_ID}" >/dev/null 2>&1; do
  sleep 30
done
log "GPT-zero process exited"

if already_phase2; then
  log "phase2 already running; watchdog exit ok"
  exit 0
fi

if ! pgrep -f "run_bcplus_test50_harness_g_gpu34567_parallel.sh" >/dev/null 2>&1; then
  log "WARN parallel coordinator gone and phase2 not started; launching phase2 manually"
  cd "${TRIM_ROOT}"
  export RUN_ID GPU_UTIL=0.75
  export WAIT_QWEN_ZERO_RUN="${WAIT_QWEN_ZERO_RUN:-20260914_073827_hg_test50_gpu04567_auditfix2}"
  export SKIP_GPT_ZERO=1
  nohup bash scripts/run_bcplus_test50_harness_g_gpu34567_parallel.sh >> "${LOG}" 2>&1 &
  log "restarted parallel script with SKIP_GPT_ZERO=1 pid=$!"
  exit 0
fi

log "coordinator alive; waiting up to 10m for phase2 START lines"
for _ in $(seq 1 20); do
  sleep 30
  if already_phase2; then
    log "phase2 confirmed started"
    exit 0
  fi
done

log "ERROR phase2 did not start; manual intervention needed"
exit 1
