#!/usr/bin/env bash
# 续跑 orchestrator：从 step80（latest checkpoint）续跑到 step200，保留 eval hooks。
# 跳过已完成的 Phase 0（step0 eval）和 Phase 1（step50 train+eval），从 Phase 2 开始。
#
# 流程（Phase 2-4）：
#   对 TARGET ∈ {100,150,200}：
#     1. 提交训练段（RESUME=1，从 latest 续训到 TARGET）
#     2. 轮询直到 step_TARGET/STEP_COMPLETE 落盘 或 job 终止
#     3. kill 训练 job
#     4. 提交 4 个 eval（TARGET × 4 benchmark），等完成
#     5. 进入下一段（RESUME=1）
#
# 注：Phase 1 的训练段（dthdo）已 kill，latest=step_000080。Phase 2 从 step80 resume to 100。
set -uo pipefail
export PATH=/root/.cloudml-cli/bin:$PATH

OUT=/mnt/songzijun/Capability_Evolution/TRIM/outputs/cloudml_custom_train_harness1_bcplus_full_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261003_retrain1
CKPT="$OUT/checkpoints/rl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="retrain1"
BENCHMARKS="bcplus_full longsealqa frames hotpotqa"
EVAL_SUBMIT="${SCRIPT_DIR}/submit_gpt_oss_20b_7nzzn_retrain_eval_8gpu.sh"
TRAIN_SUBMIT="${SCRIPT_DIR}/submit_gpt_oss_20b_rl_ddp_200step_zero_32gpu_4node_20261003_retrain1.sh"
DATE=$(date +%Y%m%d)

LOG="/tmp/orchestrate_retrain1_resume_${DATE}.log"
log() { echo "[$(date '+%m%d_%H:%M:%S')] $*" | tee -a "$LOG"; }

job_state() {
  cml custom_train describe "$1" 2>&1 | python3 -c "import sys,json;t=sys.stdin.read();i=t.find('{');d=json.loads(t[i:]) if i>=0 else {};print(d.get('state',''))" 2>/dev/null
}

wait_jobs() {
  local jobs="$1"
  local -n _done_ref="${2:-DONE}"
  while true; do
    local c=0
    for j in $jobs; do
      [ -n "${_done_ref[$j]:-}" ] && { c=$((c+1)); continue; }
      local st
      st=$(job_state "$j")
      case "$st" in
        succeed|failed) _done_ref[$j]="$st"; c=$((c+1)); log "  $j → $st" ;;
      esac
    done
    local total
    total=$(echo "$jobs" | wc -w)
    [ $c -eq $total ] && break
    sleep 300
  done
}

mkdir -p "$OUT"
log "=== orchestrate_retrain1_resume START (from step80, Phase 2-4) ==="
log "OUT=$OUT latest=$(readlink $CKPT/latest 2>/dev/null)"

# 续跑前 preflight：确认 latest checkpoint 完整
test -d "$CKPT/latest" || { log "FATAL: $CKPT/latest 不存在，无法 resume"; exit 1; }
test -f "$CKPT/latest/STEP_COMPLETE" || { log "FATAL: latest/STEP_COMPLETE 不存在"; exit 1; }
log "RESUME_TARGET latest=$(readlink -f $CKPT/latest)"

# ============================================================
# Phase 2-4: 训练到 100/150/200，每段后 eval（RESUME=1）
# ============================================================
for TARGET in 100 150 200; do
  log "=== Phase: train to step $TARGET (RESUME=1) ==="

  # 1. 提交训练段
  out=$(bash "$TRAIN_SUBMIT" "$TARGET" 1 2>&1) || { log "ERROR: train submit target $TARGET failed"; exit 1; }
  TRAIN_JOB=$(echo "$out" | grep -oE 't-[0-9]{14}-[a-z0-9]{5}' | head -1)
  [ -z "$TRAIN_JOB" ] && { log "ERROR: no train job id"; echo "$out" | tail -5 | tee -a "$LOG"; exit 1; }
  log "  train job → $TRAIN_JOB"

  # 2. 轮询直到 step_TARGET/STEP_COMPLETE 落盘
  STEP_DIR="$CKPT/step_000$(printf '%03d' "$TARGET")"
  log "  polling for $STEP_DIR/STEP_COMPLETE ..."
  TRAIN_TERMINAL=""
  while true; do
    if [ -f "$STEP_DIR/STEP_COMPLETE" ]; then
      log "  STEP_COMPLETE found at step $TARGET"
      break
    fi
    st=$(job_state "$TRAIN_JOB")
    case "$st" in
      failed|stopped|succeed)
        TRAIN_TERMINAL="$st"
        log "  train job $TRAIN_JOB → $st (before/at target $TARGET)"
        break
        ;;
    esac
    sleep 120
  done

  if [ ! -f "$STEP_DIR/STEP_COMPLETE" ]; then
    log "ERROR: train job $TRAIN_JOB ended ($TRAIN_TERMINAL) without step $TARGET STEP_COMPLETE"
    exit 1
  fi

  # 3. kill 训练 job
  log "  killing train job $TRAIN_JOB ..."
  cml custom_train kill "$TRAIN_JOB" 2>&1 | tee -a "$LOG" || true
  sleep 30

  # 4. 提交 4 个 eval
  log "  submitting 4 eval for step $TARGET ..."
  declare -A EDONE=()
  EVAL_JOBS=""
  for b in $BENCHMARKS; do
    out=$(bash "$EVAL_SUBMIT" "$TARGET" "$b" "$RUN_TAG" 2>&1) || { log "ERROR: eval submit step$TARGET $b failed"; continue; }
    j=$(echo "$out" | grep -oE 't-[0-9]{14}-[a-z0-9]{5}' | head -1)
    [ -z "$j" ] && { log "ERROR: no job id for step$TARGET $b"; continue; }
    EVAL_JOBS="$EVAL_JOBS $j"
    log "    step$TARGET $b → $j"
  done
  wait_jobs "$EVAL_JOBS" EDONE
  log "  step $TARGET eval 终态:"
  for j in $EVAL_JOBS; do log "    $j=${EDONE[$j]}"; done
done

log "=== Phase 2-4 DONE (step 100/150/200 train+eval all complete) ==="
log "汇总: 读各 eval_h1_*_retrain1_step{100,150,200}_*/upstream_api/SUMMARY.json"
log "=== orchestrate_retrain1_resume END ==="
