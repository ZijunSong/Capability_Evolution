#!/usr/bin/env bash
# 宿主侧 orchestrator：驱动 retrain1 的"分段训练 + 中途 eval"全流程。
#
# 流程：
#   Phase 0: 提交 4 个 step0 (bare) eval，等完成（训练前，无 adapter，bf16-dequant base）。
#   Phase 1-4: 对 TARGET ∈ {50,100,150,200}：
#     1. 提交训练段（RESUME=0 首段 / RESUME=1 续段），--train-steps TARGET。
#     2. 轮询直到 step_TARGET/STEP_COMPLETE 落盘 或 训练 job 终止。
#     3. kill 训练 job（checkpoint 已在 JuiceFS，kill 不丢）。
#     4. 提交 4 个 eval（TARGET × 4 benchmark），等完成。
#     5. 进入下一段（RESUME=1）。
#
# 设计要点：
#   - cml 只在 dev box（/root/.cloudml-cli/bin/cml），训练/eval pod 不调 cml。
#   - job ID 由 cml submit stdout 输出，格式 t-<14位时间戳>-<5位base36>，用 grep 捕获。
#   - poll 用 python t.find('{') 解析（cml describe 有非 JSON preamble，见 poll_step100.sh）。
#   - 训练段 priority=8，eval priority=2（eval 不抢训练资源）。
#   - 每个 eval 用 fresh RUN_ID（含 DATE+TARGET），避 assert_fresh_eval_dir 坑。
#   - 失败的 eval 不阻塞下一段（记录 failed，继续），最终汇总时人工重跑。
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

# 日志
LOG="/tmp/orchestrate_retrain1_${DATE}.log"
log() { echo "[$(date '+%m%d_%H:%M:%S')] $*" | tee -a "$LOG"; }

# 工具：解析 cml describe 的 state（python t.find('{') 模式，避 preamble）
job_state() {
  cml custom_train describe "$1" 2>&1 | python3 -c "import sys,json;t=sys.stdin.read();i=t.find('{');d=json.loads(t[i:]) if i>=0 else {};print(d.get('state',''))" 2>/dev/null
}

# 工具：等一组 job 跑完（succeed/failed），返回每个 job 的终态
# 用法: wait_jobs "job1 job2 job3" → 打印进度，全部终态后返回
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
log "=== orchestrate_retrain1 START OUT=$OUT ==="
log "eval submit: $EVAL_SUBMIT"
log "train submit: $TRAIN_SUBMIT"

# ============================================================
# Phase 0: step0 bare eval（训练前）
# ============================================================
log "=== Phase 0: step0 bare eval (4 benchmark) ==="
declare -A S0DONE=()
STEP0_JOBS=""
for b in $BENCHMARKS; do
  out=$(bash "$EVAL_SUBMIT" 0 "$b" "$RUN_TAG" 2>&1) || { log "ERROR: eval submit step0 $b failed"; continue; }
  j=$(echo "$out" | grep -oE 't-[0-9]{14}-[a-z0-9]{5}' | head -1)
  [ -z "$j" ] && { log "ERROR: no job id for step0 $b; submit output:"; echo "$out" | tail -5 | tee -a "$LOG"; continue; }
  STEP0_JOBS="$STEP0_JOBS $j"
  log "  step0 $b → $j"
done
wait_jobs "$STEP0_JOBS" S0DONE
log "=== Phase 0 done: step0 eval 终态 ==="
for j in $STEP0_JOBS; do log "  $j=${S0DONE[$j]}"; done

# ============================================================
# Phase 1-4: 训练到 50/100/150/200，每段后 eval
# ============================================================
RESUME=0
for TARGET in 50 100 150 200; do
  log "=== Phase $((TARGET/50)): train to step $TARGET (RESUME=$RESUME) ==="

  # 1. 提交训练段
  out=$(bash "$TRAIN_SUBMIT" "$TARGET" "$RESUME" 2>&1) || { log "ERROR: train submit target $TARGET failed"; exit 1; }
  TRAIN_JOB=$(echo "$out" | grep -oE 't-[0-9]{14}-[a-z0-9]{5}' | head -1)
  [ -z "$TRAIN_JOB" ] && { log "ERROR: no train job id; submit output:"; echo "$out" | tail -5 | tee -a "$LOG"; exit 1; }
  log "  train job → $TRAIN_JOB"

  # 2. 轮询直到 step_TARGET/STEP_COMPLETE 落盘 或 job 终止
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

  # 即使 STEP_COMPLETE 已落盘，job 可能还在跑后续收尾；确认 ckpt 完整
  if [ ! -f "$STEP_DIR/STEP_COMPLETE" ]; then
    log "ERROR: train job $TRAIN_JOB ended ($TRAIN_TERMINAL) without step $TARGET STEP_COMPLETE"
    exit 1
  fi

  # 3. kill 训练 job（checkpoint 已在 JuiceFS，安全）
  log "  killing train job $TRAIN_JOB (ckpt already saved) ..."
  cml custom_train kill "$TRAIN_JOB" 2>&1 | tee -a "$LOG" || true
  sleep 30

  # 4. 提交 4 个 eval（TARGET × 4 benchmark）
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

  RESUME=1
done

log "=== ALL 5 STAGES (0/50/100/150/200) DONE ==="
log "step0:   $STEP0_JOBS"
log "汇总: 读各 eval_h1_*_retrain1_step*/upstream_api/SUMMARY.json 的 final_answer_recall"
log "=== orchestrate_retrain1 END ==="
