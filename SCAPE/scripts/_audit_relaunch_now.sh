#!/usr/bin/env bash
set -euo pipefail
SCAPE=/data/ppnm/Capability_Evolution/SCAPE
cd "$SCAPE"
mkdir -p outputs/stage_l_hminus_data/logs outputs/stage_l/logs

# A collect train/512 GPU0
nohup env GPU=0 JOB_NAME=A_auto_populate_first_search COMPONENT=auto_populate_first_search \
  OUT_ROOT="$SCAPE/outputs/stage_l_hminus_data" LIMIT=512 SPLIT=train \
  MODEL_PATH=/data/ppnm/models/Qwen2.5-7B-Instruct PARALLEL=1 \
  bash "$SCAPE/scripts/run_loo_worker.sh" \
  >"$SCAPE/outputs/stage_l_hminus_data/logs/A_auto_populate_first_search.train.log" 2>&1 &
echo $! > "$SCAPE/outputs/stage_l_hminus_data/pids/A_auto_populate_first_search.pid"
echo "A=$(cat outputs/stage_l_hminus_data/pids/A_auto_populate_first_search.pid)"
sleep 2

# B collect train/512 GPU1
nohup env GPU=1 JOB_NAME=B_verify_tool COMPONENT=verify_tool \
  OUT_ROOT="$SCAPE/outputs/stage_l_hminus_data" LIMIT=512 SPLIT=train \
  MODEL_PATH=/data/ppnm/models/Qwen2.5-7B-Instruct PARALLEL=1 \
  bash "$SCAPE/scripts/run_loo_worker.sh" \
  >"$SCAPE/outputs/stage_l_hminus_data/logs/B_verify_tool.train.log" 2>&1 &
echo $! > "$SCAPE/outputs/stage_l_hminus_data/pids/B_verify_tool.pid"
echo "B=$(cat outputs/stage_l_hminus_data/pids/B_verify_tool.pid)"
sleep 2

# B L512 seed42 on GPU2-5 (user asked L512 if collect B done OR continue L200s44).
# Collect B not done → prefer NOT L512 from H_-m collect; but OPD L512 uses train split via train_opd itself (doesn't need collect).
# User: "B L512-scale (limit 512) seed42 on free GPUs if collect B done OR continue L200 seed44"
# L200s44 done → next useful is L512 OPD (train_opd generates its own rollouts; collect is separate H_-m data path).
nohup env SEED=42 LIMIT=512 PORT=8769 TP=4 CUDA_VISIBLE_DEVICES=2,3,4,5 \
  OPD_OUT="$SCAPE/outputs/stage_l/B_verify_opd_provisional" \
  OUT_CELL="$SCAPE/outputs/stage_l/B_verify_opd_provisional/L512_seed42" \
  bash "$SCAPE/scripts/run_stage_l_b_verify_opd_L200.sh" \
  >"$SCAPE/outputs/stage_l/logs/B_verify_opd_L512_s42.log" 2>&1 &
echo $! > "$SCAPE/outputs/stage_l/pids_B_opd_L512_s42.pid"
echo "B_L512=$(cat outputs/stage_l/pids_B_opd_L512_s42.pid)"
sleep 2

# A OPD L64 seed42 GPU6-7
nohup env SEED=42 LIMIT=64 PORT=8770 TP=2 CUDA_VISIBLE_DEVICES=6,7 \
  bash "$SCAPE/scripts/run_stage_l_a_auto_opd_L64.sh" \
  >"$SCAPE/outputs/stage_l/logs/A_auto_opd_L64_s42.log" 2>&1 &
echo $! > "$SCAPE/outputs/stage_l/pids_A_opd_L64_s42.pid"
echo "A_OPD=$(cat outputs/stage_l/pids_A_opd_L64_s42.pid)"

# Ensure monitor
if ! pgrep -f 'scripts/monitor_daemon.sh' >/dev/null; then
  nohup bash scripts/monitor_daemon.sh >> outputs/MONITOR_DAEMON.log 2>&1 &
  echo "monitor=$!"
else
  echo "monitor=alive"
fi

sleep 8
pgrep -af 'run_loo_worker|run_stage_l|vllm serve|train_opd|monitor_daemon' | grep -v grep || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
