#!/usr/bin/env bash
set -euo pipefail
cd /data/ppnm/Capability_Evolution/SCOPE
source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Kill stage2 trainers by matching python argv precisely
while read -r pid; do
  [[ -z "$pid" ]] && continue
  kill "$pid" 2>/dev/null || true
done < <(ps -eo pid,args | awk '/python.*run_stage2_pointer_train.py/ && !/awk/ {print $1}')
sleep 2
while read -r pid; do
  [[ -z "$pid" ]] && continue
  kill -9 "$pid" 2>/dev/null || true
done < <(ps -eo pid,args | awk '/python.*run_stage2_pointer_train.py/ && !/awk/ {print $1}')

while read -r pid; do
  [[ -z "$pid" ]] && continue
  kill "$pid" 2>/dev/null || true
done < <(ps -eo pid,args | awk '/bash.*run_stage2_gpu.sh/ && !/awk/ && !/_restart_stage2/ {print $1}')
sleep 2

rm -rf outputs/scope_round13/stage2_targeted/training/r13_ckpt_pointer_seed42
rm -rf outputs/scope_round13/stage2_targeted/training/r13_ckpt_pointer_seed43
rm -rf outputs/scope_round13/stage2_targeted/training/r13_ckpt_pointer_seed44
mkdir -p outputs/scope_round13/stage2_targeted/training
# rotate logs
for v in r13_ckpt_pointer_seed42 r13_ckpt_pointer_seed43 r13_ckpt_pointer_seed44; do
  [[ -f outputs/scope_round13/logs/stage2_${v}.log ]] && mv outputs/scope_round13/logs/stage2_${v}.log "outputs/scope_round13/logs/stage2_${v}.log.bak.$(date +%H%M%S)" || true
done

nohup bash scripts/scope_round13/run_stage2_gpu.sh 5 r13_ckpt_pointer_seed42 \
  >> outputs/scope_round13/logs/stage2_r13_ckpt_pointer_seed42_supervisor.log 2>&1 &
echo $! > outputs/scope_round13/pids/stage2_gpu5.pid
sleep 8
nohup bash scripts/scope_round13/run_stage2_gpu.sh 6 r13_ckpt_pointer_seed43 \
  >> outputs/scope_round13/logs/stage2_r13_ckpt_pointer_seed43_supervisor.log 2>&1 &
echo $! > outputs/scope_round13/pids/stage2_gpu6.pid
sleep 8
nohup bash scripts/scope_round13/run_stage2_gpu.sh 7 r13_ckpt_pointer_seed44 \
  >> outputs/scope_round13/logs/stage2_r13_ckpt_pointer_seed44_supervisor.log 2>&1 &
echo $! > outputs/scope_round13/pids/stage2_gpu7.pid

sleep 45
echo "GPU:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "PROCS:"
ps -eo pid,args | awk '/run_stage2_pointer_train.py|run_stage2_gpu.sh/ && !/awk/ && !/_restart_stage2/ {print}'
echo "LOGS:"
for v in r13_ckpt_pointer_seed42 r13_ckpt_pointer_seed43 r13_ckpt_pointer_seed44; do
  echo -n "$v: "
  if [[ -f outputs/scope_round13/stage2_targeted/training/$v/FAILED ]]; then
    cat outputs/scope_round13/stage2_targeted/training/$v/FAILED
  else
    tail -1 outputs/scope_round13/logs/stage2_${v}.log 2>/dev/null || echo starting
  fi
done
