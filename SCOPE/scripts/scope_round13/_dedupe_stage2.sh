#!/usr/bin/env bash
set -euo pipefail
cd /data/ppnm/Capability_Evolution/SCOPE

# Stop guardian briefly to avoid races
if [[ -f outputs/scope_round13/pids/guardian.pid ]]; then
  kill "$(cat outputs/scope_round13/pids/guardian.pid)" 2>/dev/null || true
fi
while read -r pid; do
  [[ -z "$pid" ]] && continue
  kill "$pid" 2>/dev/null || true
done < <(ps -eo pid,args | awk '/bash .*guardian.sh/ && !/awk/ && !/_dedupe/ {print $1}')
sleep 1

# For each variant keep the newest python trainer; kill older duplicates
for v in r13_ckpt_pointer_seed42 r13_ckpt_pointer_seed43 r13_ckpt_pointer_seed44; do
  mapfile -t pids < <(ps -eo pid,lstart,args | awk -v v="$v" '
    $0 ~ ("run_stage2_pointer_train.py --variant " v) && !/awk/ {print $1}
  ')
  if (( ${#pids[@]} <= 1 )); then
    echo "$v trainers=${#pids[@]}"
    continue
  fi
  # keep last pid (newest), kill others
  keep="${pids[-1]}"
  echo "$v keep=$keep kill=${pids[*]}"
  for pid in "${pids[@]}"; do
    if [[ "$pid" != "$keep" ]]; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
  done
done

# Kill orphan run_stage2_gpu.sh whose python child is gone
while read -r pid args; do
  [[ -z "$pid" ]] && continue
  # if no python child, kill wrapper
  if ! ps --ppid "$pid" -o args= 2>/dev/null | grep -q run_stage2_pointer; then
    # still may be parent of hb; check descendants
    if ! pstree -p "$pid" 2>/dev/null | grep -q run_stage2_pointer; then
      echo "kill orphan wrapper $pid $args"
      kill "$pid" 2>/dev/null || true
    fi
  fi
done < <(ps -eo pid,args | awk '/bash.*run_stage2_gpu.sh/ && !/awk/ && !/_dedupe/ {print $1, $0}')

sleep 2
# Restart guardian
nohup bash /data/ppnm/Capability_Evolution/SCOPE/scripts/scope_round13/guardian.sh \
  >> outputs/scope_round13/logs/guardian.log 2>&1 &
echo $! > outputs/scope_round13/pids/guardian.pid

echo '--- after dedupe ---'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ps -eo pid,args | awk '/run_stage2_pointer_train.py|run_stage2_gpu.sh/ && !/awk/ && !/_dedupe/ {print}'
