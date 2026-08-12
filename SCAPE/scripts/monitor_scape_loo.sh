#!/usr/bin/env bash
# Monitor LOCAL_CAL64 LOO; kill stuck workers and requeue missing jobs.
set -euo pipefail
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$SCAPE_ROOT/outputs/local_cal64_loo}"
STUCK_MIN="${STUCK_MIN:-45}"   # no progress for N minutes => stuck
LIMIT="${LIMIT:-64}"

mkdir -p "${OUT_ROOT}"
NOW=$(date +%s)

echo "=== SCAPE LOO monitor $(date -Iseconds) ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable in this shell"

n_done=0
n_run=0
n_fail=0
for d in "${OUT_ROOT}"/full "${OUT_ROOT}"/minus_*; do
  [[ -d "$d" ]] || continue
  name=$(basename "$d")
  if [[ -f "$d/DONE" ]]; then
    n=0
    [[ -f "$d/harness_rollouts.jsonl" ]] && n=$(wc -l < "$d/harness_rollouts.jsonl" | tr -d ' ')
    echo "DONE  ${name} n=${n}"
    n_done=$((n_done+1))
    continue
  fi
  n=0
  [[ -f "$d/harness_rollouts.jsonl" ]] && n=$(wc -l < "$d/harness_rollouts.jsonl" | tr -d ' ')
  # progress mtime
  latest="$d/logs/worker.log"
  [[ -f "$d/harness_rollouts.jsonl" ]] && latest="$d/harness_rollouts.jsonl"
  age_min=9999
  if [[ -f "$latest" ]]; then
    mtime=$(stat -c %Y "$latest")
    age_min=$(( (NOW - mtime) / 60 ))
  fi
  wpid=""
  [[ -f "$d/worker.pid" ]] && wpid=$(cat "$d/worker.pid")
  alive=0
  [[ -n "$wpid" ]] && kill -0 "$wpid" 2>/dev/null && alive=1

  if [[ "$alive" -eq 1 && "$age_min" -ge "$STUCK_MIN" && "$n" -lt "$LIMIT" ]]; then
    echo "STUCK ${name} n=${n} age_min=${age_min} pid=${wpid} — killing and requeue"
    # kill worker + vllm on its port from manifest
    port=$(python3 -c "import json;print(json.load(open('${d}/RUN_MANIFEST.json')).get('port',''))" 2>/dev/null || true)
    kill "$wpid" 2>/dev/null || true
    [[ -f "$d/vllm.pid" ]] && kill "$(cat "$d/vllm.pid")" 2>/dev/null || true
    [[ -n "$port" ]] && pkill -f "vllm serve.*--port ${port}" 2>/dev/null || true
    sleep 1
    kill -9 "$wpid" 2>/dev/null || true
    # requeue
    comp=""
    if [[ "$name" == minus_* ]]; then
      comp=${name#minus_}
    fi
    echo "${name}|${comp}" >> "${OUT_ROOT}/JOB_QUEUE.txt"
    n_fail=$((n_fail+1))
  elif [[ "$alive" -eq 1 ]]; then
    echo "RUN   ${name} n=${n}/${LIMIT} age_min=${age_min} pid=${wpid}"
    n_run=$((n_run+1))
  else
    echo "IDLE  ${name} n=${n}/${LIMIT} (no worker)"
    if [[ "$n" -lt "$LIMIT" ]]; then
      comp=""
      if [[ "$name" == minus_* ]]; then
        comp=${name#minus_}
      fi
      # only requeue if not already pending
      if ! grep -q "^${name}|" "${OUT_ROOT}/JOB_QUEUE.txt" 2>/dev/null; then
        echo "${name}|${comp}" >> "${OUT_ROOT}/JOB_QUEUE.txt"
        echo "  -> requeued"
      fi
    fi
  fi
done

# queue runners alive?
for g in 0 1 2 3 4 5 6 7; do
  pf="${OUT_ROOT}/pids/gpu${g}_queue.pid"
  if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    echo "QUEUE gpu${g} alive pid=$(cat "$pf")"
  else
    echo "QUEUE gpu${g} DEAD"
  fi
done

pending=0
[[ -f "${OUT_ROOT}/JOB_QUEUE.txt" ]] && pending=$(wc -l < "${OUT_ROOT}/JOB_QUEUE.txt" | tr -d ' ')

cat > "${OUT_ROOT}/STATUS_LIVE.md" <<EOF
# STATUS_LIVE — local_cal64_loo

- updated: $(date '+%Y-%m-%d %H:%M:%S %Z')
- done: ${n_done}
- running: ${n_run}
- stuck_killed: ${n_fail}
- queue_pending: ${pending}
- limit: ${LIMIT}
EOF

echo "summary done=${n_done} run=${n_run} stuck_killed=${n_fail} pending=${pending}"

# If all expected done, exit 0 with marker
expected=9
if [[ "${n_done}" -ge "${expected}" ]]; then
  touch "${OUT_ROOT}/ALL_DONE"
  echo "ALL_DONE"
fi
