#!/usr/bin/env bash
# Periodic monitor: stuck kill+requeue, refresh STATUS_LIVE, aggregate when quality-complete.
set -euo pipefail
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${SCAPE_ROOT}/outputs/local_cal64_loo"
LOG="${SCAPE_ROOT}/outputs/MONITOR_DAEMON.log"
# shellcheck disable=SC1091
source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop

echo "[$(date -Iseconds)] monitor_daemon start" | tee -a "$LOG"
while true; do
  bash "${SCAPE_ROOT}/scripts/monitor_scape_loo.sh" >>"$LOG" 2>&1 || true

  # Quality-aware all-done check
  python - <<'PY' >>"$LOG" 2>&1 || true
import json
from pathlib import Path
out = Path("/data/ppnm/Capability_Evolution/SCAPE/outputs/local_cal64_loo")
jobs = [out/"full", *sorted(out.glob("minus_*"))]
ok = 0
for d in jobs:
    p = d/"harness_rollouts.jsonl"
    if not p.exists():
        continue
    n=n_err=0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        row=json.loads(line)
        err=row.get("error")
        msg=str(row.get("error_message") or row.get("exception") or "")
        if err in (True,"True",1) or (isinstance(err,str) and err.strip()) or "Connection" in msg:
            n_err += 1
    if n>=64 and (n_err/n)<=0.15:
        ok += 1
        (d/"DONE").write_text("ok\n", encoding="utf-8")
print(f"quality_complete_jobs={ok}/{len(jobs)}", flush=True)
if ok >= 9:
    (out/"ALL_DONE").write_text("ok\n", encoding="utf-8")
    print("ALL_DONE_QUALITY", flush=True)
PY

  if [[ -f "${OUT}/ALL_DONE" ]]; then
    echo "[$(date -Iseconds)] ALL_DONE — aggregating" | tee -a "$LOG"
    cd "${SCAPE_ROOT}"
    python scripts/aggregate_loo_and_select.py >>"$LOG" 2>&1 || true
    OUT_ROOT="${SCAPE_ROOT}/outputs/stage_l" bash scripts/launch_stage_l_8gpu.sh >>"$LOG" 2>&1 || true
    echo "[$(date -Iseconds)] post-LOO steps kicked; daemon idling 10m" | tee -a "$LOG"
    sleep 600
    continue
  fi

  # Relaunch dead wave queues if work remains
  pending=0
  [[ -f "${OUT}/JOB_QUEUE.txt" ]] && pending=$(wc -l < "${OUT}/JOB_QUEUE.txt" | tr -d ' ')
  if [[ "$pending" -gt 0 ]]; then
    for g in 0 1 2 3 4 5 6 7; do
      pf="${OUT}/pids/gpu${g}_queue.pid"
      if [[ ! -f "$pf" ]] || ! kill -0 "$(cat "$pf")" 2>/dev/null; then
        echo "[$(date -Iseconds)] relaunch gpu${g} queue" | tee -a "$LOG"
        GPUS="$g" bash "${SCAPE_ROOT}/scripts/launch_local_cal64_loo_8gpu.sh" >>"$LOG" 2>&1 || true
      fi
    done
  fi

  # Refresh top-level STATUS_LIVE
  {
    echo "# STATUS_LIVE — SCAPE H20"
    echo
    echo "- updated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "- loo_dir: ${OUT}"
    echo "- pending_queue: ${pending}"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/- gpu: /'
    echo
    echo "## LOO jobs"
    for d in "${OUT}"/full "${OUT}"/minus_*; do
      [[ -d "$d" ]] || continue
      n=0; e=0
      if [[ -f "$d/harness_rollouts.jsonl" ]]; then
        n=$(wc -l < "$d/harness_rollouts.jsonl" | tr -d ' ')
        e=$(grep -c 'Connection\|"error": true\|"error":true' "$d/harness_rollouts.jsonl" || true)
      fi
      echo "- $(basename "$d"): n=${n} errish≈${e} done=$([[ -f $d/DONE ]] && echo yes || echo no)"
    done
  } > "${SCAPE_ROOT}/outputs/STATUS_LIVE.md"

  sleep 180
done
