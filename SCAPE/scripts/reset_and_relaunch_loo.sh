#!/usr/bin/env bash
# Purge bad LOO results (Connection-error DONE) and relaunch with stagger.
set -euo pipefail
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${SCAPE_ROOT}/outputs/local_cal64_loo"

# Stop orchestrator + queues + workers (keep harness-1 download)
pkill -f 'orchestrate_h20_loop.sh' 2>/dev/null || true
pkill -f 'launch_local_cal64_loo_8gpu.sh' 2>/dev/null || true
pkill -f 'gpu[0-7]_queue' 2>/dev/null || true
pkill -f 'run_loo_worker.sh' 2>/dev/null || true
pkill -f 'rollout_harness_browsecomp.py' 2>/dev/null || true
pkill -f 'vllm serve.*--port 195' 2>/dev/null || true
sleep 3
# force leftover engine cores
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
pkill -9 -f 'vllm serve.*scape-cal64' 2>/dev/null || true
sleep 2

python - <<PY
import json, shutil
from pathlib import Path
out = Path("${OUT}")
for d in [out/"full", *sorted(out.glob("minus_*"))]:
    if not d.is_dir():
        continue
    path = d/"harness_rollouts.jsonl"
    n = n_err = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)
            err = row.get("error")
            msg = str(row.get("error_message") or row.get("exception") or "")
            if err in (True, "True", 1) or (isinstance(err, str) and err.strip()) or "Connection" in msg:
                n_err += 1
    rate = (n_err / n) if n else 1.0
    ok = n >= 64 and rate <= 0.15
    print(f"{d.name}: n={n} err={n_err} rate={rate:.2f} keep={ok}")
    if not ok:
        # archive then wipe rollout artifacts
        arch = d/"_bad_partial"
        arch.mkdir(exist_ok=True)
        for name in ("harness_rollouts.jsonl", "DONE", "harness_rollout_manifest.json"):
            p = d/name
            if p.exists():
                shutil.move(str(p), str(arch/name))
        # keep clean rows if any
        src = arch/"harness_rollouts.jsonl"
        if src.exists():
            keep=[]
            for line in src.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row=json.loads(line)
                err=row.get("error")
                msg=str(row.get("error_message") or row.get("exception") or "")
                if err in (True,"True",1) or (isinstance(err,str) and err.strip()) or "Connection" in msg:
                    continue
                keep.append(line)
            if keep:
                (d/"harness_rollouts.jsonl").write_text("\n".join(keep)+"\n", encoding="utf-8")
                print(f"  restored {len(keep)} clean rows")
PY

rm -f "${OUT}/ALL_DONE" "${OUT}/JOB_QUEUE.txt" "${OUT}/FAILED_JOBS.txt"
rm -f "${OUT}/pids/"*.pid 2>/dev/null || true

# Staggered launch: start only 4 queue workers first (gpus 0-3), then 4-7 after delay via background
export LIMIT="${LIMIT:-64}"
export MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"
export PARALLEL=1

# Patch launch to allow MAX_PARALLEL_GPUS
GPUS="0 1 2 3" bash "${SCAPE_ROOT}/scripts/launch_local_cal64_loo_8gpu.sh"

nohup bash -c "
  sleep 900
  cd '${SCAPE_ROOT}'
  GPUS='4 5 6 7' bash scripts/launch_local_cal64_loo_8gpu.sh
" > "${OUT}/logs/stagger_second_wave.log" 2>&1 &
echo "wave2_scheduler=$!"

nohup bash "${SCAPE_ROOT}/scripts/orchestrate_h20_loop.sh" > "${SCAPE_ROOT}/outputs/ORCHESTRATOR.nohup" 2>&1 &
echo "orchestrator=$!"
echo "reset complete; wave1=gpus0-3; wave2 in ~15min"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
