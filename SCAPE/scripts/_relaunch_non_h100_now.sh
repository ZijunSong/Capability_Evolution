#!/usr/bin/env bash
# One-shot relaunch: A/B collect + S2/S3 closed-loop. Safe to re-run.
set -euo pipefail
SCAPE="$(cd "$(dirname "$0")/.." && pwd)"
COL="$SCAPE/outputs/stage_l_hminus_data"
OUT_S="$SCAPE/outputs/stage_s/B_verify_fourgrid"
HF="$SCAPE/outputs/stage_l/B_verify_opd_provisional/L64_seed42_hf/hf_model"
MODEL="${MODEL_PATH:-/data/ppnm/models/Qwen2.5-7B-Instruct}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop

mkdir -p "$COL/logs" "$COL/pids" "$OUT_S/logs" "$OUT_S/pids"

# Stop stale completion loop (will restart below with patched script)
if [[ -f "$SCAPE/outputs/COMPLETE_NON_H100.pid" ]]; then
  old="$(cat "$SCAPE/outputs/COMPLETE_NON_H100.pid" || true)"
  if [[ -n "${old:-}" ]]; then kill "$old" 2>/dev/null || true; fi
fi
kill 3705801 2>/dev/null || true

# Stop any stale workers on our ports / job names
for port in 19500 19501 19502 19503; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 2

# Purge connection-error rows from collect jsonl (keep good uniques)
python - <<'PY'
import json
from pathlib import Path
base = Path("/data/ppnm/Capability_Evolution/SCAPE/outputs/stage_l_hminus_data")
for name in ("A_auto_populate_first_search", "B_verify_tool"):
    p = base / name / "harness_rollouts.jsonl"
    if not p.exists():
        print(name, "missing")
        continue
    keep = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        err = r.get("error")
        msg = str(r.get("error_message") or r.get("exception") or "")
        if err in (True, "True", 1) or (isinstance(err, str) and err.strip()) or "Connection" in msg:
            continue
        keep.append(line)
    p.write_text(("\n".join(keep) + ("\n" if keep else "")), encoding="utf-8")
    print(name, "kept", len(keep))
    # clear DONE so worker continues
    (base / name / "DONE").unlink(missing_ok=True)
PY

# Ensure tokenizer assets on HF student
for f in vocab.json merges.txt; do
  if [[ -f /data/ppnm/models/Qwen2.5-7B-Instruct/$f && ! -f "$HF/$f" ]]; then
    cp "/data/ppnm/models/Qwen2.5-7B-Instruct/$f" "$HF/$f"
  fi
done

launch_worker() {
  local gpu="$1" job="$2" comp="$3" out_root="$4" limit="$5" split="$6" model="$7" log="$8" pidf="$9"
  # skip if already healthy
  if [[ -f "${out_root}/${job}/DONE" ]]; then
    echo "skip $job DONE"
    return 0
  fi
  if [[ -f "$pidf" ]] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "skip $job already running pid=$(cat "$pidf")"
    return 0
  fi
  nohup env GPU="$gpu" JOB_NAME="$job" COMPONENT="$comp" \
    OUT_ROOT="$out_root" LIMIT="$limit" SPLIT="$split" MODEL_PATH="$model" \
    bash "$SCAPE/scripts/run_loo_worker.sh" >"$log" 2>&1 &
  echo $! >"$pidf"
  echo "launched $job gpu=$gpu pid=$(cat "$pidf")"
}

launch_worker 0 A_auto_populate_first_search auto_populate_first_search \
  "$COL" 512 train "$MODEL" \
  "$COL/logs/A_relaunch.log" "$COL/pids/A_auto_populate_first_search.pid"

launch_worker 1 B_verify_tool verify_tool \
  "$COL" 512 train "$MODEL" \
  "$COL/logs/B_relaunch.log" "$COL/pids/B_verify_tool.pid"

launch_worker 2 S2_trained_minus_verify verify_tool \
  "$OUT_S" 64 test "$HF" \
  "$OUT_S/logs/S2_launch.log" "$OUT_S/pids/S2.pid"

launch_worker 3 S3_trained_full "" \
  "$OUT_S" 64 test "$HF" \
  "$OUT_S/logs/S3_launch.log" "$OUT_S/pids/S3.pid"

# Aggregator
if [[ ! -f "$OUT_S/pids/aggregate.pid" ]] || ! kill -0 "$(cat "$OUT_S/pids/aggregate.pid")" 2>/dev/null; then
  nohup /data/ppnm/miniconda3/envs/bishop/bin/python \
    "$SCAPE/scripts/aggregate_stage_s_closed_loop.py" \
    >"$OUT_S/logs/aggregate_closed_loop.log" 2>&1 &
  echo $! >"$OUT_S/pids/aggregate.pid"
  echo "aggregate pid=$(cat "$OUT_S/pids/aggregate.pid")"
fi

# Restart completion loop with patched script
nohup bash "$SCAPE/scripts/complete_non_h100_loop.sh" \
  >"$SCAPE/outputs/COMPLETE_NON_H100.nohup" 2>&1 &
echo $! | tee "$SCAPE/outputs/COMPLETE_NON_H100.pid"
echo "complete_loop pid=$(cat "$SCAPE/outputs/COMPLETE_NON_H100.pid")"

cat > "$OUT_S/CLOSED_LOOP_STATUS.md" <<EOF
# CLOSED_LOOP_STATUS — B verify four-grid (S2/S3)

- updated: $(date '+%Y-%m-%d %H:%M:%S %Z')
- verdict: **RUNNING_CLOSED_LOOP**
- student_ckpt: \`$HF\`
- S2: GPU2 pid=$(cat "$OUT_S/pids/S2.pid")
- S3: GPU3 pid=$(cat "$OUT_S/pids/S3.pid")
- A collect: GPU0 pid=$(cat "$COL/pids/A_auto_populate_first_search.pid")
- B collect: GPU1 pid=$(cat "$COL/pids/B_verify_tool.pid")
EOF

echo "waiting 120s for vLLM boot..."
sleep 120
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for port in 19500 19501 19502 19503; do
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/v1/models" || echo fail)
  echo "port ${port} -> ${code}"
done
for f in \
  "$COL/A_auto_populate_first_search/logs/worker.log" \
  "$COL/B_verify_tool/logs/worker.log" \
  "$OUT_S/S2_trained_minus_verify/logs/worker.log" \
  "$OUT_S/S3_trained_full/logs/worker.log"; do
  echo "==== $f ===="
  tail -6 "$f" 2>/dev/null || echo missing
done
