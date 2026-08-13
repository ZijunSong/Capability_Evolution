#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-4/SCAPE"
PY="/opt/scape-hf-scorer/bin/python"
# H100-2 generated these in its worktree outputs.
HANDOFF="/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/outputs/h100_2_utility_stability/H1004_EXACT_REPLAY_HANDOFF.json"
MANIFEST="/mnt/songzijun/Capability_Evolution/SCAPE-wt-h100-2/SCAPE/outputs/utility_stability/UTILITY_COMMON128_MANIFEST.json"
OUT="$REPO/outputs/h100_4_utility_exact_replay"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" "$OUT/shards"

cd "$REPO"

while [[ ! -f "$HANDOFF" || ! -f "$MANIFEST" ]]; do
  {
    echo "[$(date -Is)] waiting for exact replay prerequisites"
    [[ -f "$HANDOFF" ]] && echo "present: $HANDOFF" || echo "missing: $HANDOFF"
    [[ -f "$MANIFEST" ]] && echo "present: $MANIFEST" || echo "missing: $MANIFEST"
  } | tee "$OUT/WAITING_FOR_H1002.md"
  sleep 300
done

{
  echo "[$(date -Is)] exact replay prerequisites present"
  echo "handoff: $HANDOFF"
  echo "manifest: $MANIFEST"
} | tee "$OUT/WAITING_FOR_H1002.md"

run_cell() {
  local gpu="$1" component="$2" k="$3" label="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_h100_4_exact_replay.py \
    --mode utility \
    --component "$component" \
    --K "$k" \
    --device cuda:0 \
    --handoff "$HANDOFF" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT" \
    --dtype bfloat16 \
    --max-prompt-tokens 4096 \
    > "$LOGDIR/${label}.log" 2>&1
}

run_noise() {
  CUDA_VISIBLE_DEVICES="3" "$PY" scripts/run_h100_4_exact_replay.py \
    --mode noise \
    --device cuda:0 \
    --handoff "$HANDOFF" \
    --manifest "$MANIFEST" \
    --out-dir "$OUT" \
    --dtype bfloat16 \
    --max-prompt-tokens 4096 \
    > "$LOGDIR/replay_noise.log" 2>&1
}

# 4-card schedule from SCAPE-0813-H100-4.md. Each GPU runs its K2 cell then K4 cell.
(
  run_cell 0 SC 2 sc_K2
  run_cell 0 SC 4 sc_K4
) & echo $! > "$LOGDIR/gpu0_sc.pid"
(
  run_cell 1 IT 2 it_K2
  run_cell 1 IT 4 it_K4
) & echo $! > "$LOGDIR/gpu1_it.pid"
(
  run_cell 2 VT 2 vt_K2
  run_cell 2 VT 4 vt_K4
) & echo $! > "$LOGDIR/gpu2_vt.pid"
(
  run_noise
) & echo $! > "$LOGDIR/gpu3_noise.pid"

status=0
for pidfile in "$LOGDIR"/gpu*.pid; do
  pid=$(cat "$pidfile")
  if ! wait "$pid"; then
    echo "[$(date -Is)] failed: $pidfile pid=$pid" | tee -a "$OUT/FAILED_JOBS.log"
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PY" scripts/finalize_h100_4_exact_replay.py --out-dir "$OUT" > "$LOGDIR/aggregation.log" 2>&1
