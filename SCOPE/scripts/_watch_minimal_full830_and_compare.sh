#!/usr/bin/env bash
# Wait for Minimal Runtime full830 to finish, then build Phase-0 compare.
set -euo pipefail
OUT=/data/ppnm/SCOPE/outputs/minimal_runtime_browsecomp_full830
PID_FILE=$OUT/nohup_rollout.pid
LOG=$OUT/nohup_rollout.log

echo "[watch] waiting for pid=$(cat "$PID_FILE" 2>/dev/null || echo none) at $(date)"
while true; do
  if [[ -f "$OUT/summary.json" ]] && [[ -f "$OUT/episodes.jsonl" ]]; then
    n=$(wc -l < "$OUT/episodes.jsonl" | tr -d ' ')
    if [[ "$n" -ge 830 ]] && rg -q '"n": 830' "$OUT/summary.json"; then
      break
    fi
  fi
  if [[ -f "$PID_FILE" ]]; then
    pid=$(cat "$PID_FILE")
    if ! kill -0 "$pid" 2>/dev/null; then
      # process ended; give finalize a moment
      sleep 5
      break
    fi
  fi
  sleep 60
done

echo "[watch] rollout finished at $(date); building compare..."
source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
cd /data/ppnm/SCOPE
# Ensure artifacts exist even if finalize failed
if [[ ! -f "$OUT/summary.json" ]] && [[ -f "$OUT/harness_rollouts.jsonl" ]]; then
  python scripts/finalize_minimal_runtime_artifacts.py \
    --output-dir "$OUT" \
    --harness-config harness/configs/modules_minimal.yaml \
    --scope-config configs/scope/minimal_runtime.yaml
fi
python scripts/build_phase0_compare.py > "$OUT/compare_phase0_full830.build.log" 2>&1
echo "[watch] DONE at $(date)"
ls -la "$OUT/summary.json" "$OUT/compare_phase0_full830.json" "$OUT/episodes.jsonl"
