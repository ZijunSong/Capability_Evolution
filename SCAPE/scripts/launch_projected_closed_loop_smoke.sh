#!/bin/bash
set -u
SCAPE=/mnt/songzijun/Capability_Evolution/SCAPE
BASE=/mnt/songzijun/models/pat-jj_harness-1-full/harness-1
TR=$SCAPE/outputs/0818_projected_action_auto/training
ROOT=$SCAPE/outputs/0818_projected_action_auto/real_eval_smoke9
mkdir -p "$ROOT"
launch() {
  local g="$1" label="$2" dir="$3"
  CUDA_VISIBLE_DEVICES="$g" PYTHONPATH="$SCAPE:$SCAPE/external/harness-1:$SCAPE/external/harness-1/tinker-cookbook" \
  /opt/scape-projected-action/bin/python "$SCAPE/scripts/run_btp_auto_lora_real_closed_loop.py" \
    --base-model "$BASE" --out-dir "$ROOT/$label" \
    --query-manifest "$SCAPE/outputs/h100_2_real_closed_loop_bm25_0816/REAL_CLOSED_LOOP_PER_QUERY.jsonl" \
    --n-queries 16 --seed 8181 --max-steps 6 --device-map cuda:0 \
    --adapter "$label=$TR/$dir/lora_checkpoint" --smoke-only \
    >"$ROOT/$label.log" 2>&1 &
  echo $! > "$ROOT/$label.pid"
}
launch 0 CE42 PROJECTED_ACTION_CE_seed42
launch 1 CE43 PROJECTED_ACTION_CE_seed43
launch 2 PLUS42 PROJECTED_ACTION_CE_PLUS_NEXTTURN_KL_seed42
launch 3 SHUF42 SHUFFLED_PROJECTED_ACTION_CE_seed42
