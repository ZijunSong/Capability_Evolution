#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/songzijun/Capability_Evolution/SCAPE
PY=/opt/scape-h1004/bin/python
BASE_OUT="$ROOT/outputs/0818_actual_baselines_novelty"
TRAIN_JSONL="$ROOT/outputs/h100_4_privilege_representation/train_paired.jsonl"
VALID_JSONL="$ROOT/outputs/h100_4_privilege_representation/valid_paired.jsonl"
TEST_JSONL="$ROOT/outputs/h100_4_privilege_representation/test_paired.jsonl"
ROUTE_OUT="$BASE_OUT/route_level_fallback"
mkdir -p "$BASE_OUT" "$ROUTE_OUT"

run_actual() {
  local method="$1"
  local seed="$2"
  local gpu="$3"
  local out="$BASE_OUT/${method,,}_seed${seed}"
  mkdir -p "$out"
  echo "[$(date -Iseconds)] start actual ${method} seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" "$ROOT/scripts/run_h1004_actual_baseline_cell.py" \
    --method "$method" --seed "$seed" --gpu "$gpu" \
    --out "$out" --train-limit 1536 --valid-limit 256 \
    >"$out/stdout.log" 2>"$out/stderr.log"
  echo "[$(date -Iseconds)] done actual ${method} seed=${seed} gpu=${gpu}"
}

run_route() {
  local cell="$1"
  local seed="$2"
  local gpu="$3"
  local out="$ROUTE_OUT/${cell,,}_seed${seed}"
  mkdir -p "$out"
  echo "[$(date -Iseconds)] start route ${cell} seed=${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" "$ROOT/scripts/run_ophsd_route_head_cell.py" \
    --cell "$cell" --seed "$seed" --gpu "$gpu" \
    --train "$TRAIN_JSONL" --valid "$VALID_JSONL" --test "$TEST_JSONL" \
    --out "$out" >"$out/stdout.log" 2>"$out/stderr.log"
  echo "[$(date -Iseconds)] done route ${cell} seed=${seed} gpu=${gpu}"
}

pids=()
run_actual OPSD_ACTION_PI 42 0 & pids+=("$!")
run_actual OPSD_ACTION_PI 43 1 & pids+=("$!")
run_actual OPHSD_FAITHFUL 42 2 & pids+=("$!")
run_actual OPHSD_FAITHFUL 43 3 & pids+=("$!")
run_actual MATCHED_TEXT_PRIVILEGE 42 4 & pids+=("$!")
run_actual MATCHED_TEXT_PRIVILEGE 43 5 & pids+=("$!")
run_route SMRC_SD_FALLBACK 42 6 route_kl & pids+=("$!")
run_route OVCSD_FALLBACK 43 7 route_kl & pids+=("$!")

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo "one or more baseline jobs failed" >&2
  exit 1
fi

echo "all baseline jobs finished"
