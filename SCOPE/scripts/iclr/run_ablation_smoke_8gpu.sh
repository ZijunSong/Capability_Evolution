#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
DRY="${1:-}"

VARIANTS=(
  a1_same_state_on_policy
  a1_trajectory_teacher
  a3_typed_operation_o7
  a3_compact_json
  a4_full_gate
  a4_no_verification
  a8_fixed_contract_tau0
  a10_confidence_fallback
)

for i in "${!VARIANTS[@]}"; do
  v="${VARIANTS[$i]}"
  gpu="$i"
  out="outputs/iclr_ablations/smoke8/${v}/seed_42"
  cmd=(python -m experiments.common.launcher
    --experiment-id "$v"
    --seed 42
    --gpu "$gpu"
    --output-dir "$out"
    --smoke-query-limit 4
    --resume
  )
  if [[ "$DRY" == "--dry-run" ]]; then
    cmd+=(--dry-run)
  fi
  echo "[GPU$gpu] starting $v -> $out"
  if [[ "$DRY" == "--dry-run" ]]; then
    "${cmd[@]}"
  else
    # Stagger starts; do not hide exit codes
    sleep $((i * 2))
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "outputs/iclr_ablations/smoke8/${v}.log" 2>&1 &
    echo $! > "outputs/iclr_ablations/smoke8/${v}.pid"
  fi
done

if [[ "$DRY" != "--dry-run" ]]; then
  wait || true
fi
echo "smoke8 launch finished"
