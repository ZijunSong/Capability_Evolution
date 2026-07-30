#!/usr/bin/env bash
# Resume post-train: closed-loop (skip completed shards) + wave4 merge + final report
# Uses staggered parallel launch to avoid vLLM init races, then sequential retry.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${CONDA_BASE:-/data/ppnm/miniconda3}/etc/profile.d/conda.sh"
conda activate "${BISHOP_CONDA_ENV:-bishop}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

ROOT="${REPO_ROOT}/outputs/scope_round3"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"
LOG="${ROOT}/logs/resume_post_train_$(date +%Y%m%d_%H%M%S).log"
STAGGER_SEC="${STAGGER_SEC:-90}"

exec > >(tee -a "${LOG}") 2>&1
echo "=== Resume post-train started $(date -Iseconds) ==="
echo "Log: ${LOG}  stagger=${STAGGER_SEC}s"

is_merged() {
  [[ -f "${ROOT}/closed_loop/$1/merged/summary.json" ]]
}

run_cl() {
  local gpu="$1" variant="$2" port="$3"
  local merged="${ROOT}/merged/${variant}"
  local out="${ROOT}/closed_loop/${variant}"
  local log="${ROOT}/logs/cl_${variant}.log"
  if is_merged "${variant}"; then
    echo "[skip] ${variant} already merged"
    return 0
  fi
  if [[ ! -d "${merged}" ]]; then
    echo "[error] missing merged ${variant}"
    return 1
  fi
  echo "[run] ${variant} on GPU${gpu} port ${port} ($(date -Iseconds))"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/run_closed_loop_variant.py \
    --variant "${variant}" \
    --merged-path "${merged}" \
    --gpu 0 \
    --port "${port}" \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    2>&1 | tee -a "${log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -eq 0 && -f "${out}/merged/summary.json" ]]; then
    echo "[ok] ${variant} merged"
    return 0
  fi
  echo "[fail] ${variant} exit=${rc}"
  return 1
}

# gpu:variant:port
ALL_VARIANTS=(
  "0:round3_op_main_seed42:8920"
  "1:round3_op_main_seed43:8921"
  "2:round3_op_main_seed44:8922"
  "3:round3_compact_json_sample_norm:8923"
  "4:round3_legacy_full_action_token_ce:8924"
  "5:round3_correct_only_op:8925"
  "6:round3_endorse_only_op:8926"
  "7:round3_op_no_balance:8927"
)

INCOMPLETE=()
for entry in "${ALL_VARIANTS[@]}"; do
  IFS=: read -r _gpu variant _port <<< "${entry}"
  is_merged "${variant}" || INCOMPLETE+=("${entry}")
done

echo "=== Incomplete variants: ${#INCOMPLETE[@]} ==="
for entry in "${INCOMPLETE[@]}"; do
  IFS=: read -r gpu variant port <<< "${entry}"
  echo "  GPU${gpu} ${variant} port ${port}"
done

if [[ ${#INCOMPLETE[@]} -gt 0 ]]; then
  echo "=== Wave 1: staggered parallel (delay=${STAGGER_SEC}s per GPU slot) ==="
  pids=()
  for entry in "${INCOMPLETE[@]}"; do
    IFS=: read -r gpu variant port <<< "${entry}"
    (
      delay=$((gpu * STAGGER_SEC))
      echo "[stagger] ${variant} waits ${delay}s before vLLM start"
      sleep "${delay}"
      run_cl "${gpu}" "${variant}" "${port}"
    ) &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || fail=$((fail + 1))
  done
  echo "=== Wave 1 done, failures=${fail} ==="

  echo "=== Wave 2: sequential retry on GPU0 (one vLLM at a time) ==="
  for entry in "${INCOMPLETE[@]}"; do
    IFS=: read -r _gpu variant port <<< "${entry}"
    if is_merged "${variant}"; then
      continue
    fi
    echo "[retry-seq] ${variant}"
    run_cl 0 "${variant}" "${port}" || true
    sleep 10
  done
fi

echo "=== Wave4 comparison (shard0+1 only) ==="
python training/scope_round3/wave4_compare.py --root "${ROOT}/wave4_diagnostic"

echo "=== Final report ==="
python training/scope_round3/final_report.py --root "${ROOT}"

echo "=== DONE resume post-train $(date -Iseconds) ==="
for entry in "${ALL_VARIANTS[@]}"; do
  IFS=: read -r _gpu variant _port <<< "${entry}"
  if is_merged "${variant}"; then
    echo "  [merged] ${variant}"
  else
    echo "  [MISSING] ${variant}"
  fi
done
