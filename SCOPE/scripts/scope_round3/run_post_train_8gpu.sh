#!/usr/bin/env bash
# Post Barrier-C: merge + offline eval + closed-loop 100q (8 GPU parallel)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${CONDA_BASE:-/data/ppnm/miniconda3}/etc/profile.d/conda.sh"
conda activate "${BISHOP_CONDA_ENV:-bishop}"
export PYTHONPATH="${REPO_ROOT}"
cd "${REPO_ROOT}"

ROOT="${REPO_ROOT}/outputs/scope_round3"
BASE="/data/ppnm/models/Qwen2.5-7B-Instruct"
MANIFEST="${REPO_ROOT}/artifacts/datasets/round2_audit_100q/query_manifest.json"

mkdir -p "${ROOT}/merged" "${ROOT}/closed_loop" "${ROOT}/eval" "${ROOT}/logs"

echo "=== Merge + offline eval (all variants) ==="
if [[ ! -f "${ROOT}/eval/offline_capability.json" ]]; then
  python training/scope_round3/run_offline_eval_all.py
else
  echo "[skip] offline_capability.json exists"
fi

run_cl() {
  local gpu="$1" variant="$2" port="$3"
  local merged="${ROOT}/merged/${variant}"
  local out="${ROOT}/closed_loop/${variant}"
  local log="${ROOT}/logs/cl_${variant}.log"
  [[ -f "${out}/merged/summary.json" ]] && echo "[skip] ${variant}" && return 0
  [[ -d "${merged}" ]] || { echo "missing merged ${variant}"; return 1; }
  CUDA_VISIBLE_DEVICES="${gpu}" python training/scope_round3/run_closed_loop_variant.py \
    --variant "${variant}" \
    --merged-path "${merged}" \
    --gpu 0 \
    --port "${port}" \
    --output-dir "${out}" \
    --manifest "${MANIFEST}" \
    2>&1 | tee "${log}"
}

echo "=== Base closed-loop (dup operation, GPU0) — run first ==="
if [[ ! -f "${ROOT}/closed_loop/base/merged/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 python training/scope_round3/run_closed_loop_variant.py \
    --variant base \
    --merged-path "${BASE}" \
    --gpu 0 --port 8910 \
    --output-dir "${ROOT}/closed_loop/base" \
    --manifest "${MANIFEST}" \
    2>&1 | tee "${ROOT}/logs/cl_base.log"
fi

echo "=== Closed-loop 8 variants (parallel, GPU0-7) ==="
run_cl 0 round3_op_main_seed42 8920 &
run_cl 1 round3_op_main_seed43 8921 &
run_cl 2 round3_op_main_seed44 8922 &
run_cl 3 round3_compact_json_sample_norm 8923 &
run_cl 4 round3_legacy_full_action_token_ce 8924 &
run_cl 5 round3_correct_only_op 8925 &
run_cl 6 round3_endorse_only_op 8926 &
run_cl 7 round3_op_no_balance 8927 &
wait

echo "=== Wave4 (4 checkpoints, re-run) ==="
PHASE=wave4 bash "${REPO_ROOT}/scripts/scope_round3/run_all_8gpu.sh" || true

echo "=== Final report ==="
python training/scope_round3/final_report.py --root "${ROOT}"
echo "DONE post-train pipeline"
