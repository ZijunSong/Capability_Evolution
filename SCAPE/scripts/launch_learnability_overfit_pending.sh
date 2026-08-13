#!/usr/bin/env bash
# Launch pending overfit jobs only (reeval already done).
set -euo pipefail
SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${SCAPE_ROOT}/outputs/learnability_audit"
source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/data/ppnm/miniconda3/envs/bishop/bin/python
MODEL=/data/ppnm/models/harness-1
TOUR="${SCAPE_ROOT}/outputs/true_scape_candidate_b_tournament/data"
EG="${SCAPE_ROOT}/outputs/true_scape_evidence_graph/data"

run() {
  local gpu=$1 job=$2 comp=$3 shuf=$4 lr=$5 train=$6
  local out="${OUT_ROOT}/overfit/${job}"
  [[ -f "${out}/DONE" ]] && return 0
  local extra=()
  [[ "$shuf" == 1 ]] && extra+=(--shuffled-teacher)
  echo "[launch] gpu${gpu} ${job}"
  CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON} "${SCAPE_ROOT}/scripts/run_learnability_controlled_overfit.py" \
    --job-name "${job}" --component "${comp}" --train-jsonl "${train}" \
    --teacher-path "${MODEL}" --base-path "${MODEL}" --out "${out}" --gpu 0 --lr "${lr}" \
    "${extra[@]}" >"${OUT_ROOT}/logs/overfit_${job}.log" 2>&1 &
}

run 1 IT_true importance_tagging 0 1e-5 "${TOUR}/importance_tagging_TRAIN_8K.jsonl"
run 2 VT_true verify_tool 0 1e-5 "${TOUR}/verify_tool_TRAIN_8K.jsonl"
run 3 EG_narrow evidence_graph 0 1e-5 "${EG}/EG_TRAIN_8K.jsonl"
run 4 SC_shuf subtractive_curation 1 1e-5 "${TOUR}/subtractive_curation_TRAIN_8K.jsonl"
run 5 IT_shuf importance_tagging 1 1e-5 "${TOUR}/importance_tagging_TRAIN_8K.jsonl"
wait
echo "overfit batch done"
