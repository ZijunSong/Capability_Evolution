#!/usr/bin/env bash
# Learnability metric audit — 8×H20 parallel per SCAPE-0813-H20 §5–6.
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/learnability_audit}"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/harness-1}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${OUT_ROOT}/logs"
PID_DIR="${OUT_ROOT}/pids"
EG_DATA="${SCAPE_ROOT}/outputs/true_scape_evidence_graph/data"
TOUR_DATA="${SCAPE_ROOT}/outputs/true_scape_candidate_b_tournament/data"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${PID_DIR}" \
  "${OUT_ROOT}/reeval" "${OUT_ROOT}/overfit" "${OUT_ROOT}/crosscheck" \
  "${SCAPE_ROOT}/audit/learnability-metric-20260813"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# Static audit docs
cat > "${OUT_ROOT}/METRIC_DEFINITIONS.md" <<'EOF'
# METRIC_DEFINITIONS

| ID | Name | Formula | Range |
|----|------|---------|-------|
| M1 | forward_KL | KL(T \|\| S) per token, masked mean | >= 0 |
| M2 | reverse_KL | KL(S \|\| T) | >= 0 |
| M3 | JS | Jensen-Shannon | >= 0 |
| M4 | signed_gap | mean(log p_T(tok) - log p_S(tok)) teacher-forced | may be negative |

Legacy `div` / `D_pre` / `D_post` = M4 signed_gap, NOT KL. Do not call divergence.
Numeric floor: -1e-7.
EOF

cat > "${OUT_ROOT}/CODE_PATH_AUDIT.md" <<'EOF'
# CODE_PATH_AUDIT

## Finding: legacy `div` is signed log-prob gap, not KL

`scape/training/hf_tool_opd.py::score_divergence` computes:
`kl_tok = teacher_lp - student_lp` (teacher-forced token logprob difference).

This equals M4 `signed_gap`, NOT M1 forward KL. It can be negative when student assigns
higher probability to the teacher token than the teacher model does.

## Canonical metrics (new)

`scape/training/canonical_metrics.py` implements true vocab-level KL/JS via logits.

`score_canonical_metrics` in hf_tool_opd uses dual-prompt scoring:
- teacher forward: full harness prompt on frozen base teacher
- student forward: reduced prompt on student checkpoint

## Trainer

`train_step` optimizes masked mean(teacher_lp - student_lp) + anchor CE — same signed gap.

## Held-out evaluator

`mean_divergence` / `aggregate_candidate_b_tournament.py` use legacy `div`.

Historical negative D_pre values are METRIC_NAMING_BUG + METRIC_SIGN_BUG, not necessarily true negative learnability.
EOF

run_reeval() {
  local gpu="$1" families="$2"
  local out="${OUT_ROOT}/reeval/gpu${gpu}"
  mkdir -p "${out}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] reeval gpu${gpu}"
    return 0
  fi
  echo "[launch] reeval gpu${gpu} families=${families}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_learnability_historical_reeval.py" \
    --out-csv "${OUT_ROOT}/HISTORICAL_REEVAL.csv" \
    --teacher-path "${MODEL_PATH}" \
    --gpu 0 \
    --families ${families} \
    >"${LOG_DIR}/reeval_gpu${gpu}.log" 2>&1
  touch "${out}/DONE"
}

run_overfit() {
  local gpu="$1" job="$2" component="$3" shuffled="$4" lr="$5" train_jsonl="${6:-}"
  local out="${OUT_ROOT}/overfit/${job}"
  if [[ -f "${out}/DONE" ]]; then
    echo "[skip] overfit ${job}"
    return 0
  fi
  local train="${train_jsonl:-${TOUR_DATA}/${component}_TRAIN_8K.jsonl}"
  local extra=()
  if [[ "${shuffled}" == "1" ]]; then
    extra+=(--shuffled-teacher)
  fi
  echo "[launch] overfit gpu${gpu} job=${job}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_learnability_controlled_overfit.py" \
    --job-name "${job}" \
    --component "${component}" \
    --train-jsonl "${train}" \
    --teacher-path "${MODEL_PATH}" \
    --base-path "${MODEL_PATH}" \
    --out "${out}" \
    --gpu 0 \
    --lr "${lr}" \
    "${extra[@]}" \
    >"${LOG_DIR}/overfit_${job}.log" 2>&1
}

run_mask_audit() {
  local gpu=6
  local out="${OUT_ROOT}/mask_audit"
  if [[ -f "${out}/DONE" ]]; then return 0; fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_learnability_mask_audit.py" \
    --jsonl "${EG_DATA}/EG_VALID_1K.jsonl" \
    --model-path "${MODEL_PATH}" \
    --n 512 \
    --gpu 0 \
    --out "${out}/report.json" \
    >"${LOG_DIR}/mask_audit.log" 2>&1
  touch "${out}/DONE"
}

run_crosscheck() {
  local gpu=7
  if [[ -f "${OUT_ROOT}/crosscheck/DONE" ]]; then return 0; fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_learnability_metric_crosscheck.py" \
    --jsonl "${EG_DATA}/EG_VALID_1K.jsonl" \
    --model-path "${MODEL_PATH}" \
    --n 64 \
    --gpu 0 \
    --out "${OUT_ROOT}/crosscheck/report.json" \
    >"${LOG_DIR}/crosscheck.log" 2>&1
  touch "${OUT_ROOT}/crosscheck/DONE"
}

run_manual_kl() {
  local gpu=5
  mkdir -p "${OUT_ROOT}/manual_kl"
  if [[ -f "${OUT_ROOT}/manual_kl/DONE" ]]; then return 0; fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -m pytest "${SCAPE_ROOT}/tests/test_learnability_metrics.py" -v -k manual \
    >"${LOG_DIR}/manual_kl.log" 2>&1 || \
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -m pytest "${SCAPE_ROOT}/tests/test_learnability_metrics.py::test_manual_logits_match_torch_reference" -v \
    >"${LOG_DIR}/manual_kl.log" 2>&1
  touch "${OUT_ROOT}/manual_kl/DONE"
}

run_unit_tests_cpu() {
  mkdir -p "${OUT_ROOT}/unit_tests"
  if [[ -f "${OUT_ROOT}/unit_tests/DONE" ]]; then return 0; fi
  "${PYTHON_BIN}" -m pytest "${SCAPE_ROOT}/tests/test_learnability_metrics.py" -v \
    >"${LOG_DIR}/unit_tests.log" 2>&1
  touch "${OUT_ROOT}/unit_tests/DONE"
}

run_gpu_queue() {
  local gpu="$1"
  local log="${LOG_DIR}/gpu${gpu}_queue.log"
  {
    echo "[$(date -Iseconds)] gpu${gpu} audit queue start"
    case "${gpu}" in
      0)
        run_reeval 0 evidence_graph evidence_graph_uniform
        run_overfit 0 SC_true subtractive_curation 0 1e-5
        ;;
      1)
        run_reeval 1 evidence_graph_weighted evidence_graph_name_only
        run_overfit 1 IT_true importance_tagging 0 1e-5
        ;;
      2)
        run_reeval 2 subtractive_curation
        run_overfit 2 VT_true verify_tool 0 1e-5
        ;;
      3)
        run_reeval 3 importance_tagging
        run_overfit 3 EG_narrow evidence_graph 0 1e-5 "${EG_DATA}/EG_TRAIN_8K.jsonl"
        ;;
      4)
        run_reeval 4 verify_tool
        run_overfit 4 SC_shuf subtractive_curation 1 1e-5
        ;;
      5)
        run_manual_kl
        run_overfit 5 IT_shuf importance_tagging 1 1e-5
        ;;
      6)
        run_mask_audit
        run_overfit 6 VT_shuf verify_tool 1 1e-5
        ;;
      7)
        run_crosscheck
        for lr in 1e-6 3e-6 1e-5; do
          run_overfit 7 "SC_lr_${lr}" subtractive_curation 0 "${lr}"
        done
        ;;
    esac
    echo "[$(date -Iseconds)] gpu${gpu} audit queue ALL_DONE"
    touch "${OUT_ROOT}/gpu${gpu}/ALL_DONE"
  } >>"${log}" 2>&1
}

# Cleanup stale workers
for g in 0 1 2 3 4 5 6 7; do
  pf="${PID_DIR}/gpu${g}.pid"
  if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    echo "[cleanup] killing stale gpu${g} pid=$(cat "$pf")"
    kill "$(cat "$pf")" 2>/dev/null || true
  fi
  mkdir -p "${OUT_ROOT}/gpu${g}"
done
sleep 2

run_unit_tests_cpu

if [[ -z "${GPU_ONLY:-}" ]]; then
  for g in 0 1 2 3 4 5 6 7; do
    run_gpu_queue "${g}" &
    echo $! >"${PID_DIR}/gpu${g}.pid"
    echo "[bg] gpu${g} pid=$(cat "${PID_DIR}/gpu${g}.pid")"
  done
  echo "[launch] 8 GPU learnability audit queues under ${OUT_ROOT}"
else
  run_gpu_queue "${GPU_ONLY}"
  echo "[launch] GPU ${GPU_ONLY} only"
fi
