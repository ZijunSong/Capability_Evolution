#!/usr/bin/env bash
# SCAPE-0813-Next-H20 Phase A — 8×H20 metric audit + V2 rescore (no training).
set -euo pipefail

SCAPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${SCAPE_ROOT}/outputs/0813_next_h20}"
PHASE_A="${OUT_ROOT}/phase_a"
MODEL_PATH="${MODEL_PATH:-/data/ppnm/models/harness-1}"
PYTHON_BIN="${PYTHON_BIN:-/data/ppnm/miniconda3/envs/bishop/bin/python}"
LOG_DIR="${OUT_ROOT}/logs"
PID_DIR="${OUT_ROOT}/pids"
EG_DATA="${SCAPE_ROOT}/outputs/true_scape_evidence_graph/data"
AUDIT_ROOT="${SCAPE_ROOT}/outputs/learnability_audit"

mkdir -p "${OUT_ROOT}" "${PHASE_A}" "${LOG_DIR}" "${PID_DIR}"

source /data/ppnm/miniconda3/etc/profile.d/conda.sh
conda activate bishop
export PYTHONPATH="${SCAPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# Preflight record
{
  echo "# STATUS_LIVE — 0813_next_h20 Phase A"
  echo "- started: $(date -Iseconds)"
  echo "- commit: $(git -C "${SCAPE_ROOT}" rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "- legacy_scope_path_used: false"
  echo "- LOCAL_COMPAT_ONLY: true"
  echo "- official_chroma_parity: false"
} > "${OUT_ROOT}/STATUS_LIVE.md"

cat > "${OUT_ROOT}/RUN_MANIFEST.json" <<EOF
{
  "run_id": "0813_next_h20_phase_a",
  "stage": "metric_audit",
  "repo_commit": "$(git -C "${SCAPE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "model_path": "${MODEL_PATH}",
  "legacy_scope_path_used": false,
  "LOCAL_COMPAT_ONLY": true,
  "official_chroma_parity": false
}
EOF

run_controls() {
  local gpu="$1"
  if [[ -f "${PHASE_A}/controls/DONE" ]]; then return 0; fi
  echo "[launch] controls C0-C4 on gpu${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -m pytest "${SCAPE_ROOT}/tests/test_learnability_metrics_v2.py" -v \
    >"${LOG_DIR}/controls.log" 2>&1
  touch "${PHASE_A}/controls/DONE"
}

run_reeval() {
  local gpu="$1" families="$2"
  local out="${PHASE_A}/reeval_gpu${gpu}"
  if [[ -f "${out}/DONE" ]]; then return 0; fi
  echo "[launch] historical reeval gpu${gpu} families=${families}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_learnability_historical_reeval.py" \
    --out-csv "${PHASE_A}/HISTORICAL_REEVAL.csv" \
    --teacher-path "${MODEL_PATH}" \
    --gpu 0 \
    --families ${families} \
    >"${LOG_DIR}/reeval_gpu${gpu}.log" 2>&1
  touch "${out}/DONE"
}

run_rescore() {
  local gpu="$1" families="$2"
  local out="${PHASE_A}/rescore_gpu${gpu}"
  if [[ -f "${out}/DONE" ]]; then return 0; fi
  echo "[launch] V2 rescore gpu${gpu} families=${families}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/rescore_existing_stage_l_v2.py" \
    --out-csv "${PHASE_A}/RESCORE_V2.csv" \
    --teacher-path "${MODEL_PATH}" \
    --gpu 0 \
    --families ${families} \
    >"${LOG_DIR}/rescore_gpu${gpu}.log" 2>&1
  touch "${out}/DONE"
}

run_crosscheck() {
  local gpu="$1"
  if [[ -f "${PHASE_A}/independent/DONE" ]]; then return 0; fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/run_learnability_metric_crosscheck.py" \
    --jsonl "${EG_DATA}/EG_VALID_1K.jsonl" \
    --model-path "${MODEL_PATH}" \
    --n 64 \
    --gpu 0 \
    --out "${PHASE_A}/independent/report.json" \
    >"${LOG_DIR}/independent.log" 2>&1
  # Write INDEPENDENT_METRIC_CHECK.md
  "${PYTHON_BIN}" - <<'PY' "${PHASE_A}/independent/report.json" "${PHASE_A}/INDEPENDENT_METRIC_CHECK.md"
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(
  "# INDEPENDENT_METRIC_CHECK\n\n"
  f"- gap_match_rate: {r.get('trainer_evaluator_gap_match_rate')}\n"
  f"- forward_kl_nonneg_rate: {r.get('forward_kl_nonneg_rate')}\n"
  f"- pass: {r.get('pass')}\n"
)
PY
  touch "${PHASE_A}/independent/DONE"
}

run_correlation() {
  if [[ -f "${PHASE_A}/correlation/DONE" ]]; then return 0; fi
  "${PYTHON_BIN}" "${SCAPE_ROOT}/scripts/aggregate_0813_next_h20.py" \
    --out-dir "${OUT_ROOT}" \
    --python "${PYTHON_BIN}" 2>/dev/null || true
  touch "${PHASE_A}/correlation/DONE"
}

run_gpu_queue() {
  local gpu="$1"
  local log="${LOG_DIR}/gpu${gpu}_queue.log"
  {
    echo "[$(date -Iseconds)] gpu${gpu} Phase A start"
    case "${gpu}" in
      0)
        run_reeval 0 evidence_graph
        run_rescore 0 evidence_graph evidence_graph_uniform
        ;;
      1)
        run_reeval 1 evidence_graph_weighted evidence_graph_name_only
        run_rescore 1 evidence_graph_weighted evidence_graph_name_only
        ;;
      2)
        run_reeval 2 subtractive_curation
        run_rescore 2 subtractive_curation
        ;;
      3)
        run_reeval 3 importance_tagging
        run_rescore 3 importance_tagging
        ;;
      4)
        run_reeval 4 verify_tool
        run_rescore 4 verify_tool
        ;;
      5)
        run_controls 5
        ;;
      6)
        run_correlation
        ;;
      7)
        run_crosscheck 7
        ;;
    esac
    echo "[$(date -Iseconds)] gpu${gpu} Phase A ALL_DONE"
    touch "${PHASE_A}/gpu${gpu}/ALL_DONE"
  } >>"${log}" 2>&1
}

# Copy prior audit artifacts if available
if [[ -f "${AUDIT_ROOT}/HISTORICAL_REEVAL.csv" ]] && [[ ! -f "${PHASE_A}/HISTORICAL_REEVAL.csv" ]]; then
  cp "${AUDIT_ROOT}/HISTORICAL_REEVAL.csv" "${PHASE_A}/HISTORICAL_REEVAL.csv"
fi

for g in 0 1 2 3 4 5 6 7; do
  mkdir -p "${PHASE_A}/gpu${g}" "${PHASE_A}/reeval_gpu${g}" "${PHASE_A}/rescore_gpu${g}"
  pf="${PID_DIR}/phase_a_gpu${g}.pid"
  if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    kill "$(cat "$pf")" 2>/dev/null || true
    sleep 1
  fi
done

if [[ -n "${GPU_ONLY:-}" ]]; then
  run_gpu_queue "${GPU_ONLY}"
else
  for g in 0 1 2 3 4 5 6 7; do
    run_gpu_queue "${g}" &
    echo $! >"${PID_DIR}/phase_a_gpu${g}.pid"
    echo "[bg] Phase A gpu${g} pid=$(cat "${PID_DIR}/phase_a_gpu${g}.pid")"
  done
  echo "[launch] Phase A 8-GPU queues under ${OUT_ROOT}"
fi
